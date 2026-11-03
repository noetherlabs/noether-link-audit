"""End-to-end sample: simulate passes, write logs, audit the logs, validate against truth."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy  # type: ignore[import-untyped]
import yaml

from .attribution import Attribution, LossModel, shapley
from .audit import Auditor, AuditSettings, PassFeatures
from .coupling_tables import TIP_TILT, load_or_build
from .simulate import (
    CAUSES,
    LinkConfig,
    PassTruth,
    read_parquet,
    scenario_passes,
    simulate_pass,
    write_parquet,
)
from .station import StationConfig

FIX_CAUSES = {
    "fresh_ephemeris": frozenset({"ephemeris"}),
    "keyhole_tracking": frozenset({"keyhole"}),
    "vibration_fix": frozenset({"vibration"}),
    "refocus": frozenset({"static_optics"}),
}
SOURCE_FILES = (
    "projects/link-audit/src/noether_link_audit/station.py",
    "projects/link-audit/src/noether_link_audit/coupling_tables.py",
    "projects/link-audit/src/noether_link_audit/simulate.py",
    "projects/link-audit/src/noether_link_audit/audit.py",
    "projects/link-audit/src/noether_link_audit/attribution.py",
    "projects/link-audit/src/noether_link_audit/pipeline.py",
    "projects/link-audit/src/noether_link_audit/report.py",
    "packages/noether_physics/src/noether_physics/coupling.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truth_model(truth: PassTruth, station: StationConfig) -> LossModel:
    return LossModel(
        rate_gbps=station.data_rate_gbps,
        sample_s=1.0 / truth.fast_rate_hz,
        sensitivity_dbm=station.sensitivity_dbm,
        base_dbm=truth.would_be_fiber_dbm,
        locked=truth.locked,
        additive=dict(truth.additive_db),
        groups=[truth.tilt_group_db, truth.optics_group_db],
        restore=dict(truth.restore),
    )


def _fix_gains(model: LossModel, fixes: dict[str, np.ndarray], baseline: float) -> dict[str, float]:
    gains = {name: model.delivered_gb(removed) - baseline for name, removed in FIX_CAUSES.items()}
    gains["faster_tilt_loop"] = model.delivered_gb(extra_db=fixes["faster_tilt_loop"]) - baseline
    gains["all_practical_fixes"] = (
        model.delivered_gb(frozenset().union(*FIX_CAUSES.values()), extra_db=fixes["faster_tilt_loop"]) - baseline
    )
    gains["low_order_ao_upper_bound"] = model.delivered_gb(extra_db=fixes["low_order_ao"]) - baseline
    return gains


def _attribution_dict(a: Attribution) -> dict[str, Any]:
    return {
        "possible_gb": a.possible_gb,
        "delivered_gb": a.delivered_gb,
        "all_causes_removed_gb": a.all_removed_gb,
        "link_budget_residual_gb": a.link_budget_residual_gb,
        "shapley_gb": a.shapley_gb,
    }


def run_sample(config_path: Path, output: Path, work: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text())
    root = Path(__file__).resolve().parents[4]
    station = StationConfig.from_dict(raw.get("station", {}))
    link = LinkConfig(**raw.get("link", {}))
    settings = AuditSettings(**raw.get("audit", {}))
    specs = scenario_passes(raw)
    table_cfg = raw.get("coupling_tables", {})
    defocus = tuple(sorted({s.defocus_waves_rms for s in specs if s.defocus_waves_rms}))
    output.mkdir(parents=True, exist_ok=True)
    tables = load_or_build(
        output / "coupling_tables.json",
        station,
        defocus_waves_rms=defocus,
        screens=int(table_cfg.get("screens", 400)),
        grid_size=int(table_cfg.get("grid_size", 128)),
        oversampling=int(table_cfg.get("oversampling", 8)),
        outer_scale_m=link.outer_scale_m,
        seed=int(table_cfg.get("seed", 2026)),
    )

    telemetry_dir = work / "telemetry"
    truth_rows: dict[str, dict[str, Any]] = {}
    truth_slow: dict[str, dict[str, Any]] = {}
    telemetry_hashes: dict[str, str] = {}
    for spec in specs:
        fast, slow, truth = simulate_pass(spec, station, link, tables)
        fast_path = telemetry_dir / f"{spec.pass_id}_fast.parquet"
        slow_path = telemetry_dir / f"{spec.pass_id}_slow.parquet"
        write_parquet(fast_path, fast)
        write_parquet(slow_path, slow)
        telemetry_hashes[fast_path.name] = _sha256(fast_path)
        telemetry_hashes[slow_path.name] = _sha256(slow_path)
        model = _truth_model(truth, station)
        attribution = shapley(model, CAUSES)
        fixes = {"faster_tilt_loop": truth.bandwidth_fix_db, "low_order_ao": truth.low_order_ao_fix_db}
        truth_rows[spec.pass_id] = {
            "attribution": _attribution_dict(attribution),
            "fix_gains_gb": _fix_gains(model, fixes, attribution.delivered_gb),
            "parameters": truth.parameters,
            "episodes": [
                {k: v for k, v in e.items() if k in ("trigger", "start", "end", "required", "error", "truncated")}
                for e in truth.episodes
            ],
            "median_d_over_r0": float(np.median(truth.slow["d_over_r0"])),
            "median_fried_parameter_m": float(np.median(truth.slow["fried_parameter_m"])),
            "mean_cirrus_db": float(np.mean(truth.slow["cirrus_db"])),
            "max_cirrus_db": float(np.max(truth.slow["cirrus_db"])),
        }
        truth_slow[spec.pass_id] = {k: np.asarray(v) for k, v in truth.slow.items()}
        del model, truth

    # The audit reads back only the written logs and the declared station configuration.
    auditor = Auditor(station, tables, settings)
    features: list[PassFeatures] = []
    for spec in specs:
        fast = read_parquet(telemetry_dir / f"{spec.pass_id}_fast.parquet")
        slow = read_parquet(telemetry_dir / f"{spec.pass_id}_slow.parquet")
        features.append(auditor.extract(spec.pass_id, spec.night_id, fast, slow))
    calibration = auditor.calibrate(features)
    audit_rows: dict[str, dict[str, Any]] = {}
    for f in features:
        model, fixes, estimates = auditor.loss_model(f, calibration)
        attribution = shapley(model, CAUSES)
        audit_rows[f.pass_id] = {
            "attribution": _attribution_dict(attribution),
            "fix_gains_gb": _fix_gains(model, fixes, attribution.delivered_gb),
            "estimates": estimates,
            "timing": f.timing,
            "vibration": f.vibration,
            "episodes": [
                {
                    "trigger": e["trigger"],
                    "start_s": e["start"] / 10.0,
                    "stop_s": e["stop"] / 10.0,
                    "active_s": e["active_s"],
                    "predicted_error_rad": e.get("predicted_error_rad"),
                    "truncated": e["truncated"],
                }
                for e in f.episodes
            ],
        }
        del model, fixes

    # Sensitivity: the same audit assuming Kolmogorov turbulence (no outer scale).
    kolmogorov = Auditor(station, tables, replace(settings, assumed_outer_scale_m=math.inf))
    alt_features = [
        kolmogorov.extract(
            spec.pass_id,
            spec.night_id,
            read_parquet(telemetry_dir / f"{spec.pass_id}_fast.parquet"),
            read_parquet(telemetry_dir / f"{spec.pass_id}_slow.parquet"),
        )
        for spec in specs
    ]
    alt_calibration = kolmogorov.calibrate(alt_features)
    alt_ho, alt_ratio = 0.0, []
    for f, base in zip(alt_features, features, strict=True):
        model, _, estimates = kolmogorov.loss_model(f, alt_calibration)
        alt_ho += shapley(model, CAUSES).shapley_gb["higher_order_turbulence"]
        alt_ratio.append(estimates["median_fried_parameter_m"] / audit_rows[base.pass_id]["estimates"]["median_fried_parameter_m"])
        del model
    sensitivity = {
        "higher_order_gb_assumed": sum(r["attribution"]["shapley_gb"]["higher_order_turbulence"] for r in audit_rows.values()),
        "higher_order_gb_infinite_outer_scale": alt_ho,
        "r0_ratio_infinite_vs_assumed": float(np.median(alt_ratio)),
    }

    sources = {path: _sha256(root / path) for path in SOURCE_FILES}
    payload: dict[str, Any] = {
        "schema": "link_audit_sample_v1",
        "result_kind": "simulation",
        "scope": "synthetic_telemetry_from_reduced_order_model_of_hypothetical_station",
        "experiment_id": raw["experiment_id"],
        "station": station.metadata(),
        "link": asdict(link),
        "audit_settings": asdict(settings),
        "passes": [
            {
                "pass_id": s.pass_id,
                "night_id": s.night_id,
                "max_elevation_deg": s.max_elevation_deg,
                "planted": s.planted(),
                "truth": truth_rows[s.pass_id],
                "audit": audit_rows[s.pass_id],
            }
            for s in specs
        ],
        "calibration": asdict(calibration),
        "coupling_model": {
            "eta0": tables.eta0,
            "theta_c_rad": tables.theta_c_rad,
            "tilt_fit_max_relative_error": tables.tilt_fit_max_relative_error,
            "static_relative": tables.static_relative,
            "d_over_r0_grid": tables.d_over_r0.tolist(),
            "mean_log_db_tip_tilt": tables.mean_log_db[TIP_TILT].tolist(),
            "diagnostics": tables.diagnostics,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_sha256": sources,
        "config_sha256": _sha256(config_path),
        "telemetry_sha256": telemetry_hashes,
    }
    payload["outer_scale_sensitivity"] = sensitivity
    payload["summary"] = summarize(payload)
    _json_safe(payload)
    from .report import write_report

    write_report(payload, features, truth_slow, output)
    (output / "metrics.json").write_text(json.dumps(payload, indent=1, sort_keys=False, allow_nan=False) + "\n")
    manifest = {
        "result_kind": "simulation",
        "metrics_sha256": _sha256(output / "metrics.json"),
        "config_sha256": payload["config_sha256"],
        "coupling_tables_sha256": _sha256(output / "coupling_tables.json"),
        "report_sha256": _sha256(output / "REPORT.md"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return payload


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    def fleet(kind: str) -> dict[str, Any]:
        rows = [p[kind] for p in payload["passes"]]
        possible = sum(r["attribution"]["possible_gb"] for r in rows)
        delivered = sum(r["attribution"]["delivered_gb"] for r in rows)
        shap = {c: sum(r["attribution"]["shapley_gb"][c] for r in rows) for c in CAUSES}
        residual = sum(r["attribution"]["link_budget_residual_gb"] for r in rows)
        fixes = {k: sum(r["fix_gains_gb"][k] for r in rows) for k in rows[0]["fix_gains_gb"]}
        return {
            "possible_gb": possible,
            "delivered_gb": delivered,
            "lost_gb": possible - delivered,
            "delivered_fraction": delivered / possible,
            "shapley_gb": shap,
            "link_budget_residual_gb": residual,
            "fix_gains_gb": fixes,
        }

    audit, truth = fleet("audit"), fleet("truth")
    comparison = {}
    for cause in CAUSES:
        a, t = audit["shapley_gb"][cause], truth["shapley_gb"][cause]
        comparison[cause] = {"audit_gb": a, "truth_gb": t, "difference_gb": a - t}
    fix_comparison = {
        k: {"audit_gb": audit["fix_gains_gb"][k], "truth_gb": truth["fix_gains_gb"][k]} for k in audit["fix_gains_gb"]
    }
    per_pass_abs = [
        abs(p["audit"]["attribution"]["shapley_gb"][c] - p["truth"]["attribution"]["shapley_gb"][c])
        for p in payload["passes"]
        for c in CAUSES
    ]
    top_match = 0
    for p in payload["passes"]:
        a = max(CAUSES, key=lambda c: p["audit"]["attribution"]["shapley_gb"][c])
        t = max(CAUSES, key=lambda c: p["truth"]["attribution"]["shapley_gb"][c])
        top_match += int(a == t)
    return {
        "audit": audit,
        "truth": truth,
        "cause_comparison": comparison,
        "fix_comparison": fix_comparison,
        "per_pass_cause_abs_error_gb": {
            "max": float(max(per_pass_abs)),
            "mean": float(np.mean(per_pass_abs)),
        },
        "top_cause_agreement": {"passes": len(payload["passes"]), "matches": top_match},
    }


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy types and replace non-finite floats with None, in place."""
    if isinstance(value, dict):
        for key in list(value):
            value[key] = _json_safe(value[key])
        return value
    if isinstance(value, list):
        for i, item in enumerate(value):
            value[i] = _json_safe(item)
        return value
    if isinstance(value, np.floating | float):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value
