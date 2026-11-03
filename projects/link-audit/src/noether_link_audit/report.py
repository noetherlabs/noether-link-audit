"""Figures and REPORT.md for the sample audit. Every number is read from the payload."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .audit import PassFeatures  # noqa: E402
from .simulate import CAUSES  # noqa: E402

CAUSE_LABELS = {
    "higher_order_turbulence": "Higher-order turbulence",
    "tilt_residual": "Tip-tilt residual",
    "vibration": "Structural vibration",
    "static_optics": "Static coupling loss",
    "cloud": "Cloud and cirrus",
    "scintillation": "Scintillation",
    "ephemeris": "Stale-ephemeris search",
    "keyhole": "Keyhole at the mount rate limit",
    "nominal_acquisition": "Nominal acquisition",
}
COLORS = {
    "delivered": "#2b6cb0",
    "higher_order_turbulence": "#6b46c1",
    "tilt_residual": "#d69e2e",
    "vibration": "#dd6b20",
    "static_optics": "#2f855a",
    "cloud": "#718096",
    "scintillation": "#b7c3d0",
    "ephemeris": "#c53030",
    "keyhole": "#b83280",
    "nominal_acquisition": "#4a5568",
}
FIX_ORDER = ("faster_tilt_loop", "fresh_ephemeris", "refocus", "vibration_fix", "keyhole_tracking")
FIX_SHORT = {
    "faster_tilt_loop": "Faster tip-tilt loop",
    "fresh_ephemeris": "Fresh ephemeris",
    "refocus": "Refocus",
    "vibration_fix": "Vibration fix",
    "keyhole_tracking": "Keyhole-aware tracking",
    "all_practical_fixes": "All five together",
    "low_order_ao_upper_bound": "Ideal 11-mode AO bound",
}


def _gb(value: float) -> str:
    return f"{value:,.0f}"


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _signed_pct(audit: float, truth: float) -> str:
    change = 100 * (audit - truth) / truth
    return "0%" if abs(change) < 0.5 else f"{change:+.0f}%"


def _nights(rows: list[dict[str, Any]]) -> str:
    names = sorted({p["night_id"] for p in rows})
    if len(names) == 1:
        return f"night {names[0]}"
    return "nights " + ", ".join(names[:-1]) + " and " + names[-1]


def _orbital_speed_km_s(altitude_km: float) -> float:
    from .station import EARTH_MU_KM3_S2, EARTH_RADIUS_KM

    return math.sqrt(EARTH_MU_KM3_S2 / (EARTH_RADIUS_KM + altitude_km))


def _style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)


def _findings(payload: dict[str, Any]) -> dict[str, Any]:
    passes = payload["passes"]
    settings = payload["audit_settings"]
    ephemeris = [
        p for p in passes if p["audit"]["timing"].get("detected")
    ]
    static = [p for p in passes if p["audit"]["estimates"]["static_db"] <= -settings["static_report_db"]]
    vibration = [p for p in passes if p["audit"]["vibration"]]
    keyhole = [p for p in passes if any(e["trigger"] == "keyhole" for e in p["audit"]["episodes"])]
    cloud = [p for p in passes if p["audit"]["estimates"]["mean_extinction_db"] >= 0.5]
    return {
        "ephemeris": ephemeris,
        "static": static,
        "vibration": vibration,
        "keyhole": keyhole,
        "cloud": cloud,
        "vibration_hz": float(np.median([p["audit"]["vibration"]["frequency_hz"] for p in vibration])) if vibration else None,
    }


def _fix_labels(payload: dict[str, Any], found: dict[str, Any]) -> dict[str, str]:
    st = payload["station"]
    vib = f"{found['vibration_hz']:.1f} Hz" if found["vibration_hz"] else "detected"
    return {
        "faster_tilt_loop": (
            f"Faster tip-tilt loop ({st['rejection_bandwidth_hz']:.0f} to "
            f"{payload['audit_settings']['what_if_bandwidth_hz']:.0f} Hz rejection bandwidth)"
        ),
        "fresh_ephemeris": f"Fresh ephemeris before every pass ({_nights(found['ephemeris'])})",
        "refocus": f"Refocus or realign the fiber ({_nights(found['static'])})",
        "vibration_fix": f"Remove or notch the {vib} vibration ({_nights(found['vibration'])})",
        "keyhole_tracking": "Keyhole-aware tracking for near-zenith passes",
        "all_practical_fixes": "All five together",
        "low_order_ao_upper_bound": "Ideal correction of the first 11 Zernike modes (upper bound)",
    }


EFFORT = {
    "faster_tilt_loop": "Hardware: faster tracking camera and lower loop latency",
    "fresh_ephemeris": "Operations: refresh TLEs or use operator ephemeris",
    "refocus": "Maintenance: focus or fiber-alignment check",
    "vibration_fix": "Investigation: find the source, isolate or notch it",
    "keyhole_tracking": "Scheduling or tracking mode",
}


# ------------------------------------------------------------------------- figures


def _figure_waterfall(payload: dict[str, Any], path: Path) -> None:
    audit = payload["summary"]["audit"]
    causes = sorted(CAUSES, key=lambda c: -audit["shapley_gb"][c])
    labels = ["Possible"] + [CAUSE_LABELS[c] for c in causes] + ["Delivered"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    level = audit["possible_gb"]
    ax.bar(0, level, color="#cbd5e0", width=0.7)
    for i, cause in enumerate(causes, start=1):
        value = audit["shapley_gb"][cause]
        ax.bar(i, value, bottom=level - value, color=COLORS[cause], width=0.7)
        ax.text(i, level + audit["possible_gb"] * 0.01, _gb(value), ha="center", va="bottom", fontsize=7.5)
        level -= value
    residual = audit["link_budget_residual_gb"]
    if residual > 1:
        level -= residual
    ax.bar(len(labels) - 1, audit["delivered_gb"], color=COLORS["delivered"], width=0.7)
    ax.text(len(labels) - 1, audit["delivered_gb"] * 1.01, _gb(audit["delivered_gb"]), ha="center", va="bottom", fontsize=8)
    ax.text(0, audit["possible_gb"] * 1.01, _gb(audit["possible_gb"]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Data over 12 passes (Gb)")
    ax.set_title("Where the data went (audit estimate, Shapley attribution)", fontsize=10, loc="left")
    ax.set_ylim(audit["delivered_gb"] * 0.6, audit["possible_gb"] * 1.06)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_passes(payload: dict[str, Any], path: Path) -> None:
    passes = payload["passes"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(passes))
    ax.bar(x, [p["audit"]["attribution"]["delivered_gb"] for p in passes], color=COLORS["delivered"], label="Delivered", width=0.72)
    bottom = np.array([p["audit"]["attribution"]["delivered_gb"] for p in passes])
    for cause in sorted(CAUSES, key=lambda c: -payload["summary"]["audit"]["shapley_gb"][c]):
        values = np.array([p["audit"]["attribution"]["shapley_gb"][cause] for p in passes])
        ax.bar(x, values, bottom=bottom, color=COLORS[cause], label=CAUSE_LABELS[cause], width=0.72)
        bottom = bottom + values
    ax.set_xticks(x, [f"{p['pass_id']}\n{p['max_elevation_deg']:.0f}°" for p in passes], fontsize=8)
    ax.set_ylabel("Gb per pass")
    ax.set_title("Delivered and lost data per pass (label: pass, maximum elevation)", fontsize=10, loc="left")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_timing(payload: dict[str, Any], features: list[PassFeatures], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    by_night: dict[str, list[PassFeatures]] = {}
    for f in features:
        by_night.setdefault(f.night_id, []).append(f)
    palette = ["#2b6cb0", "#38a169", "#c53030", "#dd6b20", "#6b46c1", "#718096"]
    timing = {p["pass_id"]: p["audit"]["timing"] for p in payload["passes"]}
    for k, (night, rows) in enumerate(sorted(by_night.items())):
        rates, along = [], []
        for f in rows:
            s = f.slow
            locked = s["lock_state"] == 2
            rates.append(s["rate_rad_s"][locked])
            along.append((s["offload_x_rad"] * s["along_x"] + s["offload_y_rad"] * s["along_y"])[locked])
        rate, value = np.concatenate(rates), np.concatenate(along)
        offsets = [timing[f.pass_id]["offset_s"] for f in rows if timing[f.pass_id].get("offset_s") is not None]
        label = f"{night}: slope {np.mean(offsets):.2f} s" if offsets else night
        ax.scatter(rate[::5] * 1e3, value[::5] * 1e3, s=3, color=palette[k % len(palette)], alpha=0.5, label=label)
    ax.set_xlabel("Line-of-sight angular rate (mrad/s)")
    ax.set_ylabel("Mount offload, along-track (mrad)")
    ax.set_title("Offload grows with angular rate on two nights: a timing-equivalent ephemeris bias", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False, markerscale=3)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_vibration(payload: dict[str, Any], features: list[PassFeatures], found: dict[str, Any], path: Path) -> None:
    vib_ids = {p["pass_id"] for p in found["vibration"]}
    with_vib = next((f for f in features if f.pass_id in vib_ids), None)
    without = next((f for f in features if f.pass_id not in vib_ids and f.median_psd), None)
    if with_vib is None or without is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for f, color in ((without, "#718096"), (with_vib, "#dd6b20")):
        freq = f.median_psd["frequency_hz"]
        psd = f.median_psd["residual"].sum(axis=1) * 1e12
        ax.loglog(freq[1:], psd[1:], color=color, linewidth=1.2, label=f"{f.pass_id} ({f.night_id})")
    if found["vibration_hz"]:
        ax.axvline(found["vibration_hz"], color="#dd6b20", linestyle=":", linewidth=1)
        ax.text(found["vibration_hz"] * 1.08, ax.get_ylim()[1] * 0.3, f"{found['vibration_hz']:.1f} Hz", color="#dd6b20", fontsize=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Residual tilt PSD, x + y (µrad²/Hz)")
    ax.set_title("A narrow line in the tracking residual on one night", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_keyhole(payload: dict[str, Any], features: list[PassFeatures], found: dict[str, Any], path: Path) -> None:
    ids = {p["pass_id"] for p in found["keyhole"]}
    f = next((f for f in features if f.pass_id in ids), None)
    if f is None:
        return
    s = f.slow
    st = payload["station"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    t = s["t_s"]
    episode = next(e for e in next(p for p in payload["passes"] if p["pass_id"] == f.pass_id)["audit"]["episodes"] if e["trigger"] == "keyhole")
    window = (t > episode["start_s"] - 25) & (t < episode["stop_s"] + 25)
    ax.semilogy(t[window], np.maximum(s["mount_error_rad"][window], 1e-6) * 1e3, color="#b83280", label="Azimuth lag on sky")
    ax.axhline(st["fsm_range_rad"] * 1e3, color="#4a5568", linestyle="--", linewidth=1, label="FSM range")
    ax.axvspan(episode["start_s"], episode["stop_s"], color="#fed7e2", alpha=0.6, label=f"No lock ({episode['stop_s'] - episode['start_s']:.1f} s)")
    ax.set_xlabel("Time from pass start (s)")
    ax.set_ylabel("mrad")
    ax.set_title(f"{f.pass_id}: the mount cannot follow azimuth through the keyhole", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_dropout(payload: dict[str, Any], features: list[PassFeatures], found: dict[str, Any], path: Path) -> str | None:
    ephemeris_ids = {p["pass_id"] for p in found["ephemeris"]}
    best = None
    for p in payload["passes"]:
        if p["pass_id"] not in ephemeris_ids:
            continue
        for e in p["audit"]["episodes"]:
            if e["trigger"] == "cloud" and (best is None or e["stop_s"] - e["start_s"] > best[1]["stop_s"] - best[1]["start_s"]):
                best = (p["pass_id"], e)
    if best is None:
        return None
    f = next(f for f in features if f.pass_id == best[0])
    e = best[1]
    s = f.slow
    t = s["t_s"]
    window = (t > e["start_s"] - 20) & (t < e["stop_s"] + 15)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(t[window], s["clear_model_dbm"][window], color="#4a5568", linestyle="--", linewidth=1, label="Clear-sky model")
    ax.plot(t[window], s["track_dbm"][window], color="#2b6cb0", linewidth=1.2, label="Tracking-sensor power (no trace = no signal)")
    ax.axhline(payload["station"]["lock_threshold_dbm"], color="#c53030", linestyle=":", linewidth=1, label="Lock threshold")
    ax.axvspan(e["start_s"], e["stop_s"], color="#fed7d7", alpha=0.6, label=f"No lock ({e['stop_s'] - e['start_s']:.1f} s)")
    ax.set_xlabel("Time from pass start (s)")
    ax.set_ylabel("dBm")
    ax.set_title(f"{f.pass_id}: a short cloud patch, then a long search on stale ephemeris", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return best[0]


def _figure_validation(payload: dict[str, Any], path: Path) -> None:
    comparison = payload["summary"]["cause_comparison"]
    causes = sorted(CAUSES, key=lambda c: -comparison[c]["truth_gb"])
    fixes = payload["summary"]["fix_comparison"]
    fix_names = [*FIX_ORDER, "low_order_ao_upper_bound"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(causes))
    axes[0].bar(x - 0.2, [comparison[c]["truth_gb"] for c in causes], width=0.4, color="#a0aec0", label="Simulator truth")
    axes[0].bar(x + 0.2, [comparison[c]["audit_gb"] for c in causes], width=0.4, color="#2b6cb0", label="Audit from logs")
    axes[0].set_xticks(x, [CAUSE_LABELS[c] for c in causes], rotation=35, ha="right", fontsize=7.5)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Lost data attributed (Gb, log scale)")
    axes[0].set_title("Attribution by cause", fontsize=10, loc="left")
    axes[0].legend(fontsize=8, frameon=False)
    x2 = np.arange(len(fix_names))
    axes[1].bar(x2 - 0.2, [fixes[k]["truth_gb"] for k in fix_names], width=0.4, color="#a0aec0")
    axes[1].bar(x2 + 0.2, [fixes[k]["audit_gb"] for k in fix_names], width=0.4, color="#2b6cb0")
    short = {
        "faster_tilt_loop": "Faster loop",
        "fresh_ephemeris": "Ephemeris",
        "refocus": "Refocus",
        "vibration_fix": "Vibration",
        "keyhole_tracking": "Keyhole",
        "low_order_ao_upper_bound": "Ideal AO bound",
    }
    axes[1].set_xticks(x2, [short[k] for k in fix_names], rotation=35, ha="right", fontsize=7.5)
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=100)
    axes[1].set_ylabel("Data recovered by fix (Gb, log scale)")
    axes[1].set_title("Recovered data by fix", fontsize=10, loc="left")
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------- report


def write_report(
    payload: dict[str, Any], features: list[PassFeatures], truth_slow: dict[str, Any], output: Path
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    found = _findings(payload)
    _figure_waterfall(payload, figures / "waterfall.png")
    _figure_passes(payload, figures / "passes.png")
    _figure_timing(payload, features, figures / "ephemeris.png")
    _figure_vibration(payload, features, found, figures / "vibration.png")
    _figure_keyhole(payload, features, found, figures / "keyhole.png")
    dropout_pass = _figure_dropout(payload, features, found, figures / "dropout.png")
    _figure_validation(payload, figures / "validation.png")
    (output / "REPORT.md").write_text(_markdown(payload, found, dropout_pass))


def _markdown(payload: dict[str, Any], found: dict[str, Any], dropout_pass: str | None) -> str:
    summary = payload["summary"]
    audit = summary["audit"]
    st = payload["station"]
    passes = payload["passes"]
    labels = _fix_labels(payload, found)
    lost = audit["lost_gb"]
    nights = sorted({p["night_id"] for p in passes})
    fixes = audit["fix_gains_gb"]
    ranked = sorted(FIX_ORDER, key=lambda k: -fixes[k])
    ho = audit["shapley_gb"]["higher_order_turbulence"]
    low_elev = [p for p in passes if p["max_elevation_deg"] < 35]
    lines: list[str] = []
    add = lines.append

    add("# Sample link-loss audit: hypothetical station OGS-A\n")
    add(
        "> **Simulation only.** This sample runs the Noether link-loss audit on synthetic telemetry from a "
        "reduced-order model of a hypothetical optical ground station. No physical bench, live optical link or "
        "on-sky performance is represented. The station, satellite and passes are invented so that every "
        "number can be checked against the simulator's ground truth.\n"
    )
    add(
        "This is the report format we return for a free audit. The audit read only what a station logs, plus "
        "the station's declared configuration. It did not see the simulator's settings.\n"
    )

    add("## Summary\n")
    add(
        f"- **{len(passes)} passes over {len(nights)} nights.** At {st['data_rate_gbps']:.0f} Gbps above "
        f"{st['min_elevation_deg']:.0f}° elevation, the passes could have carried {_gb(audit['possible_gb'])} Gb. "
        f"The station delivered {_gb(audit['delivered_gb'])} Gb ({_pct(audit['delivered_fraction'])}), so "
        f"{_gb(lost)} Gb were lost."
    )
    add(
        f"- **Five fixes recover {_gb(fixes['all_practical_fixes'])} Gb together**, "
        f"{_pct(fixes['all_practical_fixes'] / lost)} of the lost data. They are ranked below by recovered data; "
        "the effort column is our judgment. The cheapest, fresh ephemeris, is an operations change worth "
        f"{_gb(fixes['fresh_ephemeris'])} Gb."
    )
    add(
        f"- **The largest single loss is higher-order turbulence: {_gb(ho)} Gb ({_pct(ho / lost)} of losses)**, "
        "concentrated at low elevation. Tip-tilt correction cannot recover it. Ideal correction of the first 11 "
        f"Zernike modes would recover at most {_gb(fixes['low_order_ao_upper_bound'])} Gb; a real system recovers less."
    )
    if found["cloud"]:
        add(
            f"- **Cloud cost {_gb(audit['shapley_gb']['cloud'])} Gb.** That is a site and scheduling question, not a "
            "station fault."
        )
    top = summary["top_cause_agreement"]
    dt_errors = [
        abs(p["audit"]["timing"]["offset_s"] - p["truth"]["parameters"]["ephemeris_time_offset_s"])
        for p in passes
        if p["truth"]["parameters"]["ephemeris_time_offset_s"]
    ]
    combined = summary["fix_comparison"]["all_practical_fixes"]
    add(
        f"- **Checked against ground truth.** The audit named the right largest cause on {top['matches']} of "
        f"{top['passes']} passes"
        + (f", recovered the planted ephemeris offsets within {1e3 * max(dt_errors):.0f} ms" if dt_errors else "")
        + ", and estimated the combined recovery within "
        f"{max(1, round(abs(100 * (combined['audit_gb'] - combined['truth_gb']) / combined['truth_gb'])))}% "
        "of the truth. Details are under *How we checked this sample*."
    )
    add("")
    add("| Rank | Fix | Data recovered (Gb) | Share of lost data | Typical effort |")
    add("|---|---|---:|---:|---|")
    for rank, key in enumerate(ranked, start=1):
        add(f"| {rank} | {labels[key]} | {_gb(fixes[key])} | {_pct(fixes[key] / lost)} | {EFFORT[key]} |")
    add(f"| | {labels['all_practical_fixes']} | {_gb(fixes['all_practical_fixes'])} | {_pct(fixes['all_practical_fixes'] / lost)} | |")
    add(
        "\nRecovered data for each fix is estimated alone, from the current state. Fixes overlap, so the combined "
        "row is computed directly rather than summed.\n"
    )

    add("## Where the data went\n")
    add("![Loss waterfall](figures/waterfall.png)\n")
    add(
        "Each lost gigabit is attributed to a cause with Shapley values: every cause is removed in every order, and "
        "its average marginal effect is its share. Causes interact (a cloud dropout costs more when reacquisition "
        "needs a long search), and Shapley values split that interaction fairly. They sum to the data recovered "
        "when every cause is removed.\n"
    )
    add("| Cause | Lost data (Gb) | Share |")
    add("|---|---:|---:|")
    for cause in sorted(CAUSES, key=lambda c: -audit["shapley_gb"][c]):
        add(f"| {CAUSE_LABELS[cause]} | {_gb(audit['shapley_gb'][cause])} | {_pct(audit['shapley_gb'][cause] / lost)} |")
    residual = audit["link_budget_residual_gb"]
    add(f"| Link budget (range and elevation with no other loss) | {_gb(residual)} | {_pct(residual / lost)} |")
    add("")
    add("![Per-pass losses](figures/passes.png)\n")

    add("## Findings\n")
    _finding_tilt(add, payload, labels, fixes)
    _finding_ephemeris(add, payload, found, labels, fixes, dropout_pass)
    _finding_static(add, payload, found, labels, fixes)
    _finding_vibration(add, payload, found, labels, fixes)
    _finding_keyhole(add, payload, found, labels, fixes)
    _finding_turbulence(add, payload, fixes, ho, lost, low_elev)
    _finding_cloud(add, payload, found)

    add("## Per-pass results\n")
    add("| Pass | Night | Max elevation | Possible (Gb) | Delivered (Gb) | Delivered | Largest cause | Second cause |")
    add("|---|---|---:|---:|---:|---:|---|---|")
    for p in passes:
        a = p["audit"]["attribution"]
        order = sorted(CAUSES, key=lambda c: -a["shapley_gb"][c])
        add(
            f"| {p['pass_id']} | {p['night_id']} | {p['max_elevation_deg']:.0f}° | {_gb(a['possible_gb'])} | "
            f"{_gb(a['delivered_gb'])} | {_pct(a['delivered_gb'] / a['possible_gb'])} | "
            f"{CAUSE_LABELS[order[0]]} ({_gb(a['shapley_gb'][order[0]])}) | {CAUSE_LABELS[order[1]]} ({_gb(a['shapley_gb'][order[1]])}) |"
        )
    add("")

    _validation(add, payload)
    _method(add, payload)
    _limitations(add, payload)
    _checklist(add)
    add("## Reproduce\n")
    add("```sh\nmake link-audit-sample\n```\n")
    add(
        "The command regenerates the coupling tables if absent, the synthetic telemetry (written under "
        "`results/runs/`, not committed), the audit, this report and its figures. "
        "[metrics.json](metrics.json) holds every number above, with SHA-256 hashes of the sources, the "
        "configuration and each telemetry file. [coupling_tables.json](coupling_tables.json) holds the Monte "
        "Carlo coupling tables and convergence checks.\n"
    )
    return "\n".join(lines)


def _finding_tilt(add: Any, payload: dict[str, Any], labels: dict[str, str], fixes: dict[str, float]) -> None:
    passes = payload["passes"]
    rms = [p["audit"]["estimates"]["mean_tilt_residual_rms_rad"] * 1e6 for p in passes]
    theta_c = payload["coupling_model"]["theta_c_rad"] * 1e6
    shap = payload["summary"]["audit"]["shapley_gb"]["tilt_residual"]
    add("### Tip-tilt residual: the loop is the largest fixable loss\n")
    add(
        f"The tracking residual averages {min(rms):.2f} to {max(rms):.2f} µrad RMS per axis across passes. Single-mode "
        f"coupling falls by 1/e at a {theta_c:.2f} µrad offset in this optical design, so sub-microradian jitter still "
        f"costs coupling, most where margin is thin at low elevation. Attributed loss: {_gb(shap)} Gb."
    )
    add(
        f"Raising the error-rejection bandwidth from {payload['station']['rejection_bandwidth_hz']:.0f} Hz to "
        f"{payload['audit_settings']['what_if_bandwidth_hz']:.0f} Hz would recover an estimated "
        f"{_gb(fixes['faster_tilt_loop'])} Gb. We estimated this from the pseudo-open-loop tilt (residual plus FSM "
        "command) and the declared loop model, with the same 2-frame latency at twice the frame rate and the same "
        "noise spectral density. Tracking-sensor noise is included.\n"
    )


def _finding_ephemeris(
    add: Any, payload: dict[str, Any], found: dict[str, Any], labels: dict[str, str], fixes: dict[str, float],
    dropout_pass: str | None,
) -> None:
    rows = found["ephemeris"]
    if not rows:
        return
    by_night: dict[str, list[float]] = {}
    for p in rows:
        by_night.setdefault(p["night_id"], []).append(p["audit"]["timing"]["offset_s"])
    speed = _orbital_speed_km_s(payload["station"]["altitude_km"])
    text = "; ".join(
        f"{n}: {np.mean(v):.2f} s (about {np.mean(v) * speed:.1f} km along track)" for n, v in sorted(by_night.items())
    )
    add("### Stale ephemeris: short dropouts became long outages\n")
    add(
        "On two nights the mount offload grows in proportion to the satellite's angular rate, along track. That is "
        f"the signature of pointing at where the satellite was a fixed time earlier. Fitted offsets: {text}. "
        "Either a stale TLE or a station clock offset produces this; check the TLE age and the clock logs.\n"
    )
    add("![Ephemeris bias](figures/ephemeris.png)\n")
    shap = payload["summary"]["audit"]["shapley_gb"]["ephemeris"]
    if dropout_pass:
        p = next(p for p in payload["passes"] if p["pass_id"] == dropout_pass)
        e = max((e for e in p["audit"]["episodes"] if e["trigger"] == "cloud"), key=lambda e: e["stop_s"] - e["start_s"])
        error = e.get("predicted_error_rad") or 0.0
        add(
            f"The cost shows up on reacquisition. On {dropout_pass} a short cloud patch broke lock; with the bias, "
            f"the open-loop error at reacquisition was about {error * 1e3:.1f} mrad, against a "
            f"{payload['station']['acquisition_fov_rad'] * 1e3:.1f} mrad acquisition field. The spiral search took "
            f"the outage to {e['stop_s'] - e['start_s']:.1f} s.\n"
        )
        add("![Dropout and search](figures/dropout.png)\n")
    cal = payload["calibration"]
    add(
        f"The acquisition model fitted across all passes gives {cal['acquisition_nominal_s']:.2f} s nominal plus "
        f"{cal['acquisition_search_s_per_area']:.2f} s per extra acquisition-field area. Attributed loss: {_gb(shap)} Gb. "
        f"Fresh ephemeris before every pass recovers an estimated {_gb(fixes['fresh_ephemeris'])} Gb, which "
        "includes the share of cloud dropouts that the search prolonged.\n"
    )


def _finding_static(add: Any, payload: dict[str, Any], found: dict[str, Any], labels: dict[str, str], fixes: dict[str, float]) -> None:
    rows = found["static"]
    if not rows:
        return
    values = ", ".join(f"{p['pass_id']} {p['audit']['estimates']['static_db']:.2f} dB" for p in rows)
    others = [abs(p["audit"]["estimates"]["static_db"]) for p in payload["passes"] if p not in rows]
    add("### Static coupling loss on one night\n")
    add(
        f"After the turbulence and tilt terms are removed, the fiber-to-tracking power ratio sits lower on one "
        f"night than on all others: {values}, relative to the station's median pass. The other passes agree "
        f"within ±{max(others):.2f} dB. A loss that is constant across a whole night, independent of seeing and "
        "elevation, points to focus or fiber alignment, not the atmosphere."
    )
    add(
        f"Refocusing recovers an estimated {_gb(fixes['refocus'])} Gb. This estimate treats the loss as a constant "
        "dB offset, which overstates the gain in deep turbulence fades; see the validation section.\n"
    )


def _finding_vibration(add: Any, payload: dict[str, Any], found: dict[str, Any], labels: dict[str, str], fixes: dict[str, float]) -> None:
    rows = found["vibration"]
    if not rows:
        return
    rms = "; ".join(
        f"{p['pass_id']}: {p['audit']['estimates']['mean_vibration_residual_rms_rad'] * 1e6:.2f} µrad" for p in rows
    )
    add("### Structural vibration on one night\n")
    add(
        f"The tracking residual on night {rows[0]['night_id']} carries a narrow line at {found['vibration_hz']:.1f} Hz "
        "in every analysis window, and on no other night. After the loop's partial rejection, the line leaves this "
        f"much residual tilt, RMS per axis: {rms}. Lines like this often come from fans, pumps, chillers or a dome "
        "drive."
    )
    add(
        f"Removing the source, or notching it in the controller, recovers an estimated {_gb(fixes['vibration_fix'])} Gb.\n"
    )
    add("![Vibration line](figures/vibration.png)\n")


def _finding_keyhole(add: Any, payload: dict[str, Any], found: dict[str, Any], labels: dict[str, str], fixes: dict[str, float]) -> None:
    rows = found["keyhole"]
    if not rows:
        return
    p = rows[0]
    e = next(e for e in p["audit"]["episodes"] if e["trigger"] == "keyhole")
    add("### Keyhole on the near-zenith pass\n")
    add(
        f"{p['pass_id']} peaks at {p['max_elevation_deg']:.0f}° elevation. Near zenith, an alt-azimuth mount must "
        f"swing azimuth faster than its declared {payload['station']['mount_az_rate_limit_deg_s']:.0f}°/s limit; "
        f"the lag exceeded the FSM range and lock dropped for {e['stop_s'] - e['start_s']:.1f} s."
    )
    add(
        f"Keyhole-aware tracking (pre-rotating azimuth, or over-the-top tracking where the mount allows it) recovers "
        f"{_gb(fixes['keyhole_tracking'])} Gb.\n"
    )
    add("![Keyhole](figures/keyhole.png)\n")


def _finding_turbulence(add: Any, payload: dict[str, Any], fixes: dict[str, float], ho: float, lost: float, low_elev: list[dict[str, Any]]) -> None:
    passes = payload["passes"]
    share_low = sum(p["audit"]["attribution"]["shapley_gb"]["higher_order_turbulence"] for p in low_elev) / max(ho, 1e-9)
    d_over_r0 = [p["audit"]["estimates"]["median_d_over_r0"] for p in passes]
    cm = payload["coupling_model"]
    log_grid = np.log(cm["d_over_r0_grid"])

    def mean_loss_db(value: float) -> float:
        return -float(np.interp(math.log(value), log_grid, cm["mean_log_db_tip_tilt"]))

    add("### Higher-order turbulence: the largest loss, and the hardest to fix\n")
    add(
        f"The audit estimates D/r0 at 1550 nm from the tracking data itself, between {min(d_over_r0):.2f} and "
        f"{max(d_over_r0):.2f} (median per pass). At these values the wavefront left after tip-tilt correction "
        f"costs {mean_loss_db(min(d_over_r0)):.1f} to {mean_loss_db(max(d_over_r0)):.1f} dB of single-mode coupling "
        f"on average, with deeper fades. It accounts for {_gb(ho)} Gb ({_pct(ho / lost)} of losses); passes below "
        f"35° maximum elevation carry {_pct(share_low)} of it."
    )
    add(
        f"Ideal correction of the first 11 Zernike modes would recover at most {_gb(fixes['low_order_ao_upper_bound'])} "
        "Gb. That bound has no servo lag, noise or wavefront-sensor error. A real adaptive-optics system would "
        "recover less, and this sample does not evaluate one. Cheaper levers to test first, also not evaluated "
        "here: raise the minimum elevation, or lower the data rate at low elevation.\n"
    )


def _finding_cloud(add: Any, payload: dict[str, Any], found: dict[str, Any]) -> None:
    rows = found["cloud"]
    if not rows:
        return
    p = rows[0]
    e = p["audit"]["estimates"]
    add("### Cloud: a site and scheduling question\n")
    add(
        f"{p['pass_id']} ran under patchy cirrus: the tracking power sat {e['mean_extinction_db']:.1f} dB below the "
        f"clear-sky model on average, with peaks near {e['max_extinction_db']:.0f} dB. The clear-sky model was "
        f"fitted across all passes ({payload['calibration']['clear_sky_extinction_db_per_airmass']:.2f} dB per airmass). "
        "No station change recovers this; site diversity or cloud-aware scheduling does.\n"
    )


def _validation(add: Any, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    comparison, fixes = summary["cause_comparison"], summary["fix_comparison"]
    passes = payload["passes"]
    add("## How we checked this sample\n")
    add(
        "Because the telemetry is synthetic, the simulator knows every planted cause. The audit's estimates were "
        "compared with that ground truth after the audit ran. The audit and the simulator share one coupling "
        "model: scalar diffraction overlap on HCIPy phase screens, tabulated with the simulator's outer scale. "
        "This check therefore tests estimation from telemetry, not the coupling physics, which is qualified "
        "separately in [docs/fiber-coupling.md](../../docs/fiber-coupling.md).\n"
    )
    add("![Validation](figures/validation.png)\n")
    add("**Recovered parameters**\n")
    add("| Quantity | Audit | Truth |")
    add("|---|---|---|")
    for p in passes:
        true_dt = p["truth"]["parameters"]["ephemeris_time_offset_s"]
        if true_dt:
            t = p["audit"]["timing"]
            add(
                f"| {p['pass_id']} timing-equivalent offset | {t['offset_s']:.3f} s "
                f"(regression standard error {1e3 * t['stderr_s']:.1f} ms) | {true_dt:.3f} s |"
            )
    spurious = [
        abs(p["audit"]["timing"]["offset_s"])
        for p in passes
        if not p["truth"]["parameters"]["ephemeris_time_offset_s"] and p["audit"]["timing"].get("offset_s") is not None
    ]
    if spurious:
        add(f"| Largest fitted offset on passes with no planted bias | {1e3 * max(spurious):.0f} ms | 0 |")
    ratios = [p["audit"]["estimates"]["median_d_over_r0"] / p["truth"]["median_d_over_r0"] for p in passes]
    add(
        f"| D/r0 (median per pass), audit/truth | {min(ratios):.3f} to {max(ratios):.3f} | 1 |"
    )
    for p in passes:
        if p["truth"]["parameters"]["defocus_waves_rms"]:
            add(
                f"| {p['pass_id']} static coupling loss | {p['audit']['estimates']['static_db']:.2f} dB | "
                f"{-p['truth']['parameters']['static_loss_in_pass_db']:.2f} dB |"
            )
    vib = [p for p in passes if p["truth"]["parameters"]["vibration"]]
    for p in vib:
        audit_v = p["audit"]["vibration"]
        add(
            f"| {p['pass_id']} vibration frequency | {audit_v['frequency_hz']:.2f} Hz | "
            f"{p['truth']['parameters']['vibration']['frequency_hz']:.2f} Hz |"
        )
        add(
            f"| {p['pass_id']} vibration residual RMS per axis | "
            f"{p['audit']['estimates']['mean_vibration_residual_rms_rad'] * 1e6:.3f} µrad | "
            f"{p['truth']['parameters']['mean_vibration_residual_rms_rad'] * 1e6:.3f} µrad |"
        )
    tilt_err = max(
        abs(p["audit"]["estimates"]["mean_tilt_residual_rms_rad"] - p["truth"]["parameters"]["mean_true_residual_rms_rad"])
        for p in passes
    )
    add(f"| Tip-tilt residual RMS, largest per-pass error | {tilt_err * 1e6:.3f} µrad | 0 |")
    cloud = [p for p in passes if p["truth"]["mean_cirrus_db"] > 0.5]
    for p in cloud:
        add(
            f"| {p['pass_id']} mean cloud extinction | {p['audit']['estimates']['mean_extinction_db']:.2f} dB | "
            f"{p['truth']['mean_cirrus_db']:.2f} dB |"
        )
    cal = payload["calibration"]
    add(
        f"| Acquisition model | {cal['acquisition_nominal_s']:.2f} s + {cal['acquisition_search_s_per_area']:.2f} s/area | "
        f"{payload['station']['nominal_acquisition_s']:.2f} s + {payload['station']['search_s_per_fov_area']:.2f} s/area |"
    )
    planted_false = [
        p["pass_id"]
        for p in passes
        if (p["audit"]["timing"].get("detected") and not p["truth"]["parameters"]["ephemeris_time_offset_s"])
        or (p["audit"]["vibration"] and not p["truth"]["parameters"]["vibration"])
    ]
    add(f"| False detections (timing or vibration) | {', '.join(planted_false) or 'none'} | none |")
    add("")
    add("**Attribution of lost data, all passes**\n")
    add("| Cause | Audit (Gb) | Truth (Gb) | Difference |")
    add("|---|---:|---:|---:|")
    for cause in sorted(CAUSES, key=lambda c: -comparison[c]["truth_gb"]):
        a, t = comparison[cause]["audit_gb"], comparison[cause]["truth_gb"]
        diff = _signed_pct(a, t) if t > 1 else f"{a - t:+.0f} Gb"
        add(f"| {CAUSE_LABELS[cause]} | {_gb(a)} | {_gb(t)} | {diff} |")
    add("")
    add("**Recovered data by fix**\n")
    add("| Fix | Audit (Gb) | Truth (Gb) | Difference |")
    add("|---|---:|---:|---:|")
    for key in [*FIX_ORDER, "all_practical_fixes", "low_order_ao_upper_bound"]:
        a, t = fixes[key]["audit_gb"], fixes[key]["truth_gb"]
        add(f"| {FIX_SHORT[key]} | {_gb(a)} | {_gb(t)} | {_signed_pct(a, t)} |")
    top = summary["top_cause_agreement"]
    error = summary["per_pass_cause_abs_error_gb"]
    add(
        f"\nThe audit named the same largest cause as the truth on {top['matches']} of {top['passes']} passes. "
        f"Across all passes and causes, the mean absolute attribution error was {error['mean']:.1f} Gb per pass "
        f"and cause; the largest was {error['max']:.1f} Gb.\n"
    )
    sens = payload.get("outer_scale_sensitivity")
    add("**Where the audit is weakest**\n")
    weak = []
    refocus = fixes["refocus"]
    weak.append(
        f"- The refocus gain is overstated by {100 * (refocus['audit_gb'] - refocus['truth_gb']) / refocus['truth_gb']:.0f}%. "
        "The audit models static loss as a constant dB offset, but defocus costs less in deep turbulence fades, "
        "which is where the threshold decides data."
    )
    scint = comparison["scintillation"]
    weak.append(
        f"- Scintillation is understated by {100 * (scint['truth_gb'] - scint['audit_gb']) / scint['truth_gb']:.0f}%: "
        "fluctuations slower than the 1 s reference are absorbed into the extinction estimate."
    )
    if sens:
        weak.append(
            f"- r0 depends on the assumed outer scale. The audit assumed {settings_l0(payload)} m; the simulator "
            f"used {payload['link']['outer_scale_m']:.0f} m. Assuming Kolmogorov turbulence (no outer scale) instead "
            f"raises every r0 estimate by {100 * (sens['r0_ratio_infinite_vs_assumed'] - 1):.0f}% and changes the "
            f"higher-order turbulence attribution to {_gb(sens['higher_order_gb_infinite_outer_scale'])} Gb "
            f"(from {_gb(sens['higher_order_gb_assumed'])} Gb). On a real station, a seeing monitor or "
            "a site study should anchor this."
        )
    weak.append(
        "- The timing fit's regression standard error (under a millisecond) understates its real uncertainty. "
        "The station's own pointing-model error also enters the offload, and it produced the fitted offsets on "
        "passes with no planted bias shown in the parameter table. That is why detection uses a "
        f"{1e3 * payload['audit_settings']['timing_detect_s']:.0f} ms floor, not the standard error alone."
    )
    weak.append(
        "- Lost time is restored with a proxy power series (the clear-sky model plus coupling borrowed from the "
        "nearest locked segment). The truth uses the simulator's would-be power."
    )
    for line in weak:
        add(line)
    add("")


def settings_l0(payload: dict[str, Any]) -> str:
    value = payload["audit_settings"]["assumed_outer_scale_m"]
    return f"{value:.0f}" if value is not None else "infinite"


def _method(add: Any, payload: dict[str, Any]) -> None:
    st, cm = payload["station"], payload["coupling_model"]
    diag = cm["diagnostics"]
    add("## Method in brief\n")
    add(
        f"- **Station (declared).** {st['aperture_diameter_m']:.2f} m aperture with {st['central_obscuration']:.0%} "
        f"central obscuration; {st['wavelength_m'] * 1e9:.0f} nm into single-mode fiber; tip-tilt loop at "
        f"{st['loop_rate_hz']:.0f} Hz with {st['loop_delay_frames']}-frame delay and "
        f"{st['rejection_bandwidth_hz']:.0f} Hz rejection bandwidth; FSM range ±{st['fsm_range_rad'] * 1e6:.0f} µrad; "
        f"acquisition field ±{st['acquisition_fov_rad'] * 1e3:.1f} mrad; azimuth limited to "
        f"{st['mount_az_rate_limit_deg_s']:.0f}°/s and {st['mount_az_accel_limit_deg_s2']:.0f}°/s²; receiver "
        f"sensitivity {st['sensitivity_dbm']:.0f} dBm at {st['data_rate_gbps']:.0f} Gbps."
    )
    add(
        f"- **Coupling model.** Scalar Fraunhofer overlap with an ideal Gaussian mode, using the qualified "
        f"`noether_physics.coupling` component. The mode is optimized for the obscured pupil "
        f"(zero-aberration coupling {cm['eta0']:.3f}); tilt sensitivity θc = {cm['theta_c_rad'] * 1e6:.2f} µrad "
        f"(Gaussian fit within {100 * cm['tilt_fit_max_relative_error']:.1f}%). Higher-order statistics come "
        f"from {diag['screens']} HCIPy von Kármán screens (L0 = {diag['outer_scale_m']:.0f} m, "
        f"{diag['grid_size']}² pupil, {diag['oversampling']}× oversampled), scaled over D/r0 = 0.4 to 6."
    )
    conv = diag.get("convergence", {})
    if conv:
        add(
            f"- **Convergence spot checks at D/r0 = 3.** Mean relative coupling: "
            f"{conv['grid128_over4']['mean_relative_coupling']:.3f} ± {conv['grid128_over4']['standard_error']:.3f} "
            f"(4× oversampling), {conv['grid128_over16']['mean_relative_coupling']:.3f} ± "
            f"{conv['grid128_over16']['standard_error']:.3f} (16×), {conv['grid256_over8']['mean_relative_coupling']:.3f} "
            f"± {conv['grid256_over8']['standard_error']:.3f} (256² grid), against "
            f"{conv['reference_grid128_over8_interpolated_at_3p0']:.3f} interpolated from the table."
        )
    add(
        "- **Audit steps.** Clear-sky power model fitted across passes; extinction as excess attenuation. "
        "Scintillation from fast tracking-power fluctuation. Tracking-sensor noise from the residual spectrum "
        "between 800 and 990 Hz. True residual variance corrected for noise through the loop's rejection and "
        "complementary functions. r0 from pseudo-open-loop tilt variance under an assumed outer scale. "
        "Vibration lines from the median residual spectrum. Static loss from the coupling left after the "
        "turbulence and tilt terms. Timing-equivalent bias by regressing the offload on angular rate. "
        "Keyhole from encoder lag. Acquisition time fitted against predicted open-loop error."
    )
    add(
        "- **Data accounting.** A sample counts as delivered when the station is locked and fiber power is at or "
        "above sensitivity. Removing a cause adds its estimated dB back and restores any lock time it blocked. "
        "Shapley values attribute lost data; fixes are evaluated alone and together.\n"
    )


def _limitations(add: Any, payload: dict[str, Any]) -> None:
    add("## Scope and limitations\n")
    for line in (
        "- Simulation only. A hypothetical station, satellite and atmosphere; no bench, live link or on-sky data.",
        "- The simulator is reduced-order. Tilt spectra are third-order low-pass shapes per layer group; "
        "higher-order coupling is a quantile-mapped process with a Greenwood-time correlation; scintillation is "
        "weak-fluctuation lognormal with aperture averaging; the pass geometry ignores Earth rotation.",
        "- A Gaussian mode stands in for a real fiber eigenmode. The light is monochromatic and polarization is ignored. "
        "Coupling is modeled as tilt and higher-order factors that multiply.",
        "- Data is modeled from power above a fixed sensitivity, with no FEC, interleaving or rate adaptation. A real "
        "audit would use the modem's frame counters.",
        "- Fix estimates hold the rest of the system fixed. A faster loop assumes the same latency in frames and the "
        "same noise spectral density. The adaptive-optics bound is ideal modal correction with no servo error.",
        "- Stations with different receivers (multi-mode or photonic-lantern coupling, mode-converter optics, other "
        "adaptive optics) need their own coupling model. The timing, acquisition, keyhole, cloud and accounting "
        "steps carry over unchanged.",
    ):
        add(line)
    add("")


def _checklist(add: Any) -> None:
    add("## What we need from a real station\n")
    add("| Data | Rate | Used for |")
    add("|---|---|---|")
    for row in (
        ("Fiber (or receiver) power", "≥ 1 kHz", "Delivered data, fades, coupling"),
        ("Tracking-sensor power", "Loop rate", "Extinction, scintillation, coupling ratio"),
        ("Tracking residual x/y and FSM command x/y", "Loop rate, in bursts if needed", "Tilt loss, noise, vibration, r0, loop what-ifs"),
        ("Commanded and encoder az/el", "≥ 10 Hz", "Angular rate, keyhole, mount lag"),
        ("Mount offload or pointing offsets", "≥ 1 Hz", "Ephemeris and clock bias"),
        ("Lock or tracking state", "≥ 10 Hz", "Acquisition, dropouts, outage causes"),
        ("Modem frame counters", "≥ 1 Hz", "Delivered data (replaces the power threshold)"),
        ("Station configuration", "Once", "Aperture, obscuration, fiber, loop rate and latency, FSM range, acquisition field, mount limits, sensitivity"),
        ("Ephemeris used per pass, and its age", "Per pass", "Confirms ephemeris findings"),
        ("Weather, seeing monitor, all-sky camera (optional)", "Any", "Anchors r0 and cloud attribution"),
    ):
        add(f"| {row[0]} | {row[1]} | {row[2]} |")
    add("\nTen to twenty passes are enough for a first report. Nothing runs on, or connects to, the station.\n")
