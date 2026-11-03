"""Unit and smoke checks for the sample link-loss audit (simulation only)."""

import math
from pathlib import Path

import numpy as np
import pytest
from noether_link_audit.attribution import LossModel, shapley
from noether_link_audit.audit import Auditor, AuditSettings
from noether_link_audit.coupling_tables import CouplingTables, build_tables
from noether_link_audit.simulate import CAUSES, LinkConfig, PassSpec, simulate_pass
from noether_link_audit.station import (
    StationConfig,
    gain_for_bandwidth,
    pass_geometry,
    r0_from_tilt,
    rejection,
    von_karman_tilt_ratio,
)


def test_gain_sets_requested_rejection_bandwidth() -> None:
    gain = gain_for_bandwidth(2000.0, 2, 100.0)
    magnitude = abs(rejection(np.array([100.0]), 2000.0, 2, gain)[0])
    assert magnitude == pytest.approx(1 / math.sqrt(2), rel=1e-6)
    assert abs(rejection(np.array([1.0]), 2000.0, 2, gain)[0]) < 0.02


def test_outer_scale_tilt_reduction_matches_winker_approximation() -> None:
    assert von_karman_tilt_ratio(0.4, math.inf) == 1.0
    approximation = 1 - 0.77 * (2 * math.pi * 0.4 / 25.0) ** (1 / 3)
    assert von_karman_tilt_ratio(0.4, 25.0) == pytest.approx(approximation, abs=0.01)


def test_r0_inversion_round_trips_tilt_variance() -> None:
    wavelength, diameter, r0 = 1.55e-6, 0.4, 0.2
    variance = 0.1816 * (wavelength / diameter) ** 2 * (diameter / r0) ** (5 / 3) * von_karman_tilt_ratio(diameter, 20.0)
    assert r0_from_tilt(np.array([variance]), wavelength, diameter, 20.0)[0] == pytest.approx(r0, rel=1e-9)


@pytest.mark.parametrize("max_elevation", [23.0, 55.0, 87.0])
def test_pass_geometry_reaches_requested_elevation(max_elevation: float) -> None:
    geometry = pass_geometry(
        max_elevation, min_elevation_deg=20.0, altitude_km=550.0, azimuth_offset_deg=0.0, sample_rate_hz=10.0
    )
    assert math.degrees(geometry.elevation_rad.max()) == pytest.approx(max_elevation, abs=0.05)
    assert math.degrees(geometry.elevation_rad.min()) >= 20.0 - 1e-6
    assert geometry.range_km.min() >= 550.0 - 1e-6


def test_shapley_values_sum_and_split_symmetric_causes() -> None:
    n = 1000
    base = np.full(n, -41.0, dtype=np.float32)
    model = LossModel(
        rate_gbps=10.0,
        sample_s=1e-3,
        sensitivity_dbm=-40.0,
        base_dbm=base,
        locked=np.ones(n, dtype=bool),
        additive={"a": np.full(n, 0.6, dtype=np.float32), "b": np.full(n, 0.6, dtype=np.float32)},
    )
    result = shapley(model, ("a", "b"))
    total = result.all_removed_gb - result.delivered_gb
    assert sum(result.shapley_gb.values()) == pytest.approx(total)
    assert result.shapley_gb["a"] == pytest.approx(result.shapley_gb["b"])
    assert result.delivered_gb == 0.0 and result.all_removed_gb == pytest.approx(10.0)


def _minimal_tables(theta_c: float = 2.5e-6) -> CouplingTables:
    grid = np.array([0.5, 1.0, 2.0])
    probabilities = np.linspace(0.005, 0.995, 100)
    table = np.tile(np.linspace(0.5, 1.0, 100), (3, 1))
    return CouplingTables(
        d_over_r0=grid,
        probabilities=probabilities,
        quantiles={"tip_tilt": table, "low_order_ao_11": table},
        mean_log_db={"tip_tilt": np.zeros(3), "low_order_ao_11": np.zeros(3)},
        static_relative={"tip_tilt": 1.0},
        eta0=0.65,
        theta_c_rad=theta_c,
        tilt_fit_max_relative_error=0.0,
        mode_radius_m=1e-6,
    )


def test_timing_regression_recovers_planted_offset() -> None:
    auditor = Auditor(StationConfig(), _minimal_tables(), AuditSettings())
    rng = np.random.default_rng(3)
    rate = np.linspace(3e-3, 13e-3, 2000)
    angle = np.linspace(0.0, 1.0, 2000)
    along_x, along_y = np.cos(angle), np.sin(angle)
    offload = 0.45 * rate
    slow = {
        "lock_state": np.full(2000, 2),
        "rate_rad_s": rate,
        "along_x": along_x,
        "along_y": along_y,
        "offload_x_rad": offload * along_x + rng.normal(scale=3e-6, size=2000),
        "offload_y_rad": offload * along_y + rng.normal(scale=3e-6, size=2000),
    }
    assert auditor._timing(slow)["offset_s"] == pytest.approx(0.45, abs=1e-3)


def test_vibration_line_is_detected_at_its_frequency() -> None:
    station = StationConfig()
    auditor = Auditor(station, _minimal_tables(), AuditSettings())
    rng = np.random.default_rng(5)
    n = 40000
    t = np.arange(n) / station.loop_rate_hz
    resid = rng.normal(scale=0.3, size=(n, 2)) + 0.8 * np.sin(2 * np.pi * 61.3 * t)[:, None]
    fast = {
        "resid_x_urad": resid[:, 0],
        "resid_y_urad": resid[:, 1],
        "fsm_x_urad": np.zeros(n),
        "fsm_y_urad": np.zeros(n),
    }
    _, _, vibration, _ = auditor._tilt_windows(fast, np.ones(n, dtype=bool), np.zeros(n))
    assert vibration is not None
    assert vibration["frequency_hz"] == pytest.approx(61.3, abs=0.6)


def test_smoke_audit_matches_truth_on_one_short_pass(tmp_path: Path) -> None:
    station = StationConfig()
    tables = build_tables(station, defocus_waves_rms=(), screens=12, grid_size=48, oversampling=2, convergence_checks=False)
    spec = PassSpec("T01", "N0", 26.0, 0.0, 11, 0.09, 4.0, 21.0)
    fast, slow, truth = simulate_pass(spec, station, LinkConfig(), tables)
    auditor = Auditor(station, tables, AuditSettings())
    features = auditor.extract("T01", "N0", fast, slow)
    calibration = auditor.calibrate([features])
    model, _, estimates = auditor.loss_model(features, calibration)
    audit = shapley(model, CAUSES)
    truth_delivered = float(np.count_nonzero(truth.locked & (truth.would_be_fiber_dbm >= station.sensitivity_dbm)))
    truth_delivered *= station.data_rate_gbps / station.loop_rate_hz
    assert audit.delivered_gb == pytest.approx(truth_delivered, rel=0.01)
    assert audit.delivered_gb + sum(audit.shapley_gb.values()) == pytest.approx(audit.all_removed_gb, rel=1e-9)
    assert estimates["median_d_over_r0"] == pytest.approx(float(np.median(truth.slow["d_over_r0"])), rel=0.1)
