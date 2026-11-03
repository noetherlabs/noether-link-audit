"""Synthetic pass telemetry for the hypothetical station OGS-A, with planted loss causes.

The simulator knows every factor that shapes received power. It writes only what
a station would log (powers, tracking residuals, FSM commands, pointing, offload
and lock state) and keeps the factors as ground truth for validating the audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy import signal, special  # type: ignore[import-untyped]

from .coupling_tables import LOW_ORDER_AO, TIP_TILT, CouplingTables, defocus_variant
from .station import (
    DB_PER_NEPER_POWER,
    Atmosphere,
    FloatArray,
    StationConfig,
    gain_for_bandwidth,
    loop_coefficients,
    pass_geometry,
)

SLOW_RATE_HZ = 10.0
BLOCK_SECONDS = 0.5
STATE_ACQUIRING = 1
STATE_LOCKED = 2
CAUSES = (
    "cloud",
    "scintillation",
    "tilt_residual",
    "vibration",
    "static_optics",
    "higher_order_turbulence",
    "ephemeris",
    "keyhole",
    "nominal_acquisition",
)


@dataclass(frozen=True)
class LinkConfig:
    reference_power_dbm: float = -32.0
    outer_scale_m: float = 25.0
    clear_extinction_db_per_airmass: float = 0.1
    monitor_noise_db: float = 0.03
    no_signal_dbm: float = -75.0


@dataclass(frozen=True)
class PassSpec:
    pass_id: str
    night_id: str
    max_elevation_deg: float
    azimuth_offset_deg: float
    seed: int
    r0_500nm_zenith_m: float
    ground_wind_m_s: float
    upper_wind_rms_m_s: float
    ephemeris_time_offset_s: float = 0.0
    defocus_waves_rms: float = 0.0
    vibration: dict[str, float] | None = None
    cirrus: dict[str, float] | None = None

    def planted(self) -> list[str]:
        causes = []
        if self.ephemeris_time_offset_s:
            causes.append(f"along-track ephemeris bias equivalent to {self.ephemeris_time_offset_s:.2f} s")
        if self.defocus_waves_rms:
            causes.append(f"static defocus {self.defocus_waves_rms:.3f} waves RMS")
        if self.vibration:
            causes.append(f"{self.vibration['frequency_hz']:.0f} Hz structural vibration")
        if self.cirrus and "mean_db" in self.cirrus:
            causes.append("patchy cirrus deck")
        if self.cirrus and self.cirrus.get("patches"):
            causes.append("short cloud patch near culmination")
        return causes


def scenario_passes(raw: dict[str, Any]) -> list[PassSpec]:
    base_seed = int(raw["seed"])
    passes: list[PassSpec] = []
    for night in raw["nights"]:
        for item in night["passes"]:
            passes.append(
                PassSpec(
                    pass_id=str(item["id"]),
                    night_id=str(night["id"]),
                    max_elevation_deg=float(item["max_elevation_deg"]),
                    azimuth_offset_deg=float(item.get("azimuth_offset_deg", 0.0)),
                    seed=base_seed + 1000 * (len(passes) + 1),
                    r0_500nm_zenith_m=float(night["r0_500nm_zenith_m"]),
                    ground_wind_m_s=float(night["ground_wind_m_s"]),
                    upper_wind_rms_m_s=float(night.get("upper_wind_rms_m_s", 21.0)),
                    ephemeris_time_offset_s=float(night.get("ephemeris_time_offset_s", 0.0)),
                    defocus_waves_rms=float(night.get("defocus_waves_rms", 0.0)),
                    vibration=night.get("vibration"),
                    cirrus=item.get("cirrus"),
                )
            )
    return passes


@dataclass
class PassTruth:
    """Ground truth retained in memory for validation; never given to the audit."""

    spec: PassSpec
    fast_rate_hz: float
    would_be_fiber_dbm: NDArray[np.float32]
    locked: NDArray[np.bool_]
    additive_db: dict[str, NDArray[np.float32]]
    tilt_group_db: dict[frozenset[str], NDArray[np.float32]]
    optics_group_db: dict[frozenset[str], NDArray[np.float32]]
    restore: dict[str, NDArray[np.bool_]]
    bandwidth_fix_db: NDArray[np.float32]
    low_order_ao_fix_db: NDArray[np.float32]
    slow: dict[str, FloatArray]
    episodes: list[dict[str, Any]]
    parameters: dict[str, Any] = field(default_factory=dict)


def _ar1(rng: np.random.Generator, rho: FloatArray, block: int, total: int) -> FloatArray:
    """Unit-variance AR(1) series with a coefficient that changes per block."""
    out = np.empty(total)
    state = np.array([rng.normal()])
    for b, coefficient in enumerate(rho):
        lo, hi = b * block, min((b + 1) * block, total)
        if lo >= total:
            break
        innovations = rng.normal(size=hi - lo)
        y, state = signal.lfilter(
            [math.sqrt(max(1.0 - coefficient * coefficient, 0.0))], [1.0, -coefficient], innovations,
            zi=state * coefficient,
        )
        out[lo:hi] = y
        state = np.array([y[-1]])
    return out


class _ButterworthBank:
    """Third-order low-pass filters and their white-noise variance gains."""

    def __init__(self, rate_hz: float) -> None:
        self.rate_hz = rate_hz
        self.cutoffs = np.geomspace(0.2, 0.4 * rate_hz, 160)
        impulse = np.zeros(int(40 * rate_hz))
        impulse[0] = 1.0
        gains = []
        for cutoff in self.cutoffs:
            sos = signal.butter(3, cutoff, fs=rate_hz, output="sos")
            gains.append(float(np.sum(signal.sosfilt(sos, impulse) ** 2)))
        self.gains = np.asarray(gains)

    def design(self, cutoff_hz: float) -> tuple[FloatArray, float]:
        cutoff = float(np.clip(cutoff_hz, self.cutoffs[0], self.cutoffs[-1]))
        gain = float(np.exp(np.interp(math.log(cutoff), np.log(self.cutoffs), np.log(self.gains))))
        return signal.butter(3, cutoff, fs=self.rate_hz, output="sos"), gain


def _mount_track(
    time_s: FloatArray, azimuth_rad: FloatArray, rate_limit: float, accel_limit: float
) -> FloatArray:
    """Rate- and acceleration-limited azimuth tracking at 100 Hz."""
    dt = 0.01
    fine_t = np.arange(time_s[0], time_s[-1] + dt / 2, dt)
    command = np.interp(fine_t, time_s, azimuth_rad)
    command_rate = np.gradient(command, dt)
    actual = np.empty_like(command)
    angle, rate = command[0], command_rate[0]
    for i in range(len(fine_t)):
        desired = np.clip(command_rate[i] + 2.0 * (command[i] - angle), -rate_limit, rate_limit)
        rate += np.clip(desired - rate, -accel_limit * dt, accel_limit * dt)
        angle += rate * dt
        actual[i] = angle
    return np.asarray(np.interp(time_s, fine_t, actual))


def _lock_logic(
    track_dbm: FloatArray,
    bias: FloatArray,
    mount: FloatArray,
    station: StationConfig,
    dt: float,
) -> tuple[NDArray[np.int8], FloatArray, list[dict[str, Any]], dict[str, NDArray[np.bool_]]]:
    """Acquisition/tracking state machine at the slow log rate, with per-sample cause labels."""
    n = len(track_dbm)
    state = np.full(n, STATE_ACQUIRING, dtype=np.int8)
    offload = np.zeros((n, 2))
    labels = {name: np.zeros(n, dtype=bool) for name in ("nominal_acquisition", "ephemeris", "keyhole", "cloud")}
    episodes: list[dict[str, Any]] = []
    fov = station.acquisition_fov_rad
    current = {"start": 0, "trigger": "pass_start", "timer": None, "required": None, "error": None}
    elapsed = 0.0
    locked = False
    hold = max(1, round(station.lock_loss_hold_s / dt))
    low = 0
    o = np.zeros(2)
    for i in range(n):
        if not locked:
            power_ok = track_dbm[i] >= station.lock_threshold_dbm + 1.0
            mount_ok = float(np.hypot(*mount[i])) < fov
            if power_ok and mount_ok:
                if current["timer"] is None:
                    error = float(np.hypot(*(bias[i] + mount[i])))
                    current["timer"], current["error"] = i, error
                    current["required"] = station.nominal_acquisition_s + station.search_s_per_fov_area * max(
                        0.0, (error / fov) ** 2 - 1.0
                    )
                if elapsed < station.nominal_acquisition_s - 1e-9:
                    labels["nominal_acquisition" if current["trigger"] == "pass_start" else current["trigger"]][i] = True
                else:
                    labels["ephemeris"][i] = True
                elapsed += dt
            else:
                labels["cloud" if not power_ok else "keyhole"][i] = True
            if current["trigger"] != "pass_start":
                labels[current["trigger"]][i] = True
            if current["timer"] is not None and elapsed >= float(current["required"]) - 1e-9:
                locked = True
                o = bias[i].copy()
                episodes.append({**current, "end": i + 1, "elapsed_s": elapsed})
        else:
            state[i] = STATE_LOCKED
            o = o + (bias[i] - o) * (dt / station.offload_time_constant_s)
            offload[i] = o
            slow = bias[i] - o + mount[i]
            low = low + 1 if track_dbm[i] < station.lock_threshold_dbm else 0
            if low >= hold or float(np.hypot(*slow)) > station.fsm_range_rad:
                locked = False
                trigger = "cloud" if low >= hold else "keyhole"
                current = {"start": i + 1, "trigger": trigger, "timer": None, "required": None, "error": None}
                elapsed, low, o = 0.0, 0, np.zeros(2)
    if not locked:
        episodes.append({**current, "end": n, "elapsed_s": elapsed, "truncated": True})
    return state, offload, episodes, labels


def simulate_pass(
    spec: PassSpec, station: StationConfig, link: LinkConfig, tables: CouplingTables
) -> tuple[dict[str, FloatArray], dict[str, FloatArray], PassTruth]:
    rng = np.random.default_rng(spec.seed)
    fs = station.loop_rate_hz
    per_slow = int(round(fs / SLOW_RATE_HZ))
    geometry = pass_geometry(
        spec.max_elevation_deg,
        min_elevation_deg=station.min_elevation_deg,
        altitude_km=station.altitude_km,
        azimuth_offset_deg=spec.azimuth_offset_deg,
        sample_rate_hz=SLOW_RATE_HZ,
    )
    n_slow = len(geometry.time_s)
    n_fast = n_slow * per_slow
    slow_t = geometry.time_s
    fast_t = np.arange(n_fast) / fs
    elevation = geometry.elevation_rad
    airmass = 1.0 / np.sin(elevation)

    atmosphere = Atmosphere(
        r0_500nm_zenith_m=spec.r0_500nm_zenith_m,
        ground_wind_m_s=spec.ground_wind_m_s,
        upper_wind_rms_m_s=spec.upper_wind_rms_m_s,
        outer_scale_m=link.outer_scale_m,
        wavelength_m=station.wavelength_m,
        diameter_m=station.aperture_diameter_m,
    )
    cond = atmosphere.conditions(elevation, geometry.rate_rad_s)
    d_over_r0_slow = station.aperture_diameter_m / cond.fried_parameter_m

    # Extinction: clear air plus optional patchy cirrus (lognormal, AR(1) in time).
    cirrus_slow = np.zeros(n_slow)
    if spec.cirrus and "mean_db" in spec.cirrus:
        rho = math.exp(-1.0 / (SLOW_RATE_HZ * float(spec.cirrus["correlation_s"])))
        process = _ar1(rng, np.full(1, rho), n_slow, n_slow)
        sigma = float(spec.cirrus["log_sigma"])
        cirrus_slow = float(spec.cirrus["mean_db"]) * np.exp(sigma * process - 0.5 * sigma * sigma)
    for patch in (spec.cirrus or {}).get("patches", []):
        centre = float(patch["centre_fraction"]) * slow_t[-1]
        cirrus_slow = cirrus_slow + float(patch["peak_db"]) * np.exp(
            -0.5 * ((slow_t - centre) / float(patch["width_s"])) ** 2
        )
    clear_db = link.clear_extinction_db_per_airmass * airmass

    # Scintillation (aperture averaged, lognormal) at the loop rate.
    blocks = int(math.ceil(n_fast / (BLOCK_SECONDS * fs)))
    block = int(BLOCK_SECONDS * fs)
    centres = np.minimum((np.arange(blocks) * block + block // 2) // per_slow, n_slow - 1)
    tau_scint = station.aperture_diameter_m / np.maximum(cond.scintillation_speed_m_s[centres], 0.5)
    z_scint = _ar1(rng, np.exp(-1.0 / (fs * tau_scint)), block, n_fast)
    log_var = np.interp(fast_t, slow_t, cond.log_intensity_variance)
    log_intensity = np.sqrt(log_var) * z_scint - 0.5 * log_var
    scint_db = DB_PER_NEPER_POWER * log_intensity

    range_db = 20 * np.log10(geometry.range_km / 1000.0)
    aperture_slow = link.reference_power_dbm - range_db - clear_db - cirrus_slow
    aperture_dbm = np.interp(fast_t, slow_t, aperture_slow) + scint_db
    track_true = aperture_dbm + 10 * np.log10(station.tracking_split)
    track_slow = 10 * np.log10(np.mean(10 ** (track_true.reshape(n_slow, per_slow) / 10), axis=1))

    # Pointing: mount lag (keyhole) and along-track ephemeris bias, in the tangent frame.
    limit = math.radians(station.mount_az_rate_limit_deg_s)
    accel = math.radians(station.mount_az_accel_limit_deg_s2)
    azimuth_actual = _mount_track(slow_t, geometry.azimuth_rad, limit, accel)
    mount = np.stack([(geometry.azimuth_rad - azimuth_actual) * np.cos(elevation), np.zeros(n_slow)], 1)
    bias = spec.ephemeris_time_offset_s * geometry.rate_rad_s[:, None] * np.stack(
        [geometry.along_x, geometry.along_y], 1
    )
    # Residual station pointing-model error (fixed per station, tens of microradians),
    # which the offload also absorbs; it is not a planted cause.
    azimuth, el = geometry.azimuth_rad, elevation
    pointing_model = np.stack(
        [25e-6 * np.cos(azimuth - 0.7) + 12e-6 * np.cos(el), 18e-6 * np.sin(azimuth + 0.3) - 10e-6 * np.sin(el)], 1
    )
    pointing = bias + pointing_model
    state, offload, episodes, labels = _lock_logic(track_slow, pointing, mount, station, 1.0 / SLOW_RATE_HZ)
    locked_fast = np.repeat(state == STATE_LOCKED, per_slow)
    offload_log = np.where((state == STATE_LOCKED)[:, None], offload + rng.normal(scale=3e-6, size=offload.shape), 0.0)

    # Tip-tilt input: two layer groups with Butterworth spectra, plus optional vibration.
    bank = _ButterworthBank(fs)
    tilt = np.zeros((n_fast, 2))
    spectra = []
    for axis in range(2):
        for group in ("ground", "free"):
            zi = None
            for b in range(blocks):
                lo, hi = b * block, min((b + 1) * block, n_fast)
                c = centres[b]
                fraction = cond.ground_fraction[c] if group == "ground" else 1.0 - cond.ground_fraction[c]
                variance = cond.tilt_variance_axis_rad2[c] * fraction
                speed = cond.ground_speed_m_s[c] if group == "ground" else cond.free_speed_m_s[c]
                cutoff = 0.3 * speed / station.aperture_diameter_m
                sos, gain = bank.design(cutoff)
                drive = rng.normal(size=hi - lo) * math.sqrt(variance / gain)
                if zi is None:
                    zi = signal.sosfilt_zi(sos) * 0.0
                out, zi = signal.sosfilt(sos, drive, zi=zi)
                tilt[lo:hi, axis] += out
                if axis == 0:
                    spectra.append((b, group, variance, cutoff))
    vibration = np.zeros((n_fast, 2))
    if spec.vibration:
        phase = (
            2 * math.pi * float(spec.vibration["frequency_hz"]) * fast_t
            + rng.uniform(0, 2 * math.pi)
            + np.cumsum(rng.normal(scale=0.02 * math.sqrt(1.0 / fs), size=n_fast))
        )
        vibration[:, 0] = float(spec.vibration["amplitude_x_rad"]) * np.sin(phase)
        vibration[:, 1] = float(spec.vibration["amplitude_y_rad"]) * np.sin(phase + 1.1)
    noise_sigma = station.tracking_noise_rad * 10 ** ((station.tracking_noise_reference_dbm - track_true) / 20)
    noise = rng.normal(size=(n_fast, 2)) * noise_sigma[:, None]

    gain = gain_for_bandwidth(fs, station.loop_delay_frames, station.rejection_bandwidth_hz)
    numerator, denominator = loop_coefficients(station.loop_delay_frames, gain)

    def complementary(x: FloatArray) -> FloatArray:
        return np.asarray(signal.lfilter(numerator, denominator, x, axis=0))

    t_tilt, t_vib, t_noise = complementary(tilt), complementary(vibration), complementary(noise)
    r_turb, r_vib, r_noise = tilt - t_tilt, vibration - t_vib, -t_noise
    residual = r_turb + r_vib + r_noise
    command = t_tilt + t_vib + t_noise
    measured = residual + noise

    theta_c2 = tables.theta_c_rad**2

    def tilt_db(r: FloatArray) -> FloatArray:
        return np.asarray(-DB_PER_NEPER_POWER * np.sum(r * r, axis=1) / theta_c2)

    # Higher-order coupling by quantile mapping of a Greenwood-time AR(1) process.
    tau_ho = 0.314 * cond.fried_parameter_m[centres] / np.maximum(cond.greenwood_speed_m_s[centres], 1.0)
    z_ho = _ar1(rng, np.exp(-1.0 / (fs * tau_ho)), block, n_fast)
    probability = special.ndtr(z_ho)
    d_over_r0 = np.interp(fast_t, slow_t, d_over_r0_slow)
    variant = defocus_variant(spec.defocus_waves_rms) if spec.defocus_waves_rms else TIP_TILT
    eta_ho = tables.quantile(variant, d_over_r0, probability)
    eta_clean = tables.quantile(TIP_TILT, d_over_r0, probability)
    eta_ao = tables.quantile(LOW_ORDER_AO, d_over_r0, probability)
    static_only = tables.static_relative.get(variant, 1.0)

    constant_db = 10 * np.log10(station.optical_throughput * (1 - station.tracking_split) * tables.eta0)
    base_tilt = tilt_db(residual)
    base_optics = 10 * np.log10(eta_ho)
    would_be = aperture_dbm + constant_db + base_optics + base_tilt

    monitor = link.monitor_noise_db
    pointing_ok = np.repeat(
        np.hypot(*(pointing - offload).T) + np.hypot(*mount.T) < station.acquisition_fov_rad, per_slow
    )
    track_log = np.where(
        pointing_ok, track_true + rng.normal(scale=monitor, size=n_fast), link.no_signal_dbm + rng.normal(scale=0.5, size=n_fast)
    )
    fiber_log = np.where(
        locked_fast, would_be + rng.normal(scale=monitor, size=n_fast), link.no_signal_dbm + rng.normal(scale=0.5, size=n_fast)
    )
    measured_log = np.where(locked_fast[:, None], measured, np.nan)
    command_log = np.where(locked_fast[:, None], command, np.nan)

    fast = {
        "t_s": fast_t.astype(np.float32),
        "fiber_power_dbm": fiber_log.astype(np.float32),
        "track_power_dbm": track_log.astype(np.float32),
        "resid_x_urad": (measured_log[:, 0] * 1e6).astype(np.float32),
        "resid_y_urad": (measured_log[:, 1] * 1e6).astype(np.float32),
        "fsm_x_urad": (command_log[:, 0] * 1e6).astype(np.float32),
        "fsm_y_urad": (command_log[:, 1] * 1e6).astype(np.float32),
    }
    slow = {
        "t_s": slow_t,
        "az_cmd_deg": np.degrees(geometry.azimuth_rad),
        "el_cmd_deg": np.degrees(elevation),
        "az_enc_deg": np.degrees(azimuth_actual),
        "el_enc_deg": np.degrees(elevation),
        "offload_x_urad": offload_log[:, 0] * 1e6,
        "offload_y_urad": offload_log[:, 1] * 1e6,
        "lock_state": state.astype(np.int8),
        "ground_wind_m_s": np.full(n_slow, spec.ground_wind_m_s),
    }

    def as32(x: FloatArray) -> NDArray[np.float32]:
        return np.asarray(x, dtype=np.float32)

    tilt_group = {
        frozenset(): as32(np.zeros(n_fast)),
        frozenset({"vibration"}): as32(tilt_db(residual - r_vib) - base_tilt),
        frozenset({"tilt_residual"}): as32(tilt_db(r_vib) - base_tilt),
        frozenset({"vibration", "tilt_residual"}): as32(-base_tilt),
    }
    optics_group = {
        frozenset(): as32(np.zeros(n_fast)),
        frozenset({"static_optics"}): as32(10 * np.log10(eta_clean) - base_optics),
        frozenset({"higher_order_turbulence"}): as32(10 * np.log10(static_only) - base_optics),
        frozenset({"static_optics", "higher_order_turbulence"}): as32(-base_optics),
    }
    restore = {name: np.repeat(mask, per_slow) for name, mask in labels.items()}

    # Expected-value gain of a faster tilt loop from the true (declared) input spectra.
    fix_gain = _bandwidth_fix_truth(
        spectra, blocks, block, n_fast, station, gain, vibration_spec=spec.vibration,
        noise_sigma=noise_sigma, theta_c2=theta_c2,
    )
    truth = PassTruth(
        spec=spec,
        fast_rate_hz=fs,
        would_be_fiber_dbm=as32(would_be),
        locked=locked_fast,
        additive_db={"cloud": as32(np.interp(fast_t, slow_t, cirrus_slow)), "scintillation": as32(-scint_db)},
        tilt_group_db=tilt_group,
        optics_group_db=optics_group,
        restore=restore,
        bandwidth_fix_db=as32(fix_gain),
        low_order_ao_fix_db=as32(10 * np.log10(eta_ao) - base_optics),
        slow={
            "d_over_r0": d_over_r0_slow,
            "fried_parameter_m": cond.fried_parameter_m,
            "cirrus_db": cirrus_slow,
            "mount_error_rad": np.hypot(*mount.T),
            "elevation_rad": elevation,
        },
        episodes=episodes,
        parameters={
            "ephemeris_time_offset_s": spec.ephemeris_time_offset_s,
            "defocus_waves_rms": spec.defocus_waves_rms,
            "static_loss_db": float(-10 * math.log10(static_only)),
            "static_loss_in_pass_db": float(np.mean(optics_group[frozenset({"static_optics"})][locked_fast])),
            "vibration": spec.vibration,
            "loop_gain": gain,
            "tilt_variance_ratio_outer_scale": atmosphere.tilt_ratio,
            "mean_true_residual_rms_rad": float(np.sqrt(np.mean(np.sum(residual[locked_fast] ** 2, axis=1) / 2))),
            "mean_vibration_residual_rms_rad": float(np.sqrt(np.mean(np.sum(r_vib[locked_fast] ** 2, axis=1) / 2))),
            "keyhole": any(e["trigger"] == "keyhole" for e in episodes),
            "planted": spec.planted(),
        },
    )
    return fast, slow, truth


def _bandwidth_fix_truth(
    spectra: list[tuple[int, str, float, float]],
    blocks: int,
    block: int,
    n_fast: int,
    station: StationConfig,
    gain: float,
    *,
    vibration_spec: dict[str, float] | None,
    noise_sigma: FloatArray,
    theta_c2: float,
    new_rate_hz: float = 4000.0,
    new_bandwidth_hz: float = 200.0,
) -> FloatArray:
    from .station import rejection

    fs = station.loop_rate_hz
    delay = station.loop_delay_frames
    new_gain = gain_for_bandwidth(new_rate_hz, delay, new_bandwidth_hz)
    frequency = np.linspace(0.05, new_rate_hz / 2, 20000)
    df = frequency[1] - frequency[0]
    s_old = np.abs(rejection(frequency, fs, delay, gain)) ** 2
    s_old[frequency > fs / 2] = 0.0
    s_new = np.abs(rejection(frequency, new_rate_hz, delay, new_gain)) ** 2
    t_old = np.mean(np.abs(1 - rejection(np.linspace(0, fs / 2, 4001), fs, delay, gain)) ** 2)
    t_new = np.mean(np.abs(1 - rejection(np.linspace(0, new_rate_hz / 2, 4001), new_rate_hz, delay, new_gain)) ** 2)
    by_block: dict[int, list[tuple[float, float]]] = {}
    for b, _group, variance, cutoff in spectra:
        by_block.setdefault(b, []).append((variance, cutoff))
    gain_db = np.zeros(n_fast)
    vib_term = 0.0
    if vibration_spec:
        f_v = float(vibration_spec["frequency_hz"])
        amp2 = 0.5 * (float(vibration_spec["amplitude_x_rad"]) ** 2 + float(vibration_spec["amplitude_y_rad"]) ** 2)
        vib_term = amp2 * (
            float(np.abs(rejection(np.array([f_v]), new_rate_hz, delay, new_gain))[0] ** 2)
            - float(np.abs(rejection(np.array([f_v]), fs, delay, gain))[0] ** 2)
        )
    for b in range(blocks):
        lo, hi = b * block, min((b + 1) * block, n_fast)
        delta = 0.0
        for variance, cutoff in by_block.get(b, []):
            shape = 1.0 / (1.0 + (frequency / cutoff) ** 6)
            norm = float(np.sum(shape) * df)
            psd = variance * shape / norm
            delta += float(np.sum(psd * (s_new - s_old)) * df)
        noise_var = float(np.mean(noise_sigma[lo:hi] ** 2))
        # Same noise PSD level at the faster rate: per-frame variance doubles, band doubles.
        delta += noise_var * (2.0 * t_new - t_old)
        total = 2.0 * delta + vib_term
        gain_db[lo:hi] = DB_PER_NEPER_POWER * total / theta_c2 * -1.0
    return gain_db


def write_parquet(path: Path, columns: dict[str, FloatArray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: np.asarray(v) for k, v in columns.items()}), path, compression="zstd")


def read_parquet(path: Path) -> dict[str, FloatArray]:
    table = pq.read_table(path)
    return {name: table.column(name).to_numpy() for name in table.column_names}
