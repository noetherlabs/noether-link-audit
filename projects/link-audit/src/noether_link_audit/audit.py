"""Telemetry-only loss audit for an optical ground station.

Inputs are what a station logs (fiber and tracking-sensor power, tracking
residuals, FSM commands, commanded and encoder pointing, offload offsets, lock
state) plus its declared configuration. Ground truth is never read here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal  # type: ignore[import-untyped]

from .attribution import LossModel
from .coupling_tables import LOW_ORDER_AO, TIP_TILT, CouplingTables
from .station import (
    DB_PER_NEPER_POWER,
    EARTH_RADIUS_KM,
    FloatArray,
    StationConfig,
    gain_for_bandwidth,
    loop_noise_integrals,
    r0_from_tilt,
    rejection,
    sky_rates,
)

SLOW_RATE_HZ = 10.0
STATE_ACQUIRING = 1
STATE_LOCKED = 2
VALID_TRACK_DBM = -70.0


@dataclass(frozen=True)
class AuditSettings:
    assumed_outer_scale_m: float = 20.0
    window_s: float = 2.0
    what_if_rate_hz: float = 4000.0
    what_if_bandwidth_hz: float = 200.0
    vibration_peak_ratio: float = 20.0
    vibration_detect_fraction: float = 0.5
    timing_detect_s: float = 0.05
    timing_detect_sigma: float = 5.0
    static_report_db: float = 0.3


@dataclass
class PassFeatures:
    pass_id: str
    night_id: str
    fs: float
    per_slow: int
    slow: dict[str, FloatArray]
    locked_fast: NDArray[np.bool_]
    fiber_fast: FloatArray
    track_fast: FloatArray
    coupling_fast: FloatArray
    tilt_power_fast: FloatArray
    windows: list[dict[str, Any]]
    window_of_sample: NDArray[np.int64]
    vibration: dict[str, Any] | None
    timing: dict[str, float]
    episodes: list[dict[str, Any]]
    median_psd: dict[str, FloatArray] = field(default_factory=dict)


@dataclass
class FleetCalibration:
    clear_sky_c0_db: float
    clear_sky_extinction_db_per_airmass: float
    clear_sky_points: int
    coupling_reference_db: float
    acquisition_nominal_s: float
    acquisition_search_s_per_area: float
    acquisition_episodes: int


def _range_km(elevation: FloatArray, altitude_km: float) -> FloatArray:
    radius = EARTH_RADIUS_KM + altitude_km
    return np.asarray(
        np.sqrt(radius**2 - (EARTH_RADIUS_KM * np.cos(elevation)) ** 2) - EARTH_RADIUS_KM * np.sin(elevation)
    )


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


class Auditor:
    def __init__(self, station: StationConfig, tables: CouplingTables, settings: AuditSettings) -> None:
        self.station = station
        self.tables = tables
        self.settings = settings
        fs = station.loop_rate_hz
        self.gain = gain_for_bandwidth(fs, station.loop_delay_frames, station.rejection_bandwidth_hz)
        self.new_gain = gain_for_bandwidth(
            settings.what_if_rate_hz, station.loop_delay_frames, settings.what_if_bandwidth_hz
        )
        self.mean_s2_old, self.mean_t2_old = loop_noise_integrals(fs, station.loop_delay_frames, self.gain)
        _, self.mean_t2_new = loop_noise_integrals(
            settings.what_if_rate_hz, station.loop_delay_frames, self.new_gain
        )
        self.theta_c2 = tables.theta_c_rad**2

    # ------------------------------------------------------------------ phase 1

    def extract(self, pass_id: str, night_id: str, fast: dict[str, Any], slow: dict[str, Any]) -> PassFeatures:
        st, fs = self.station, self.station.loop_rate_hz
        per_slow = int(round(fs / SLOW_RATE_HZ))
        n_slow = len(slow["t_s"])
        lock = np.asarray(slow["lock_state"])
        locked_slow = lock == STATE_LOCKED
        locked_fast = np.repeat(locked_slow, per_slow)[: len(fast["t_s"])]
        elevation = np.radians(np.asarray(slow["el_cmd_deg"], dtype=np.float64))
        azimuth = np.radians(np.asarray(slow["az_cmd_deg"], dtype=np.float64))
        t = np.asarray(slow["t_s"], dtype=np.float64)
        rate, along_x, along_y = sky_rates(t, azimuth, elevation)
        mount = np.abs(
            np.radians(np.asarray(slow["az_cmd_deg"]) - np.asarray(slow["az_enc_deg"])) * np.cos(elevation)
        )
        track = np.asarray(fast["track_power_dbm"], dtype=np.float64)
        fiber = np.asarray(fast["fiber_power_dbm"], dtype=np.float64)
        valid = track > VALID_TRACK_DBM
        linear = np.where(valid, 10 ** (track / 10), np.nan).reshape(n_slow, per_slow)
        with np.errstate(invalid="ignore"):
            frac_valid = np.mean(valid.reshape(n_slow, per_slow), axis=1)
            track_slow = np.where(frac_valid > 0.5, 10 * np.log10(np.nanmean(linear, axis=1)), np.nan)
        coupling = np.where(locked_fast & valid, fiber - track, np.nan)
        slow_out = {
            "t_s": t,
            "elevation_rad": elevation,
            "range_km": _range_km(elevation, st.altitude_km),
            "airmass": 1.0 / np.sin(elevation),
            "rate_rad_s": rate,
            "along_x": along_x,
            "along_y": along_y,
            "lock_state": lock,
            "track_dbm": track_slow,
            "mount_error_rad": mount,
            "offload_x_rad": np.asarray(slow["offload_x_urad"]) * 1e-6,
            "offload_y_rad": np.asarray(slow["offload_y_urad"]) * 1e-6,
        }
        windows, window_of_sample, vibration, median_psd = self._tilt_windows(fast, locked_fast, coupling)
        tilt_power = (
            np.asarray(fast["resid_x_urad"], dtype=np.float64) ** 2 + np.asarray(fast["resid_y_urad"], dtype=np.float64) ** 2
        ) * 1e-12
        return PassFeatures(
            pass_id=pass_id,
            night_id=night_id,
            fs=fs,
            per_slow=per_slow,
            slow=slow_out,
            locked_fast=locked_fast,
            fiber_fast=fiber,
            track_fast=track,
            coupling_fast=coupling,
            tilt_power_fast=tilt_power,
            windows=windows,
            window_of_sample=window_of_sample,
            vibration=vibration,
            timing=self._timing(slow_out),
            episodes=self._episodes(slow_out),
            median_psd=median_psd,
        )

    def _tilt_windows(
        self, fast: dict[str, Any], locked_fast: NDArray[np.bool_], coupling: FloatArray
    ) -> tuple[list[dict[str, Any]], NDArray[np.int64], dict[str, Any] | None, dict[str, FloatArray]]:
        st, fs, settings = self.station, self.station.loop_rate_hz, self.settings
        length = int(settings.window_s * fs)
        n = len(locked_fast)
        resid = np.stack([fast["resid_x_urad"], fast["resid_y_urad"]], 1).astype(np.float64) * 1e-6
        command = np.stack([fast["fsm_x_urad"], fast["fsm_y_urad"]], 1).astype(np.float64) * 1e-6
        freq = np.fft.rfftfreq(1000, 1.0 / fs)
        df = freq[1] - freq[0]
        s2_old = np.abs(rejection(freq, fs, st.loop_delay_frames, self.gain)) ** 2
        s2_new = np.abs(rejection(freq, settings.what_if_rate_hz, st.loop_delay_frames, self.new_gain)) ** 2
        floor_band = (freq >= 800) & (freq <= 990)
        raw: list[dict[str, Any]] = []
        for start in range(0, n - length + 1, length):
            stop = start + length
            if not np.all(locked_fast[start:stop]):
                continue
            r, u = resid[start:stop], command[start:stop]
            if not (np.all(np.isfinite(r)) and np.all(np.isfinite(u))):
                continue
            _, pr = signal.welch(r, fs, nperseg=1000, axis=0)
            _, pp = signal.welch(r + u, fs, nperseg=1000, axis=0)
            raw.append({"start": start, "stop": stop, "r": r, "u": u, "pr": pr, "pp": pp})
        if not raw:
            return [], np.full(n, -1), None, {}
        median_pr = np.median(np.stack([w["pr"] for w in raw]), axis=0)
        median_pp = np.median(np.stack([w["pp"] for w in raw]), axis=0)
        vibration = self._detect_vibration(freq, raw)
        windows = []
        for w in raw:
            pr, pp, r, u = w["pr"], w["pp"], w["r"], w["u"]
            noise_psd = np.median(pr[floor_band] / s2_old[floor_band, None], axis=0)
            sigma_n2 = noise_psd * fs / 2
            var_true = np.maximum(np.var(r, axis=0) - sigma_n2 * self.mean_s2_old + sigma_n2 * self.mean_t2_old, 0.0)
            vib_resid = np.zeros(2)
            vib_input = np.zeros(2)
            line = np.zeros_like(pp)
            s2_old_v = s2_new_v = 0.0
            if vibration:
                band = np.abs(freq - vibration["frequency_hz"]) <= 4.0
                bg_r = ndimage.median_filter(pr, size=(15, 1), mode="nearest")
                bg_p = ndimage.median_filter(pp, size=(15, 1), mode="nearest")
                vib_resid = np.sum(np.maximum(pr - bg_r, 0.0)[band], axis=0) * df
                line[band] = np.maximum(pp - bg_p, 0.0)[band]
                vib_input = np.sum(line, axis=0) * df
                f_v = np.array([vibration["frequency_hz"]])
                s2_old_v = float(np.abs(rejection(f_v, fs, st.loop_delay_frames, self.gain))[0] ** 2)
                s2_new_v = float(
                    np.abs(rejection(f_v, settings.what_if_rate_hz, st.loop_delay_frames, self.new_gain))[0] ** 2
                )
            pol = signal.detrend(r + u, axis=0)
            turbulent_tilt = np.maximum(np.var(pol, axis=0) - sigma_n2 - vib_input, 1e-18)
            r0 = float(
                r0_from_tilt(
                    np.array([turbulent_tilt.mean()]), st.wavelength_m, st.aperture_diameter_m,
                    settings.assumed_outer_scale_m,
                )[0]
            )
            tilt_psd = np.maximum(pp - noise_psd[None, :] - line, 0.0)
            old_fd = np.sum(tilt_psd * s2_old[:, None], axis=0) * df + vib_input * s2_old_v + sigma_n2 * self.mean_t2_old
            new_fd = (
                np.sum(tilt_psd * s2_new[:, None], axis=0) * df
                + vib_input * s2_new_v
                + 2.0 * sigma_n2 * self.mean_t2_new
            )
            windows.append(
                {
                    "start": w["start"],
                    "stop": w["stop"],
                    "d_over_r0": st.aperture_diameter_m / r0,
                    "fried_parameter_m": r0,
                    "var_true_sum": float(var_true.sum()),
                    "var_vib_sum": float(min(vib_resid.sum(), var_true.sum())),
                    "vib_input_sum": float(vib_input.sum()),
                    "delta_bandwidth_var_sum": float((new_fd - old_fd).sum()),
                    "noise_rms_rad": float(np.sqrt(sigma_n2.mean())),
                    "coupling_mean_db": float(np.nanmean(coupling[w["start"] : w["stop"]])),
                }
            )
        centres = np.array([(w["start"] + w["stop"]) // 2 for w in windows])
        index = np.clip(np.searchsorted(centres, np.arange(n)), 0, len(centres) - 1)
        prev = np.clip(index - 1, 0, len(centres) - 1)
        nearer_prev = np.abs(np.arange(n) - centres[prev]) < np.abs(centres[index] - np.arange(n))
        window_of_sample = np.where(nearer_prev, prev, index).astype(np.int64)
        return windows, window_of_sample, vibration, {"frequency_hz": freq, "residual": median_pr, "pol": median_pp}

    def _detect_vibration(self, freq: FloatArray, raw: list[dict[str, Any]]) -> dict[str, Any] | None:
        settings = self.settings
        search = (freq >= 5.0) & (freq <= 500.0)
        peaks = []
        for w in raw:
            psd = w["pr"].sum(axis=1)
            background = ndimage.median_filter(psd, size=15, mode="nearest")
            ratio = np.where(search, psd / np.maximum(background, 1e-30), 0.0)
            k = int(np.argmax(ratio))
            peaks.append((freq[k], ratio[k]))
        strong = [f for f, ratio in peaks if ratio >= settings.vibration_peak_ratio]
        if len(strong) < settings.vibration_detect_fraction * len(raw):
            return None
        frequency = float(np.median(strong))
        consistent = sum(abs(f - frequency) <= 2.0 for f in strong)
        if consistent < settings.vibration_detect_fraction * len(raw):
            return None
        # Sub-bin estimate: parabola through the log median PSD around the peak bin.
        median_psd = np.log(np.median(np.stack([w["pr"].sum(axis=1) for w in raw]), axis=0))
        k = int(np.argmin(np.abs(freq - frequency)))
        left, centre, right = median_psd[k - 1], median_psd[k], median_psd[k + 1]
        curvature = left - 2 * centre + right
        delta = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
        refined = float(freq[k] + np.clip(delta, -0.5, 0.5) * (freq[1] - freq[0]))
        return {"frequency_hz": refined, "bin_frequency_hz": frequency, "detected_fraction": consistent / len(raw)}

    def _timing(self, slow: dict[str, FloatArray]) -> dict[str, float]:
        locked = slow["lock_state"] == STATE_LOCKED
        rate = slow["rate_rad_s"][locked]
        along = (slow["offload_x_rad"] * slow["along_x"] + slow["offload_y_rad"] * slow["along_y"])[locked]
        cross = (slow["offload_x_rad"] * -slow["along_y"] + slow["offload_y_rad"] * slow["along_x"])[locked]
        if len(rate) < 20 or np.var(rate) < 1e-12:
            return {"offset_s": float("nan"), "stderr_s": float("nan"), "samples": float(len(rate))}
        design = np.stack([np.ones_like(rate), rate], 1)
        coef, *_ = np.linalg.lstsq(design, along, rcond=None)
        residual = along - design @ coef
        dof = max(len(rate) - 2, 1)
        cov = np.linalg.inv(design.T @ design) * float(residual @ residual) / dof
        cross_coef, *_ = np.linalg.lstsq(design, cross, rcond=None)
        return {
            "offset_s": float(coef[1]),
            "stderr_s": float(math.sqrt(cov[1, 1])),
            "intercept_rad": float(coef[0]),
            "cross_track_slope_s": float(cross_coef[1]),
            "samples": float(len(rate)),
            "max_bias_rad": float(abs(coef[1]) * np.max(slow["rate_rad_s"])),
        }

    def _episodes(self, slow: dict[str, FloatArray]) -> list[dict[str, Any]]:
        st = self.station
        acquiring = slow["lock_state"] == STATE_ACQUIRING
        episodes = []
        n = len(acquiring)
        for start, stop in _runs(acquiring):
            if start == 0:
                trigger = "pass_start"
            else:
                before = slice(max(0, start - 3), start)
                track_before = slow["track_dbm"][before]
                if np.any(np.isfinite(track_before) & (track_before < st.lock_threshold_dbm + 1.0)):
                    trigger = "cloud"
                elif np.any(slow["mount_error_rad"][max(0, start - 3) : start + 3] > 0.5 * st.fsm_range_rad):
                    trigger = "keyhole"
                else:
                    trigger = "unknown"
            track = slow["track_dbm"][start:stop]
            mount = slow["mount_error_rad"][start:stop]
            mount_blocked = mount >= st.acquisition_fov_rad
            cloud_blocked = np.isfinite(track) & (track < st.lock_threshold_dbm + 1.0) & ~mount_blocked
            active = ~(mount_blocked | cloud_blocked)
            episodes.append(
                {
                    "start": start,
                    "stop": stop,
                    "trigger": trigger,
                    "truncated": stop >= n,
                    "mount_blocked": mount_blocked,
                    "cloud_blocked": cloud_blocked,
                    "active": active,
                    "active_s": float(active.sum()) / SLOW_RATE_HZ,
                    "first_active": int(start + np.argmax(active)) if active.any() else None,
                }
            )
        return episodes

    # ------------------------------------------------------------------ phase 2

    def calibrate(self, features: list[PassFeatures]) -> FleetCalibration:
        st = self.station
        ys, xs = [], []
        for f in features:
            s = f.slow
            ok = (s["lock_state"] == STATE_LOCKED) & np.isfinite(s["track_dbm"])
            ys.append(s["track_dbm"][ok] + 20 * np.log10(s["range_km"][ok] / 1000.0))
            xs.append(s["airmass"][ok])
        y, x = np.concatenate(ys), np.concatenate(xs)
        keep = np.ones_like(y, dtype=bool)
        coef = np.zeros(2)
        for _ in range(8):
            design = np.stack([np.ones(keep.sum()), -x[keep]], 1)
            coef, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
            residual = y - (coef[0] - coef[1] * x)
            keep = residual > -0.4
        self._clear = (float(coef[0]), float(coef[1]))
        for f in features:
            self._extinction(f)
        offsets = []
        for f in features:
            k_w = [self._coupling_offset(w) for w in f.windows]
            offsets.append(float(np.median(k_w)) if k_w else float("nan"))
            f.timing["coupling_offset_db"] = offsets[-1]
        reference = float(np.nanmedian(offsets))
        # Acquisition model: active time = T0 + k * max(0, (error/FOV)^2 - 1).
        rows = []
        for f in features:
            offset = self._pass_offset(f)
            for e in f.episodes:
                if e["truncated"] or e["first_active"] is None:
                    continue
                i = e["first_active"]
                s = f.slow
                bias = offset * s["rate_rad_s"][i] * np.array([s["along_x"][i], s["along_y"][i]])
                error = float(np.hypot(bias[0] + s["mount_error_rad"][i], bias[1]))
                x_val = max(0.0, (error / st.acquisition_fov_rad) ** 2 - 1.0)
                e["predicted_error_rad"] = error
                e["search_area"] = x_val
                rows.append((x_val, e["active_s"]))
        data = np.asarray(rows)
        if len(data) >= 3 and np.ptp(data[:, 0]) > 0.5:
            design = np.stack([np.ones(len(data)), data[:, 0]], 1)
            (t0, k), *_ = np.linalg.lstsq(design, data[:, 1], rcond=None)
        else:
            t0, k = float(np.median(data[:, 1])) if len(data) else st.nominal_acquisition_s, float("nan")
        return FleetCalibration(
            clear_sky_c0_db=self._clear[0],
            clear_sky_extinction_db_per_airmass=self._clear[1],
            clear_sky_points=int(keep.sum()),
            coupling_reference_db=reference,
            acquisition_nominal_s=float(t0),
            acquisition_search_s_per_area=float(k),
            acquisition_episodes=len(data),
        )

    def _pass_offset(self, f: PassFeatures) -> float:
        timing, settings = f.timing, self.settings
        offset, stderr = timing.get("offset_s", float("nan")), timing.get("stderr_s", float("nan"))
        if not math.isfinite(offset) or not math.isfinite(stderr):
            return 0.0
        detected = abs(offset) >= settings.timing_detect_s and abs(offset) >= settings.timing_detect_sigma * stderr
        timing["detected"] = float(detected)
        return float(offset) if detected else 0.0

    def _extinction(self, f: PassFeatures) -> None:
        s = f.slow
        model = self._clear[0] - 20 * np.log10(s["range_km"] / 1000.0) - self._clear[1] * s["airmass"]
        excess = model - s["track_dbm"]
        smoothed = ndimage.median_filter(np.nan_to_num(excess, nan=0.0), size=11, mode="nearest")
        smoothed = np.where(np.isfinite(excess), smoothed, np.nan)
        smoothed = np.where(smoothed < 0.2, 0.0, smoothed)
        filled = np.where(np.isfinite(smoothed), smoothed, 0.0)
        # Unobservable gaps (no signal while searching) take the clearer neighbour,
        # not a ramp: the last reading before a dropout is usually the cloud's worst.
        for a, b in _runs(~np.isfinite(smoothed)):
            neighbours = [smoothed[i] for i in (a - 1, b) if 0 <= i < len(smoothed) and np.isfinite(smoothed[i])]
            filled[a:b] = min(neighbours) if neighbours else 0.0
        s["clear_model_dbm"] = model
        s["extinction_db"] = filled

    def _coupling_offset(self, w: dict[str, Any]) -> float:
        ho = float(self.tables.mean_db(TIP_TILT, np.array([w["d_over_r0"]]))[0])
        tilt = -DB_PER_NEPER_POWER * w["var_true_sum"] / self.theta_c2
        return w["coupling_mean_db"] - ho - tilt

    # ------------------------------------------------------------------ phase 3

    def loss_model(
        self, f: PassFeatures, cal: FleetCalibration
    ) -> tuple[LossModel, dict[str, NDArray[np.float32]], dict[str, Any]]:
        st, fs, per_slow = self.station, f.fs, f.per_slow
        n = len(f.fiber_fast)
        s = f.slow
        locked = f.locked_fast
        slow_t = s["t_s"]
        fast_t = np.arange(n) / fs
        wins = f.windows
        w_idx = f.window_of_sample

        def per_window(key: str) -> FloatArray:
            return np.asarray([w[key] for w in wins], dtype=np.float64)[w_idx]

        # Map unlocked samples to a locked donor segment for coupling and per-sample terms.
        source = np.arange(n)
        locked_runs = _runs(locked)
        for a, b in _runs(~locked):
            after = [r for r in locked_runs if r[0] >= b]
            before = [r for r in locked_runs if r[1] <= a]
            donor = after[0] if after else (before[-1] if before else None)
            if donor is None:
                continue
            span = np.arange(donor[0], donor[1])
            source[a:b] = np.resize(span, b - a)
        offset_p = float(f.timing.get("coupling_offset_db", cal.coupling_reference_db))
        static_p = offset_p - cal.coupling_reference_db
        static_loss = max(0.0, -static_p)
        # Instantaneous tilt power from the measured residual, rescaled per window so
        # its mean equals the noise-corrected true residual variance.
        var_total = per_window("var_true_sum")
        tilt_power = f.tilt_power_fast[source]
        scale = np.ones(len(wins))
        for k, w in enumerate(wins):
            measured = float(np.nanmean(f.tilt_power_fast[w["start"] : w["stop"]]))
            scale[k] = w["var_true_sum"] / measured if measured > 0 else 0.0
        tilt_inst = np.where(np.isfinite(tilt_power), tilt_power * scale[w_idx], var_total)
        tilt_total = -DB_PER_NEPER_POWER * tilt_inst / self.theta_c2
        coupling = np.where(np.isfinite(f.coupling_fast), f.coupling_fast, np.nan)[source]
        coupling = np.where(np.isfinite(coupling), coupling, offset_p + tilt_total)
        ho_inst = coupling - offset_p - tilt_total
        track_model = np.interp(fast_t, slow_t, s["clear_model_dbm"] - s["extinction_db"])
        proxy = track_model + coupling
        base = np.where(locked, f.fiber_fast, proxy).astype(np.float32)

        # Scintillation: fast tracking-power fluctuation about a 1 s running mean.
        valid = locked & (f.track_fast > VALID_TRACK_DBM)
        width = int(fs)
        numerator = ndimage.uniform_filter1d(np.where(valid, 10 ** (f.track_fast / 10), 0.0), width)
        denominator = ndimage.uniform_filter1d(valid.astype(np.float64), width)
        with np.errstate(divide="ignore", invalid="ignore"):
            track_smooth = 10 * np.log10(numerator / denominator)
        scint = np.where(valid & (denominator > 0.5), f.track_fast - track_smooth, 0.0)
        vib_share = (
            np.clip(per_window("var_vib_sum") / np.maximum(var_total, 1e-30), 0.0, 1.0) if f.vibration else np.zeros(n)
        )
        additive = {
            "cloud": np.interp(fast_t, slow_t, s["extinction_db"]).astype(np.float32),
            "scintillation": (-scint).astype(np.float32),
            "tilt_residual": (-tilt_total * (1.0 - vib_share)).astype(np.float32),
            "vibration": (-tilt_total * vib_share).astype(np.float32),
            "static_optics": np.full(n, static_loss, dtype=np.float32),
            "higher_order_turbulence": (-ho_inst).astype(np.float32),
        }
        restore = self._restore_masks(f, cal, n, per_slow)

        # Fixes that improve a term without removing it.
        bandwidth_fix = (-DB_PER_NEPER_POWER * per_window("delta_bandwidth_var_sum") / self.theta_c2).astype(np.float32)
        ao_fix = np.zeros(n, dtype=np.float32)
        d_over_r0 = per_window("d_over_r0")
        for k in range(len(wins)):
            members = np.flatnonzero(w_idx == k)
            values = ho_inst[members]
            ranks = (np.argsort(np.argsort(values)) + 0.5) / max(len(values), 1)
            ao_db = 10 * np.log10(self.tables.quantile(LOW_ORDER_AO, np.full(len(values), d_over_r0[members[0]]), ranks))
            ao_fix[members] = (ao_db - values + static_loss).astype(np.float32)
        model = LossModel(
            rate_gbps=st.data_rate_gbps,
            sample_s=1.0 / fs,
            sensitivity_dbm=st.sensitivity_dbm,
            base_dbm=base,
            locked=locked.copy(),
            additive=additive,
            restore=restore,
        )
        observed = float(np.count_nonzero(locked & (f.fiber_fast >= st.sensitivity_dbm))) / fs * st.data_rate_gbps
        estimates = {
            "static_db": static_p,
            "median_d_over_r0": float(np.median([w["d_over_r0"] for w in wins])) if wins else float("nan"),
            "median_fried_parameter_m": float(np.median([w["fried_parameter_m"] for w in wins])) if wins else float("nan"),
            "mean_tilt_residual_rms_rad": float(np.sqrt(np.mean([w["var_true_sum"] for w in wins]) / 2)) if wins else float("nan"),
            "mean_vibration_residual_rms_rad": float(np.sqrt(np.mean([w["var_vib_sum"] for w in wins]) / 2)) if (wins and f.vibration) else 0.0,
            "observed_delivered_gb": observed,
            "max_extinction_db": float(np.max(s["extinction_db"])),
            "mean_extinction_db": float(np.mean(s["extinction_db"])),
        }
        fixes = {"faster_tilt_loop": bandwidth_fix, "low_order_ao": ao_fix}
        return model, fixes, estimates

    def _restore_masks(
        self, f: PassFeatures, cal: FleetCalibration, n: int, per_slow: int
    ) -> dict[str, NDArray[np.bool_]]:
        n_slow = len(f.slow["t_s"])
        masks = {name: np.zeros(n_slow, dtype=bool) for name in ("nominal_acquisition", "ephemeris", "keyhole", "cloud")}
        offset = self._pass_offset(f)
        nominal = max(cal.acquisition_nominal_s, 0.0)
        for e in f.episodes:
            idx = np.arange(e["start"], e["stop"])
            trigger = e["trigger"] if e["trigger"] in masks else "nominal_acquisition"
            if trigger in ("cloud", "keyhole"):
                masks[trigger][idx] = True
            masks["cloud"][idx[e["cloud_blocked"]]] = True
            masks["keyhole"][idx[e["mount_blocked"]]] = True
            active = idx[e["active"]]
            elapsed = np.arange(len(active)) / SLOW_RATE_HZ
            first = active[elapsed < nominal]
            rest = active[elapsed >= nominal]
            masks["nominal_acquisition" if trigger == "nominal_acquisition" else trigger][first] = True
            if offset:
                masks["ephemeris"][rest] = True
            else:
                masks["nominal_acquisition" if trigger == "nominal_acquisition" else trigger][rest] = True
        return {name: np.repeat(mask, per_slow)[:n] for name, mask in masks.items()}
