"""Monte Carlo single-mode fiber coupling tables for the sample station.

Phase screens come from HCIPy (finite FFT layers with a von Karman outer scale).
Each screen is scaled to a grid of D/r0, residual modes are removed by least
squares over the illuminated annulus, and coupling is evaluated with the
qualified scalar overlap in ``noether_physics.coupling``. Tables are relative to
the zero-aberration coupling of the same pupil and mode.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from noether_physics.coupling import gaussian_fiber_coupling
from scipy import optimize  # type: ignore[import-untyped]

from .station import FloatArray, StationConfig

D_OVER_R0_GRID = np.array([0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.4, 2.8, 3.3, 4.0, 5.0, 6.0])
PROBABILITIES = np.linspace(0.005, 0.995, 100)
FOCAL_LENGTH_M = 1.0
TIP_TILT = "tip_tilt"
LOW_ORDER_AO = "low_order_ao_11"


def defocus_variant(waves_rms: float) -> str:
    return f"tip_tilt+defocus_{waves_rms:.3f}"


@dataclass(frozen=True)
class CouplingTables:
    d_over_r0: FloatArray
    probabilities: FloatArray
    quantiles: dict[str, FloatArray]
    mean_log_db: dict[str, FloatArray]
    static_relative: dict[str, float]
    eta0: float
    theta_c_rad: float
    tilt_fit_max_relative_error: float
    mode_radius_m: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def _indices(self, d_over_r0: FloatArray, probability: FloatArray) -> tuple[Any, ...]:
        log_grid = np.log(self.d_over_r0)
        log_value = np.clip(np.log(np.maximum(d_over_r0, 1e-9)), log_grid[0], log_grid[-1])
        gi = np.clip(np.searchsorted(log_grid, log_value) - 1, 0, len(log_grid) - 2)
        gw = (log_value - log_grid[gi]) / (log_grid[gi + 1] - log_grid[gi])
        p = np.clip(probability, self.probabilities[0], self.probabilities[-1])
        step = self.probabilities[1] - self.probabilities[0]
        pf = (p - self.probabilities[0]) / step
        pi = np.clip(np.floor(pf).astype(int), 0, len(self.probabilities) - 2)
        pw = pf - pi
        return gi, gw, pi, pw

    def quantile(self, variant: str, d_over_r0: FloatArray, probability: FloatArray) -> FloatArray:
        """Relative coupling at the given quantile; bilinear in (log D/r0, p)."""
        table = self.quantiles[variant]
        gi, gw, pi, pw = self._indices(np.asarray(d_over_r0), np.asarray(probability))
        low = table[gi, pi] * (1 - pw) + table[gi, pi + 1] * pw
        high = table[gi + 1, pi] * (1 - pw) + table[gi + 1, pi + 1] * pw
        return np.asarray(low * (1 - gw) + high * gw)

    def probability_of(self, variant: str, d_over_r0: float, relative_coupling: FloatArray) -> FloatArray:
        """Inverse of ``quantile`` at one D/r0 (monotone interpolation)."""
        row = self.quantile(variant, np.full_like(self.probabilities, d_over_r0), self.probabilities)
        return np.asarray(np.interp(relative_coupling, row, self.probabilities))

    def mean_db(self, variant: str, d_over_r0: FloatArray) -> FloatArray:
        log_grid = np.log(self.d_over_r0)
        log_value = np.clip(np.log(np.maximum(np.asarray(d_over_r0), 1e-9)), log_grid[0], log_grid[-1])
        return np.asarray(np.interp(log_value, log_grid, self.mean_log_db[variant]))

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "link_audit_coupling_tables_v1",
            "d_over_r0": self.d_over_r0.tolist(),
            "probabilities": self.probabilities.tolist(),
            "quantiles": {k: np.round(v, 7).tolist() for k, v in self.quantiles.items()},
            "mean_log_db": {k: np.round(v, 7).tolist() for k, v in self.mean_log_db.items()},
            "static_relative": self.static_relative,
            "eta0": self.eta0,
            "theta_c_rad": self.theta_c_rad,
            "tilt_fit_max_relative_error": self.tilt_fit_max_relative_error,
            "mode_radius_m": self.mode_radius_m,
            "focal_length_m": FOCAL_LENGTH_M,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> CouplingTables:
        return cls(
            d_over_r0=np.asarray(payload["d_over_r0"]),
            probabilities=np.asarray(payload["probabilities"]),
            quantiles={k: np.asarray(v) for k, v in payload["quantiles"].items()},
            mean_log_db={k: np.asarray(v) for k, v in payload["mean_log_db"].items()},
            static_relative={k: float(v) for k, v in payload["static_relative"].items()},
            eta0=float(payload["eta0"]),
            theta_c_rad=float(payload["theta_c_rad"]),
            tilt_fit_max_relative_error=float(payload["tilt_fit_max_relative_error"]),
            mode_radius_m=float(payload["mode_radius_m"]),
            diagnostics=dict(payload.get("diagnostics", {})),
        )


class _Pupil:
    def __init__(self, station: StationConfig, grid_size: int) -> None:
        import hcipy  # optional physics dependency

        self.hcipy = hcipy
        self.size = grid_size
        self.diameter = station.aperture_diameter_m
        self.wavelength = station.wavelength_m
        self.grid = hcipy.make_pupil_grid(grid_size, self.diameter)
        aperture = hcipy.make_obstructed_circular_aperture(self.diameter, station.central_obscuration)
        self.amplitude = np.asarray(aperture(self.grid), dtype=np.float64).reshape(grid_size, grid_size)
        self.mask = self.amplitude > 0
        self.pitch = self.diameter / grid_size
        x = np.asarray(self.grid.x).reshape(grid_size, grid_size)
        y = np.asarray(self.grid.y).reshape(grid_size, grid_size)
        self.x, self.y = x, y
        zernike = hcipy.make_zernike_basis(11, self.diameter, self.grid)
        modes = np.stack([np.asarray(mode).reshape(grid_size, grid_size)[self.mask] for mode in zernike], 1)
        self.tilt_projector = self._projector(modes[:, :3])
        self.ao_projector = self._projector(modes)
        shape = modes[:, 3] - modes[:, 3].mean()
        self.defocus_unit = shape / math.sqrt(float(np.mean(shape**2)))

    @staticmethod
    def _projector(basis: FloatArray) -> FloatArray:
        q, _ = np.linalg.qr(basis)
        return np.asarray(q)

    def remove(self, projector: FloatArray, values: FloatArray) -> FloatArray:
        return np.asarray(values - projector @ (projector.T @ values))

    def coupling(self, phase_masked: FloatArray, mode_radius_m: float) -> float:
        opd = np.zeros_like(self.amplitude)
        opd[self.mask] = phase_masked * self.wavelength / (2 * math.pi)
        return gaussian_fiber_coupling(
            self.amplitude,
            opd,
            pixel_pitch_m=self.pitch,
            wavelength_m=self.wavelength,
            focal_length_m=FOCAL_LENGTH_M,
            mode_radius_m=mode_radius_m,
        )

    def layer(self, outer_scale_m: float, oversampling: int, seed: int) -> Any:
        cn2 = self.hcipy.Cn_squared_from_fried_parameter(self.diameter, self.wavelength)
        return self.hcipy.FiniteAtmosphericLayer(
            self.grid, cn2, L0=outer_scale_m, velocity=0, oversampling=oversampling, seed=seed
        )


def _mean_relative_tip_tilt(
    pupil: _Pupil, mode_radius: float, eta0: float, d_over_r0: float, screens: int,
    outer_scale_m: float, oversampling: int, seed: int,
) -> tuple[float, float]:
    layer = pupil.layer(outer_scale_m, oversampling, seed)
    scale = d_over_r0 ** (5.0 / 6.0)
    values = []
    for _ in range(screens):
        layer.reset(make_independent_realization=True)
        phase = np.asarray(layer.phase_for(pupil.wavelength)).reshape(pupil.size, pupil.size)[pupil.mask]
        values.append(pupil.coupling(scale * pupil.remove(pupil.tilt_projector, phase), mode_radius) / eta0)
    array = np.asarray(values)
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(len(array)))


def build_tables(
    station: StationConfig,
    *,
    defocus_waves_rms: tuple[float, ...],
    screens: int = 400,
    grid_size: int = 128,
    oversampling: int = 8,
    outer_scale_m: float = 25.0,
    seed: int = 2026,
    convergence_checks: bool = True,
) -> CouplingTables:
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    pupil = _Pupil(station, grid_size)
    zero = np.zeros(int(pupil.mask.sum()))
    radius_guess = station.wavelength_m * FOCAL_LENGTH_M / (math.pi * 0.45 * station.aperture_diameter_m)
    best = optimize.minimize_scalar(
        lambda w: -pupil.coupling(zero, w),
        bounds=(0.4 * radius_guess, 2.5 * radius_guess),
        method="bounded",
        options={"xatol": radius_guess * 1e-6},
    )
    mode_radius = float(best.x)
    eta0 = pupil.coupling(zero, mode_radius)

    # Tilt sensitivity: eta(theta)/eta0 ~ exp(-theta^2 / theta_c^2) for small offsets.
    lam_over_d = station.wavelength_m / station.aperture_diameter_m
    angles = np.linspace(0.05, 0.5, 10) * lam_over_d
    ratios = []
    for angle in angles:
        phase = 2 * math.pi * angle * pupil.x[pupil.mask] / station.wavelength_m
        ratios.append(pupil.coupling(phase, mode_radius) / eta0)
    ratios_array = np.asarray(ratios)
    inv_theta_c2 = float(np.sum(-np.log(ratios_array) * angles**2) / np.sum(angles**4))
    theta_c = 1.0 / math.sqrt(inv_theta_c2)
    fit = np.exp(-(angles**2) / theta_c**2)
    fit_error = float(np.max(np.abs(fit - ratios_array) / ratios_array))

    static_relative = {TIP_TILT: 1.0, LOW_ORDER_AO: 1.0}
    defocus_phase = {}
    for waves in defocus_waves_rms:
        name = defocus_variant(waves)
        defocus_phase[name] = 2 * math.pi * waves * pupil.defocus_unit
        static_relative[name] = pupil.coupling(defocus_phase[name], mode_radius) / eta0

    variants = [TIP_TILT, *defocus_phase.keys(), LOW_ORDER_AO]
    samples = {name: np.zeros((len(D_OVER_R0_GRID), screens)) for name in variants}
    tilt_removed_variance = np.zeros(screens)
    layer = pupil.layer(outer_scale_m, oversampling, seed)
    scales = D_OVER_R0_GRID ** (5.0 / 6.0)
    for s in range(screens):
        layer.reset(make_independent_realization=True)
        phase = np.asarray(layer.phase_for(station.wavelength_m)).reshape(grid_size, grid_size)[pupil.mask]
        tip_tilt = pupil.remove(pupil.tilt_projector, phase)
        low_order = pupil.remove(pupil.ao_projector, phase)
        tilt_removed_variance[s] = float(np.mean(tip_tilt**2))
        for g, scale in enumerate(scales):
            samples[TIP_TILT][g, s] = pupil.coupling(scale * tip_tilt, mode_radius) / eta0
            samples[LOW_ORDER_AO][g, s] = pupil.coupling(scale * low_order, mode_radius) / eta0
            for name, static in defocus_phase.items():
                samples[name][g, s] = pupil.coupling(scale * tip_tilt + static, mode_radius) / eta0

    quantiles = {name: np.quantile(values, PROBABILITIES, axis=1).T for name, values in samples.items()}
    mean_log_db = {
        name: np.mean(10 * np.log10(np.maximum(values, 1e-12)), axis=1) for name, values in samples.items()
    }
    diagnostics: dict[str, Any] = {
        "screens": screens,
        "grid_size": grid_size,
        "oversampling": oversampling,
        "outer_scale_m": outer_scale_m,
        "seed": seed,
        "tilt_removed_phase_variance_rad2_at_d_over_r0_1": float(tilt_removed_variance.mean()),
        "noll_tip_tilt_removed_kolmogorov_clear_pupil_rad2": 0.134,
        "mean_relative_coupling": {
            name: np.round(values.mean(axis=1), 6).tolist() for name, values in samples.items()
        },
    }
    if convergence_checks:
        checks = {}
        for label, size, over in (("grid128_over4", 128, 4), ("grid128_over16", 128, 16), ("grid256_over8", 256, 8)):
            check_pupil = pupil if size == grid_size else _Pupil(station, size)
            check_radius = mode_radius
            check_eta0 = check_pupil.coupling(np.zeros(int(check_pupil.mask.sum())), check_radius)
            mean, stderr = _mean_relative_tip_tilt(
                check_pupil, check_radius, check_eta0, 3.0, 150, outer_scale_m, over, seed + 17
            )
            checks[label] = {"d_over_r0": 3.0, "mean_relative_coupling": mean, "standard_error": stderr}
        index = int(np.argmin(np.abs(D_OVER_R0_GRID - 3.3)))
        checks["reference_grid128_over8_at_3p3"] = float(samples[TIP_TILT][index].mean())
        checks["reference_grid128_over8_interpolated_at_3p0"] = float(
            np.interp(math.log(3.0), np.log(D_OVER_R0_GRID), samples[TIP_TILT].mean(axis=1))
        )
        diagnostics["convergence"] = checks
    return CouplingTables(
        d_over_r0=D_OVER_R0_GRID.copy(),
        probabilities=PROBABILITIES.copy(),
        quantiles=quantiles,
        mean_log_db=mean_log_db,
        static_relative=static_relative,
        eta0=eta0,
        theta_c_rad=theta_c,
        tilt_fit_max_relative_error=fit_error,
        mode_radius_m=mode_radius,
        diagnostics=diagnostics,
    )


def load_or_build(path: Path, station: StationConfig, **kwargs: Any) -> CouplingTables:
    if path.exists():
        return CouplingTables.from_json(json.loads(path.read_text()))
    tables = build_tables(station, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tables.to_json(), indent=1, allow_nan=False) + "\n")
    return tables
