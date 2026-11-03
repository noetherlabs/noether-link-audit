"""Station, pass geometry, atmosphere and tip-tilt loop models for the sample audit.

Validity: reduced-order models for a hypothetical LEO downlink station. They are
not calibrated to a site, terminal or instrument.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, optimize, special  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]

EARTH_RADIUS_KM = 6371.0
EARTH_MU_KM3_S2 = 398600.4418
# One-axis Z-tilt angle variance for Kolmogorov turbulence, in (lambda/D)^2 (D/r0)^(5/3).
Z_TILT_COEFFICIENT = 0.1816
DB_PER_NEPER_POWER = 10.0 / math.log(10.0)


@dataclass(frozen=True, slots=True)
class StationConfig:
    """Declared configuration of the hypothetical station OGS-A (SI units)."""

    aperture_diameter_m: float = 0.40
    central_obscuration: float = 0.30
    wavelength_m: float = 1.55e-6
    optical_throughput: float = 0.70
    tracking_split: float = 0.10
    loop_rate_hz: float = 2000.0
    loop_delay_frames: int = 2
    rejection_bandwidth_hz: float = 100.0
    tracking_noise_rad: float = 0.25e-6
    tracking_noise_reference_dbm: float = -42.0
    fsm_range_rad: float = 400e-6
    acquisition_fov_rad: float = 1.0e-3
    nominal_acquisition_s: float = 3.0
    search_s_per_fov_area: float = 0.5
    lock_threshold_dbm: float = -54.0
    lock_loss_hold_s: float = 0.2
    mount_az_rate_limit_deg_s: float = 8.0
    mount_az_accel_limit_deg_s2: float = 4.0
    offload_time_constant_s: float = 1.0
    data_rate_gbps: float = 10.0
    sensitivity_dbm: float = -40.0
    min_elevation_deg: float = 20.0
    altitude_km: float = 550.0

    def __post_init__(self) -> None:
        if self.loop_delay_frames < 1:
            raise ValueError("loop_delay_frames must be at least one")
        if not 0.0 <= self.central_obscuration < 0.9:
            raise ValueError("central_obscuration must be in [0, 0.9)")
        if not 0.0 < self.tracking_split < 1.0:
            raise ValueError("tracking_split must be in (0, 1)")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StationConfig:
        return cls(**values)

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- loop


def loop_coefficients(delay_frames: int, gain: float) -> tuple[FloatArray, FloatArray]:
    """Numerator/denominator of T = C/(1+C) for an integrator with pure frame delay."""
    denominator = np.zeros(delay_frames + 1)
    denominator[0], denominator[1] = 1.0, -1.0
    denominator[delay_frames] += gain
    numerator = np.zeros(delay_frames + 1)
    numerator[delay_frames] = gain
    return numerator, denominator


def rejection(frequency_hz: FloatArray, rate_hz: float, delay_frames: int, gain: float) -> NDArray[np.complex128]:
    """Error rejection S(f) = 1/(1 + L) with L = g z^-d / (1 - z^-1)."""
    z_inv = np.exp(-2j * np.pi * np.asarray(frequency_hz, dtype=np.float64) / rate_hz)
    integrator = 1.0 - z_inv
    return np.asarray(integrator / (integrator + gain * z_inv**delay_frames))


def gain_for_bandwidth(rate_hz: float, delay_frames: int, bandwidth_hz: float) -> float:
    """Integrator gain whose -3 dB error-rejection frequency equals bandwidth_hz."""

    def excess(gain: float) -> float:
        value = abs(rejection(np.array([bandwidth_hz]), rate_hz, delay_frames, gain)[0])
        return float(value - 1.0 / math.sqrt(2.0))

    return float(optimize.brentq(excess, 1e-4, 0.9))


def loop_noise_integrals(rate_hz: float, delay_frames: int, gain: float) -> tuple[float, float]:
    """Mean of |S|^2 and |T|^2 over 0..Nyquist: white-noise variance gains."""
    frequency = np.linspace(0.0, rate_hz / 2.0, 20001)
    s = rejection(frequency, rate_hz, delay_frames, gain)
    t = 1.0 - s
    return float(np.mean(np.abs(s) ** 2)), float(np.mean(np.abs(t) ** 2))


# ----------------------------------------------------------------------- geometry


@dataclass(frozen=True, slots=True)
class PassGeometry:
    """Pass sampled at a fixed rate from minimum elevation to minimum elevation."""

    time_s: FloatArray
    elevation_rad: FloatArray
    azimuth_rad: FloatArray
    range_km: FloatArray
    rate_rad_s: FloatArray
    along_x: FloatArray
    along_y: FloatArray
    az_rate_rad_s: FloatArray
    el_rate_rad_s: FloatArray


def pass_geometry(
    max_elevation_deg: float,
    *,
    min_elevation_deg: float,
    altitude_km: float,
    azimuth_offset_deg: float,
    sample_rate_hz: float,
) -> PassGeometry:
    """Great-circle pass over a spherical, non-rotating Earth.

    The tangent frame is x along increasing azimuth (scaled by cos E) and y along
    increasing elevation. Earth rotation and orbit eccentricity are ignored.
    """
    radius = EARTH_RADIUS_KM + altitude_km
    omega_orbit = math.sqrt(EARTH_MU_KM3_S2 / radius**3)
    e_max, e_min = math.radians(max_elevation_deg), math.radians(min_elevation_deg)
    psi0 = math.acos(EARTH_RADIUS_KM / radius * math.cos(e_max)) - e_max
    gamma_min = math.acos(EARTH_RADIUS_KM / radius * math.cos(e_min)) - e_min
    theta_max = math.acos(math.cos(gamma_min) / math.cos(psi0))
    half_duration = theta_max / omega_orbit
    dt = 1.0 / sample_rate_hz
    time = np.arange(-half_duration, half_duration + 0.5 * dt, dt)
    theta = omega_orbit * time
    sat = radius * np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    station = EARTH_RADIUS_KM * np.array([math.cos(psi0), 0.0, math.sin(psi0)])
    line = sat - station
    up = station / EARTH_RADIUS_KM
    east = np.array([0.0, 1.0, 0.0])
    north = np.array([-math.sin(psi0), 0.0, math.cos(psi0)])
    distance = np.linalg.norm(line, axis=1)

    def project(axis: FloatArray) -> FloatArray:
        return np.asarray(np.einsum("ij,j->i", line, axis))

    elevation = np.arcsin(np.clip(project(up) / distance, -1.0, 1.0))
    azimuth = np.unwrap(np.arctan2(project(east), project(north))) + math.radians(azimuth_offset_deg)
    keep = elevation >= e_min - 1e-9
    time, elevation, azimuth, distance = time[keep], elevation[keep], azimuth[keep], distance[keep]
    az_rate = np.gradient(azimuth, time)
    el_rate = np.gradient(elevation, time)
    sky_x = az_rate * np.cos(elevation)
    rate = np.hypot(sky_x, el_rate)
    return PassGeometry(
        time_s=time - time[0],
        elevation_rad=elevation,
        azimuth_rad=azimuth,
        range_km=distance,
        rate_rad_s=rate,
        along_x=sky_x / np.maximum(rate, 1e-12),
        along_y=el_rate / np.maximum(rate, 1e-12),
        az_rate_rad_s=az_rate,
        el_rate_rad_s=el_rate,
    )


def sky_rates(
    time_s: FloatArray, azimuth_rad: FloatArray, elevation_rad: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """LOS angular rate and its tangent-frame unit vector from pointing angles."""
    sky_x = np.gradient(np.unwrap(azimuth_rad), time_s) * np.cos(elevation_rad)
    sky_y = np.gradient(elevation_rad, time_s)
    rate = np.hypot(sky_x, sky_y)
    safe = np.maximum(rate, 1e-12)
    return rate, sky_x / safe, sky_y / safe


# --------------------------------------------------------------------- atmosphere


def von_karman_tilt_ratio(diameter_m: float, outer_scale_m: float) -> float:
    """Z-tilt variance with a von Karman outer scale, relative to Kolmogorov."""
    if not math.isfinite(outer_scale_m):
        return 1.0
    x0 = math.pi * diameter_m / outer_scale_m

    def kernel(x: float, offset: float) -> float:
        return float(x * (x * x + offset * offset) ** (-11.0 / 6.0) * (special.jv(2, x) / x) ** 2)

    def total(offset: float) -> float:
        parts = [(0.0, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, np.inf)]
        return float(sum(integrate.quad(kernel, a, b, args=(offset,), limit=400)[0] for a, b in parts))

    return total(x0) / total(0.0)


@dataclass(frozen=True, slots=True)
class AtmosphereConditions:
    """Per-pass-sample atmospheric quantities at the station wavelength."""

    fried_parameter_m: FloatArray
    tilt_variance_axis_rad2: FloatArray
    ground_fraction: FloatArray
    ground_speed_m_s: FloatArray
    free_speed_m_s: FloatArray
    greenwood_speed_m_s: FloatArray
    log_intensity_variance: FloatArray
    scintillation_speed_m_s: FloatArray
    scintillation_height_m: FloatArray


class Atmosphere:
    """Hufnagel-Valley Cn^2 with a Bufton wind profile and tracking pseudo-wind."""

    def __init__(
        self,
        *,
        r0_500nm_zenith_m: float,
        ground_wind_m_s: float,
        upper_wind_rms_m_s: float,
        outer_scale_m: float,
        wavelength_m: float,
        diameter_m: float,
    ) -> None:
        self.wavelength_m = wavelength_m
        self.diameter_m = diameter_m
        self.outer_scale_m = outer_scale_m
        self.height_m = np.concatenate([[0.0], np.geomspace(1.0, 30000.0, 3000)])
        self.wind_m_s = ground_wind_m_s + 30.0 * np.exp(-(((self.height_m - 9400.0) / 4800.0) ** 2))
        self.upper_wind_rms_m_s = upper_wind_rms_m_s

        def r0_zenith(ground: float, wavelength: float) -> float:
            k = 2 * math.pi / wavelength
            integral = float(integrate.trapezoid(self._cn2(ground), self.height_m))
            return float((0.423 * k * k * integral) ** (-3.0 / 5.0))

        self.ground_cn2 = float(
            optimize.brentq(
                lambda g: r0_zenith(g, 500e-9) - r0_500nm_zenith_m, 1e-19, 1e-11, xtol=1e-22
            )
        )
        self.cn2 = self._cn2(self.ground_cn2)
        self.tilt_ratio = von_karman_tilt_ratio(diameter_m, outer_scale_m)

    def _cn2(self, ground: float) -> FloatArray:
        h = self.height_m
        upper = 0.00594 * (self.upper_wind_rms_m_s / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
        return np.asarray(upper + 2.7e-16 * np.exp(-h / 1500.0) + ground * np.exp(-h / 100.0))

    def conditions(self, elevation_rad: FloatArray, rate_rad_s: FloatArray) -> AtmosphereConditions:
        elevation = np.atleast_1d(np.asarray(elevation_rad, dtype=np.float64))
        rate = np.atleast_1d(np.asarray(rate_rad_s, dtype=np.float64))
        secant = 1.0 / np.sin(elevation)
        h, cn2 = self.height_m, self.cn2
        k = 2 * math.pi / self.wavelength_m
        total = float(integrate.trapezoid(cn2, h))
        r0 = (0.423 * k * k * secant * total) ** (-3.0 / 5.0)
        # Slant pseudo-wind from tracking, added in quadrature to the natural wind.
        pseudo = rate[:, None] * h[None, :] * secant[:, None]
        speed = np.sqrt(self.wind_m_s[None, :] ** 2 + pseudo**2)
        ground = h < 1000.0
        weight = cn2[None, :]
        ground_integral = float(integrate.trapezoid(cn2[ground], h[ground]))
        free_integral = total - ground_integral

        def weighted(power: float, mask: NDArray[np.bool_], norm: float) -> FloatArray:
            values = integrate.trapezoid(weight[:, mask] * speed[:, mask] ** power, h[mask], axis=1)
            return np.asarray(values / norm)

        ground_speed = np.sqrt(weighted(2.0, ground, ground_integral))
        free_speed = np.sqrt(weighted(2.0, ~ground, free_integral))
        all_heights = np.ones_like(h, dtype=bool)
        greenwood = weighted(5.0 / 3.0, all_heights, total) ** (3.0 / 5.0)
        tilt = (
            Z_TILT_COEFFICIENT
            * (self.wavelength_m / self.diameter_m) ** 2
            * (self.diameter_m / r0) ** (5.0 / 3.0)
            * self.tilt_ratio
        )
        moment = float(integrate.trapezoid(cn2 * h ** (5.0 / 6.0), h))
        rytov = 2.25 * k ** (7.0 / 6.0) * secant ** (11.0 / 6.0) * moment
        scint_height = (moment / total) ** (6.0 / 5.0)
        averaging = (1.0 + 1.062 * k * self.diameter_m**2 / (4.0 * scint_height * secant)) ** (-7.0 / 6.0)
        log_variance = np.log1p(rytov * averaging)
        scint_speed = np.array(
            [np.interp(scint_height, h, row) for row in speed], dtype=np.float64
        )
        return AtmosphereConditions(
            fried_parameter_m=np.asarray(r0),
            tilt_variance_axis_rad2=np.asarray(tilt),
            ground_fraction=np.full_like(elevation, ground_integral / total),
            ground_speed_m_s=ground_speed,
            free_speed_m_s=free_speed,
            greenwood_speed_m_s=np.asarray(greenwood),
            log_intensity_variance=np.asarray(log_variance),
            scintillation_speed_m_s=scint_speed,
            scintillation_height_m=np.full_like(elevation, scint_height),
        )


def r0_from_tilt(
    tilt_variance_axis_rad2: FloatArray,
    wavelength_m: float,
    diameter_m: float,
    assumed_outer_scale_m: float,
) -> FloatArray:
    """Invert the one-axis Z-tilt variance for r0 under an assumed von Karman outer scale."""
    reduction = von_karman_tilt_ratio(diameter_m, assumed_outer_scale_m)
    ratio = np.maximum(np.asarray(tilt_variance_axis_rad2), 1e-30) / (
        Z_TILT_COEFFICIENT * (wavelength_m / diameter_m) ** 2 * reduction
    )
    return np.asarray(diameter_m * ratio ** (-3.0 / 5.0))
