"""Analytic and independent Fourier checks for Gaussian fiber coupling."""

import math

import numpy as np
import pytest
from noether_physics.coupling import gaussian_fiber_coupling
from numpy.typing import NDArray
from scipy.integrate import quad

WAVELENGTH_M = 1.55e-6
FOCAL_LENGTH_M = 0.1
PUPIL_MODE_RADIUS_M = 0.001
MODE_RADIUS_M = WAVELENGTH_M * FOCAL_LENGTH_M / (np.pi * PUPIL_MODE_RADIUS_M)


def grid(n: int, width_m: float) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    pitch = width_m / n
    axis = (np.arange(n, dtype=np.float64) - (n - 1) / 2) * pitch
    x, y = np.meshgrid(axis, axis)
    return x, y, pitch


def coupling(amplitude: NDArray[np.float64], opd: NDArray[np.float64], pitch: float) -> float:
    return gaussian_fiber_coupling(
        amplitude,
        opd,
        pixel_pitch_m=pitch,
        wavelength_m=WAVELENGTH_M,
        focal_length_m=FOCAL_LENGTH_M,
        mode_radius_m=MODE_RADIUS_M,
    )


@pytest.mark.parametrize("pupil_radius_ratio", [0.6, 1.0, 1.8])
@pytest.mark.parametrize("tilt_rad", [0.0, 1.0e-4, -2.5e-4])
def test_gaussian_width_and_tilt_analytic_limit(pupil_radius_ratio: float, tilt_rad: float) -> None:
    x, y, pitch = grid(384, 12 * PUPIL_MODE_RADIUS_M)
    wi = pupil_radius_ratio * PUPIL_MODE_RADIUS_M
    wg = PUPIL_MODE_RADIUS_M
    amplitude = np.exp(-(x**2 + y**2) / wi**2)
    q = 2 * np.pi * tilt_rad / WAVELENGTH_M
    expected = 4 * wi**2 * wg**2 / (wi**2 + wg**2) ** 2
    expected *= np.exp(-q**2 * wi**2 * wg**2 / (2 * (wi**2 + wg**2)))
    assert coupling(amplitude, tilt_rad * x, pitch) == pytest.approx(expected, abs=2e-4)


def test_piston_amplitude_and_unit_scaling_invariance() -> None:
    x, y, pitch = grid(128, 8 * PUPIL_MODE_RADIUS_M)
    amplitude = np.exp(-(x**2 + y**2) / PUPIL_MODE_RADIUS_M**2)
    opd = 1e-4 * x
    before_amplitude, before_opd = amplitude.copy(), opd.copy()
    expected = coupling(amplitude, opd, pitch)
    assert coupling(17 * amplitude, opd + 0.371 * WAVELENGTH_M, pitch) == pytest.approx(expected)
    scale = 1000.0
    assert gaussian_fiber_coupling(
        amplitude, opd * scale, pixel_pitch_m=pitch * scale,
        wavelength_m=WAVELENGTH_M * scale, focal_length_m=FOCAL_LENGTH_M * scale,
        mode_radius_m=MODE_RADIUS_M * scale,
    ) == pytest.approx(expected)
    np.testing.assert_array_equal(amplitude, before_amplitude)
    np.testing.assert_array_equal(opd, before_opd)


def test_circular_pupil_gaussian_analytic_optimum_and_convergence() -> None:
    # Nonzero solution of exp(q) = 1 + 2q; q = (pupil radius / mode radius)^2.
    q = 1.2564312086261697
    radius = np.sqrt(q) * PUPIL_MODE_RADIUS_M
    expected = 2 * (1 - np.exp(-q)) ** 2 / q
    values = []
    for n in (64, 128, 256):
        x, y, pitch = grid(n, 2.4 * radius)
        amplitude = (x**2 + y**2 <= radius**2).astype(float)
        values.append(coupling(amplitude, np.zeros_like(x), pitch))
    assert expected == pytest.approx(0.814528755178, abs=1e-12)
    assert abs(values[-1] - expected) < 1e-4
    assert abs(values[-1] - expected) < abs(values[0] - expected)


@pytest.mark.parametrize("obscuration", [0.3, 0.6])
def test_annular_pupil_matches_full_power_analytic_limit(obscuration: float) -> None:
    q = 1.2564312086261697
    radius = np.sqrt(q) * PUPIL_MODE_RADIUS_M
    x, y, pitch = grid(384, 2.4 * radius)
    radius2 = x**2 + y**2
    amplitude = ((radius2 <= radius**2) & (radius2 >= (obscuration * radius)**2)).astype(float)
    expected = 2 * (np.exp(-q * obscuration**2) - np.exp(-q))**2 / (q * (1 - obscuration**2))
    assert coupling(amplitude, np.zeros_like(x), pitch) == pytest.approx(expected, abs=4e-4)


def test_coarse_sampling_is_not_a_continuous_field_accuracy_certificate() -> None:
    errors = []
    for n in (8, 32, 128):
        x, y, pitch = grid(n, 8 * PUPIL_MODE_RADIUS_M)
        amplitude = np.exp(-(x**2 + y**2) / PUPIL_MODE_RADIUS_M**2)
        value = coupling(amplitude, np.zeros_like(x), pitch)
        assert 0 <= value <= 1
        errors.append(1 - value)
    assert errors[0] > 0.1
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 7e-4


def test_full_mode_power_is_not_truncated_to_the_pupil_window() -> None:
    x, y, pitch = grid(64, 0.5 * PUPIL_MODE_RADIUS_M)
    amplitude = np.exp(-(x**2 + y**2) / PUPIL_MODE_RADIUS_M**2)
    value = coupling(amplitude, np.zeros_like(x), pitch)
    # Renormalizing the Gaussian to this window would give approximately 1.
    assert 0 < value < 0.2


@pytest.mark.parametrize("row,column", [(8, 14), (8, 1), (14, 8), (1, 8)])
def test_representable_gaussian_tails_survive_cell_integration(row: int, column: int) -> None:
    amplitude = np.zeros((16, 16))
    amplitude[row, column] = 1
    # Each pixel is one pupil-mode radius wide; these cells lie 6--7
    # radii from one axis. erf endpoint subtraction alone returns zero.
    integral_x = quad(lambda x: np.exp(-x*x), column - 8, column - 7, epsabs=1e-40)[0]
    integral_y = quad(lambda y: np.exp(-y*y), row - 8, row - 7, epsabs=1e-40)[0]
    expected = (integral_x * integral_y)**2 / (np.pi / 2)
    assert expected > 1e-35
    actual = coupling(amplitude, np.zeros_like(amplitude), PUPIL_MODE_RADIUS_M)
    assert actual == pytest.approx(expected, rel=1e-12, abs=0)


def test_independent_focal_plane_fft_agrees() -> None:
    # An odd grid includes the origin exactly. Zero padding increases focal
    # sampling; the FFT includes dx**2/(lambda*f) for the unitary lens scaling.
    x, y, pitch = grid(129, 8 * PUPIL_MODE_RADIUS_M)
    amplitude = (x**2 + y**2 < (1.1 * PUPIL_MODE_RADIUS_M) ** 2).astype(float)
    opd = 1e-4 * x + 0.08 * WAVELENGTH_M * (x**2 - y**2) / PUPIL_MODE_RADIUS_M**2
    field = amplitude * np.exp(2j * np.pi * opd / WAVELENGTH_M)
    padded = np.pad(field, ((129, 129), (129, 129)))
    focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    focal *= pitch**2 / (WAVELENGTH_M * FOCAL_LENGTH_M)
    axis = np.fft.fftshift(np.fft.fftfreq(padded.shape[0], pitch))
    axis *= WAVELENGTH_M * FOCAL_LENGTH_M
    fx, fy = np.meshgrid(axis, axis)
    focal_pitch = WAVELENGTH_M * FOCAL_LENGTH_M / (padded.shape[0] * pitch)
    mode = np.exp(-(fx**2 + fy**2) / MODE_RADIUS_M**2)
    input_power = np.sum(amplitude**2) * pitch**2
    np.testing.assert_allclose(np.sum(np.abs(focal)**2) * focal_pitch**2, input_power, rtol=1e-12)
    # Fourier transform of each constant square pixel contributes a sinc.
    # Without this factor the FFT describes midpoint quadrature instead of
    # the implementation's explicitly piecewise-constant incident field.
    focal *= np.sinc(fx * pitch / (WAVELENGTH_M * FOCAL_LENGTH_M))
    focal *= np.sinc(fy * pitch / (WAVELENGTH_M * FOCAL_LENGTH_M))
    overlap = np.sum(focal * mode) * focal_pitch**2
    expected = abs(overlap)**2 / (input_power * np.pi * MODE_RADIUS_M**2 / 2)
    assert coupling(amplitude, opd, pitch) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("parameter", ["pixel_pitch_m", "wavelength_m", "focal_length_m", "mode_radius_m"])
def test_invalid_optical_scales_rejected(parameter: str, bad: float) -> None:
    kwargs = dict(pixel_pitch_m=1e-4, wavelength_m=WAVELENGTH_M,
                  focal_length_m=FOCAL_LENGTH_M, mode_radius_m=MODE_RADIUS_M)
    kwargs[parameter] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        gaussian_fiber_coupling(np.ones((4, 4)), np.zeros((4, 4)), **kwargs)


@pytest.mark.parametrize("bad", [np.zeros((4, 4)), -np.ones((4, 4)),
                                np.full((4, 4), np.nan), np.ones((4, 4), dtype=complex)])
def test_invalid_amplitude_rejected(bad: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        coupling(bad, np.zeros((4, 4)), 1e-4)


def test_invalid_opd_and_shape_rejected() -> None:
    for opd in (np.zeros((4, 3)), np.full((4, 4), np.inf), np.ones((4, 4), dtype=complex)):
        with pytest.raises(ValueError):
            coupling(np.ones((4, 4)), opd, 1e-4)


@pytest.mark.parametrize("pitch", [1e-200, 1e200])
def test_unrepresentable_power_rejected(pitch: float) -> None:
    with pytest.raises(ValueError, match="power is not representable"):
        coupling(np.ones((4, 4)), np.zeros((4, 4)), pitch)


@pytest.mark.parametrize("normalized_pitch", [1e-100, 6e153])
def test_representable_efficiency_survives_extreme_field_scales(normalized_pitch: float) -> None:
    # Uniform square spans [-p,p]^2 in mode-radius units. Arrange the closed
    # form to retain small results without squaring an unnormalized integral.
    p = normalized_pitch
    expected = (np.sqrt(np.pi / 2) * (math.erf(p) / p) * math.erf(p))**2
    assert expected > 0
    actual = coupling(np.ones((2, 2)), np.zeros((2, 2)), p * PUPIL_MODE_RADIUS_M)
    assert actual == pytest.approx(expected, rel=1e-12, abs=0)
