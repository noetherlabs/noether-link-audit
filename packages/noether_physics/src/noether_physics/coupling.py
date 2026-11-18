"""Scalar Fraunhofer overlap with an ideal, centered Gaussian fiber mode.

This component is independent of the reduced-order plant's coupling proxy.
It integrates a piecewise-constant pupil field against the back-propagated
Gaussian, retaining the analytic mode power over the entire plane.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf, erfc  # type: ignore[import-untyped]


def _real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real numbers")
    return np.asarray(array, dtype=np.float64)


def _gaussian_overlap_components(
    pupil_amplitude: ArrayLike,
    *,
    pixel_pitch_m: float,
    wavelength_m: float,
    focal_length_m: float,
    mode_radius_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Prepare fixed geometry shared by the reference and OPD objective.

    Return max-normalized amplitude, exact Gaussian cell integrals, and
    incident power in back-propagated mode-radius units. The full receiving
    mode power in these units is pi/2, independent of the supplied window.
    """
    amplitude = _real_array(pupil_amplitude, "pupil_amplitude")
    if amplitude.ndim != 2 or min(amplitude.shape) < 2:
        raise ValueError("pupil_amplitude must have a two-dimensional shape >= 2")
    if np.any(amplitude < 0) or not np.any(amplitude > 0):
        raise ValueError("pupil_amplitude must be nonnegative with nonzero power")
    for name, value in (
        ("pixel_pitch_m", pixel_pitch_m),
        ("wavelength_m", wavelength_m),
        ("focal_length_m", focal_length_m),
        ("mode_radius_m", mode_radius_m),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    # Normalize lengths by the back-propagated mode radius and amplitudes
    # by their maximum; the overlap fraction is invariant to either scale.
    radius = wavelength_m * (focal_length_m / mode_radius_m) / np.pi
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("back-propagated mode radius is not representable")
    pitch = pixel_pitch_m / radius
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("normalized pixel pitch is not representable")
    amplitude = amplitude / np.max(amplitude)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        power = float(np.sum(amplitude**2) * np.square(np.float64(pitch)))
    if not np.isfinite(power) or power <= 0:
        raise ValueError("sampled pupil power is not representable")

    def cell_integrals(count: int) -> NDArray[np.float64]:
        edges = (np.arange(count + 1, dtype=np.float64) - count / 2) * pitch
        left, right = edges[:-1], edges[1:]
        integrals = np.asarray(erf(right) - erf(left), dtype=np.float64)
        # erf rounds to +/-1 in the tails. Complementary error functions
        # preserve small but representable overlap on either side of the axis.
        # Keep erf near zero: erfc would instead round very small arguments
        # to 1 and lose narrow central-cell integrals.
        positive, negative = left >= 1, right <= -1
        integrals[positive] = erfc(left[positive]) - erfc(right[positive])
        integrals[negative] = erfc(-right[negative]) - erfc(-left[negative])
        return np.asarray(integrals * (np.sqrt(np.pi) / 2), dtype=np.float64)

    weights = cell_integrals(amplitude.shape[0])[:, None] * cell_integrals(
        amplitude.shape[1]
    )[None, :]
    return amplitude, weights, power


def gaussian_fiber_coupling(
    pupil_amplitude: ArrayLike,
    opd_m: ArrayLike,
    *,
    pixel_pitch_m: float,
    wavelength_m: float,
    focal_length_m: float,
    mode_radius_m: float,
) -> float:
    """Fraction of pupil power coupled into a centered Gaussian focal mode.

    All lengths are in metres. ``mode_radius_m`` is the focal-plane 1/e
    *field* radius (also the 1/e**2 intensity radius); it is not a diameter.
    The pupil samples form rectangular, square-pitch cells centered on the
    optical axis. Amplitude is nonnegative and dimensionless. OPD is the
    optical path difference, not a reflective surface displacement. The
    field is zero outside the supplied grid, and constant within each cell.

    For a scalar, monochromatic, paraxial ideal lens, Fourier reciprocity
    gives a pupil-plane Gaussian radius ``w = wavelength*f/(pi*mode_radius)``.
    Its full-plane power is ``pi*w**2/2``. Gaussian cell integrals are exact;
    the continuous input field is approximated by its sampled cell values.
    Refine the pupil sampling to establish convergence, especially for
    aperture edges or rapidly varying OPD. No sampling-accuracy certificate
    is inferred from a result in [0, 1]. No clipping is performed.

    The fraction excludes upstream transmission, polarization mismatch,
    Fresnel reflection, fiber loss and propagation away from best focus.
    A Gaussian is an idealized mode, not a measured fiber eigenmode.
    """
    amplitude, weights, power = _gaussian_overlap_components(
        pupil_amplitude,
        pixel_pitch_m=pixel_pitch_m,
        wavelength_m=wavelength_m,
        focal_length_m=focal_length_m,
        mode_radius_m=mode_radius_m,
    )
    opd = _real_array(opd_m, "opd_m")
    if opd.shape != amplitude.shape:
        raise ValueError("amplitude and OPD must have matching two-dimensional shapes >= 2")
    # Reducing the OPD modulo wavelength avoids overflow in the phase scale.
    phase = 2 * np.pi * (np.remainder(opd, wavelength_m) / wavelength_m)
    field = amplitude * np.exp(1j * phase)
    overlap = np.sum(field * weights)
    # Normalize before squaring so that representable efficiencies survive
    # even when the unnormalized overlap square or power product would not.
    efficiency = float(np.abs(overlap / np.sqrt(power)) ** 2 / (np.pi / 2))
    if not np.isfinite(efficiency) or not 0 <= efficiency <= 1:
        raise ValueError("overlap violates the power bound; inspect numerical scales")
    return efficiency
