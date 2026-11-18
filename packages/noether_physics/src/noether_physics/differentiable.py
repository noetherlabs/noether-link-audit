"""Optional JAX objective for the fixed-geometry Gaussian receiving mode."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .coupling import _gaussian_overlap_components

if TYPE_CHECKING:
    import jax


def make_gaussian_fiber_objective(
    pupil_amplitude: ArrayLike,
    *,
    pixel_pitch_m: float,
    wavelength_m: float,
    focal_length_m: float,
    mode_radius_m: float,
) -> Callable[[ArrayLike], jax.Array]:
    """Construct a scalar coupling objective ``eta(opd_m)`` for real OPD.

    Amplitude and geometry are validated and fixed at construction; construct
    the objective outside JAX transformations. The optical assumptions and
    exact Gaussian cell integrals are those of ``gaussian_fiber_coupling``.
    The receiving mode has its full-plane power, not power renormalized to
    the supplied pupil window. Amplitude or geometry derivatives are not
    provided by this factory.

    Enable JAX float64 before construction and evaluation. This module does
    not change JAX configuration or select a device. CPU float64 is the
    qualification target. The optional ``differentiable`` dependency supplies
    JAX; importing this module alone does not require it.

    The returned function accepts a real float64 OPD array in metres with
    the same two-dimensional shape as the amplitude. It composes with
    ``jax.grad``, ``jax.jit`` and ``jax.vmap``. For a command-to-OPD map B,
    use ``eta(opd0 - B @ commands)`` with the appropriate pupil reshape.
    Any reflective surface-to-OPD factor belongs in B, not in this objective.

    Static shape/dtype errors raise during evaluation or tracing. Eager
    nonfinite OPD raises; under a transformation nonfinite OPD propagates
    into a nonfinite objective. Callers must reject any nonfinite value or
    gradient, including derivatives unrepresentable at extreme finite
    physical scales. There are no
    callbacks, epsilon floors, clipping, or replacement values. Phase is
    reduced modulo wavelength before conversion to radians; its complex
    exponential and the coupling remain periodic across the wrap.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as error:
        if error.name in {"jax", "jaxlib"}:
            raise ImportError(
                "Install noether-link-audit[differentiable] to construct this objective"
            ) from error
        raise

    def require_x64() -> None:
        if not jax.config.x64_enabled:
            raise RuntimeError("Enable JAX float64 before constructing or evaluating the objective")

    require_x64()
    amplitude, weights, power = _gaussian_overlap_components(
        pupil_amplitude,
        pixel_pitch_m=pixel_pitch_m,
        wavelength_m=wavelength_m,
        focal_length_m=focal_length_m,
        mode_radius_m=mode_radius_m,
    )
    shape = amplitude.shape
    fixed_amplitude = jnp.array(amplitude, dtype=jnp.float64, copy=True)
    fixed_weights = jnp.array(weights, dtype=jnp.float64, copy=True)
    root_power = jnp.asarray(np.sqrt(power), dtype=jnp.float64)
    wavelength = float(wavelength_m)

    def objective(opd_m: ArrayLike) -> jax.Array:
        require_x64()
        opd = jnp.asarray(opd_m)
        if opd.shape != shape:
            raise ValueError("opd_m must match the fixed two-dimensional pupil shape")
        if opd.dtype != jnp.dtype("float64"):
            raise TypeError("opd_m must have real float64 dtype")
        if not isinstance(opd, jax.core.Tracer) and not bool(jnp.all(jnp.isfinite(opd))):
            raise ValueError("opd_m must contain finite real numbers")
        phase = 2 * jnp.pi * (jnp.remainder(opd, wavelength) / wavelength)
        field = fixed_amplitude * jnp.exp(1j * phase)
        normalized_overlap = jnp.sum(field * fixed_weights) / root_power
        # Squared components have a finite derivative at destructive overlap;
        # differentiating an intermediate complex magnitude is unnecessary.
        return (
            jnp.real(normalized_overlap) ** 2 + jnp.imag(normalized_overlap) ** 2
        ) / (jnp.pi / 2)

    return objective
