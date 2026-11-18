"""Fixed-geometry OPD derivatives against analytic and NumPy references."""

import math

import numpy as np
import pytest
from noether_physics.coupling import gaussian_fiber_coupling
from noether_physics.differentiable import make_gaussian_fiber_objective

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

WAVELENGTH = 1.55e-6
FOCAL_LENGTH = 0.1
PUPIL_RADIUS = 1e-3
MODE_RADIUS = WAVELENGTH * FOCAL_LENGTH / (np.pi * PUPIL_RADIUS)


@pytest.fixture(autouse=True)
def float64_context():
    # The caller, not the optical library, owns JAX configuration.
    with jax.enable_x64(True):
        yield


def geometry(pitch, scale=1.0):
    return {
        "pixel_pitch_m": pitch * scale,
        "wavelength_m": WAVELENGTH * scale,
        "focal_length_m": FOCAL_LENGTH * scale,
        "mode_radius_m": MODE_RADIUS * scale,
    }


def objective(amplitude, pitch):
    return make_gaussian_fiber_objective(amplitude, **geometry(pitch))


def reference(amplitude, opd, pitch):
    return gaussian_fiber_coupling(amplitude, opd, **geometry(pitch))


def grid(n, width):
    pitch = width / n
    axis = (np.arange(n, dtype=np.float64) - (n - 1) / 2) * pitch
    x, y = np.meshgrid(axis, axis)
    return x, y, pitch


def small_case():
    row, column = np.indices((5, 6), dtype=np.float64)
    amplitude = 0.4 + 0.03 * row + 0.07 * column
    amplitude[0, 0] = 0.0
    waves = 0.11 * np.sin(0.7 * row + 0.3 * column) + 0.02 * column
    waves[2, 3] = 0.0  # Finite differences cross a wavelength wrap here.
    return amplitude, waves, 0.25 * PUPIL_RADIUS


def test_qualified_backend_is_pinned_float64_cpu():
    assert jax.__version__ == "0.10.2"
    assert jax.default_backend() == "cpu"
    assert jax.config.x64_enabled


@pytest.mark.parametrize("step_waves", [1e-4, 1e-5])
def test_every_opd_derivative_matches_numpy_central_difference(step_waves):
    amplitude, waves, pitch = small_case()
    eta = objective(amplitude, pitch)
    # Differentiate dimensionless q=d/lambda, then compare with independent
    # NumPy evaluations. Both predeclared steps must pass, including q=0.
    gradient = np.asarray(jax.grad(lambda q: eta(q * WAVELENGTH))(waves))
    finite_difference = np.empty_like(waves)
    for index in np.ndindex(waves.shape):
        plus, minus = waves.copy(), waves.copy()
        plus[index] += step_waves
        minus[index] -= step_waves
        finite_difference[index] = (
            reference(amplitude, plus * WAVELENGTH, pitch)
            - reference(amplitude, minus * WAVELENGTH, pitch)
        ) / (2 * step_waves)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-6, atol=2e-9)
    assert float(eta(waves * WAVELENGTH)) == pytest.approx(
        reference(amplitude, waves * WAVELENGTH, pitch), rel=2e-13, abs=2e-14
    )


@pytest.mark.parametrize("difference_waves", [0.0, 0.17, 0.5])
def test_exact_two_cell_value_gradient_and_destructive_overlap(difference_waves):
    # Equal active cells have weight w=(sqrt(pi)/2*erf(h))**2. Thus
    # eta(q)=eta0*cos(pi*(q00-q01))**2, eta0=2*w**2/(h**2*(pi/2)).
    # This is a finite-cell identity, independent of the preparation helper.
    h = 0.3
    amplitude = np.array([[1.0, 1.0], [0.0, 0.0]])
    waves = np.array([[difference_waves, 0.0], [0.13, -0.21]])
    eta = objective(amplitude, h * PUPIL_RADIUS)
    value, gradient = jax.value_and_grad(lambda q: eta(q * WAVELENGTH))(waves)
    weight = (math.sqrt(math.pi) / 2 * math.erf(h)) ** 2
    eta0 = 2 * weight**2 / (h**2 * (math.pi / 2))
    expected = eta0 * math.cos(math.pi * difference_waves) ** 2
    slope = -math.pi * eta0 * math.sin(2 * math.pi * difference_waves)
    assert float(value) == pytest.approx(expected, rel=2e-13, abs=2e-14)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    np.testing.assert_allclose(gradient, [[slope, -slope], [0, 0]], rtol=2e-13, atol=2e-13)
    if difference_waves == 0.5:
        assert abs(float(value)) < 1e-28
        assert np.max(np.abs(gradient)) < 2e-13


@pytest.mark.parametrize("radius_ratio", [0.8, 1.3])
@pytest.mark.parametrize("tilt", [-1e-4, 1e-4])
def test_gaussian_tilt_derivative_converges_to_continuous_analytic_limit(radius_ratio, tilt):
    wi, wg = radius_ratio * PUPIL_RADIUS, PUPIL_RADIUS
    beta = (2 * math.pi / WAVELENGTH) ** 2 * wi**2 * wg**2 / (2 * (wi**2 + wg**2))
    expected_value = 4 * wi**2 * wg**2 / (wi**2 + wg**2) ** 2 * math.exp(-beta * tilt**2)
    expected_slope = -2 * beta * tilt * expected_value
    errors = []
    # Keep the same physical window on both grids. At its edge, even the
    # widest input's intensity is below exp(-2*(6/1.3)**2) < 4e-19.
    # These tolerances cover cell discretization, not AD versus FD error.
    for n, relative_tolerance in ((128, 4e-3), (256, 1e-3)):
        x, y, pitch = grid(n, 12 * wg)
        amplitude = np.exp(-(x**2 + y**2) / wi**2)
        eta = objective(amplitude, pitch)
        value, slope = jax.value_and_grad(lambda angle, eta=eta, x=x: eta(angle * jnp.asarray(x)))(
            np.float64(tilt)
        )
        assert float(value) == pytest.approx(expected_value, rel=relative_tolerance)
        assert float(slope) == pytest.approx(expected_slope, rel=relative_tolerance)
        errors.append(abs(float(slope) - expected_slope))
    assert errors[1] < errors[0]


def test_piston_null_direction_dark_pixels_and_periodic_wrapping():
    amplitude, waves, pitch = small_case()
    eta = objective(amplitude, pitch)
    value_and_gradient = jax.value_and_grad(lambda q: eta(q * WAVELENGTH))
    value, gradient = value_and_gradient(waves)
    assert abs(float(jnp.sum(gradient))) < 2e-12
    assert float(gradient[0, 0]) == 0.0
    wraps = np.arange(waves.size).reshape(waves.shape) % 5 - 2
    for shifted in (waves + 0.371, waves + wraps):
        other_value, other_gradient = value_and_gradient(shifted)
        assert float(other_value) == pytest.approx(float(value), abs=2e-14)
        np.testing.assert_allclose(other_gradient, gradient, rtol=2e-12, atol=2e-12)


def test_phase_conjugation_preserves_value_and_reverses_gradient():
    amplitude, waves, pitch = small_case()
    eta = objective(amplitude, pitch)
    evaluate = jax.value_and_grad(lambda q: eta(q * WAVELENGTH))
    value, gradient = evaluate(waves)
    conjugate_value, conjugate_gradient = evaluate(-waves)
    assert float(conjugate_value) == pytest.approx(float(value), rel=2e-13, abs=2e-14)
    np.testing.assert_allclose(conjugate_gradient, -gradient, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("step_waves", [1e-4, 1e-5])
def test_normal_gaussian_tail_value_and_gradient_match_numpy(step_waves):
    # Symmetric cells 6--7 mode radii from the axis retain a tiny but normal
    # overlap. Two illuminated cells give a nonzero phase derivative.
    # This checks tail integration, not IEEE subnormal behavior under XLA.
    amplitude = np.zeros((16, 16))
    amplitude[8, 14] = amplitude[8, 1] = 1.0
    waves = np.zeros_like(amplitude)
    waves[8, 14] = 0.17
    eta = objective(amplitude, PUPIL_RADIUS)
    value, gradient = jax.jit(jax.value_and_grad(lambda q: eta(q * WAVELENGTH)))(waves)
    expected = reference(amplitude, waves * WAVELENGTH, PUPIL_RADIUS)
    assert 1e-36 < expected < 1e-30
    assert float(value) == pytest.approx(expected, rel=2e-12, abs=0)
    for index in ((8, 14), (8, 1)):
        plus, minus = waves.copy(), waves.copy()
        plus[index] += step_waves
        minus[index] -= step_waves
        finite_difference = (
            reference(amplitude, plus * WAVELENGTH, PUPIL_RADIUS)
            - reference(amplitude, minus * WAVELENGTH, PUPIL_RADIUS)
        ) / (2 * step_waves)
        assert finite_difference != 0
        assert float(gradient[index]) == pytest.approx(finite_difference, rel=2e-6, abs=0)
    assert np.all(np.isfinite(gradient))
    np.testing.assert_array_equal(np.asarray(gradient)[amplitude == 0], 0.0)


@pytest.mark.parametrize("step_command", [1e-4, 1e-5])
def test_modal_command_chain_matches_numpy_finite_difference(step_command):
    amplitude, waves, pitch = small_case()
    row, column = np.indices(waves.shape, dtype=np.float64)
    # Each map is OPD in metres per unit command. A reflective surface
    # conversion, if applicable to an actuator, would be included here.
    basis_m = WAVELENGTH * np.stack((0.2 * row, 0.15 * column, 0.1 * (row - column) ** 2))
    commands = np.array([0.21, -0.13, 0.07])
    opd0 = waves * WAVELENGTH
    eta = objective(amplitude, pitch)

    def reward(u):
        return eta(jnp.asarray(opd0) - jnp.einsum("k,kij->ij", u, jnp.asarray(basis_m)))

    residual = opd0 - np.einsum("k,kij->ij", commands, basis_m)
    actual = np.asarray(jax.grad(reward)(commands))
    chain = -np.einsum("ij,kij->k", np.asarray(jax.grad(eta)(residual)), basis_m)
    np.testing.assert_allclose(actual, chain, rtol=2e-13, atol=2e-13)
    finite_difference = []
    for index in range(len(commands)):
        plus, minus = commands.copy(), commands.copy()
        plus[index] += step_command
        minus[index] -= step_command
        finite_difference.append((
            reference(amplitude, opd0 - np.einsum("k,kij->ij", plus, basis_m), pitch)
            - reference(amplitude, opd0 - np.einsum("k,kij->ij", minus, basis_m), pitch)
        ) / (2 * step_command))
    np.testing.assert_allclose(actual, finite_difference, rtol=3e-6, atol=2e-9)


def test_jit_vmap_and_real_scalar_dtype_agree_with_numpy():
    amplitude, waves, pitch = small_case()
    eta = objective(amplitude, pitch)
    opds = WAVELENGTH * np.stack((waves, -0.7 * waves, waves + 0.3))
    batched = jax.jit(jax.vmap(jax.value_and_grad(eta)))
    values, gradients = batched(opds)
    assert values.dtype == jnp.float64 and gradients.dtype == jnp.float64
    assert values.shape == (3,) and gradients.shape == opds.shape
    assert eta(opds[0]).shape == ()
    for index, opd in enumerate(opds):
        assert float(values[index]) == pytest.approx(reference(amplitude, opd, pitch), rel=2e-13, abs=2e-14)
        np.testing.assert_allclose(gradients[index], jax.grad(eta)(opd), rtol=2e-12, atol=2e-8)


def test_full_plane_mode_power_is_not_window_renormalized():
    x, y, pitch = grid(8, 0.5 * PUPIL_RADIUS)
    amplitude = np.exp(-(x**2 + y**2) / PUPIL_RADIUS**2)
    eta = objective(amplitude, pitch)
    actual = float(eta(np.zeros_like(x)))
    assert 0 < actual < 0.2
    assert actual == pytest.approx(reference(amplitude, np.zeros_like(x), pitch), abs=2e-14)


def test_amplitude_unit_scaling_and_frozen_input_custody():
    amplitude, waves, pitch = small_case()
    opd = waves * WAVELENGTH
    original_amplitude, original_opd = amplitude.copy(), opd.copy()
    eta = objective(amplitude, pitch)
    value, gradient = jax.value_and_grad(eta)(opd)
    for factor in (1e-100, 17.0, 1e100):
        other_value, other_gradient = jax.value_and_grad(objective(amplitude * factor, pitch))(opd)
        assert float(other_value) == pytest.approx(float(value), abs=2e-14)
        np.testing.assert_allclose(other_gradient, gradient, rtol=2e-12, atol=2e-8)
    scale = 1000.0
    scaled = make_gaussian_fiber_objective(amplitude, **geometry(pitch, scale))
    other_value, other_gradient = jax.value_and_grad(scaled)(opd * scale)
    assert float(other_value) == pytest.approx(float(value), abs=2e-14)
    np.testing.assert_allclose(other_gradient * scale, gradient, rtol=2e-12, atol=2e-8)
    np.testing.assert_array_equal(amplitude, original_amplitude)
    np.testing.assert_array_equal(opd, original_opd)
    amplitude.fill(0.0)
    assert float(eta(opd)) == pytest.approx(float(value), abs=2e-14)


def test_x64_is_required_at_construction_and_use_without_global_changes():
    amplitude, waves, pitch = small_case()
    eta = objective(amplitude, pitch)
    assert jax.config.x64_enabled
    with jax.enable_x64(False):
        with pytest.raises(RuntimeError, match="Enable JAX float64"):
            objective(amplitude, pitch)
        with pytest.raises(RuntimeError, match="Enable JAX float64"):
            eta(waves * WAVELENGTH)
        with pytest.raises(RuntimeError, match="Enable JAX float64"):
            jax.jit(eta)(waves * WAVELENGTH)
        assert not jax.config.x64_enabled
    assert jax.config.x64_enabled


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int64, np.bool_, np.complex128])
@pytest.mark.parametrize("compiled", [False, True])
def test_invalid_opd_dtype_raises_eagerly_or_at_tracing(dtype, compiled):
    eta = objective(np.ones((4, 4)), 1e-4)
    evaluate = jax.jit(eta) if compiled else eta
    with pytest.raises(TypeError, match="float64"):
        evaluate(np.zeros((4, 4), dtype=dtype))


@pytest.mark.parametrize("shape", [(4,), (4, 3), (1, 4, 4), (0, 4)])
def test_invalid_opd_shape_rejected_at_tracing(shape):
    eta = objective(np.ones((4, 4)), 1e-4)
    with pytest.raises(ValueError, match="shape"):
        jax.jit(eta)(np.zeros(shape, dtype=np.float64))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("dark_pixel", [False, True])
def test_nonfinite_opd_is_never_sanitized_under_jit(bad, dark_pixel):
    amplitude, waves, pitch = small_case()
    eta = objective(amplitude, pitch)
    opd = waves * WAVELENGTH
    # Invalid inputs remain invalid even on an unilluminated sample.
    opd[(0, 0) if dark_pixel else (1, 1)] = bad
    with pytest.raises(ValueError, match="finite"):
        eta(opd)
    value, gradient = jax.jit(jax.value_and_grad(eta))(opd)
    assert not np.isfinite(value)
    assert not np.all(np.isfinite(gradient))


@pytest.mark.parametrize("bad", [
    np.zeros((4, 4)), -np.ones((4, 4)), np.full((4, 4), np.nan),
    np.ones((4, 4), dtype=complex), np.ones(4), np.ones((1, 1)),
])
def test_invalid_fixed_amplitude_rejected(bad):
    with pytest.raises(ValueError):
        objective(bad, 1e-4)


@pytest.mark.parametrize("name", ["pixel_pitch_m", "wavelength_m", "focal_length_m", "mode_radius_m"])
@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_fixed_geometry_rejected(name, bad):
    parameters = geometry(1e-4)
    parameters[name] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        make_gaussian_fiber_objective(np.ones((4, 4)), **parameters)


@pytest.mark.parametrize("name", ["pixel_pitch_m", "wavelength_m", "focal_length_m", "mode_radius_m"])
def test_geometry_cannot_be_silently_differentiated_through_factory(name):
    parameters = geometry(1e-4)

    def changing_geometry(value):
        changed = {**parameters, name: value}
        eta = make_gaussian_fiber_objective(np.ones((4, 4)), **changed)
        return eta(jnp.zeros((4, 4), dtype=jnp.float64))

    with pytest.raises((jax.errors.TracerArrayConversionError, jax.errors.TracerBoolConversionError)):
        jax.grad(changing_geometry)(np.float64(parameters[name]))
