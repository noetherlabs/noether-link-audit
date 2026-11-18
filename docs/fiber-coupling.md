# Gaussian fiber overlap

`noether_physics.coupling.gaussian_fiber_coupling` computes monochromatic,
scalar, paraxial coupling into an ideal centered Gaussian mode at the focal
plane of an ideal lens. In this repository it computes the Monte Carlo
coupling tables that the link-loss audit uses.

The implementation is distributed separately as
[`noether-physics`](../packages/noether_physics/README.md). It can be installed
without Link Audit; the numerical source and optical conventions are shared.

## Optical definition

For pupil amplitude $A$, optical path difference $d$, wavelength $\lambda$
and focal length $f$, the pupil field is $E=A\exp(2\pi i d/\lambda)$.
The focal mode has field amplitude $\exp(-\rho^2/w_f^2)$, where $w_f$ is
the **1/e field radius**, equivalently the 1/e² intensity radius. It is
half the mode-field diameter.

Fourier reciprocity permits the normalized focal overlap to be evaluated
in the pupil using $g=\exp(-r^2/w_p^2)$ with $w_p=\lambda f/(\pi w_f)$:

$$\eta=\frac{|\int E g^*\,d^2r|^2}
{\int |E|^2\,d^2r\;\pi w_p^2/2}.$$

The mode denominator is its **full-plane analytic power**, including the
mode outside the pupil and computational window. Normalizing it only over
the illuminated aperture would overestimate coupling. The fraction is
relative to incident pupil power; throughput, reflection, polarization
mismatch and fiber propagation loss are separate factors. A Gaussian mode
does not model the exact eigenmode of a particular manufactured fiber.

The overlap definition and pupil-plane Fourier transformation follow
[Robbe-Dubois et al. (2009), §2.2](https://doi.org/10.1111/j.1365-2966.2009.14587.x).
[Shaklan and Roddier (1988)](https://doi.org/10.1364/AO.27.002334) analyze
the dependence on seeing, focal ratio and displacement for single-mode
fibers. Their physical fiber-mode efficiency is not the ideal Gaussian
limit tested here.

## Discretization and limits

All input lengths are in metres. OPD is optical path, not mirror surface
displacement. Array samples are centers of square cells; the array midpoint
is the optical axis. The input complex field is constant inside each cell
and zero outside the array. The Gaussian is integrated exactly over each
cell using separable error functions. This preserves the power bound for
the declared piecewise-constant field, without clipping computed values.

A physically bounded value is not a sampling-accuracy certificate. Fine
structure in pupil transmission or OPD can be absent or aliased. Refine the
input grid at fixed physical aperture and check convergence. A truncated
input Gaussian must also be checked by enlarging the window. The tests
include a deliberately coarse grid whose overlap is bounded but inaccurate
for the underlying continuous Gaussian.

For a uniform clear circular pupil of radius $R$, put $q=R^2/w_p^2$.
Direct integration gives $\eta=2(1-e^{-q})^2/q$. Its maximum is
approximately 0.81453, at the positive solution of $e^q=1+2q$,
$q\simeq1.25643$. Piston changes only the overlap phase, so it cannot
change the coupled fraction.

For an untruncated Gaussian incident pupil of field radius $w_i$ and an
OPD ramp $d=\alpha x$, the analytic limit is

$$\eta=\frac{4w_i^2w_p^2}{(w_i^2+w_p^2)^2}
\exp\left[-\frac{(2\pi\alpha/\lambda)^2w_i^2w_p^2}
{2(w_i^2+w_p^2)}\right].$$

These limits, amplitude/length rescaling, invalid inputs and a separate
focal-plane FFT overlap are checked in `tests/test_fiber_coupling.py`.
The Fourier check includes the sinc transform of each square constant-field
cell, the lens scale and the analytic full focal-mode power.

## Reproduce the spatial comparison

```sh
.venv/bin/python -m pytest tests/test_fiber_coupling.py
.venv/bin/python scripts/compare_fiber_coupling.py > comparison.json
```

The [retained comparison](../results/fiber-coupling-v1/comparison.json)
contains 13 deterministic cases at 64², 128² and 256² pupil samples. It
varies tilt, defocus, astigmatism, central obscuration and Gaussian radius.
Phase RMS is normalized over illuminated samples at each resolution. It
records absolute efficiency, efficiency relative to that aperture's
zero-aberration reference, the final refinement change and source hashes.
There is no timing benchmark, random ensemble or confidence interval.

The comparison also evaluates **spatial analogues** of a scalar proxy: an
exponential of phase variance, with an additional tilt penalty. On these
explicit phase maps it shows what information a scalar variance summary
loses. It does not evaluate a controller.

This qualifies a numerical component on stated analytic and sampled-field
checks. It does not establish real-fiber accuracy, hardware performance,
an end-to-end optical link model or safe autonomous control.


## Differentiating optical path

The optional JAX interface prepares the same fixed pupil and Gaussian geometry,
then returns a scalar function of **real optical path difference in metres**:

```sh
uv sync --locked --extra dev --extra differentiable
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --locked --extra differentiable python your_model.py
```

```python
import jax
import jax.numpy as jnp
import numpy as np

from noether_physics.differentiable import make_gaussian_fiber_objective

# An illustrative sampled pupil; replace these constants with receiver geometry.
n = 32
pitch = 8e-5
axis = (np.arange(n) - (n - 1) / 2) * pitch
x, y = np.meshgrid(axis, axis)
amplitude = (x*x + y*y <= 1e-6).astype(float)
eta = make_gaussian_fiber_objective(
    amplitude,
    pixel_pitch_m=pitch,
    wavelength_m=1.55e-6,
    focal_length_m=0.1,
    mode_radius_m=5e-5,
)
opd = jnp.asarray(8e-8 * (x*x - y*y) / 1e-6, dtype=jnp.float64)
efficiency, gradient_per_metre = jax.jit(jax.value_and_grad(eta))(opd)
```

Enable JAX's 64-bit mode before constructing arrays or tracing the model. The
library does not change process-wide configuration. Shape and dtype errors
raise at evaluation or tracing. Eager evaluation rejects nonfinite OPD;
under a JAX transformation it propagates to nonfinite output. The calling
optimizer or study must reject and retain nonfinite values or gradients,
including derivatives made unrepresentable by extreme physical scales. Finite output alone is not a calibration or sampling certificate.
The optional interface is qualified with CPU float64 checks. Other precision
or accelerator settings require separate numerical checks.

Pupil transmission, pixel pitch, wavelength, focal length and Gaussian radius
are **fixed preparation inputs**, not differentiable arguments. Preparing a
new factory for a changed geometry does not supply a geometry derivative.
This interface differentiates the specified optical objective; it does not
identify an instrument or implement sensing, delay, actuator dynamics or a
closed-loop controller.

For exact cell weights $w_j$, fixed normalized amplitudes $A_j$, incident
power $P$, full-plane mode power $Q=\pi/2$, and
$s=\sum_j A_jw_j\exp(2\pi i d_j/\lambda)$, the derivative is

$$
\frac{\partial\eta}{\partial d_j}
=-\frac{4\pi}{\lambda P Q}
\operatorname{Im}\left[s^* A_jw_j e^{2\pi i d_j/\lambda}\right].
$$

The API accepts real OPD and returns a real objective, so an optimizer acts on
real parameters even though the optical field is complex. It supports JAX
`grad`, `jit` and `vmap`. For an actuator influence map $B$ converting commands
$u$ to optical path, residual $d=d_0-Bu$ gives
$\nabla_u\eta=-B^\mathsf{T}\nabla_d\eta$.
Include a reflective mirror's surface-to-path factor of two exactly once in
that map. Differentiating $1-\eta$ reverses the sign again.

At destructive interference, nonzero incident power can give zero overlap
and zero first derivative. This is a valid stationary point, not an error or
a promise that gradient descent will escape it. Zero incident power is
undefined and rejected. No denominator epsilon or clipping changes either
case. OPD is reduced modulo wavelength before forming phase; the composed
optical objective remains periodic across those wraps.

The derivative checks include an exact two-cell interference solution,
independent central differences of the NumPy reference, Gaussian tilt,
piston and length scaling, blocked pupil samples, modal-command composition,
and compiled/batched evaluation. These checks qualify numerical derivatives;
they do not measure a learned controller's benefit or delivered link data.
See [JAX's differentiation conventions](https://docs.jax.dev/en/latest/301/cookbook.html)
and [64-bit configuration](https://docs.jax.dev/en/latest/101/default_dtypes.html).
