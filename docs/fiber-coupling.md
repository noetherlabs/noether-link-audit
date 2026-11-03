# Gaussian fiber overlap

`noether_physics.coupling.gaussian_fiber_coupling` computes monochromatic,
scalar, paraxial coupling into an ideal centered Gaussian mode at the focal
plane of an ideal lens. In this repository it computes the Monte Carlo
coupling tables that the link-loss audit uses.

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
