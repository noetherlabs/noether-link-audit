# Noether Physics

Scalar coupling into an ideal centered Gaussian fiber mode, with optional
derivatives with respect to optical path. This package contains the shared
receiver model used by [Noether Link Audit](https://github.com/noetherlabs/noether-link-audit).
It does not require the audit, HCIPy, plotting libraries or a controller.

## Installation

Requires Python 3.11 or 3.12. From a checkout of the repository:

```sh
python -m pip install ./packages/noether_physics
```

For the optional JAX interface:

```sh
python -m pip install './packages/noether_physics[differentiable]'
```

These commands install the independent distribution into the active environment
with its dependencies. The repository's complete development workspace uses
Python 3.12 and preserves Link Audit's exact numerical pins. The package itself
has broader dependency bounds to allow use in other optical simulation environments;
those bounds are not a claim that every allowed version has been numerically qualified.
The retained Link Audit reference results use NumPy 2.1.3 and SciPy 1.14.1.
The optional differentiable backend pins JAX and JAXlib 0.10.2.

To build a portable wheel and source archive from the repository root:

```sh
uv build packages/noether_physics --out-dir dist
python -m pip install dist/noether_physics-0.1.0-py3-none-any.whl
```

No package-index release is required for these local installations. For a fixed
analysis, record the repository revision, wheel hash and resolved environment.

## Optical contract

```python
from noether_physics import gaussian_fiber_coupling
from noether_physics.differentiable import make_gaussian_fiber_objective
```

The numerical reference accepts pupil amplitude and optical path difference
in **metres**, sampled at square-cell centers. Optical path is not mirror
surface displacement. The receiver wavelength, pupil pitch, focal length and
Gaussian 1/e field radius are explicit inputs. The Gaussian is normalized over
its full plane, including the mode outside the finite pupil and array.

The JAX factory fixes amplitude and receiver geometry, returning a scalar
function of real optical path. Set `JAX_ENABLE_X64=1` before starting Python;
the factory requires float64 and does not change process-wide JAX settings.
Its output supports differentiation, JIT compilation and batching.

See the [method and qualification](https://github.com/noetherlabs/noether-link-audit/blob/main/docs/fiber-coupling.md)
for equations, examples, analytic limits, sampling checks and independent
Fourier comparisons. Coupling is a fraction of incident pupil power. It does
not include throughput, polarization mismatch, a measured fiber eigenmode,
scintillation dynamics, modem goodput or demonstrated controller performance.

## Source and license

`noether-physics` is the sole distribution that installs the `noether_physics`
namespace. Link Audit depends on this package; it does not bundle another copy.
The source remains in this directory so the retained source hashes and numerical
comparison paths continue to identify the same implementation.

Apache-2.0; see [LICENSE](LICENSE).
