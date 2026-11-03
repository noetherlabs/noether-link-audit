# Gaussian fiber-overlap comparison

A deterministic spatial sensitivity check of a scalar diffraction component.
The [methods](../../docs/fiber-coupling.md) define optical normalization, analytic
checks, sampling assumptions and reproduction. Numerical values and source
SHA-256 hashes are retained in [comparison.json](comparison.json).

At 1 rad RMS phase over a clear circular pupil, the fraction coupled relative
to the zero-aberration reference is 0.36829 for defocus and 0.50416 for
astigmatism. An exponential using only phase variance gives 0.36788 for both.
The discrepancy demonstrates information lost by that scalar summary in
these specific monochromatic phase maps; it does not measure controller gain.

The zero-aberration clear-pupil efficiency is 0.814512 at the declared Gaussian
radius, compared with the analytic ideal-Gaussian limit 0.814529. A central
obscuration of 0.3 times the pupil radius lowers the sampled value to 0.646975
at the same mode radius. These are fractions of the transmitted pupil power,
so the latter does not also include the obscuration's lost incident flux.

All 13 cases retain 64², 128² and 256² pupil grids. The largest absolute
128²-to-256² change is 0.001049, for the obscured pupil. This refinement
difference is a convergence diagnostic, not a certified error bound. The
reported examples use the finest retained grid.

The exponential comparisons are spatial analogues of a scalar variance proxy.
They score no controller. No real fiber eigenmode, controller, hardware link,
polarization or broadband performance is evaluated.
