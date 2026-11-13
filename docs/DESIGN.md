# Design notes

## Core idea
Treat wavelength as an extra dimension of path space: a path is
`x = (lambda_0..3, u_1..u_n)`. ReSTIR resamples whole paths, so reservoirs carry
the wavelength set, not just geometry.

## Why it's hard
With dispersion, the same camera ray refracts differently per wavelength. A
neighbor's path is therefore *not* a valid path for this pixel's wavelengths
beyond the first dispersive vertex. Standard reconnection-shift Jacobians assume
the path geometry is wavelength-independent.

## Shift mapping candidates (hypotheses to test, not results)
1. **Carry wavelengths with the sample.** Reuse a neighbor's path at the
   neighbor's wavelengths. Jacobian = 1 for wavelengths; costs color variance
   because the pixel receives foreign wavelengths. Simplest baseline.
2. **Random replay including wavelength.** Replay the neighbor's RNG stream
   with this pixel's wavelengths. Handles dispersion exactly (paths re-traced),
   Jacobian in random-number space is 1, but costs a full re-trace.
3. **Hybrid (ReSTIR PT style).** Random replay up to `lastDispersiveVertex`,
   reconnect after it. Reconnection is only valid where geometry no longer
   depends on wavelength. This is the field `lastDispersiveVertex` supports.

## Estimator
Companions are stratified in u-space, so per-wavelength weight is
`(1/N) / p(l_i)` (see `stratifiedWeights`). If dispersion terminates companions,
use the hero only: `1 / p(l_0)`.

## Target function p_hat
Starting point: `sum_i ybar(l_i) * f(l_i)`. Luminance-based p_hat may starve
blue/red extremes where ybar is small. Alternative: weight by full XYZ norm.

## Validation plan
- White-furnace test (energy conservation across wavelengths)
- Reference convergence per scene; RMSE vs equal-time RGB-ReSTIR
- Bias check: resampled mean vs long-run reference on prism and diamond scenes
- Per-candidate ablation of shift mappings 1/2/3

## Open questions
- Does wavelength-aware reuse beat simply denoising spectral noise?
- How much does reservoir size (80 B) cost in bandwidth at 4K?
- Does MIS between neighbor wavelengths need a new weight (not just M)?
