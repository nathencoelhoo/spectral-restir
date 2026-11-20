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


## Toy-model results (tests/restir_toy_test.cpp, experiments/restir_iter_toy.cpp)

Setup: floor points along a line, glass (a sphere, or the renderer's prism), the area light, BK7
dispersion. A path is a vector u of uniform numbers that includes the wavelength; reuse replays a
neighbour's u at the current floor point (random replay, Jacobian 1; dispersion is exact because the
wavelength rides along in u). Only paths with >= 1 glass vertex count (the caustic channel; direct
light is NEE+MIS in the renderer). Reservoirs carry an unbiased contribution weight W and follow
generalized RIS (Lin et al. 2022).

**Correction.** An earlier version of this section counted 29 path evaluations per pixel for balance
MIS; the correct count is 24 (diagonal entries are known from the initial RIS). The numbers below use
the corrected count. It made plain MC look about 20% better than it was.

### 1. Correctness (measured, CI-asserted)
- Balance MIS and pairwise MIS are unbiased: worst |z| about 2.3 over every configuration tried.
- Naive reuse (divide by sum of M, no MIS) is biased: -7% to -77% on the sphere, about -6% on the
  prism. The MIS terms are required.

### 2. Single reuse pass (4 candidates + 4 neighbours): NOT worth it
Efficiency = variance x path evaluations per pixel / plain-MC variance (below 1.00 = ReSTIR wins).
Balance MIS costs 24 evaluations per pixel, pairwise 12.

| geometry, point | spacing 0.12 | 0.03 | 0.01 |
|---|---|---|---|
| sphere, light 1.5: balance / pairwise | 1.51 / 1.14 | 1.27 / 0.98 | 1.22 / 0.95 |
| sphere, light 0.2: balance / pairwise | 4.05 / 2.34 | 2.58 / 1.74 | 1.46 / 1.19 |
| sphere, light 0.08: balance / pairwise | 5.47 / 2.86 | 4.47 / 2.46 | 2.44 / 1.74 |
| prism, x=1.3: balance / pairwise | 1.49 / 1.22 | 1.24 / 1.00 | 1.20 / 0.94 |
| prism, x=1.9: balance / pairwise | 1.40 / 1.17 | 1.27 / 1.02 | 1.20 / 0.97 |

- A sphere is a lens: it magnifies the offset between neighbours, so reuse works poorly there.
  A prism (flat faces) is much friendlier: variance falls about 3x with 5 reservoirs.
- Pairwise MIS halves the cost and is unbiased, but a single pass is at best break-even (0.94-1.02x).
- A 256x192 render has roughly 0.03 floor units per pixel (my estimate); 0.01 is an 800x600 render.

### 3. Iterated reuse (temporal + spatial, pairwise MIS): a real gain
Strip of 41 pixels, 40 iterations, 2 fresh candidates + 2 spatial neighbours (radius 3) + the previous
frame's reservoir, M capped at 20. Cost 6 evaluations per frame (temporal reuse is free in a static
scene: p_hat is stored). Prism, caustic channel, light 1.5.

| run | single frame | mean of last 16 frames |
|---|---|---|
| x=1.3, spacing 0.03, temporal+spatial | 0.15x | 0.40x |
| x=1.3, spacing 0.03, spatial only | 0.26x | 0.50x |
| x=1.3, spacing 0.01, temporal+spatial | 0.08x | 0.25x |
| x=1.9 (rarer caustic), spacing 0.03 | 0.10x | 0.35x |

"Mean of last 16 frames" compares against plain MC given 16x the evaluations, which is the fair
comparison for a static render. Reading: iterated reuse needs 2.5x to 4x fewer path evaluations for
the same noise. Reuse that recycles reservoirs works where a single pass does not.

**Bias.** No bias detected, but the resolution is limited: the standard errors were about 1-2% of the
signal even with 6000-14000 chains, so a bias below roughly 2% cannot be ruled out. Iterated reuse
feeds reservoirs into each other, which breaks the independence the unbiasedness proof assumes, so
this needs a stronger test on the GPU, where every pixel and frame adds statistical power.

### 4. Limits of the toy (read before trusting any of this)
1D strip of pixels (a 2D image has more and differently placed neighbours); one wavelength per path;
the static-scene assumption that makes temporal reuse free; blotchy spatial correlation in the output
is not measured; the real renderer has companion wavelengths, Russian roulette and a diffuse floor
interreflection that the toy leaves out. Not tested: a better shift than random replay (for specular
chains this means manifold walks, and dispersion makes the solution wavelength dependent).
