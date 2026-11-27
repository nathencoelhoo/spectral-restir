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

### 3. Iterated reuse in the 1D toy (SUPERSEDED: the GPU result and the 2D toy below contradict the temporal part)
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


## GPU port: status

**Part 1 (written, awaiting CI): reservoir buffers + candidate pass, no reuse (mode 3).**
- Pixels whose primary hit is the diffuse floor split their light exactly into a *base* estimate (all
  paths except x -> 1..8 glass vertices -> light) and a *caustic channel* (exactly those paths).
- Pass 0 draws N0 replay seeds per pixel, evaluates each with evalCaustic() (mirrors the toy's evalPath),
  RIS-selects one and stores (seed, W, M, p_hat, f) in a 32-byte reservoir. Pass 1 shades: base path
  tracer (which skips precisely the events evalCaustic counts) + f * W.
- Expected result: unbiased, noise like mode 1 (RIS alone has efficiency 1.00x in the toy). It exists
  to prove the plumbing: if mode 3 disagrees with mode 1 beyond noise, the split or the replay is wrong.
- Not yet: temporal/spatial reuse, ping-pong reservoir buffers, pairwise MIS (part 2).
- Scope limits: the caustic channel only serves pixels that see the floor directly; floor seen through
  the prism, and chains that start at a later diffuse vertex, stay in the base estimator.

**Part 2 (written, awaiting CI): ping-pong reservoirs, temporal and spatial reuse with pairwise MIS.**
- Reservoir and primary-hit buffers are doubled and flipped by frame parity. Frame passes: 0 candidates,
  2 reuse (optional), 1 shading. Mode 4 outputs the caustic channel alone so reuse noise can be measured
  without the base estimator.
- Reuse (reusePass) mirrors experiments/restir_iter_toy.cpp: canonical = fresh candidates; sources = same
  pixel last frame (temporal) and 2 random neighbours in a 7x7 window (spatial); pairwise MIS; M capped at 20.
- Difference from the toy: camera jitter moves each pixel's floor point every frame, so the temporal source
  must be replayed at the new point (2 evaluations), unlike the toy where it was free.
- Cost per pixel per frame: candidates + 2 (temporal) + 4 (two spatial). CI compares equal-cost configs
  (8 evaluations): 8 candidates / 6 + temporal / 2 + temporal + spatial.
- Statistics: reuse makes noise spatially correlated, which breaks the per-pixel z-test (a synthetic check
  gave a false FAIL, z = -5.95, with no bias). Reuse tests use REGION_TILE=16 (tiles as independent units).
  The tile test has less power; its real resolution is to be read off the CI tables, not assumed.

### Part 2 GPU results (CI run, caustic channel only, 160x120, equal cost of 8 evaluations/pixel/frame)
- Bias: none detected. Temporal reuse and temporal+spatial reuse pass in every region with |z| <= 1.5
  (tile-based z-test). Standard errors were 1-3% of the signal, so only bias above roughly 2-5% is excluded.
- Efficiency: reuse is WORSE than plain candidates. MSE ratio vs baseline: temporal 1.62 / 2.67 / 2.20 and
  temporal+spatial 1.34 / 1.97 / 1.67 at 32 / 128 / 512 frames. (The reference has its own noise, which pulls
  these ratios toward 1, so the true ratios at 512 frames are somewhat worse, about 2.0 and 2.8.)
  This contradicts the 1D toy, which predicted roughly 0.4.

**Diagnosis by toy (experiments/restir_iter2d_toy.cpp; 2D grid, per-frame jitter, real pixel footprints).**
One pixel at the caustic covers (floor units, across x deep): 160x120: 0.047 x 0.165; 256x192: 0.029 x 0.102;
800x600: 0.009 x 0.033. The 1D toy had assumed 0.03 x 0.03. Variance relative to the same-cost baseline
(below 1 = reuse wins; 80 chains, so each ratio is uncertain by about 25%):

| footprint | temporal only | temporal + spatial |
|---|---|---|
| 160x120, jitter on (the CI case) | 1.48 | 1.02 |
| 160x120, jitter off | 1.51 | 0.71 |
| isotropic 0.047 x 0.047 | 1.24 | 0.41 |
| 800x600, jitter on | 1.37 | 0.35 |

- The toy reproduces the SIGN of the GPU result at 160x120 (reuse does not win), which argues against a GPU
  bug. The GPU is worse than the toy by up to ~2x, unexplained (toy noise alone is about 25%).
- Temporal reuse alone loses in every case once its 2 evaluations are charged, even with jitter off.
- Temporal+spatial wins when neighbours are close: the deep (z) direction at coarse resolution puts neighbours
  0.165 apart. Prediction to test on the GPU: at 400x300 and above, temporal+spatial should win clearly.
- Not diagnosed: why temporal reuse hurts. Strong temporal correlation (a sample persisting across frames)
  is the likely reason but is not demonstrated.

### Part 2 results (CI run #40) and what they mean

**GPU, caustic channel only (mode 4), 160x120, equal cost of 8 path evaluations per pixel per frame.**
MSE ratio vs the 8-candidate baseline (below 1 = reuse wins). The reference has its own noise, which
compresses ratios toward 1, so the true ratios are somewhat further from 1 than shown.

| frames | temporal (6 cand + T) | temporal + spatial (2 cand + T + 2 S) |
|---|---|---|
| 32 | 1.62 | 1.34 |
| 128 | 2.67 | 1.97 |
| 512 | 2.20 | 1.67 |

Reuse was WORSE than plain candidates in this configuration, contradicting the 1D toy (which predicted
about 0.4). Spatial-only reuse has not run on the GPU yet (added to CI after this result).

**Bias (tile z-test, caustic channel).** None detected: temporal: prism +0.98% (z 0.54), near floor +0.64%
(z 0.82), far floor -0.45% (z -0.16); temporal+spatial: +1.34% (z 0.84), -0.61% (z -0.48), +4.51% (z 1.46).
The implied standard errors are 0.8% to 3%, so a bias below roughly 2% to 5% is NOT excluded.

**2D toy with camera jitter and real pixel footprints** (experiments/restir_iter2d_toy.cpp; 100 chains,
48 frames, ratio of frame-mean variance vs the 8-candidate baseline, +-15% noise):

| footprint | temporal | temporal + spatial |
|---|---|---|
| 160x120 (0.047 x 0.165) | 1.24 | 0.56 |
| isotropic 0.047 | 2.24 | 1.18 |
| 800x600 (0.009 x 0.033) | 2.20 | 0.83 |

Diagnostics at the 160x120 footprint: without camera jitter, temporal 1.35, temporal+spatial 1.04,
spatial-only 0.56. With jitter, spatial-only 0.33.

**Reading (hypotheses are marked).**
- Camera jitter is NOT the cause of the temporal loss: it persists without jitter.
- Temporal reuse hurt in every variant, on the GPU and in the toy. Hypothesis, untested: with M capped at 20
  the old sample dominates the MIS weights and persists across frames, so frame-to-frame estimates are
  strongly correlated and the mean over many frames converges slowly. Temporal reuse is built for per-frame
  quality when not accumulating, not for converging a static image.
- Spatial-only reuse looks good in the toy (0.33 to 0.56). The GPU has to confirm it.
- The toy does not fully reproduce the GPU numbers for temporal + spatial (toy 0.56 to 1.18, GPU 1.3 to 2.0),
  so a GPU-specific cause cannot be ruled out.


## Light tracing for the caustic channel (CPU toy, experiments/light_tracing_toy.cpp)

Motivation: with ReSTIR the full contribution is its own target function, so resampling can only pool
candidates, and only about 7% of camera-side candidates reach the light at all (see "Part 2 results").
Light tracing attacks the real problem: photons start at the light, are aimed at the prism's bounding
sphere, follow the Fresnel-sampled glass chain with their own wavelength (dispersion needs no special
handling), land on the floor, and are splatted to the camera pixel with weight
(albedo/pi) * flux * h / (z^3 * A_pix) (h = eye height above the landing point, z = depth along the optical
axis, A_pix = pixel area on the unit-distance image plane), after a visibility ray from the camera.
It estimates exactly the events the renderer's base estimator already excludes for floor-primary pixels,
so it can replace the ReSTIR caustic channel without MIS.

Single CPU core, 80x60, 8 candidates or photons per pixel per frame (toy, not the GPU):

| estimator | RMSE vs guided reference, 256 frames | CPU time, 256 frames |
|---|---|---|
| cosine path tracing (current candidates) | 0.005619 | 1.9 s |
| path tracing aimed at the prism | 0.004718 | 2.5 s |
| light tracing | 0.001611 | 5.0 s |

- The reference itself has noise of roughly 0.0017, so light tracing is at the reference noise floor: its
  true advantage is larger than the numbers show. Lower bound on the MSE ratio vs cosine path tracing: about 11x
  per sample, about 4x per CPU second (photons cost about 2.6x a candidate here).
- Across 10 independent replicates the run-to-run spread of region means was 7% to 13% of path tracing's, i.e.
  roughly 60x to 200x lower variance. (These replicate files were lost in a sandbox reset; the numbers come from
  the earlier run's output.)
- Aiming path-tracing directions at the prism gives only about 1.5x.

**Bias.** None detected, at a resolution of roughly 1% to 3%. Tile z-test against two guided references
(2048 frames each): light tracing prism -3.62% (z -1.94), near floor +1.06% (z 2.03), far floor +1.14% (z 1.90).
Cosine path tracing, unbiased by construction, shows deviations of the same size against the same references
(prism -1.71%, near floor -1.27% with z -2.29, far floor +2.38%), so the test cannot separate them. Global means:
cosine path tracing 0.004278 and 0.004291, guided 0.004322 and 0.004338, light tracing 0.004342 and 0.004330.

**Limits.** Prism-only: the aiming cone uses the prism's bounding sphere, the splat assumes a pinhole camera and
a diffuse floor seen directly, and events are capped at 11 glass vertices as in the toy. Not yet on the GPU.


## Light tracing on the GPU (modes 5 and 6): status

**Written, awaiting CI (never compiled or run).** Design:
- Pass 3 clears a per-pixel fixed-point accumulator (3 ints per pixel, scale 2^20; integer atomics because
  float atomics are not portable). Pass 4: every pixel thread emits `photons` photons. Each photon: uniform point
  on the light, direction uniform in the cone toward the prism's bounding sphere, hero wavelength from the visible
  distribution, Fresnel-sampled glass chain (1..8 vertices, same cap as the base estimator's exclusion), landing
  on the floor, visibility ray from the camera, splat weight (albedo/pi) * flux * h / (z^3 * A_pix).
- Mode 6 outputs the splat alone (measurement). Mode 5 = mode 1 with the base estimator skipping the same events
  as in mode 3 for floor-primary pixels, plus the splat for every pixel (visibility is inside the splat).
- Scene-specific: the aiming cone uses the prism's bounding sphere (constants PRISM_CENTER and PRISM_R in the
  shader), the camera is a pinhole, the floor is the only diffuse surface.
- CI: bias test of mode 6 against the mode-4 references (tile z-test), efficiency vs 8 candidates at equal counts
  (scripts/lt_efficiency.py), and a full-image check of mode 5 against the NEE+MIS references.
- Expected from the CPU toy (not yet confirmed on the GPU): large noise reduction at equal sample counts; the toy
  could only resolve bias at the 1% to 3% level.


### Light tracing: GPU results (CI run #53, lavapipe, 160x120, caustic channel, 8 photons or candidates per pixel per frame)

Compiled and ran on the first build. MSE ratio vs 8 candidates (mode 4), below 1 = light tracing wins:

| frames | RMSE (light tracing) | MSE ratio | wall time, candidates / photons |
|---|---|---|---|
| 16 | 0.006795 | 0.022 | 0.4 s / 0.9 s |
| 64 | 0.005678 | 0.081 | 0.8 s / 2.9 s |
| 256 | 0.005285 | 0.222 | 2.4 s / 10.5 s |

- The ratios understate the advantage: light tracing's RMSE flattens near 0.0053 because the reference (8
  candidates, 1024 frames) has its own noise of about 0.005. Correcting for that with a simple scaling
  assumption (my estimate, not measured) gives light-tracing errors of roughly 0.004, 0.003 and 0.0017 against
  baseline errors of 0.045, 0.019 and 0.010, i.e. an MSE advantage of tens of times per sample, and roughly
  8x to 60x at equal wall time after dividing by the 2x to 4x photon cost.
- Bias (tile z-test vs the mode-4 references): prism +1.15% (z 1.67), near floor +1.28% (z 2.99), far floor
  +1.93% (z 1.47). Not flagged, all positive. The CPU toy showed the same pattern against short references
  (near floor +1.9%, z 3.14) and it vanished against longer ones (8192 frames x2): prism -0.32% (z -0.34),
  near floor +0.33% (z 1.68), far floor -1.04% (z -1.01); the two long path-tracing references differ from each
  other by up to 1.05% in the prism region. So the GPU's +1.2% to +1.9% is probably reference noise too, but this
  is not proven. The z-test is anti-conservative for heavy-tailed noise: z near 3 occurred from noise alone.
- Full image (mode 5 vs the NEE+MIS reference, 4096 vs 8192 frames): prism -0.029% (z -0.30), near floor +0.018%
  (z 1.43), far floor +0.009% (z 0.74). The caustic channel is only about 2% of region means, so this cannot
  resolve a 1% bias in the caustic channel itself.
- Open: lengthen the CI references (c_ref and c_ref2) to tighten the bias test; inspect lt5.png for splat
  artifacts; floor seen through the prism and chains that start at a later bounce are still handled by the
  base estimator.
