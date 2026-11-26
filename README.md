# spectral-restir

Real-time **spectral ReSTIR path tracing** in Vulkan.

Existing real-time ReSTIR renderers resample light paths in RGB. This project
resamples over **wavelength as part of the path**, so dispersion, prisms, gems,
and thin films get ReSTIR-level sample reuse instead of falling back to
brute-force spectral path tracing.

> **Status: milestones 1 and 2 done; measurement harness added.**
> CI builds everything and runs on Mesa's CPU Vulkan driver (lavapipe): furnace test,
> NEE+MIS render, a bias check against brute-force path tracing, and a convergence sweep.

## Related work (read before claiming novelty)
- Waseda thesis proposing resampling for spectral path tracing (offline-oriented)
- GraphiCon 2025, "Spatiotemporal resampling for physical light transport rendering" (wave optics, not real-time)
- "Spectra" Blender addon (Rust + Vulkan, ReSTIR BDPT + spectral materials): check what it resamples
- ReSTIR PT (Lin et al. 2022) and its hybrid shift mapping: the base this builds on

## Build
```
cmake -B build && cmake --build build
ctest --test-dir build --output-on-failure   # CPU spectral tests
./build/sr_probe                              # lists RT-capable GPUs
./build/sr_render 0 256 256 64 build/pathtrace.spv furnace.ppm   # furnace: expect Y = 1.0
./build/sr_render 1 800 600 512 build/pathtrace.spv prism.ppm    # prism under an area light
```
Needs the Vulkan SDK (headers + `glslc`) and a GPU with ray query support.

## Measuring error
```
./build/sr_render 1 256 192 8192 build/pathtrace.spv ref.ppm ref.pfm 100000   # reference (offset seeds)
./build/sr_render 1 256 192 64   build/pathtrace.spv t.ppm t.pfm 0            # test render
./build/sr_compare ref.pfm t.pfm                                              # RMSE, relMSE, bias
python3 scripts/convergence.py                                                # RMSE vs spp, slope check
```
Modes: 0 furnace, 1 prism with NEE+MIS, 2 prism brute-force (unbiased reference for mode 1),
3 prism with the caustic channel split out and sampled by ReSTIR (extra args: candidates, reuse bits: 1 = temporal, 2 = spatial),
4 caustic channel only (measurement mode: no base estimator),
5 prism with the caustic channel estimated by LIGHT TRACING (extra arg after reuse: photons per pixel per frame),
6 light-traced caustic channel only (measurement mode).
`.pfm` files are linear sRGB; `.ppm` files are tone-mapped (ACES fit) for viewing.
Always render the reference with a different frame offset than the test, or they share samples.

## Roadmap
1. **Spectral path tracer** (no ReSTIR): HWSS, Sellmeier dispersion, XYZ output, furnace test (done)
1b. **NEE + MIS** on the area light, neutral white light spectrum
2. **Scenes that prove the point**: prism, diamond, thin-film bubble, glass caustics
3. **ReSTIR DI** in spectral form (baseline)
4. **ReSTIR PT, RGB-style**: wavelength fixed per pixel/frame (your control group)
5. **Wavelength-aware shift mappings** (see docs/DESIGN.md), the novel part
6. **Denoiser + temporal accumulation** that respects spectral samples
7. **Benchmarks**: RMSE vs reference at equal ms, RGB-ReSTIR vs spectral-ReSTIR

## Layout
```
src/main.cpp              Vulkan RT capability probe
src/render.cpp            headless renderer: BLAS/TLAS, compute dispatch, PPM + linear PFM output
tools/compare.cpp         RMSE / relMSE / bias between two PFM images
scripts/convergence.py    spp sweep + 1/sqrt(N) slope check (used by CI)
scripts/region_bias.py    per-region (prism / floor) bias z-test; REGION_TILE=16 for correlated (reuse) noise
scripts/restir_efficiency.py  caustic-channel noise at equal path evaluations: baseline vs temporal vs temporal+spatial
scripts/lt_efficiency.py  light tracing vs candidates at equal sample counts and wall time
shaders/pathtrace.comp    spectral path tracer (ray queries, hero wavelengths, dispersion)
tests/transport_test.cpp  CPU mirror of the transport + glass furnace test
tests/nee_test.cpp        NEE / BSDF / MIS estimators vs brute-force integration
tests/restir_toy_test.cpp CPU toy of ReSTIR reuse for caustic paths (unbiasedness + efficiency, see docs/DESIGN.md)
experiments/restir_iter_toy.cpp  iterated temporal+spatial reuse experiment (not in CI)
experiments/restir_iter2d_toy.cpp  2D / jitter variant with real pixel footprints (not in CI)
experiments/light_tracing_toy.cpp  light tracing vs path tracing for the caustic channel (not in CI)
shaders/spectral.glsl     HWSS, CIE fit, Sellmeier
shaders/reservoir.glsl    80-byte spectral reservoir + update/finalize
tests/spectral_test.cpp   CPU reference: unbiasedness + variance checks
docs/DESIGN.md            algorithm notes and open questions
```
