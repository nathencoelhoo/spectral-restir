# spectral-restir

Real-time **spectral ReSTIR path tracing** in Vulkan.

Existing real-time ReSTIR renderers resample light paths in RGB. This project
resamples over **wavelength as part of the path**, so dispersion, prisms, gems,
and thin films get ReSTIR-level sample reuse instead of falling back to
brute-force spectral path tracing.

> **Status: milestone 1 written, GPU code not yet compiled.**
> Verified on CPU: spectral sampling, HWSS estimator, Fresnel, and a glass-sphere
> furnace test of the transport logic. **Not yet compiled:** `src/render.cpp`,
> `src/main.cpp`, and the GLSL (written without a Vulkan SDK). Expect small
> first-build fixes.

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

## Roadmap
1. **Spectral path tracer** (no ReSTIR): HWSS, Sellmeier dispersion, XYZ output, furnace test
2. **Scenes that prove the point**: prism, diamond, thin-film bubble, glass caustics
3. **ReSTIR DI** in spectral form (baseline)
4. **ReSTIR PT, RGB-style**: wavelength fixed per pixel/frame (your control group)
5. **Wavelength-aware shift mappings** (see docs/DESIGN.md), the novel part
6. **Denoiser + temporal accumulation** that respects spectral samples
7. **Benchmarks**: RMSE vs reference at equal ms, RGB-ReSTIR vs spectral-ReSTIR

## Layout
```
src/main.cpp              Vulkan RT capability probe
src/render.cpp            headless renderer: BLAS/TLAS, compute dispatch, PPM output
shaders/pathtrace.comp    spectral path tracer (ray queries, hero wavelengths, dispersion)
tests/transport_test.cpp  CPU mirror of the transport + glass furnace test
shaders/spectral.glsl     HWSS, CIE fit, Sellmeier
shaders/reservoir.glsl    80-byte spectral reservoir + update/finalize
tests/spectral_test.cpp   CPU reference: unbiasedness + variance checks
docs/DESIGN.md            algorithm notes and open questions
```
