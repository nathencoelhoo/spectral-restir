#!/usr/bin/env python3
"""Light tracing vs camera-side candidates for the caustic channel, at equal sample counts and equal wall time.

Baseline : mode 4, 8 candidates per pixel per frame, no reuse (= plain Monte Carlo of the caustic channel)
Light    : mode 6, 8 photons per pixel per frame (aimed at the prism, splatted to the camera)
Both are compared with c_ref.pfm (the baseline rendered with many frames and independent seeds).
MSE ratio = MSE(light) / MSE(baseline); below 1 means light tracing wins. The reference has its own noise, which
pulls ratios toward 1, so a ratio near the noise floor understates the real advantage. The time column is wall
clock on the CI runner (software Vulkan, shared machine: indicative only).

usage: lt_efficiency.py [render_bin] [compare_bin] [spv]
"""
import os, re, subprocess, sys, time

render = sys.argv[1] if len(sys.argv) > 1 else "build/sr_render"
compare = sys.argv[2] if len(sys.argv) > 2 else "build/sr_compare"
spv = sys.argv[3] if len(sys.argv) > 3 else "build/pathtrace.spv"
W, H = "160", "120"
configs = [("baseline: 8 candidates (mode 4)", "4", 8, 0, 0), ("light tracing: 8 photons (mode 6)", "6", 0, 0, 8)]
frame_counts = (16, 64, 256)

rows = []
for fi, frames in enumerate(frame_counts):
    base_mse = base_t = None
    for ci, (name, mode, cands, reuse, photons) in enumerate(configs):
        off = 4_000_000 + 100_000 * fi + 10_000 * ci
        t0 = time.time()
        subprocess.run([render, mode, W, H, str(frames), spv, "l.ppm", "l.pfm", str(off), str(cands), str(reuse), str(photons)],
                       check=True, stdout=subprocess.DEVNULL)
        dt = time.time() - t0
        out = subprocess.run([compare, "c_ref.pfm", "l.pfm"], check=True, capture_output=True, text=True).stdout
        rmse = float(re.search(r"RMSE=([0-9.eE+-]+)", out).group(1)); mse = rmse * rmse
        if ci == 0: base_mse, base_t = mse, dt
        rows.append((frames, name, rmse, mse / base_mse, dt))
        print(f"frames={frames:4d}  {name:<36s} RMSE={rmse:.6f}  MSE ratio={mse / base_mse:.3f}  time={dt:.1f}s", flush=True)

summ = os.environ.get("GITHUB_STEP_SUMMARY")
if summ:
    with open(summ, "a") as f:
        f.write("### Light tracing vs candidates for the caustic channel (8 samples per pixel per frame), 160x120\n\n"
                "| frames | config | RMSE | MSE ratio vs baseline (below 1 = light tracing wins) | wall time (s) |\n|---|---|---|---|---|\n")
        for frames, name, rmse, ratio, dt in rows:
            f.write(f"| {frames} | {name} | {rmse:.6f} | {ratio:.3f} | {dt:.1f} |\n")
        f.write("\nThe reference has its own noise, which pulls ratios toward 1.\n")
