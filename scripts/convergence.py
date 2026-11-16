#!/usr/bin/env python3
"""Render the prism scene at increasing sample counts, measure error against a reference
(ref.pfm, rendered with a different frame offset), and check that RMSE falls like 1/sqrt(N).
Writes a markdown table to $GITHUB_STEP_SUMMARY when running in CI.

usage: convergence.py [render_bin] [compare_bin] [spv] [mode]
"""
import math, os, re, subprocess, sys

render = sys.argv[1] if len(sys.argv) > 1 else "build/sr_render"
compare = sys.argv[2] if len(sys.argv) > 2 else "build/sr_compare"
spv = sys.argv[3] if len(sys.argv) > 3 else "build/pathtrace.spv"
mode = sys.argv[4] if len(sys.argv) > 4 else "1"
W, H = "256", "192"
counts = (16, 64, 256, 1024)

rows = []
for n in counts:
    subprocess.run([render, mode, W, H, str(n), spv, f"s_{n}.ppm", f"s_{n}.pfm", "0"],
                   check=True, stdout=subprocess.DEVNULL)
    out = subprocess.run([compare, "ref.pfm", f"s_{n}.pfm"], check=True,
                         capture_output=True, text=True).stdout
    rmse = float(re.search(r"RMSE=([0-9.eE+-]+)", out).group(1))
    rel = float(re.search(r"relMSE=([0-9.eE+-]+)", out).group(1))
    rows.append((n, rmse, rel))
    print(f"spp={n:5d}  RMSE={rmse:.5f}  relMSE={rel:.5f}", flush=True)

# least-squares slope of log(RMSE) vs log(N); pure Monte Carlo noise gives -0.5.
xs = [math.log(n) for n, _, _ in rows]; ys = [math.log(r) for _, r, _ in rows]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
ok = -0.65 <= slope <= -0.35
print(f"log-log slope = {slope:.3f} (expect about -0.5)  ->  {'PASS' if ok else 'FAIL'}")

summ = os.environ.get("GITHUB_STEP_SUMMARY")
if summ:
    with open(summ, "a") as f:
        f.write(f"### Convergence vs reference (mode {mode}, {W}x{H})\n\n| spp | RMSE | relMSE |\n|---|---|---|\n")
        for n, r, rel in rows:
            f.write(f"| {n} | {r:.5f} | {rel:.5f} |\n")
        f.write(f"\nlog-log slope: **{slope:.3f}** (expect about -0.5): {'PASS' if ok else 'FAIL'}\n")
sys.exit(0 if ok else 1)
