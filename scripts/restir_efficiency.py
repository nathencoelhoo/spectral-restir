#!/usr/bin/env python3
"""Noise of the ReSTIR caustic channel at equal path evaluations (mode 4 = caustic channel only).

Every configuration costs 8 path evaluations per pixel per frame:
  baseline            : 8 candidates, no reuse                      (= plain Monte Carlo, 8 samples/frame)
  temporal            : 6 candidates + temporal source (2 evals)
  temporal + spatial  : 2 candidates + temporal (2) + 2 spatial neighbours (4)
Each is rendered at several frame counts and compared to a high-sample reference (c_ref.pfm, baseline
config, independent seeds). Ratio = MSE(config) / MSE(baseline); below 1 means reuse wins.

Caveats printed with the table: the reference has its own noise (so ratios at high spp are slightly
pessimistic for the reuse configs) and reuse noise is spatially correlated (blotchy), which RMSE does not show.

usage: restir_efficiency.py [render_bin] [compare_bin] [spv]
"""
import os, re, subprocess, sys

render = sys.argv[1] if len(sys.argv) > 1 else "build/sr_render"
compare = sys.argv[2] if len(sys.argv) > 2 else "build/sr_compare"
spv = sys.argv[3] if len(sys.argv) > 3 else "build/pathtrace.spv"
W, H = "160", "120"
configs = [("baseline: 8 cand, no reuse", 8, 0), ("temporal: 6 cand + temporal", 6, 1),
           ("temporal+spatial: 2 cand + temporal + 2 spatial", 2, 3)]
spps = (32, 128, 512)

rows = []
for si, spp in enumerate(spps):
    base_mse = None
    for ci, (name, cands, reuse) in enumerate(configs):
        off = 3_000_000 + 100_000 * si + 10_000 * ci
        subprocess.run([render, "4", W, H, str(spp), spv, "e.ppm", "e.pfm", str(off), str(cands), str(reuse)],
                       check=True, stdout=subprocess.DEVNULL)
        out = subprocess.run([compare, "c_ref.pfm", "e.pfm"], check=True, capture_output=True, text=True).stdout
        rmse = float(re.search(r"RMSE=([0-9.eE+-]+)", out).group(1))
        mse = rmse * rmse
        if ci == 0: base_mse = mse
        ratio = mse / base_mse
        rows.append((spp, name, rmse, ratio))
        print(f"spp={spp:4d}  {name:<52s} RMSE={rmse:.6f}  MSE ratio vs baseline = {ratio:.2f}", flush=True)

summ = os.environ.get("GITHUB_STEP_SUMMARY")
if summ:
    with open(summ, "a") as f:
        f.write("### Caustic-channel noise at equal cost (8 path evaluations per pixel per frame)\n\n"
                "| frames | config | RMSE | MSE ratio vs baseline (below 1 = reuse wins) |\n|---|---|---|---|\n")
        for spp, name, rmse, ratio in rows:
            f.write(f"| {spp} | {name} | {rmse:.6f} | {ratio:.2f} |\n")
        f.write("\nThe reference has its own noise and reuse noise is spatially correlated (blotchy); "
                "RMSE does not show that.\n")
