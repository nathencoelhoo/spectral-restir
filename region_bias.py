#!/usr/bin/env python3
"""Per-region bias test: is `test` (e.g. brute-force path tracing) statistically consistent with
two independent renders A and B of the reference estimator (NEE+MIS)?

usage: region_bias.py A.pfm B.pfm TEST.pfm [label]

A whole-image mean can hide a local error (the caustic is a small part of the frame), so this
compares regions separately. Regions come from the camera geometry in shaders/pathtrace.comp
(mode 1/2 camera), so keep EYE/TARGET/FOV in sync with it:
  prism_bbox : screen bounding box of the prism
  floor_near : floor with |x|,|z| <= 3 (where the caustic lands)
  floor_far  : rest of the floor (|x|,|z| <= 6)
Statistics: D = TEST - (A+B)/2 per pixel. z = mean(D) / (std(D)/sqrt(N)). Calibration: z0 from
A - B, which contains pure noise, so |z0| should be small. FAIL if |z| > 5 and |relative bias| > 0.5%.
"""
import math, os, struct, sys

EYE, TARGET = (0.0, 2.2, 7.5), (0.0, 0.9, 0.0)
TAN_HALF = math.tan(math.radians(40.0) * 0.5)
Z_FAIL, BIAS_FAIL = 5.0, 0.005

def sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a):
    n = math.sqrt(dot(a, a)); return (a[0]/n, a[1]/n, a[2]/n)
FWD = norm(sub(TARGET, EYE)); RIGHT = norm(cross(FWD, (0.0, 1.0, 0.0))); UP = cross(RIGHT, FWD)

def project(P, W, H):
    v = sub(P, EYE); z = dot(v, FWD)
    u = (dot(v, RIGHT) / z) / (TAN_HALF * W / H); w = -(dot(v, UP) / z) / TAN_HALF
    return (u + 1) / 2 * W, (w + 1) / 2 * H

def ray_dir(px, py, W, H):
    u = ((px + 0.5) / W) * 2 - 1; v = ((py + 0.5) / H) * 2 - 1
    return norm(tuple(FWD[i] + RIGHT[i] * (u * TAN_HALF * W / H) - UP[i] * (v * TAN_HALF) for i in range(3)))

def build_masks(W, H):
    s3 = math.sqrt(3.0)
    pts = [(x, y, z) for (x, y) in ((-1, 0), (1, 0), (0, s3)) for z in (-1, 1)]
    pp = [project(p, W, H) for p in pts]
    x0, x1 = min(p[0] for p in pp), max(p[0] for p in pp)
    y0, y1 = min(p[1] for p in pp), max(p[1] for p in pp)
    labels = []
    for y in range(H):
        for x in range(W):
            if x0 <= x + 0.5 <= x1 and y0 <= y + 0.5 <= y1:
                labels.append("prism_bbox"); continue
            d = ray_dir(x, y, W, H)
            if d[1] < 0:
                t = -EYE[1] / d[1]; hx = EYE[0] + t * d[0]; hz = EYE[2] + t * d[2]
                if abs(hx) <= 3 and abs(hz) <= 3: labels.append("floor_near")
                elif abs(hx) <= 6 and abs(hz) <= 6: labels.append("floor_far")
                else: labels.append("other")
            else: labels.append("sky")
    return labels

def read_pfm(path):
    with open(path, "rb") as f:
        magic = f.readline().strip(); w, h = map(int, f.readline().split()); scale = float(f.readline())
        if magic != b"PF" or scale >= 0: raise ValueError("need little-endian RGB PFM")
        data = struct.unpack(f"<{w*h*3}f", f.read(w * h * 12))
    lum = [0.0] * (w * h)
    for r in range(h):                      # file rows run bottom-to-top
        y, base = h - 1 - r, r * w * 3
        for x in range(w):
            R, G, B = data[base + 3*x: base + 3*x + 3]; lum[y*w + x] = 0.2126*R + 0.7152*G + 0.0722*B
    return w, h, lum

def zstat(diff):
    n = len(diff); m = sum(diff) / n
    var = sum((d - m) ** 2 for d in diff) / max(1, n - 1)
    return m, (m / math.sqrt(var / n)) if var > 0 else 0.0

def main():
    if len(sys.argv) not in (4, 5): print(__doc__); return 1
    label = sys.argv[4] if len(sys.argv) == 5 else "brute-force"
    (w, h, A), (w2, h2, B), (w3, h3, T) = (read_pfm(p) for p in sys.argv[1:4])
    assert (w, h) == (w2, h2) == (w3, h3), "size mismatch"
    labels = build_masks(w, h)
    rows, fail = [], False
    for name in ("prism_bbox", "floor_near", "floor_far"):
        idx = [i for i, l in enumerate(labels) if l == name]
        if len(idx) < 50: continue
        ref = [(A[i] + B[i]) / 2 for i in idx]; mref = sum(ref) / len(ref)
        mt = sum(T[i] for i in idx) / len(idx)
        md, z = zstat([T[i] - r for i, r in zip(idx, ref)])
        _, z0 = zstat([A[i] - B[i] for i in idx])
        rel = md / mref if mref > 0 else 0.0
        bad = abs(z) > Z_FAIL and abs(rel) > BIAS_FAIL
        fail |= bad
        rows.append((name, len(idx), mref, mt, rel, z, z0, bad))
    hdr = f"{'region':<11}{'pixels':>7}{'mean(ref)':>11}{'mean(test)':>11}{'bias':>9}{'z':>8}{'null z':>8}  verdict"
    print(hdr)
    for n, c, mr, mt, rel, z, z0, bad in rows:
        print(f"{n:<11}{c:>7}{mr:>11.5f}{mt:>11.5f}{rel*100:>+8.3f}%{z:>8.2f}{z0:>8.2f}  {'FAIL' if bad else 'ok'}")
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as f:
            f.write(f"### Per-region bias ({label} vs NEE+MIS)\n\n| region | pixels | mean ref | mean test | bias | z | null z | verdict |\n|---|---|---|---|---|---|---|---|\n")
            for n, c, mr, mt, rel, z, z0, bad in rows:
                f.write(f"| {n} | {c} | {mr:.5f} | {mt:.5f} | {rel*100:+.3f}% | {z:.2f} | {z0:.2f} | {'FAIL' if bad else 'ok'} |\n")
            f.write(f"\nFAIL needs |z| > {Z_FAIL} and |bias| > {BIAS_FAIL*100:.1f}%. 'null z' compares A with B (noise only) and should be small.\n")
    print("region bias check:", "FAIL" if fail else "PASS")
    return 3 if fail else 0

if __name__ == "__main__":
    sys.exit(main())
