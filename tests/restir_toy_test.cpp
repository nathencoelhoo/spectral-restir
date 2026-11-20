// CPU toy model of ReSTIR for caustic paths, with wavelength as part of the path.
//
// Domain: a floor point x (normal +y). A "path" is a vector u of uniform random numbers:
//   u[0],u[1] cosine-sampled direction, u[2] wavelength, u[3..] Fresnel decisions at glass hits.
// Replaying the same u at a *different* floor point x' gives a different path (random replay shift,
// Jacobian 1 in u-space; the wavelength is part of u so dispersion is handled exactly, at the price
// of re-tracing). Only paths with >= 1 glass interaction count (the "caustic channel": direct
// light is handled by NEE+MIS in the renderer, glass blocks NEE).
//
// Reservoirs hold (u, W) with W an unbiased contribution weight: E[W h(y)] = int h(u) du over {p_hat>0}.
// Reuse combines a pixel's own reservoir with neighbours' via generalized RIS (Lin et al. 2022):
//   w_i = m_i(y_i) * p_hat_q(y_i) * W_i,  W_new = (sum_i w_i) / p_hat_q(y_sel)
// with generalized-balance MIS m_i(y) = M_i p_hat_i(y) / sum_j M_j p_hat_j(y). The naive variant
// divides by sum_i M_i instead (no MIS); it is biased when neighbours' supports differ.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

struct V3 { double x, y, z; };
static V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
static double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// ---- spectral helpers (mirror shaders/spectral.glsl) ----
static double lobe(double l, double mu, double s1, double s2) {
  double t = (l - mu) / (l < mu ? s1 : s2); return std::exp(-0.5 * t * t);
}
static V3 xyzBar(double l) {
  return {1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7) - 0.065 * lobe(l, 501.1, 20.4, 26.2),
          0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1),
          1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8)};
}
static double pdfVis(double l) { double c = std::cosh(0.0072 * (l - 538.0)); return 0.0039398042 / (c * c); }
static double sampleVis(double u) { return 538.0 - 138.888889 * std::atanh(0.85691062 - 1.82750197 * u); }
static double bk7(double nm) {
  const double B[3] = {1.03961212, 0.231792344, 1.01046945}, C[3] = {0.00600069867, 0.0200179144, 103.560653};
  double l2 = nm * 1e-3; l2 *= l2; double s = 1; for (int i = 0; i < 3; ++i) s += B[i] * l2 / (l2 - C[i]); return std::sqrt(s);
}
static double lightSpectrum(double l) {
  const double C2 = 1.4387769e7, T = 6540.0;
  double p = std::pow(l / 560.0, -5.0) * (std::exp(C2 / (560.0 * T)) - 1.0) / (std::exp(C2 / (l * T)) - 1.0);
  double u = (l - 540.0) / 45.0;
  return 0.958039 * p * (1.0 + 0.07 * std::exp(-0.5 * u * u));
}
static double fresnel(double cosI, double eta) {
  double s2 = std::fmax(0.0, 1 - cosI * cosI) / (eta * eta); if (s2 >= 1) return 1;
  double cosT = std::sqrt(1 - s2), rp = (eta * cosI - cosT) / (eta * cosI + cosT), rs = (cosI - eta * cosT) / (cosI + eta * cosT);
  return 0.5 * (rp * rp + rs * rs);
}

// ---- scene: glass sphere above a diffuse floor, area light at y=4 (as in the renderer) ----
static const V3 SC{0, 1.2, 0}; static const double SR = 1.0, ALBEDO = 0.6, EMISSION = 12.0, EPS = 1e-3;
static V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static int GEOM = 0;   // 0 = glass sphere (a lens: magnifies offsets), 1 = the renderer's prism (flat faces)
struct Tri { V3 a, b, c, n; };
static std::vector<Tri> PRISM;
static void buildPrism() {   // same prism as src/render.cpp: side 2, length 2, base on the floor
  const double s3 = std::sqrt(3.0);
  V3 v[6] = {{-1, 0, -1}, {1, 0, -1}, {0, s3, -1}, {-1, 0, 1}, {1, 0, 1}, {0, s3, 1}}, cen{0, s3 / 3, 0};
  int f[8][3] = {{0, 1, 2}, {3, 4, 5}, {0, 1, 4}, {0, 4, 3}, {1, 2, 5}, {1, 5, 4}, {2, 0, 3}, {2, 3, 5}};
  for (auto& t : f) {
    V3 a = v[t[0]], b = v[t[1]], c = v[t[2]], n = cross(b - a, c - a); n = n * (1 / std::sqrt(dot(n, n)));
    if (dot(n, (a + b + c) * (1.0 / 3) - cen) < 0) n = n * -1.0;       // outward
    PRISM.push_back({a, b, c, n});
  }
}
static double yInt = 0, LH = 1.5;   // LH = light half-size (smaller light = rarer caustic)
static const int NU = 16;

static bool intersectGlass(V3 o, V3 d, double& tOut, V3& nOut) {
  if (GEOM == 0) {
    V3 oc = o - SC; double b = dot(oc, d), c = dot(oc, oc) - SR * SR, disc = b * b - c, t = -1;
    if (disc > 0) { double sq = std::sqrt(disc); t = (-b - sq > 1e-6) ? -b - sq : ((-b + sq > 1e-6) ? -b + sq : -1); }
    if (t < 0) return false;
    tOut = t; nOut = ((o + d * t) - SC) * (1.0 / SR); return true;
  }
  double best = 1e30; const Tri* hit = nullptr;
  for (const Tri& T : PRISM) {                                  // Moller-Trumbore
    V3 e1 = T.b - T.a, e2 = T.c - T.a, pv = cross(d, e2); double det = dot(e1, pv);
    if (std::fabs(det) < 1e-12) continue;
    double inv = 1 / det; V3 tv = o - T.a; double u = dot(tv, pv) * inv; if (u < 0 || u > 1) continue;
    V3 qv = cross(tv, e1); double v = dot(d, qv) * inv; if (v < 0 || u + v > 1) continue;
    double t = dot(e2, qv) * inv; if (t > 1e-6 && t < best) { best = t; hit = &T; }
  }
  if (!hit) return false;
  tOut = best; nOut = hit->n; return true;
}

// Caustic-channel contribution (XYZ) of random vector u replayed at floor point (px, 0, pz).
static V3 evalPath(double px, double pz, const double* u) {
  double r = std::sqrt(u[0]), phi = 2 * M_PI * u[1];
  V3 d{r * std::cos(phi), std::sqrt(std::fmax(0.0, 1 - u[0])), r * std::sin(phi)};
  double lam = sampleVis(u[2]), pdf = pdfVis(lam), thr = ALBEDO;
  V3 o{px, EPS, pz}; int glass = 0;
  for (int depth = 0; depth < 12; ++depth) {
    double t = -1; V3 n{0, 0, 0};
    if (!intersectGlass(o, d, t, n)) {                   // left the glass: light or nothing
      if (glass > 0 && d.y > 1e-9) {
        double tl = (4.0 - o.y) / d.y, hx = o.x + tl * d.x, hz = o.z + tl * d.z;
        if (std::fabs(hx) <= LH && std::fabs(hz) <= LH) {
          double s = thr * EMISSION * lightSpectrum(lam) / pdf / yInt;
          V3 xb = xyzBar(lam); return {xb.x * s, xb.y * s, xb.z * s};
        }
      }
      return {0, 0, 0};
    }
    V3 p = o + d * t; bool front = dot(d, n) < 0; V3 nf = front ? n : n * -1.0;
    ++glass; double nG = bk7(lam), eta = front ? nG : 1.0 / nG, cosI = -dot(d, nf), F = fresnel(cosI, eta);
    if (u[3 + depth] < F) { d = d - nf * (2 * dot(nf, d)); o = p + nf * EPS; }
    else {
      double k = 1 - (1 / eta) * (1 / eta) * (1 - dot(nf, d) * dot(nf, d)); double e = 1 / eta;
      V3 rd = d * e - nf * (e * dot(nf, d) + std::sqrt(std::fmax(0.0, k)));
      double len = std::sqrt(dot(rd, rd)); d = rd * (1 / len); thr /= eta * eta; o = p - nf * EPS;
    }
  }
  return {0, 0, 0};
}

struct Res { std::vector<double> u; double W = 0, M = 0; };

int main(int argc, char** argv) {
  for (double l = 360; l < 830; l += 0.01) yInt += xyzBar(l).y * 0.01;
  std::mt19937_64 rng(2026); std::uniform_real_distribution<double> U(0, 1);
  auto fillU = [&](std::vector<double>& v) { v.resize(NU); for (auto& x : v) x = U(rng); };

  const int N0 = 4, K = 4;                  // initial candidates per pixel; neighbours reused (K=4 -> 5 reservoirs)
  double DX = 0.12;                         // floor spacing between neighbouring pixels (256x192 render: ~0.03 per pixel)
  const long TRIALS = argc > 1 ? std::atol(argv[1]) : 300000, NREF = 3000000;
  if (argc > 2) LH = std::atof(argv[2]);
  if (argc > 3) DX = std::atof(argv[3]);
  std::printf("neighbour spacing = %.3f\n", DX);
  std::printf("light half-size = %.3f\n", LH);
  double xs[2] = {0.0, 0.55};               // sphere: on-axis (bright caustic) and off-axis (partial support)
  if (argc > 4 && std::string(argv[4]) == "prism") { GEOM = 1; buildPrism(); xs[0] = 1.3; xs[1] = 1.9; }
  if (argc > 6) { xs[0] = std::atof(argv[5]); xs[1] = std::atof(argv[6]); }
  std::printf("geometry = %s\n", GEOM ? "prism" : "sphere");
  int fails = 0;

  for (double xq : xs) {
    // ---- reference: plain Monte Carlo of the caustic-channel integral at xq ----
    double rs[3] = {0, 0, 0}, rs2[3] = {0, 0, 0}; std::vector<double> u(NU);
    for (long i = 0; i < NREF; ++i) { for (auto& x : u) x = U(rng); V3 f = evalPath(xq, 0, u.data());
      rs[0] += f.x; rs[1] += f.y; rs[2] += f.z; rs2[0] += f.x * f.x; rs2[1] += f.y * f.y; rs2[2] += f.z * f.z; }
    double ref[3], refSE[3]; for (int c = 0; c < 3; ++c) { ref[c] = rs[c] / NREF; refSE[c] = std::sqrt(std::fmax(0.0, rs2[c] / NREF - ref[c] * ref[c]) / NREF); }

    // ---- trials ----
    const int S = K + 1, self = K / 2;                      // reservoirs at xq + (j - self)*DX
    double sum[4][3] = {}, sum2[4][3] = {}; double hitFrac = 0;   // [variant][component]; variants: none, balance, naive, pairwise
    for (long tr = 0; tr < TRIALS; ++tr) {
      std::vector<Res> R(S); double px[S];
      for (int j = 0; j < S; ++j) {                          // initial RIS per pixel (uniform source pdf)
        px[j] = xq + (j - self) * DX;
        double wsum = 0, pick = -1; Res& r = R[j]; r.M = N0; fillU(r.u);
        std::vector<double> cu(NU);
        for (int c = 0; c < N0; ++c) {
          fillU(cu); double w = evalPath(px[j], 0, cu.data()).y; wsum += w;
          if (w > 0 && U(rng) * wsum < w) { r.u = cu; pick = w; }
        }
        r.W = (pick > 0) ? wsum / (N0 * pick) : 0.0; if (pick <= 0) r.W = 0;
      }
      V3 F[S][S];                                            // F[j][i] = path of reservoir i replayed at pixel j
      for (int j = 0; j < S; ++j) for (int i = 0; i < S; ++i) F[j][i] = evalPath(px[j], 0, R[i].u.data());
      hitFrac += (F[self][self].y > 0);
      // variant 0: own reservoir only (no reuse)
      V3 e0 = F[self][self] * R[self].W;
      // variants 1/2: reuse across all S reservoirs
      double wb[S], wn[S], sb = 0, sn = 0, Msum = 0;
      for (int i = 0; i < S; ++i) {
        double den = 0; for (int j = 0; j < S; ++j) den += R[j].M * F[j][i].y;
        double m = den > 0 ? R[i].M * F[i][i].y / den : 0.0;
        wb[i] = m * F[self][i].y * R[i].W; sb += wb[i];
        wn[i] = F[self][i].y * R[i].W * R[i].M; sn += wn[i]; Msum += R[i].M;
      }
      auto select = [&](double* w, double s) { double t = U(rng) * s, a = 0; for (int i = 0; i < S; ++i) { a += w[i]; if (t < a) return i; } return S - 1; };
      V3 e1{0, 0, 0}, e2{0, 0, 0};
      if (sb > 0) { int i = select(wb, sb); if (F[self][i].y > 0) e1 = F[self][i] * (sb / F[self][i].y); }
      if (sn > 0) { int i = select(wn, sn); if (F[self][i].y > 0) e2 = F[self][i] * (sn / Msum / F[self][i].y); }
      // variant 3: pairwise MIS (Lin et al. 2022). Neighbour i competes only against the canonical
      // sample c (this pixel's own), so only p_hat_q(X_i) and p_hat_i(X_c) are needed: 2 evaluations per
      // neighbour instead of K. m_i(x) = (1/K) M_i p_i(x) / (M_i p_i(x) + M_c p_q(x)); m_c = 1 - sum_i m_i.
      double wp[S], sp = 0; const double Mc = R[self].M;
      for (int i = 0; i < S; ++i) wp[i] = 0;
      double mc = 1.0;
      for (int i = 0; i < S; ++i) {
        if (i == self) continue;
        double den = R[i].M * F[i][i].y + Mc * F[self][i].y;                  // at x = X_i
        double mi = den > 0 ? (1.0 / K) * R[i].M * F[i][i].y / den : 0.0;
        wp[i] = mi * F[self][i].y * R[i].W; sp += wp[i];
        double denc = R[i].M * F[i][self].y + Mc * F[self][self].y;           // at x = X_c
        mc -= denc > 0 ? (1.0 / K) * R[i].M * F[i][self].y / denc : 0.0;
      }
      wp[self] = mc * F[self][self].y * R[self].W; sp += wp[self];
      V3 e3{0, 0, 0};
      if (sp > 0) { int i = select(wp, sp); if (F[self][i].y > 0) e3 = F[self][i] * (sp / F[self][i].y); }
      V3 e[4] = {e0, e1, e2, e3};
      for (int v = 0; v < 4; ++v) { double c[3] = {e[v].x, e[v].y, e[v].z}; for (int k = 0; k < 3; ++k) { sum[v][k] += c[k]; sum2[v][k] += c[k] * c[k]; } }
    }
    std::printf("\nfloor point x=%.2f  (caustic-channel XYZ; reference Y = %.5f +- %.5f; p(initial sample has p_hat>0) = %.3f)\n",
                xq, ref[1], refSE[1], hitFrac / TRIALS);
    const char* names[4] = {"no reuse (RIS only)", "reuse, balance MIS ", "reuse, naive (1/sumM)", "reuse, pairwise MIS "};
    double var[4], biasPct[4], zz[4];
    for (int v = 0; v < 4; ++v) {
      double worstZ = 0, biasY = 0;
      for (int k = 0; k < 3; ++k) {
        double m = sum[v][k] / TRIALS, vr = std::fmax(0.0, sum2[v][k] / TRIALS - m * m), se = std::sqrt(vr / TRIALS);
        double z = (m - ref[k]) / std::sqrt(se * se + refSE[k] * refSE[k] + 1e-300); worstZ = std::fmax(worstZ, std::fabs(z));
        if (k == 1) { biasY = (m - ref[1]) / ref[1]; var[v] = vr; }
      }
      std::printf("  %-22s bias(Y) = %+7.3f%%   worst |z| over X,Y,Z = %5.2f   per-pixel variance(Y) = %.4f\n", names[v], biasY * 100, worstZ, var[v]);
      biasPct[v] = biasY * 100; zz[v] = worstZ;
      bool unbiasedExpected = (v != 2);
      if (unbiasedExpected && worstZ > 4.5) { ++fails; std::printf("  FAIL: expected unbiased\n"); }
      if (v == 2 && ref[1] > 0.05 && worstZ < 3) { ++fails; std::printf("  FAIL: naive reuse should be visibly biased (test lacks power)\n"); }
    }
    // plain MC variance of a single path sample, for scale
    double v1 = rs2[1] / NREF - ref[1] * ref[1];
    std::printf("  variance of one plain-MC path sample(Y) = %.4f\n", v1);
    // Cost-normalised: efficiency = variance x path evaluations per pixel (lower is better; plain MC = v1 x 1).
    // Toy cost of full balance MIS, no sharing across pixels: N0 initial + S replays at q + S*(S-1) MIS terms.
    // New path evaluations per pixel beyond the initial N0 candidates. Balance MIS needs every
    // off-diagonal entry of the SxS replay matrix: S(S-1) (an earlier version of this test also
    // counted S replays at q, double-counting; corrected here). Pairwise needs 2 per neighbour.
    const double evals[4] = {double(N0), double(N0 + S * (S - 1)), double(N0 + S * (S - 1)), double(N0 + 2 * K)};
    for (int v = 0; v < 4; ++v) {
      if (v == 2) continue;
      double eff = var[v] * evals[v] / v1;
      std::printf("  efficiency vs plain MC, %-20s: %.2fx %s  (evals/pixel = %.0f)\n", names[v], eff,
                  eff < 1 ? "(ReSTIR better)" : "(plain MC better)", evals[v]);
    }
    for (int v = 0; v < 4; ++v)
      std::printf("CSV,%.3f,%.3f,%.2f,%d,%.3f,%.2f,%.5f,%.3f\n", LH, DX, xq, v, biasPct[v], zz[v], var[v], var[v] * evals[v] / v1);
  }
  std::printf(fails ? "\n%d check(s) failed\n" : "\nunbiasedness checks passed\n", fails);
  return fails ? 1 : 0;
}
