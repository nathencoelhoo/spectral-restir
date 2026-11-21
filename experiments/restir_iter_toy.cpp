// Experiment (not a CI test): iterated temporal+spatial ReSTIR reuse on a strip of floor pixels under the prism,
// pairwise MIS, random replay with wavelength in u. Reports bias z-test and variance vs plain MC at equal evals.
// Build: g++ -std=c++20 -O2 experiments/restir_iter_toy.cpp -o restir_iter_toy
// Run:   ./restir_iter_toy DX N0 NS Mcap T chains xc temporal   (defaults: 0.03 2 2 20 40 800 1.3 1)
#define main toy_main
#include "../tests/restir_toy_test.cpp"
#undef main
#include <cstdlib>

struct R2 { double u[NU]; double W = 0, M = 0, pY = 0; V3 f{0, 0, 0}; };
struct Src { const R2* r; double M, pq_Xi, pi_Xc, pi_Xi; };

int main(int argc, char** argv) {
  double DX = argc > 1 ? atof(argv[1]) : 0.03; int N0 = argc > 2 ? atoi(argv[2]) : 2, NS = argc > 3 ? atoi(argv[3]) : 2;
  double Mcap = argc > 4 ? atof(argv[4]) : 20; int T = argc > 5 ? atoi(argv[5]) : 40, RCH = argc > 6 ? atoi(argv[6]) : 800;
  double xc = argc > 7 ? atof(argv[7]) : 1.3; bool temporal = argc > 8 ? atoi(argv[8]) : 1;
  for (double l = 360; l < 830; l += 0.01) yInt += xyzBar(l).y * 0.01;
  GEOM = 1; buildPrism(); LH = 1.5;
  const int P = 41, C = P / 2, F = 16, RAD = 3;
  std::mt19937_64 rng(77); std::uniform_real_distribution<double> U(0, 1);
  std::vector<double> tmp(NU);
  // reference and plain-MC variance at the centre pixel
  double rs = 0, rs2 = 0, rx = 0, rz = 0; const long NREF = 3000000;
  for (long i = 0; i < NREF; ++i) { for (auto& a : tmp) a = U(rng); V3 f = evalPath(xc, 0, tmp.data()); rs += f.y; rs2 += f.y * f.y; }
  double refY = rs / NREF, v1 = rs2 / NREF - refY * refY, refSE = std::sqrt(v1 / NREF);
  const double cost = N0 + 2.0 * NS;   // spatial neighbours cost 2 evals each; temporal costs 0 (static scene)
  double sA = 0, sA2 = 0, sL = 0, sL2 = 0;
  for (int ch = 0; ch < RCH; ++ch) {
    std::vector<R2> cur(P), nxt(P); double acc = 0, last = 0;
    for (int t = 0; t < T; ++t) {
      for (int j = 0; j < P; ++j) {
        double pxj = xc + (j - C) * DX;
        R2 cand; cand.M = N0; double wsum = 0, pick = -1; R2 dummy;
        for (auto& a : cand.u) a = U(rng);
        for (int c = 0; c < N0; ++c) {
          for (auto& a : tmp) a = U(rng); V3 f = evalPath(pxj, 0, tmp.data()); wsum += f.y;
          if (f.y > 0 && U(rng) * wsum < f.y) { std::copy(tmp.begin(), tmp.end(), cand.u); pick = f.y; cand.f = f; }
        }
        cand.pY = pick > 0 ? pick : 0; cand.W = pick > 0 ? wsum / (N0 * pick) : 0;
        std::vector<Src> src;
        if (t > 0 && temporal && cur[j].M > 0) src.push_back({&cur[j], std::fmin(cur[j].M, Mcap), cur[j].pY, cand.pY, cur[j].pY});
        if (t > 0) {
          int offs[2 * RAD]; int n = 0; for (int o = -RAD; o <= RAD; ++o) if (o) offs[n++] = o;
          for (int a = n - 1; a > 0; --a) std::swap(offs[a], offs[int(U(rng) * (a + 1))]);
          int used = 0;
          for (int a = 0; a < n && used < NS; ++a) {
            int jn = j + offs[a]; if (jn < 0 || jn >= P || cur[jn].M <= 0) continue;
            double pxn = xc + (jn - C) * DX;
            V3 fq = evalPath(pxj, 0, cur[jn].u);
            double piXc = cand.pY > 0 ? evalPath(pxn, 0, cand.u).y : 0.0;
            src.push_back({&cur[jn], std::fmin(cur[jn].M, Mcap), fq.y, piXc, cur[jn].pY}); ++used;
            (void)fq;
          }
        }
        int k = (int)src.size(); double Mc = cand.M, mc = 1.0, tot = Mc; std::vector<double> w(k + 1); double ws = 0;
        std::vector<V3> fqs(k);
        for (int i = 0; i < k; ++i) {
          const Src& s = src[i]; tot += s.M;
          double den = s.M * s.pi_Xi + Mc * s.pq_Xi, mi = den > 0 ? (1.0 / k) * s.M * s.pi_Xi / den : 0.0;
          w[i] = mi * s.pq_Xi * s.r->W; ws += w[i];
          double denc = s.M * s.pi_Xc + Mc * cand.pY; mc -= denc > 0 ? (1.0 / k) * s.M * s.pi_Xc / denc : 0.0;
        }
        w[k] = mc * cand.pY * cand.W; ws += w[k];
        R2& out = nxt[j]; out.M = std::fmin(tot, Mcap);
        if (ws > 0) {
          double th = U(rng) * ws, a = 0; int sel = k; for (int i = 0; i <= k; ++i) { a += w[i]; if (th < a) { sel = i; break; } }
          const double* us = sel == k ? cand.u : src[sel].r->u; std::copy(us, us + NU, out.u);
          V3 f = evalPath(pxj, 0, out.u);   // (a real implementation stores this; counted as free)
          out.f = f; out.pY = f.y; out.W = f.y > 0 ? ws / f.y : 0;
        } else { out.W = 0; out.pY = 0; out.f = {0, 0, 0}; std::copy(cand.u, cand.u + NU, out.u); }
      }
      std::swap(cur, nxt);
      double e = cur[C].f.y * cur[C].W; if (t >= T - F) acc += e; if (t == T - 1) last = e;
    }
    double A = acc / F; sA += A; sA2 += A * A; sL += last; sL2 += last * last;
  }
  double mA = sA / RCH, vA = sA2 / RCH - mA * mA, mL = sL / RCH, vL = sL2 / RCH - mL * mL;
  double zA = (mA - refY) / std::sqrt(vA / RCH + refSE * refSE), zL = (mL - refY) / std::sqrt(vL / RCH + refSE * refSE);
  std::printf("DX=%.3f N0=%d NS=%d Mcap=%.0f T=%d chains=%d temporal=%d | cost/frame=%.0f evals | ref Y=%.5f\n", DX, N0, NS, Mcap, T, RCH, temporal, cost, refY);
  std::printf("  last-frame estimate:  bias %+.2f%% (z %.2f)  variance %.5f  vs plain MC %.5f  -> efficiency %.2fx\n",
              100 * (mL - refY) / refY, zL, vL, v1 / cost, vL * cost / v1);
  std::printf("  mean of last %d frames: bias %+.2f%% (z %.2f)  variance %.6f  vs plain MC %.6f  -> efficiency %.2fx\n",
              F, 100 * (mA - refY) / refY, zA, vA, v1 / (cost * F), vA * cost * F / v1);
}
