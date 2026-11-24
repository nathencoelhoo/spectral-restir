// Experiment (not in CI): 2D grid of floor pixels with camera jitter and real pixel footprints, equal-cost
// comparison of baseline / temporal / spatial-only / temporal+spatial reuse. Built to check whether the toy
// reproduces the GPU result (see docs/DESIGN.md). Variance estimates from 100 chains carry about +-15% noise.
// Build: g++ -std=c++20 -O2 experiments/restir_iter2d_toy.cpp -o restir_iter2d_toy
// Run:   ./restir_iter2d_toy DX DZ cands reuse T chains [jitter] [seed]   reuse bits: 1 temporal, 2 spatial
// Pixel footprints at the caustic: 160x120 -> 0.047 x 0.165; 256x192 -> 0.029 x 0.102; 800x600 -> 0.009 x 0.033
#define main toy_main
#include "../tests/restir_toy_test.cpp"
#undef main
#include <cstdlib>

struct PRes { double u[NU]; double W = 0, M = 0, pY = 0, px = 0, pz = 0; V3 f{0, 0, 0}; bool valid = false; };

int main(int argc, char** argv) {
  double DX = atof(argv[1]), DZ = atof(argv[2]); int cands = atoi(argv[3]), reuse = atoi(argv[4]), T = atoi(argv[5]), CH = atoi(argv[6]);
  int jitter = argc > 7 ? atoi(argv[7]) : 1; unsigned seed = argc > 8 ? atoi(argv[8]) : 1;
  for (double l = 360; l < 830; l += 0.01) yInt += xyzBar(l).y * 0.01;
  GEOM = 1; buildPrism(); LH = 1.5;
  const int P = 31, C = 15; const double XC = 1.3, MCAP = 20;
  std::mt19937_64 rng(1000 + seed); std::uniform_real_distribution<double> U(0, 1); std::vector<double> tmp(NU);
  // reference: pixel-integrated mean at the centre pixel (jitter over the footprint), plain MC
  double rs = 0, rs2 = 0; const long NREF = 3000000;
  for (long i = 0; i < NREF; ++i) { for (auto& a : tmp) a = U(rng);
    V3 f = evalPath(XC + (U(rng) - 0.5) * DX, (U(rng) - 0.5) * DZ, tmp.data()); rs += f.y; rs2 += f.y * f.y; }
  double refY = rs / NREF, refSE = std::sqrt((rs2 / NREF - refY * refY) / NREF);
  double sA = 0, sA2 = 0;
  for (int ch = 0; ch < CH; ++ch) {
    std::vector<PRes> cur(P * P), nxt(P * P); double acc = 0;
    for (int t = 0; t < T; ++t) {
      for (int j = 0; j < P; ++j) for (int i = 0; i < P; ++i) {
        int idx = j * P + i;
        double px = XC + (i - C) * DX + (jitter ? (U(rng) - 0.5) * DX : 0), pz = (j - C) * DZ + (jitter ? (U(rng) - 0.5) * DZ : 0);
        PRes cand; cand.M = cands; cand.px = px; cand.pz = pz; double wsum = 0, pick = -1; for (auto& a : cand.u) a = U(rng);
        for (int c = 0; c < cands; ++c) { for (auto& a : tmp) a = U(rng); V3 f = evalPath(px, pz, tmp.data()); wsum += f.y;
          if (f.y > 0 && U(rng) * wsum < f.y) { std::copy(tmp.begin(), tmp.end(), cand.u); pick = f.y; cand.f = f; } }
        cand.pY = pick > 0 ? pick : 0; cand.W = pick > 0 ? wsum / (cands * pick) : 0; cand.valid = true;
        std::vector<const PRes*> src;
        if (t > 0 && (reuse & 1) && cur[idx].valid) src.push_back(&cur[idx]);
        if (t > 0 && (reuse & 2)) { int got = 0;
          for (int tr = 0; tr < 12 && got < 2; ++tr) { int ox = int(U(rng) * 7) - 3, oy = int(U(rng) * 7) - 3; if (!ox && !oy) continue;
            int ni = i + ox, nj = j + oy; if (ni < 0 || nj < 0 || ni >= P || nj >= P) continue; const PRes* n = &cur[nj * P + ni]; if (!n->valid) continue;
            bool dup = false; for (auto* s : src) if (s == n) dup = true; if (dup) continue; src.push_back(n); ++got; } }
        int k = src.size(); double Mc = cand.M, mc = 1, tot = Mc, ws = 0; std::vector<double> w(k + 1); std::vector<V3> fq(k);
        for (int s = 0; s < k; ++s) { const PRes& r = *src[s]; double sM = std::fmin(r.M, MCAP); tot += sM; w[s] = 0;
          if (r.W > 0) { fq[s] = evalPath(px, pz, r.u); double den = sM * r.pY + Mc * fq[s].y; double mi = den > 0 ? (1.0 / k) * sM * r.pY / den : 0; w[s] = mi * fq[s].y * r.W; }
          double piXc = cand.pY > 0 ? evalPath(r.px, r.pz, cand.u).y : 0; double denc = sM * piXc + Mc * cand.pY; mc -= denc > 0 ? (1.0 / k) * sM * piXc / denc : 0; ws += w[s]; }
        w[k] = mc * cand.pY * cand.W; ws += w[k];
        PRes& o = nxt[idx]; o.valid = true; o.px = px; o.pz = pz; o.M = std::fmin(tot, MCAP);
        if (ws > 0) { double th = U(rng) * ws, a = 0; int sel = k; for (int s = 0; s <= k; ++s) { a += w[s]; if (th < a) { sel = s; break; } }
          if (sel == k) { std::copy(cand.u, cand.u + NU, o.u); o.f = cand.f; } else { std::copy(src[sel]->u, src[sel]->u + NU, o.u); o.f = fq[sel]; }
          o.pY = o.f.y; o.W = o.pY > 0 ? ws / o.pY : 0; }
        else { o.W = 0; o.pY = 0; o.f = {0, 0, 0}; std::copy(cand.u, cand.u + NU, o.u); }
      }
      std::swap(cur, nxt); acc += cur[C * P + C].f.y * cur[C * P + C].W;
    }
    double A = acc / T; sA += A; sA2 += A * A;
  }
  double mA = sA / CH, vA = sA2 / CH - mA * mA, z = (mA - refY) / std::sqrt(vA / CH + refSE * refSE);
  std::printf("DX=%.3f DZ=%.3f cands=%d reuse=%d T=%d chains=%d jitter=%d | bias %+.2f%% (z %.2f)  Var(mean of %d frames)=%.6e\n", DX, DZ, cands, reuse, T, CH, jitter, 100 * (mA - refY) / refY, z, T, vA);
}
