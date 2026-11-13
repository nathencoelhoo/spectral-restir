// CPU mirror of the transport logic in shaders/pathtrace.comp (Fresnel, refraction,
// dispersion with hero-only continuation, Russian roulette, HWSS weighting).
// Furnace test: a lossless glass sphere in a flat-spectrum environment of radiance 1
// must render as exactly 1 everywhere. Catches sign/scale errors in the estimator.
#include <cmath>
#include <cstdio>
#include <random>

struct V { double x, y, z; };
static V operator+(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static V operator-(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V operator*(V a, double s) { return {a.x * s, a.y * s, a.z * s}; }
static double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V norm(V a) { return a * (1.0 / std::sqrt(dot(a, a))); }
static V reflectV(V i, V n) { return i - n * (2 * dot(n, i)); }
static bool refractV(V i, V n, double eta, V& out) {   // GLSL semantics: eta = n_i/n_t
  double k = 1 - eta * eta * (1 - dot(n, i) * dot(n, i));
  if (k < 0) return false;
  out = i * eta - n * (eta * dot(n, i) + std::sqrt(k));
  return true;
}

static double lobe(double l, double mu, double s1, double s2) {
  double t = (l - mu) / (l < mu ? s1 : s2); return std::exp(-0.5 * t * t);
}
static double ybar(double l) { return 0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1); }
static double pdfVis(double l) {
  if (l < 360 || l > 830) return 0; double c = std::cosh(0.0072 * (l - 538.0)); return 0.0039398042 / (c * c);
}
static double sampleVis(double u) { return 538.0 - 138.888889 * std::atanh(0.85691062 - 1.82750197 * u); }
static double bk7(double nm) {
  const double B[3] = {1.03961212, 0.231792344, 1.01046945}, C[3] = {0.00600069867, 0.0200179144, 103.560653};
  double l2 = nm * 1e-3; l2 *= l2; double s = 1; for (int i = 0; i < 3; ++i) s += B[i] * l2 / (l2 - C[i]); return std::sqrt(s);
}
static double fresnel(double cosI, double eta) {
  double s2 = std::fmax(0.0, 1 - cosI * cosI) / (eta * eta);
  if (s2 >= 1) return 1;
  double cosT = std::sqrt(1 - s2);
  double rp = (eta * cosI - cosT) / (eta * cosI + cosT), rs = (cosI - eta * cosT) / (cosI + eta * cosT);
  return 0.5 * (rp * rp + rs * rs);
}

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++fails; std::printf("FAIL: "); } else std::printf("ok:   "); std::printf(__VA_ARGS__); std::printf("\n"); } while (0)

int main() {
  CHECK(std::fabs(fresnel(1.0, 1.5) - 0.04) < 1e-9, "normal-incidence Fresnel = %.5f (expect 0.04)", fresnel(1.0, 1.5));
  CHECK(fresnel(0.5, 1.0 / 1.5) == 1.0, "total internal reflection beyond critical angle");
  {   // F(cosI, n) == F(cosT, 1/n)
    double n = 1.5, cI = 0.8, sT = std::sqrt(1 - cI * cI) / n, cT = std::sqrt(1 - sT * sT);
    CHECK(std::fabs(fresnel(cI, n) - fresnel(cT, 1 / n)) < 1e-12, "Fresnel reciprocity");
  }

  // Furnace: glass sphere radius 1 at origin; parallel camera rays along -z.
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> U(0, 1);
  double yInt = 0; for (double l = 360; l < 830; l += 0.01) yInt += ybar(l) * 0.01;

  const int N = 1500000, MAXD = 32; const double EPS = 1e-3;
  double sumY = 0, sumLhero = 0; long nHero = 0;
  for (int k = 0; k < N; ++k) {
    double u = U(rng), lam[4], pd[4];
    for (int i = 0; i < 4; ++i) { lam[i] = sampleVis(std::fmod(u + i / 4.0, 1.0)); pd[i] = pdfVis(lam[i]); }
    double r = 1.2 * std::sqrt(U(rng)), phi = 6.283185307 * U(rng);
    V o{r * std::cos(phi), r * std::sin(phi), 5}, d{0, 0, -1};
    double thr[4] = {1, 1, 1, 1}, L[4] = {0, 0, 0, 0}; bool hero = false;
    for (int depth = 0; depth < MAXD; ++depth) {
      double b = dot(o, d), c = dot(o, o) - 1, disc = b * b - c;
      double t = -1;
      if (disc > 0) { double sq = std::sqrt(disc); t = (-b - sq > 1e-6) ? -b - sq : ((-b + sq > 1e-6) ? -b + sq : -1); }
      if (t < 0) { for (int i = 0; i < 4; ++i) L[i] += thr[i]; break; }   // env = 1
      V p = o + d * t, n = p;                                            // outward normal
      bool front = dot(d, n) < 0; V nf = front ? n : n * -1.0;
      hero = true;
      double nG = bk7(lam[0]), eta = front ? nG : 1.0 / nG, cosI = dot(d * -1.0, nf);
      double F = fresnel(cosI, eta);
      if (U(rng) < F) { d = reflectV(d, nf); o = p + nf * EPS; }
      else { V rd; refractV(d, nf, 1.0 / eta, rd); d = norm(rd); for (int i = 0; i < 4; ++i) thr[i] /= eta * eta; o = p - nf * EPS; }
      if (depth >= 3) { double q = std::fmin(std::fmax(thr[0], 0.05), 0.95); if (U(rng) >= q) break; for (int i = 0; i < 4; ++i) thr[i] /= q; }
    }
    double Y;
    if (hero) { Y = L[0] / pd[0] * ybar(lam[0]); sumLhero += L[0]; ++nHero; }
    else { Y = 0; for (int i = 0; i < 4; ++i) Y += 0.25 * L[i] / pd[i] * ybar(lam[i]); }
    sumY += Y;
  }
  double meanY = sumY / N / yInt;
  CHECK(std::fabs(meanY - 1.0) < 0.01, "glass-sphere furnace: mean Y = %.4f (expect 1.0)", meanY);
  CHECK(std::fabs(sumLhero / nHero - 1.0) < 0.01, "mean hero radiance through glass = %.4f (expect 1.0)", sumLhero / nHero);

  std::printf(fails ? "\n%d test(s) failed\n" : "\nall tests passed\n", fails);
  return fails ? 1 : 0;
}
