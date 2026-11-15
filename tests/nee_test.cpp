// CPU check of the NEE + MIS math used in shaders/pathtrace.comp.
// Scene: diffuse floor point at the origin (normal +y), area light 3x3 at y=4 facing down.
// Reference radiance comes from brute-force numerical integration over the light.
// NEE-only, BSDF-only, and MIS-combined estimators must all agree with it.
#include <cmath>
#include <cstdio>
#include <random>

static const double PI = 3.14159265358979323846;
static const double ALBEDO = 0.6, AREA = 9.0, LY = 4.0, HALF = 1.5;

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++fails; std::printf("FAIL: "); } else std::printf("ok:   "); std::printf(__VA_ARGS__); std::printf("\n"); } while (0)

int main() {
  // Reference: Lo = albedo/pi * integral of cosS * cosL / dist^2 dA  (Le = 1)
  double ref = 0; const int G = 3000; const double dA = (2 * HALF / G) * (2 * HALF / G);
  for (int i = 0; i < G; ++i) for (int j = 0; j < G; ++j) {
    double x = -HALF + (i + 0.5) * (2 * HALF / G), z = -HALF + (j + 0.5) * (2 * HALF / G);
    double d2 = x * x + LY * LY + z * z, d = std::sqrt(d2), cosS = LY / d, cosL = LY / d;
    ref += cosS * cosL / d2 * dA;
  }
  ref *= ALBEDO / PI;

  std::mt19937_64 rng(11); std::uniform_real_distribution<double> U(0, 1);
  const long N = 4000000;
  double nee1 = 0, bsdf1 = 0, neeM = 0, bsdfM = 0, v1 = 0, vM = 0;
  for (long k = 0; k < N; ++k) {
    // --- NEE sample ---
    double x = -HALF + 2 * HALF * U(rng), z = -HALF + 2 * HALF * U(rng);
    double d2 = x * x + LY * LY + z * z, d = std::sqrt(d2), cosS = LY / d, cosL = LY / d;
    double pl = d2 / (cosL * AREA), pb = cosS / PI;
    double f = ALBEDO / PI;
    double c = f * cosS / pl;                                // unweighted NEE contribution
    nee1 += c;
    double wN = pl * pl / (pl * pl + pb * pb);
    neeM += c * wN;
    // --- BSDF sample (cosine-weighted) ---
    double u1 = U(rng), u2 = U(rng), r = std::sqrt(u1), phi = 2 * PI * u2;
    double dx = r * std::cos(phi), dz = r * std::sin(phi), dy = std::sqrt(std::fmax(0.0, 1 - u1));
    double b = 0, bm = 0;
    if (dy > 1e-9) {
      double t = LY / dy, hx = dx * t, hz = dz * t;
      if (std::fabs(hx) <= HALF && std::fabs(hz) <= HALF) {
        double pbs = dy / PI, cl = dy, pls = t * t / (cl * AREA);   // cosL at light = dy
        b = ALBEDO;                                                 // f*cos/pdf = albedo
        bm = b * (pbs * pbs) / (pbs * pbs + pls * pls);
      }
    }
    bsdf1 += b; bsdfM += bm;
    double comb = c * wN + bm; vM += comb * comb; v1 += c * c;
  }
  nee1 /= N; bsdf1 /= N; neeM /= N; bsdfM /= N; vM = vM / N - std::pow(neeM + bsdfM, 2);
  v1 = v1 / N - nee1 * nee1;
  double mis = neeM + bsdfM;
  std::printf("reference Lo = %.6f\n", ref);
  CHECK(std::fabs(nee1 - ref) / ref < 0.005, "NEE only      = %.6f", nee1);
  CHECK(std::fabs(bsdf1 - ref) / ref < 0.005, "BSDF only     = %.6f", bsdf1);
  CHECK(std::fabs(mis - ref) / ref < 0.005, "MIS combined  = %.6f", mis);
  std::printf("info: per-sample variance NEE-only=%.3e, MIS=%.3e\n", v1, vM);
  std::printf(fails ? "\n%d test(s) failed\n" : "\nall tests passed\n", fails);
  return fails ? 1 : 0;
}
