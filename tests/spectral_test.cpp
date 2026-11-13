// CPU reference for shaders/spectral.glsl. Validates the math before any GPU work.
#include <cmath>
#include <cstdio>
#include <random>

static double lobe(double l, double mu, double s1, double s2) {
  double t = (l - mu) / (l < mu ? s1 : s2);
  return std::exp(-0.5 * t * t);
}
static double ybar(double l) {
  return 0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1);
}
static double pdfVis(double l) {
  if (l < 360 || l > 830) return 0;
  double c = std::cosh(0.0072 * (l - 538.0));
  return 0.0039398042 / (c * c);
}
static double sampleVis(double u) {
  return 538.0 - 138.888889 * std::atanh(0.85691062 - 1.82750197 * u);
}
static double sellmeier(const double B[3], const double C[3], double nm) {
  double l2 = nm * 1e-3; l2 *= l2;
  double s = 1;
  for (int i = 0; i < 3; ++i) s += B[i] * l2 / (l2 - C[i]);
  return std::sqrt(s);
}

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++fails; std::printf("FAIL: "); } else { std::printf("ok:   "); } std::printf(__VA_ARGS__); std::printf("\n"); } while (0)

int main() {
  // 1. Sampling pdf integrates to 1 over the visible range.
  double integ = 0;
  for (double l = 360; l < 830; l += 0.01) integ += pdfVis(l) * 0.01;
  CHECK(std::fabs(integ - 1.0) < 0.01, "pdf integral = %.5f", integ);

  // 2. Inverse-CDF endpoints hit the range limits.
  CHECK(std::fabs(sampleVis(0.0) - 360) < 1.0 && std::fabs(sampleVis(1.0) - 830) < 1.5,
        "sample(0)=%.2f sample(1)=%.2f", sampleVis(0.0), sampleVis(1.0));

  // 3. Single-wavelength and hero-wavelength estimators of int ybar are unbiased.
  double ref = 0;
  for (double l = 360; l < 830; l += 0.01) ref += ybar(l) * 0.01;
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> U(0, 1);
  const int N = 400000;
  double s1 = 0, s1sq = 0, s4 = 0, s4sq = 0;
  for (int k = 0; k < N; ++k) {
    double u = U(rng);
    double l = sampleVis(u);
    double e1 = ybar(l) / pdfVis(l);
    s1 += e1; s1sq += e1 * e1;
    double e4 = 0;
    for (int i = 0; i < 4; ++i) {
      double li = sampleVis(std::fmod(u + i / 4.0, 1.0));
      e4 += 0.25 * ybar(li) / pdfVis(li);   // stratifiedWeights
    }
    s4 += e4; s4sq += e4 * e4;
  }
  double m1 = s1 / N, m4 = s4 / N;
  double v1 = s1sq / N - m1 * m1, v4 = s4sq / N - m4 * m4;
  CHECK(std::fabs(m1 - ref) / ref < 0.01, "single-lambda mean %.3f vs ref %.3f", m1, ref);
  CHECK(std::fabs(m4 - ref) / ref < 0.01, "HWSS mean %.3f vs ref %.3f", m4, ref);
  std::printf("info: variance single=%.3f HWSS=%.3f\n", v1, v4);

  // 4. Sellmeier BK7 at the sodium d-line (587.56 nm) is ~1.5168.
  const double B[3] = {1.03961212, 0.231792344, 1.01046945};
  const double C[3] = {0.00600069867, 0.0200179144, 103.560653};
  double n = sellmeier(B, C, 587.56);
  CHECK(std::fabs(n - 1.5168) < 1e-3, "BK7 n(587.56nm) = %.5f", n);
  CHECK(sellmeier(B, C, 450) > sellmeier(B, C, 650), "dispersion: n(450) > n(650)");

  std::printf(fails ? "\n%d test(s) failed\n" : "\nall tests passed\n", fails);
  return fails ? 1 : 0;
}
