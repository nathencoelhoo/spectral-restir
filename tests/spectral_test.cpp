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


  // 5. White light spectrum: linear sRGB of the light must be (1,1,1) with Y = 1.
  {
    auto planckLight = [](double l) {
      const double C2 = 1.4387769e7, T = 6540.0;
      double p = std::pow(l / 560.0, -5.0) * (std::exp(C2 / (560.0 * T)) - 1.0) / (std::exp(C2 / (l * T)) - 1.0);
      double u = (l - 540.0) / 45.0;
      return 0.958039 * p * (1.0 + 0.07 * std::exp(-0.5 * u * u));
    };
    auto xbar = [](double l) { return 1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7) - 0.065 * lobe(l, 501.1, 20.4, 26.2); };
    auto zbar = [](double l) { return 1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8); };
    double X = 0, Y = 0, Z = 0;
    for (double l = 360; l < 830; l += 0.01) { double s = planckLight(l); X += xbar(l) * s * 0.01; Y += ybar(l) * s * 0.01; Z += zbar(l) * s * 0.01; }
    X /= ref; Y /= ref; Z /= ref;   // ref = integral of ybar
    double R = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
    double Gc = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
    double Bc = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
    CHECK(std::fabs(Y - 1) < 0.005, "light luminance Y = %.4f", Y);
    CHECK(std::fabs(R - 1) < 0.01 && std::fabs(Gc - 1) < 0.01 && std::fabs(Bc - 1) < 0.01,
          "light linear sRGB = (%.4f, %.4f, %.4f)", R, Gc, Bc);
  }

  std::printf(fails ? "\n%d test(s) failed\n" : "\nall tests passed\n", fails);
  return fails ? 1 : 0;
}
