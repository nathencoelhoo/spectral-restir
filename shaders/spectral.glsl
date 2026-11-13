// spectral.glsl: wavelength sampling, color matching, dispersion.
// CPU mirror lives in tests/spectral_test.cpp. Keep the two in sync.
#ifndef SPECTRAL_GLSL
#define SPECTRAL_GLSL

const float LAMBDA_MIN = 360.0;   // nm
const float LAMBDA_MAX = 830.0;   // nm
const int   NUM_LAMBDA = 4;       // hero + 3 rotated companions

// Visible-range importance sampling (same fit pbrt-v4 uses).
float visiblePdf(float l) {
  if (l < LAMBDA_MIN || l > LAMBDA_MAX) return 0.0;
  float c = cosh(0.0072 * (l - 538.0));
  return 0.0039398042 / (c * c);
}
float sampleVisible(float u) {
  return 538.0 - 138.888889 * atanh(0.85691062 - 1.82750197 * u);
}

// Hero wavelength spectral sampling (Wilkie et al. 2014):
// one random u, four stratified wavelengths.
vec4 sampleHeroWavelengths(float u, out vec4 pdf) {
  vec4 l;
  for (int i = 0; i < NUM_LAMBDA; ++i) {
    l[i]   = sampleVisible(fract(u + float(i) / float(NUM_LAMBDA)));
    pdf[i] = visiblePdf(l[i]);
  }
  return l;
}
// Companions are stratified in u-space, so every l_i has marginal pdf p(l_i)
// and the unbiased estimate is the mean of f(l_i)/p(l_i):
//   pixel contribution = dot(f, stratifiedWeights(pdf))
// (Verified in tests/spectral_test.cpp. A 1/sum(p) weight is only valid when
// companions are spaced in wavelength space, as in the original paper.)
vec4 stratifiedWeights(vec4 pdf) { return (1.0 / float(NUM_LAMBDA)) / pdf; }
// If dispersion terminates the companions, only the hero survives:
float heroOnlyWeight(vec4 pdf) { return 1.0 / pdf.x; }

// CIE 1931 analytic multi-lobe fit (Wyman, Sloan, Shirley 2013).
float cieLobe(float l, float mu, float s1, float s2) {
  float t = (l - mu) / (l < mu ? s1 : s2);
  return exp(-0.5 * t * t);
}
vec3 xyzBar(float l) {
  float x = 1.056 * cieLobe(l, 599.8, 37.9, 31.0)
          + 0.362 * cieLobe(l, 442.0, 16.0, 26.7)
          - 0.065 * cieLobe(l, 501.1, 20.4, 26.2);
  float y = 0.821 * cieLobe(l, 568.8, 46.9, 40.5)
          + 0.286 * cieLobe(l, 530.9, 16.3, 31.1);
  float z = 1.217 * cieLobe(l, 437.0, 11.8, 36.0)
          + 0.681 * cieLobe(l, 459.0, 26.0, 13.8);
  return vec3(x, y, z);
}
vec3 xyzToLinearSRGB(vec3 c) {
  return vec3( 3.2404542 * c.x - 1.5371385 * c.y - 0.4985314 * c.z,
              -0.9692660 * c.x + 1.8760108 * c.y + 0.0415560 * c.z,
               0.0556434 * c.x - 0.2040259 * c.y + 1.0572252 * c.z);
}

// Dispersion: Sellmeier n^2 = 1 + sum B_i l^2 / (l^2 - C_i), l in micrometres.
struct Sellmeier { vec3 B; vec3 C; };
float sellmeierIor(Sellmeier s, float lambdaNm) {
  float l2 = lambdaNm * 1e-3; l2 *= l2;
  return sqrt(1.0 + dot(s.B, vec3(l2) / (vec3(l2) - s.C)));
}
const Sellmeier GLASS_BK7 = Sellmeier(vec3(1.03961212, 0.231792344, 1.01046945),
                                      vec3(0.00600069867, 0.0200179144, 103.560653));
const Sellmeier GLASS_FUSED_SILICA = Sellmeier(vec3(0.6961663, 0.4079426, 0.8974794),
                                               vec3(0.00467914826, 0.0135120631, 97.9340025));
#endif
