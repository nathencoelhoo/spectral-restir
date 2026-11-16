// sr_compare: error metrics between two linear-RGB PFM images (little-endian, as written by sr_render).
// Usage: sr_compare ref.pfm test.pfm [max_abs_bias_fraction]
// Prints RMSE, relMSE and the relative difference of mean luminance ("bias").
// With a tolerance, exits 3 if |bias| exceeds it. Bias is the signal that two supposedly
// unbiased estimators disagree; RMSE alone cannot tell noise from bias.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

struct Img { int w = 0, h = 0; std::vector<float> d; };

static bool readPFM(const char* path, Img& im) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::string magic; double scale = 0;
  f >> magic >> im.w >> im.h >> scale;
  f.get();                                    // single whitespace after the header
  if (magic != "PF" || scale >= 0 || im.w <= 0 || im.h <= 0) return false;
  im.d.resize(size_t(im.w) * im.h * 3);
  f.read(reinterpret_cast<char*>(im.d.data()), std::streamsize(im.d.size() * sizeof(float)));
  return bool(f);
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: sr_compare ref.pfm test.pfm [max_abs_bias_fraction]\n"); return 1; }
  Img a, b;
  if (!readPFM(argv[1], a) || !readPFM(argv[2], b)) { std::fprintf(stderr, "cannot read PFM input\n"); return 1; }
  if (a.w != b.w || a.h != b.h) { std::fprintf(stderr, "size mismatch\n"); return 1; }

  const size_t n = size_t(a.w) * a.h;
  double se = 0, rel = 0, ma = 0, mb = 0;
  for (size_t i = 0; i < n; ++i) {
    for (int c = 0; c < 3; ++c) {
      double r = a.d[3 * i + c], t = b.d[3 * i + c], e = t - r;
      se += e * e; rel += e * e / (r * r + 0.01);
    }
    ma += 0.2126 * a.d[3 * i] + 0.7152 * a.d[3 * i + 1] + 0.0722 * a.d[3 * i + 2];
    mb += 0.2126 * b.d[3 * i] + 0.7152 * b.d[3 * i + 1] + 0.0722 * b.d[3 * i + 2];
  }
  ma /= n; mb /= n;
  double rmse = std::sqrt(se / (3.0 * n)), relmse = rel / (3.0 * n), bias = ma > 0 ? (mb - ma) / ma : 0;
  std::printf("RMSE=%.6f relMSE=%.6f meanLumRef=%.6f meanLumTest=%.6f bias=%+.4f%%\n",
              rmse, relmse, ma, mb, bias * 100.0);
  if (argc > 3 && std::fabs(bias) > std::atof(argv[3])) {
    std::printf("BIAS CHECK FAIL: |bias| exceeds %.4f%%\n", std::atof(argv[3]) * 100.0);
    return 3;
  }
  if (argc > 3) std::printf("bias check: PASS\n");
  return 0;
}
