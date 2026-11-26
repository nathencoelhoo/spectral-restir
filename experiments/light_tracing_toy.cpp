// Experiment (not in CI): CPU toy comparing three estimators of the caustic channel
// (floor seen directly; light -> 1..11 glass vertices -> floor -> eye), written as linear PFM images
// (Y in all channels) so scripts/region_bias.py and tools/compare.cpp can be reused.
//   pt  : camera-side path tracing, cosine-sampled directions (the renderer's current candidates)
//   ptg : camera-side path tracing with directions aimed at the prism's bounding sphere
//   lt  : light tracing: photons from the light aimed at the prism, splatted to the camera
// Build: g++ -std=c++20 -O2 experiments/light_tracing_toy.cpp -o light_tracing_toy
// Run:   ./light_tracing_toy pt|ptg|lt W H frames perPixel seed out.pfm
//        perPixel = candidates (pt, ptg) or photons (lt) per pixel per frame.
// Light-tracing splat weight: a photon landing at floor point x contributes
//   (albedo/pi) * flux * h / (z^3 * A_pix) to its pixel, h = eye height above x, z = depth along the optical axis,
//   A_pix = pixel area on the unit-distance image plane. (Derived from dA_img = dA_x cos(theta_x) / (r^2 cos^3(theta_off)).)
#define main toy_main
#include "../tests/restir_toy_test.cpp"
#undef main
#include <cstdlib>
#include <fstream>

static const V3 EYE{0, 2.2, 7.5}, TARGET{0, 0.9, 0};
static V3 nrm(V3 a) { return a * (1 / std::sqrt(dot(a, a))); }
struct Cam { V3 fwd, right, up; double th, asp; int W, H; };
static Cam makeCam(int W, int H) { Cam c; c.fwd = nrm(TARGET - EYE); c.right = nrm(cross(c.fwd, V3{0, 1, 0})); c.up = cross(c.right, c.fwd);
  c.th = std::tan(20.0 * M_PI / 180); c.asp = double(W) / H; c.W = W; c.H = H; return c; }
static V3 camDir(const Cam& c, double px, double py) {
  double u = px / c.W * 2 - 1, v = py / c.H * 2 - 1; return nrm(c.fwd + c.right * (u * c.th * c.asp) - c.up * (v * c.th)); }
static bool project(const Cam& c, V3 P, double& px, double& py, double& z) {
  V3 v = P - EYE; z = dot(v, c.fwd); if (z <= 1e-9) return false;
  px = (dot(v, c.right) / z / (c.th * c.asp) + 1) / 2 * c.W; py = (-dot(v, c.up) / z / c.th + 1) / 2 * c.H; return true; }

// Caustic-channel contribution when the first direction is given (guided sampling): u[2] wavelength, u[3..] Fresnel.
static V3 evalPathDir(double px, double pz, V3 d, double thr0, const double* u) {
  double lam = sampleVis(u[2]), pdf = pdfVis(lam), thr = thr0; V3 o{px, EPS, pz}; int glass = 0;
  for (int depth = 0; depth < 12; ++depth) {
    double t = -1; V3 n{0, 0, 0};
    if (!intersectGlass(o, d, t, n)) {
      if (glass > 0 && d.y > 1e-9) { double tl = (4.0 - o.y) / d.y, hx = o.x + tl * d.x, hz = o.z + tl * d.z;
        if (std::fabs(hx) <= LH && std::fabs(hz) <= LH) { double s = thr * EMISSION * lightSpectrum(lam) / pdf / yInt; V3 xb = xyzBar(lam); return {xb.x * s, xb.y * s, xb.z * s}; } }
      return {0, 0, 0}; }
    V3 p = o + d * t; bool front = dot(d, n) < 0; V3 nf = front ? n : n * -1.0; ++glass;
    double nG = bk7(lam), eta = front ? nG : 1.0 / nG, F = fresnel(-dot(d, nf), eta);
    if (u[3 + depth] < F) { d = d - nf * (2 * dot(nf, d)); o = p + nf * EPS; }
    else { double k = 1 - (1 / eta) * (1 / eta) * (1 - dot(nf, d) * dot(nf, d)), e = 1 / eta;
           V3 rd = d * e - nf * (e * dot(nf, d) + std::sqrt(std::fmax(0.0, k))); d = nrm(rd); thr /= eta * eta; o = p - nf * EPS; }
  }
  return {0, 0, 0};
}

int main(int argc, char** argv) {
  std::string mode = argv[1]; int W = atoi(argv[2]), H = atoi(argv[3]), F = atoi(argv[4]), per = atoi(argv[5]); unsigned seed = atoi(argv[6]);
  for (double l = 360; l < 830; l += 0.01) yInt += xyzBar(l).y * 0.01;
  GEOM = 1; buildPrism(); LH = 1.5;
  Cam cam = makeCam(W, H); std::mt19937_64 rng(5000 + seed); std::uniform_real_distribution<double> U(0, 1);
  const double Apix = (2 * cam.th * cam.asp / W) * (2 * cam.th / H), A_LIGHT = 9.0, MAXG = 11;
  const V3 cen{0, std::sqrt(3.0) / 3, 0}; const double RB = 1.5275;      // bounding sphere of the prism
  std::vector<double> img(size_t(W) * H, 0.0); std::vector<double> u(NU);
  for (int f = 0; f < F; ++f) {
    if (mode == "pt" || mode == "ptg") {
      for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        V3 d = camDir(cam, x + U(rng), y + U(rng)); if (d.y >= 0) continue;
        double tf = -EYE.y / d.y; V3 h = EYE + d * tf; if (std::fabs(h.x) > 6 || std::fabs(h.z) > 6) continue;
        double tp; V3 n; if (intersectGlass(EYE, d, tp, n) && tp < tf) continue;     // primary hit must be the floor
        double acc = 0;
        V3 axis = cen - V3{h.x, 0, h.z}; double Dg = std::sqrt(dot(axis, axis));
        bool guided = (mode == "ptg") && Dg > RB * 1.001;                     // inside the bounding sphere: cosine fallback
        for (int k = 0; k < per; ++k) {
          for (auto& a : u) a = U(rng);
          if (!guided) { acc += evalPath(h.x, h.z, u.data()).y; continue; }
          V3 ax = axis * (1 / Dg); double ca = std::sqrt(1 - (RB / Dg) * (RB / Dg)), cosT = 1 - U(rng) * (1 - ca), sinT = std::sqrt(1 - cosT * cosT), phi = 2 * M_PI * U(rng);
          V3 t1 = nrm(std::fabs(ax.x) > 0.9 ? cross(ax, V3{0, 1, 0}) : cross(ax, V3{1, 0, 0})), t2 = cross(ax, t1);
          V3 dd = ax * cosT + t1 * (sinT * std::cos(phi)) + t2 * (sinT * std::sin(phi));
          if (dd.y <= 0) continue;                                            // below the horizon: zero contribution, pdf unchanged
          double pdfOm = 1 / (2 * M_PI * (1 - ca));
          acc += evalPathDir(h.x, h.z, dd, (ALBEDO / M_PI) * dd.y / pdfOm, u.data()).y;
        }
        img[size_t(y) * W + x] += acc / per;
      }
    } else {                                                              // light side: per*W*H photons
      long N = long(per) * W * H;
      for (long i = 0; i < N; ++i) {
        V3 l{-1.5 + 3 * U(rng), 4.0, -1.5 + 3 * U(rng)}; V3 axis = cen - l; double D = std::sqrt(dot(axis, axis)); axis = axis * (1 / D);
        double ca = std::sqrt(1 - (RB / D) * (RB / D)), cosT = 1 - U(rng) * (1 - ca), sinT = std::sqrt(1 - cosT * cosT), phi = 2 * M_PI * U(rng);
        V3 t1 = nrm(std::fabs(axis.x) > 0.9 ? cross(axis, V3{0, 1, 0}) : cross(axis, V3{1, 0, 0})), t2 = cross(axis, t1);
        V3 d = axis * cosT + t1 * (sinT * std::cos(phi)) + t2 * (sinT * std::sin(phi));
        double cosL = -d.y; if (cosL <= 0) continue; double pdfOm = 1 / (2 * M_PI * (1 - ca));
        double lam = sampleVis(U(rng)), pl = pdfVis(lam);
        double wph = EMISSION * lightSpectrum(lam) * cosL * A_LIGHT / pdfOm / pl;      // flux carried by the photon
        V3 o = l; int glass = 0; bool ok = false; V3 land{0, 0, 0};
        for (int depth = 0; depth < 14; ++depth) {
          double t; V3 n; if (!intersectGlass(o, d, t, n)) { if (glass >= 1 && d.y < -1e-9) { land = o + d * (-o.y / d.y); ok = true; } break; }
          if (glass >= MAXG) break;
          V3 p = o + d * t; bool front = dot(d, n) < 0; V3 nf = front ? n : n * -1.0; ++glass;
          double nG = bk7(lam), eta = front ? nG : 1.0 / nG, F_ = fresnel(-dot(d, nf), eta);
          if (U(rng) < F_) { d = d - nf * (2 * dot(nf, d)); o = p + nf * EPS; }
          else { double e = 1 / eta, k = 1 - e * e * (1 - dot(nf, d) * dot(nf, d)); V3 rd = d * e - nf * (e * dot(nf, d) + std::sqrt(std::fmax(0.0, k)));
                 d = nrm(rd); o = p - nf * EPS; }                                      // radiance eta^2 factors cancel over a closed chain
        }
        if (!ok || std::fabs(land.x) > 6 || std::fabs(land.z) > 6) continue;
        double px, py, z; if (!project(cam, land, px, py, z)) continue;
        int ix = int(std::floor(px)), iy = int(std::floor(py)); if (ix < 0 || iy < 0 || ix >= W || iy >= H) continue;
        V3 dir = nrm(land - EYE); double tp, dist = std::sqrt(dot(land - EYE, land - EYE)); V3 n;
        if (intersectGlass(EYE, dir, tp, n) && tp < dist - 1e-4) continue;             // camera must see the landing point
        double hgt = EYE.y - land.y, wgeom = hgt / (z * z * z * Apix);                  // image-plane measure Jacobian
        img[size_t(iy) * W + ix] += (ALBEDO / M_PI) * wph * wgeom * xyzBar(lam).y / yInt / double(N);
      }
    }
  }
  std::ofstream out(argv[7], std::ios::binary); out << "PF\n" << W << " " << H << "\n-1.0\n";
  for (int y = H - 1; y >= 0; --y) for (int x = 0; x < W; ++x) { float v = float(img[size_t(y) * W + x] / F); for (int c = 0; c < 3; ++c) out.write((char*)&v, 4); }
  double tot = 0; for (double v : img) tot += v; std::printf("%s: mean pixel Y = %.6f\n", mode.c_str(), tot / F / (double(W) * H));
}
