// reservoir.glsl: spectral path reservoir for ReSTIR PT.
// 80 bytes, std430-friendly. See docs/DESIGN.md for the shift-mapping plan.
#ifndef RESERVOIR_GLSL
#define RESERVOIR_GLSL
#include "spectral.glsl"

struct SpectralReservoir {
  vec4  lambda;            // hero + companions (nm)
  vec4  radiance;          // unweighted path contribution f(l_i)
  vec3  reconnectPos;  float W;   // reconnection vertex, unbiased contribution weight
  vec3  reconnectNormal; float M; // confidence / sample count
  uint  rngSeed;           // for random-replay shift (wavelength-agnostic)
  uint  lastDispersiveVertex; // reconnection allowed only AFTER this vertex
  uint  pathFlags;
  uint  _pad;
};

// Scalar target function p_hat. Starting choice: luminance of the unweighted
// spectral contribution. Expected to need tuning (see DESIGN.md, open questions).
float targetPdf(vec4 radiance, vec4 lambda) {
  return radiance.x * xyzBar(lambda.x).y + radiance.y * xyzBar(lambda.y).y
       + radiance.z * xyzBar(lambda.z).y + radiance.w * xyzBar(lambda.w).y;
}

SpectralReservoir reservoirEmpty() {
  SpectralReservoir r;
  r.lambda = vec4(0); r.radiance = vec4(0);
  r.reconnectPos = vec3(0); r.W = 0.0;
  r.reconnectNormal = vec3(0); r.M = 0.0;
  r.rngSeed = 0u; r.lastDispersiveVertex = 0u; r.pathFlags = 0u; r._pad = 0u;
  return r;
}

// Streaming weighted reservoir sampling. w = p_hat * W_cand * M_cand (resampling weight).
bool reservoirUpdate(inout SpectralReservoir r, inout float wSum,
                     SpectralReservoir cand, float w, float u) {
  float Mtot = r.M + cand.M;
  wSum += w;
  bool take = (u * wSum < w);
  if (take) r = cand;
  r.M = Mtot;
  return take;
}

void reservoirFinalize(inout SpectralReservoir r, float wSum) {
  float ph = targetPdf(r.radiance, r.lambda);
  r.W = (ph > 0.0 && r.M > 0.0) ? wSum / (r.M * ph) : 0.0;
}
#endif
