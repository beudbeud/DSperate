// SPDX-License-Identifier: GPL-3.0-or-later
// DSperate - Nintendo DS emulator. Copyright (C) 2026 DSperate contributors.
//
// xBRZ on the 2D (frontend/video/smooth2d.h): a diagonal edge is smoothed,
// flat areas are kept, and of the internal-resolution frame the 3D samples
// are kept while the 2D (repeated) blocks are smoothed as without it.
#include "frontend/video/smooth2d.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ds;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
  constexpr u32 W = 256, H = 192, Bk = 0xFF000000, Wh = 0xFFFFFFFF, R = 0xFFFF0000, M = 0x00FFFFFF;
  // White screen with a black staircase (a 1:1 diagonal edge) in the top-left 40x40.
  std::vector<u32> fb(W * H, Wh);
  for (u32 y = 0; y < 40; ++y) for (u32 x = 0; x < 40; ++x) if (x < y) fb[y * W + x] = Bk;
  for (u32 n : {2u, 3u, 4u}) {
    const size_t p = W * n;
    std::vector<u32> out(p * H * n, 0);
    frontend::smooth2d_screen(fb.data(), nullptr, nullptr, n, out.data(), p);
    // Smoothed: somewhere along the edge a sample differs from nearest-neighbour.
    u32 changed = 0;
    for (u32 y = 0; y < 40 * n; ++y) for (u32 x = 0; x < 40 * n; ++x) changed += ((out[y * p + x] ^ fb[(y / n) * W + x / n]) & M) != 0;
    CHECK(changed > 0);
    CHECK((out[(100 * n) * p + 100 * n] & M) == (Wh & M));   // flat area kept
    // From the internal-resolution frame: one 3D sample kept, the rest as without it.
    std::vector<u32> hfb(p * H * n), out2(p * H * n, 0);
    for (u32 y = 0; y < H * n; ++y) for (u32 x = 0; x < p; ++x) hfb[y * p + x] = fb[(y / n) * W + x / n];
    hfb[(100 * n) * p + 100 * n] = R;
    bool rows[H]; for (bool& r : rows) r = true;
    frontend::smooth2d_screen(fb.data(), hfb.data(), rows, n, out2.data(), p);
    CHECK(out2[(100 * n) * p + 100 * n] == R);
    CHECK(out2[(100 * n) * p + 100 * n + 1] == Wh);   // the rest of that 3D block, as drawn
    u32 diff = 0;
    for (u32 y = 0; y < H * n; ++y) for (u32 x = 0; x < p; ++x) if (y / n != 100 || x / n != 100) diff += ((out[y * p + x] ^ out2[y * p + x]) & M) != 0;
    CHECK(diff == 0);
    // At 50% a smoothed sample lies halfway between the pixel's colour and xBRZ's.
    std::vector<u32> half(p * H * n, 0);
    frontend::smooth2d_screen(fb.data(), nullptr, nullptr, n, half.data(), p, 50);
    u32 checked = 0;
    for (u32 y = 0; y < 40 * n; ++y) for (u32 x = 0; x < 40 * n; ++x) {
      const u32 e = fb[(y / n) * W + x / n], f = out[y * p + x], h = half[y * p + x];
      if (((e ^ f) & M) == 0) { CHECK(h == f); continue; }
      for (u32 sh : {0u, 8u, 16u}) { const int ce = (e >> sh) & 255, cf = (f >> sh) & 255, ch = (h >> sh) & 255; CHECK(std::abs(2 * ch - ce - cf) <= 2); }
      ++checked;
    }
    CHECK(checked > 0);
  }
  if (fails) return 1;
  std::printf("smooth2d: ok\n");
  return 0;
}
