// SPDX-License-Identifier: GPL-3.0-or-later
// DSperate - Nintendo DS emulator. Copyright (C) 2026 DSperate contributors.
#include "frontend/video/smooth2d.h"

#include "core/gpu/line_worker.h"
#include "frontend/video/xbrz/xbrz.h"

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

namespace ds::frontend {

namespace {
constexpr u32 W = 256, H = 192;

struct Job {
  const u32* fb; const u32* hfb; const bool* hrow; u32 n; u32* dst; size_t pitch; u32* big;
  u32 y0, y1;   // the slice of DS lines
  u32 mix;      // xBRZ's share of a sample, of 256
};
// a * (256 - m) + b * m, per channel, over 256.
u32 lerp(u32 a, u32 b, u32 m) {
  const u32 rb = (((a & 0xFF00FF) * (256 - m) + (b & 0xFF00FF) * m) >> 8) & 0xFF00FF;
  const u32 g = (((a & 0x00FF00) * (256 - m) + (b & 0x00FF00) * m) >> 8) & 0x00FF00;
  return 0xFF000000 | rb | g;
}

void run(const Job& j) {
  const u32 n = j.n;
  const size_t hw = size_t(W) * n;
  xbrz::scale(n, j.fb, j.big, W, H, xbrz::ColorFormat::RGB, xbrz::ScalerCfg(), static_cast<int>(j.y0), static_cast<int>(j.y1));
  for (u32 y = j.y0; y < j.y1; ++y) {
    for (u32 sy = 0; sy < n; ++sy) {
      u32* o = j.dst + (size_t(y) * n + sy) * j.pitch;
      const u32* s = j.big + (size_t(y) * n + sy) * hw;
      if (j.mix >= 256) { std::memcpy(o, s, hw * 4); continue; }
      const u32* e = j.fb + y * W;
      for (u32 x = 0; x < hw; ++x) o[x] = lerp(e[x / n], s[x], j.mix);
    }
    if (!j.hfb || !j.hrow || !j.hrow[y]) continue;
    const u32* hr = j.hfb + size_t(y) * n * hw;
    u32* o = j.dst + size_t(y) * n * j.pitch;
    for (u32 x = 0; x < W; ++x) {
      const u32 E = j.fb[y * W + x];
      bool repeat = true;
      for (u32 sy = 0; sy < n && repeat; ++sy) for (u32 sx = 0; sx < n; ++sx) if ((hr[sy * hw + x * n + sx] ^ E) & 0x00FFFFFF) { repeat = false; break; }
      if (!repeat) for (u32 sy = 0; sy < n; ++sy) std::memcpy(o + sy * j.pitch + x * n, hr + sy * hw + x * n, n * 4);
    }
  }
}

// xBRZ is ~5 ms a frame for both screens at 2x on one Pi 5 core: the screen
// goes in slices, one on the caller and one on each worker (up to three, one
// core left to the rest). xBRZ's slices are exact: each reads the source rows around it.
struct Pool {
  static constexpr u32 kMax = 3;
  gpu::LineWorker w[kMax];
  Job job[kMax];
  u32 workers = 0;
  Pool() {
    const u32 hc = std::thread::hardware_concurrency();
    workers = std::min<u32>(kMax, hc > 1 ? hc - 1 : 0);
    for (u32 i = 0; i < workers; ++i) w[i].start([](void* p) { run(*static_cast<Job*>(p)); }, &job[i], ThreadRole::Aux, "smooth2d");
  }
};
}

void smooth2d_screen(const u32* fb, const u32* hfb, const bool* hrow, u32 n, u32* dst, size_t pitch, u32 strength) {
  static thread_local std::vector<u32> big;   // the whole screen smoothed (n is at most 4: 3 MB)
  big.resize(size_t(W) * n * H * n);
  static Pool pool;
  const u32 slices = pool.workers + 1;
  Job base{fb, hfb, hrow, n, dst, pitch, big.data(), 0, 0, std::clamp<u32>(strength, 1, 100) * 256 / 100};
  auto edge = [&](u32 k) { return H * k / slices; };
  for (u32 i = 0; i < pool.workers; ++i) { pool.job[i] = base; pool.job[i].y0 = edge(i); pool.job[i].y1 = edge(i + 1); pool.w[i].dispatch(); }
  Job own = base; own.y0 = edge(slices - 1); own.y1 = H;
  run(own);
  for (u32 i = 0; i < pool.workers; ++i) pool.w[i].wait();
}

} // namespace ds::frontend
