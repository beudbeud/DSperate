// SPDX-License-Identifier: GPL-3.0-or-later
// DSperate - Nintendo DS emulator. Copyright (C) 2026 DSperate contributors.
//
// The GPU raster's whole record above 1x (Raster3D::create's `fine`, what the
// libretro core composes from), on each backend this machine has: a frame
// with no polygon is the clear colour in the 1x record and at every texel of
// the whole one. A backend that cannot start is skipped, not failed.
#include "core/gpu/gpu_raster.h"
#include "core/gpu/vk/vk_layout.h"
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace ds;

int main() {
  int fails = 0, ran = 0;
  for (const char* api : {"vk", "gles"}) {
    setenv("DS_GPU3D_API", api, 1);
    for (u32 s : {2u, 3u, 4u}) {
      std::string why;
      auto L = gpu::Raster3D::create(false, s, &why, true);
      if (!L || !L->ready()) { std::printf("%s %ux: skipped (%s)\n", api, s, why.c_str()); continue; }
      ++ran;
      gpu::vk::GpuFrame f{};
      f.clear_color = 0x1F00003Fu;   // red, alpha 31
      f.clear_depth = 0x7FFF;
      *L->post_buffer() = gpu::vk::GpuPost{};
      if (!L->submit(0, 0, 0, f)) { std::printf("%s %ux: submit refused\n", api, s); ++fails; continue; }
      L->wait_all();
      const u32* out = L->output();
      const u32* fine = out ? L->fine_for(out) : nullptr;
      u32 bad = 0;
      if (!out || !fine || L->scale() != s) bad = 1;
      else {
        for (u32 i = 0; i < 256 * 192; ++i) bad += out[i] != f.clear_color;
        for (u32 i = 0; i < 256 * s * 192 * s; ++i) bad += fine[i] != f.clear_color;
      }
      std::printf("%s %ux (%s): %s\n", api, s, L->api(), bad ? "FAIL" : "ok");
      fails += bad != 0;
    }
  }
  if (!ran) std::printf("no GPU raster here: nothing checked\n");
  return fails ? 1 : 0;
}
