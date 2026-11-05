// SPDX-License-Identifier: GPL-3.0-or-later
// Smoke test for the libretro core: load it as RetroArch would, boot with no
// game, run three frames, and check that video and audio come out.
#include "../src/frontend/libretro/libretro.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <dlfcn.h>

static unsigned frames = 0, width = 0, height = 0; static bool audio = false;
static bool env(unsigned cmd, void* data) {
  switch (cmd) {
    case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT: return *static_cast<retro_pixel_format*>(data) == RETRO_PIXEL_FORMAT_XRGB8888;
    case RETRO_ENVIRONMENT_GET_VARIABLE: { auto* v = static_cast<retro_variable*>(data); v->value = !std::strcmp(v->key, "dsperate_layout") ? "horizontal" : nullptr; return v->value != nullptr; }
    case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY: case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY: *static_cast<const char**>(data) = "/nonexistent"; return true;
    default: return false;
  }
}
static void video(const void* px, unsigned w, unsigned h, size_t pitch) { assert(px && pitch == w * 4); width = w; height = h; ++frames; }
static size_t audio_batch(const int16_t*, size_t n) { audio = audio || n > 0; return n; }
static void poll() {}
static int16_t state(unsigned, unsigned, unsigned, unsigned) { return 0; }

int main(int argc, char** argv) {
  assert(argc == 2);
  void* so = dlopen(argv[1], RTLD_NOW);
  if (!so) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
  #define SYM(name) auto name = reinterpret_cast<decltype(&::name)>(dlsym(so, #name)); assert(name)
  SYM(retro_api_version); SYM(retro_get_system_info); SYM(retro_set_environment); SYM(retro_set_video_refresh);
  SYM(retro_set_audio_sample_batch); SYM(retro_set_input_poll); SYM(retro_set_input_state); SYM(retro_init);
  SYM(retro_load_game); SYM(retro_get_system_av_info); SYM(retro_run); SYM(retro_serialize_size); SYM(retro_serialize);
  SYM(retro_unserialize); SYM(retro_unload_game); SYM(retro_deinit);
  assert(retro_api_version() == RETRO_API_VERSION);
  retro_system_info si; retro_get_system_info(&si); assert(!std::strcmp(si.library_name, "DSperate") && si.need_fullpath);
  retro_set_environment(env); retro_set_video_refresh(video); retro_set_audio_sample_batch(audio_batch);
  retro_set_input_poll(poll); retro_set_input_state(state);
  retro_init();
  assert(retro_load_game(nullptr));
  retro_system_av_info av; retro_get_system_av_info(&av); assert(av.geometry.base_width == 512 && av.geometry.base_height == 192);
  for (int i = 0; i < 3; ++i) retro_run();
  assert(frames == 3 && width == 512 && height == 192 && audio);
  const size_t n = retro_serialize_size(); assert(n > 8);
  std::vector<unsigned char> st(n);
  assert(retro_serialize(st.data(), n) && retro_unserialize(st.data(), n));
  retro_run(); assert(frames == 4);
  retro_unload_game(); retro_deinit();
  std::puts("libretro: ok");
  return 0;
}
