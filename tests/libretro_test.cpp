// SPDX-License-Identifier: GPL-3.0-or-later
// The libretro core as RetroArch drives it: loaded with dlopen, a cartridge
// from a path, options through the environment callback.
#include "../src/frontend/libretro/libretro.h"
#include "check.h"

#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

std::map<std::string, std::string> options;
bool options_changed = false;
unsigned frames = 0, width = 0, height = 0;
std::uint64_t picture = 0, sound = 0;   // FNV-1a of the last frame's pixels and samples
size_t samples = 0;
std::string dir;

std::uint64_t fnv(const void* p, size_t n, std::uint64_t h) {
  const auto* b = static_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

bool env(unsigned cmd, void* data) {
  switch (cmd) {
    case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT: return *static_cast<retro_pixel_format*>(data) == RETRO_PIXEL_FORMAT_XRGB8888;
    case RETRO_ENVIRONMENT_GET_VARIABLE: {
      auto* v = static_cast<retro_variable*>(data);
      const auto it = options.find(v->key);
      v->value = it == options.end() ? nullptr : it->second.c_str();
      return v->value != nullptr;
    }
    case RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE: *static_cast<bool*>(data) = options_changed; options_changed = false; return true;
    case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY: case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY: *static_cast<const char**>(data) = dir.c_str(); return true;
    default: return false;
  }
}
void video(const void* px, unsigned w, unsigned h, size_t pitch) {
  CHECK(px && pitch == w * 4);
  width = w; height = h; ++frames;
  picture = fnv(px, pitch * h, 1469598103934665603ull);
}
size_t audio(const int16_t* d, size_t n) { sound = fnv(d, n * 4, sound); samples += n; return n; }
void poll() {}
int16_t input(unsigned, unsigned, unsigned, unsigned) { return 0; }

struct Core {
  void* so;
#define SYM(name) decltype(&::name) name = reinterpret_cast<decltype(&::name)>(dlsym(so, #name))
  explicit Core(void* h) : so(h) {}
  SYM(retro_api_version); SYM(retro_get_system_info); SYM(retro_set_environment); SYM(retro_set_video_refresh);
  SYM(retro_set_audio_sample_batch); SYM(retro_set_input_poll); SYM(retro_set_input_state); SYM(retro_init);
  SYM(retro_load_game); SYM(retro_get_system_av_info); SYM(retro_run); SYM(retro_serialize_size); SYM(retro_serialize);
  SYM(retro_unserialize); SYM(retro_unload_game); SYM(retro_deinit);
#undef SYM
};

void set(const std::string& key, const std::string& value) { options[key] = value; options_changed = true; }

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc == 2);
  void* so = dlopen(argv[1], RTLD_NOW);
  if (!so) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
  Core c(so);
  CHECK(c.retro_api_version && c.retro_deinit);
  CHECK(c.retro_api_version() == RETRO_API_VERSION);
  retro_system_info si; c.retro_get_system_info(&si);
  CHECK(!std::strcmp(si.library_name, "DSperate") && si.need_fullpath && si.block_extract);

  // A blank cartridge: the CPUs run zeros, which is enough for the core's plumbing.
  dir = (std::filesystem::temp_directory_path() / ("dsperate-libretro-test-" + std::to_string(::getpid()))).string();
  std::filesystem::create_directories(dir);
  const std::string rom = dir + "/blank.nds";
  { std::ofstream(rom, std::ios::binary).write(std::string(0x4000, '\0').data(), 0x4000); }

  c.retro_set_environment(env); c.retro_set_video_refresh(video); c.retro_set_audio_sample_batch(audio);
  c.retro_set_input_poll(poll); c.retro_set_input_state(input);
  c.retro_init();
  retro_game_info gi{rom.c_str(), nullptr, 0, nullptr};
  CHECK(c.retro_load_game(&gi));

  { // The default layout stacks the two screens.
    retro_system_av_info av; c.retro_get_system_av_info(&av);
    CHECK(av.geometry.base_width == 256 && av.geometry.base_height == 384);
    c.retro_run();
    CHECK(frames == 1 && width == 256 && height == 384 && samples > 0);
  }

  { // Each layout gives its documented frame size, the gap included.
    const struct { const char* layout; int gap; unsigned w, h; } cases[] = {
      {"horizontal", 0, 512, 192}, {"hybrid", 0, 768, 384}, {"hybrid", 16, 784, 384},
      {"pip", 0, 512, 384}, {"top", 0, 256, 192}, {"bottom", 0, 256, 192}, {"vertical", 8, 256, 392}, {"vertical", 0, 256, 384}};
    for (const auto& k : cases) {
      set("dsperate_layout", k.layout); set("dsperate_screen_gap", std::to_string(k.gap));
      c.retro_run();
      CHECK(width == k.w && height == k.h);
    }
  }

  { // The internal resolution multiplies the frame with the GPU raster on, and only then.
    set("dsperate_internal_res", "2");
    c.retro_run();
    CHECK(width == 256 && height == 384);
    set("dsperate_gpu3d", "enabled");
    c.retro_run();
    CHECK(width == 512 && height == 768);
    set("dsperate_layout", "hybrid");
    c.retro_run();
    CHECK(width == 768 && height == 384);   // hybrid is 2x already: the large screen at the scale
    set("dsperate_layout", "vertical"); set("dsperate_internal_res", "1"); set("dsperate_gpu3d", "disabled");
    c.retro_run();
    CHECK(width == 256 && height == 384);
  }

  const size_t size = c.retro_serialize_size();
  { // The state size is fixed for the session, whatever the machine holds.
    CHECK(size > 1024 * 1024);
    for (int i = 0; i < 30; ++i) c.retro_run();
    CHECK(c.retro_serialize_size() == size);
  }

  { // A buffer smaller than the announced size is refused rather than overrun.
    std::vector<unsigned char> small(size - 1);
    CHECK(!c.retro_serialize(small.data(), small.size()));
  }

  { // A state restores the same pictures and the same sound, from its first frame on.
    std::vector<unsigned char> st(size);
    CHECK(c.retro_serialize(st.data(), st.size()));
    std::vector<std::uint64_t> pics, snds;
    for (int i = 0; i < 30; ++i) { sound = 1469598103934665603ull; c.retro_run(); pics.push_back(picture); snds.push_back(sound); }
    CHECK(c.retro_unserialize(st.data(), st.size()));
    for (int i = 0; i < 30; ++i) {
      sound = 1469598103934665603ull; c.retro_run();
      CHECK(picture == pics[i]);
      CHECK(sound == snds[i]);
    }
  }

  { // A truncated or garbled state is refused, and the session goes on.
    std::vector<unsigned char> st(size);
    CHECK(c.retro_serialize(st.data(), st.size()));
    CHECK(!c.retro_unserialize(st.data(), 11));
    std::vector<unsigned char> bad(st);
    std::memset(bad.data(), 0xFF, 8);   // a machine state longer than the buffer
    CHECK(!c.retro_unserialize(bad.data(), bad.size()));
    CHECK(c.retro_unserialize(st.data(), st.size()));
    c.retro_run();
  }

  c.retro_unload_game(); c.retro_deinit();
  std::filesystem::remove_all(dir);
  std::puts("libretro: ok");
  return 0;
}
