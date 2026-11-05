// SPDX-License-Identifier: GPL-3.0-or-later
// DSperate - Nintendo DS emulator. Copyright (C) 2026 DSperate contributors.
//
// libretro core: the headless boot flow (frontend/headless/main.cpp) behind
// the libretro ABI. Video is the two software framebuffers laid out into one
// XRGB8888 frame; the GPU 3D raster, when enabled, draws in its own EGL/GLES
// context (gpu/gl/gl_lean.cpp) and reads the layer back, so no frontend
// hardware context is needed.
// ponytail: no slot 2 GBA cart, no local wireless, no DSi SD card; add when a game needs one.
#include "libretro.h"
#include "core/nds.h"
#include "core/cheat/database.h"
#include "core/input/input_log.h"
#include "core/io/dsi_title_install.h"
#include "core/state/state.h"
#if DSPERATE_JIT
#include "core/cpu/jit/jit.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace ds;

retro_environment_t env_cb;
retro_video_refresh_t video_cb;
retro_audio_sample_batch_t audio_batch_cb;
retro_input_poll_t input_poll_cb;
retro_input_state_t input_state_cb;
retro_log_printf_t log_cb;
retro_rumble_interface rumble_if{};
retro_microphone_interface mic_if{};
retro_microphone_t* mic = nullptr;
unsigned mic_rate = 0;

void log(retro_log_level lvl, const char* fmt, ...) {
  char buf[512]; va_list ap; va_start(ap, fmt); std::vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
  if (log_cb) log_cb(lvl, "[DSperate] %s\n", buf); else std::fprintf(stderr, "[DSperate] %s\n", buf);
}

std::unique_ptr<NDS> nds;
std::string save_path;          // <saves>/<rom>.sav, as the SDL frontend names it
u32 save_writes_seen = 0;       // cart->sram_writes() when the file last matched the chip
int save_quiet_frames = 0;      // frames since the chip last changed
std::vector<u32> frame;         // the laid-out output frame
std::vector<cheat::Code> analog_codes;   // the game's CPAD patch from the built-in database
std::vector<cheat::Code> user_codes;     // retro_cheat_set, by index
u32 rumble_seen = 0; int rumble_frames = 0;
u64 noise = 0x9E3779B97F4A7C15ull;       // fake mic xorshift state
float cursor_x = SCREEN_W / 2.0f, cursor_y = SCREEN_H / 2.0f;   // stick-driven stylus

// Options, read by apply_vars.
enum Layout { Vertical, Horizontal, Top, Bottom } layout = Vertical;
u32 gap = 0; bool swap_screens = false;
bool gpu3d = false, aa = false; int frameskip = 0;
enum TouchStick { NoStick, Right, Left } touch_stick = NoStick;
enum Mic { NoMic, HostMic, NoiseL2, NoiseR2 } mic_mode = NoMic;
bool lid_closed = false, lid_btn_was = false;

constexpr double FPS = 59.8261;
constexpr double SAMPLE_RATE = 32728.5;   // the SPU's real mixer rate (spu/spu.h)

retro_core_option_v2_category CATS[] = {
  {"video",  "Video",  "Screens and the 3D raster."},
  {"system", "System", "Boot, console and firmware settings (take effect on the next load)."},
  {"input",  "Input",  "Sticks, stylus, microphone and rumble."},
  {nullptr, nullptr, nullptr},
};

#define ON_OFF {"disabled", "Off"}, {"enabled", "On"}
retro_core_option_v2_definition DEFS[] = {
  {"dsperate_layout", "Video > Screen layout", "Screen layout", "How the two DS screens are arranged in the frame.", nullptr, "video",
   {{"vertical", "Vertical"}, {"horizontal", "Horizontal"}, {"top", "Top screen only"}, {"bottom", "Bottom screen only"}, {nullptr, nullptr}}, "vertical"},
  {"dsperate_screen_gap", "Video > Screen gap", "Screen gap", "Pixels between the two screens in the vertical and horizontal layouts.", nullptr, "video",
   {{"0", nullptr}, {"8", nullptr}, {"16", nullptr}, {"32", nullptr}, {"64", nullptr}, {"90", "90 (hinge)"}, {nullptr, nullptr}}, "0"},
  {"dsperate_swap_screens", "Video > Swap screens", "Swap screens", "Draw the bottom screen first (top or left). Touch follows the bottom screen wherever it is.", nullptr, "video",
   {ON_OFF, {nullptr, nullptr}}, "disabled"},
  {"dsperate_gpu3d", "Video > GPU 3D", "GPU 3D", "Draw the 3D layer on the GPU: Vulkan, else OpenGL ES 3.1 (the Pi 5's V3D). The software rasteriser is the exact reference; falls back to it when no driver works.", nullptr, "video",
   {ON_OFF, {nullptr, nullptr}}, "disabled"},
  {"dsperate_gpu3d_api", "Video > GPU 3D backend", "GPU 3D backend", "Auto tries Vulkan then OpenGL ES. Vulkan has 4x MSAA anti-aliasing; OpenGL ES has none.", nullptr, "video",
   {{"auto", "Auto"}, {"vulkan", "Vulkan"}, {"opengl", "OpenGL ES"}, {nullptr, nullptr}}, "auto"},
  {"dsperate_aa", "Video > 3D anti-aliasing", "3D anti-aliasing", "The DS's own edge blend on the CPU raster, 4x MSAA on Vulkan. The OpenGL ES raster has none.", nullptr, "video",
   {ON_OFF, {nullptr, nullptr}}, "disabled"},
  {"dsperate_frameskip", "Video > Frameskip", "Frameskip", "Frames skipped for each one drawn. The game runs as usual; only drawing is skipped.", nullptr, "video",
   {{"0", "Off"}, {"1", nullptr}, {"2", nullptr}, {"3", nullptr}, {nullptr, nullptr}}, "0"},
  {"dsperate_jit", "System > Recompiler", "Recompiler", "ARM to native recompiler for both CPUs. Off: the interpreter (slow). Takes effect on the next load.", nullptr, "system",
   {{"enabled", "On"}, {"disabled", "Off"}, {nullptr, nullptr}}, "enabled"},
  {"dsperate_boot", "System > Boot", "Boot", "Direct: jump into the game. Firmware: boot the DS menu with the cart inserted; needs bios9.bin, bios7.bin and firmware.bin in the system directory.", nullptr, "system",
   {{"direct", "Direct boot"}, {"firmware", "Firmware (DS menu)"}, {nullptr, nullptr}}, "direct"},
  {"dsperate_console", "System > Console", "Console", "Auto: DSi for DSiWare and DSi-enhanced carts when bios9i.bin and bios7i.bin are in the system directory, else DS. DSiWare is handed over as the launcher would, no NAND needed.", nullptr, "system",
   {{"auto", "Auto"}, {"ds", "DS"}, {"dsi", "DSi"}, {nullptr, nullptr}}, "auto"},
  {"dsperate_language", "System > Language", "Language", "The console language games read from the firmware. Firmware: the dump's own setting, or English with the generated firmware.", nullptr, "system",
   {{"firmware", "Firmware setting"}, {"japanese", "Japanese"}, {"english", "English"}, {"french", "French"}, {"german", "German"}, {"italian", "Italian"}, {"spanish", "Spanish"}, {nullptr, nullptr}}, "firmware"},
  {"dsperate_analog", "Input > Analog controls (CPAD patch)", "Analog controls (CPAD patch)", "For the games the built-in database has a Circle Pad patch for: left stick steers, right stick the camera, L2/R2 are ZL/ZR. Takes effect on the next load.", nullptr, "input",
   {ON_OFF, {nullptr, nullptr}}, "disabled"},
  {"dsperate_touch_stick", "Input > Stylus on a stick", "Stylus on a stick", "A stick moves a stylus cursor over the bottom screen; R3 taps. For pads without a touch surface.", nullptr, "input",
   {{"none", "Off"}, {"right", "Right stick"}, {"left", "Left stick"}, {nullptr, nullptr}}, "none"},
  {"dsperate_mic", "Input > Microphone", "Microphone", "Host: the frontend's microphone. Noise: white noise while a trigger is held, enough to blow into the mic.", nullptr, "input",
   {{"none", "Off"}, {"host", "Host microphone"}, {"noise on L2", "Noise while L2 is held"}, {"noise on R2", "Noise while R2 is held"}, {nullptr, nullptr}}, "none"},
  {"dsperate_rumble", "Input > Rumble Pak", "Rumble Pak", "A Rumble Pak in slot 2; its motor pulses the controller's. Takes effect on the next load.", nullptr, "input",
   {ON_OFF, {nullptr, nullptr}}, "disabled"},
  {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, {{nullptr, nullptr}}, nullptr},
};
#undef ON_OFF
retro_core_options_v2 OPTIONS = {CATS, DEFS};

// Frontends before core options v2 get the same options as "desc; a|b" variables.
void set_options(retro_environment_t cb) {
  unsigned version = 0;
  if (cb(RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION, &version) && version >= 2) { cb(RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2, &OPTIONS); return; }
  static std::vector<std::string> text; static std::vector<retro_variable> vars;
  text.clear(); vars.clear();
  for (const retro_core_option_v2_definition* d = DEFS; d->key; ++d) {
    std::string t = std::string(d->desc) + "; ";
    for (const retro_core_option_value* v = d->values; v->value; ++v) t += (v == d->values ? "" : "|") + std::string(v->value);
    text.push_back(t);
  }
  for (size_t i = 0; text.size() > i; ++i) vars.push_back({DEFS[i].key, text[i].c_str()});
  vars.push_back({nullptr, nullptr});
  cb(RETRO_ENVIRONMENT_SET_VARIABLES, vars.data());
}


const retro_input_descriptor DESCS[] = {
  {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_UP, "Up"}, {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_DOWN, "Down"},
  {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_LEFT, "Left"}, {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_RIGHT, "Right"},
  {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_A, "A"}, {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_B, "B"},
  {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_X, "X"}, {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_Y, "Y"},
  {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_L, "L"}, {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_R, "R"},
  {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_START, "Start"}, {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_SELECT, "Select"},
  {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_L2, "ZL (analog patch) / mic noise"}, {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_R2, "ZR (analog patch) / mic noise"},
  {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_L3, "Close/open lid"}, {0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_R3, "Stylus tap (stick stylus)"},
  {0, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_X, "Circle Pad X / stylus"}, {0, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_Y, "Circle Pad Y / stylus"},
  {0, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X, "C-Stick X / stylus"}, {0, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y, "C-Stick Y / stylus"},
  {0, 0, 0, 0, nullptr},
};

std::string var(const char* key) {
  retro_variable v{key, nullptr};
  return env_cb(RETRO_ENVIRONMENT_GET_VARIABLE, &v) && v.value ? v.value : "";
}

void frame_size(u32& w, u32& h) {
  w = layout == Horizontal ? SCREEN_W * 2 + gap : SCREEN_W;
  h = layout == Vertical ? SCREEN_H * 2 + gap : SCREEN_H;
}

// Where DS screen `s` (0 top, 1 bottom) sits in the output frame; false when it's not shown.
bool screen_origin(int s, u32& x, u32& y) {
  x = y = 0;
  if (layout == Top) return s == 0;
  if (layout == Bottom) return s == 1;
  const int slot = swap_screens ? 1 - s : s;   // 0 first (top/left), 1 second
  if (slot) { if (layout == Horizontal) x = SCREEN_W + gap; else y = SCREEN_H + gap; }
  return true;
}

retro_game_geometry geometry() {
  u32 w, h; frame_size(w, h);
  return {w, h, SCREEN_W * 2 + 90, SCREEN_H * 2 + 90, static_cast<float>(w) / static_cast<float>(h)};
}

bool open_mic() {
  if (mic || !mic_if.open_mic) return mic != nullptr;
  retro_microphone_params_t p{44100};
  mic = mic_if.open_mic(&p);
  if (!mic) { log(RETRO_LOG_WARN, "mic: the frontend has no microphone"); return false; }
  if (mic_if.get_params && mic_if.get_params(mic, &p)) mic_rate = p.rate;
  if (!mic_rate) mic_rate = 44100;
  mic_if.set_mic_state(mic, true);
  return true;
}

void close_mic() {
  if (mic && mic_if.close_mic) mic_if.close_mic(mic);
  mic = nullptr;
}

void apply_vars(bool at_load) {
  const std::string l = var("dsperate_layout");
  const Layout nl = l == "horizontal" ? Horizontal : l == "top" ? Top : l == "bottom" ? Bottom : Vertical;
  const u32 ng = static_cast<u32>(std::atoi(var("dsperate_screen_gap").c_str()));
  const bool ns = var("dsperate_swap_screens") == "enabled";
  if (nl != layout || ng != gap || ns != swap_screens || at_load) {
    layout = nl; gap = ng; swap_screens = ns;
    u32 w, h; frame_size(w, h); frame.assign(static_cast<size_t>(w) * h, 0xFF000000u);
    if (!at_load) { retro_game_geometry g = geometry(); env_cb(RETRO_ENVIRONMENT_SET_GEOMETRY, &g); }
  }
  const bool g = var("dsperate_gpu3d") == "enabled";
  // The backend pick goes through the raster's own knob (gpu/gpu_raster.cpp reads DS_GPU3D_API).
  const std::string api = var("dsperate_gpu3d_api");
  const char* want = api == "vulkan" ? "vk" : api == "opengl" ? "gles" : "";
  const bool api_changed = std::strcmp(std::getenv("DS_GPU3D_API") ? std::getenv("DS_GPU3D_API") : "", want) != 0;
  if (api_changed) setenv("DS_GPU3D_API", want, 1);
  if (nds && (g != gpu3d || at_load || (g && api_changed))) {
    std::string why;
    if (g && gpu3d && api_changed) nds->gpu3d.renderer().set_gpu(false);   // a new backend needs a fresh raster
    if (!nds->gpu3d.renderer().set_gpu(g, &why)) log(RETRO_LOG_WARN, "gpu3d: unavailable (%s), drawing on the CPU", why.c_str());
    else if (g) log(RETRO_LOG_INFO, "gpu3d: on (%s)", nds->gpu3d.renderer().gpu_api());
    gpu3d = g;
  }
  aa = var("dsperate_aa") == "enabled";
  if (nds) nds->gpu3d.renderer().set_aa(aa);
  frameskip = std::atoi(var("dsperate_frameskip").c_str());
  const std::string ts = var("dsperate_touch_stick");
  touch_stick = ts == "right" ? Right : ts == "left" ? Left : NoStick;
  const std::string m = var("dsperate_mic");
  mic_mode = m == "host" ? HostMic : m == "noise on L2" ? NoiseL2 : m == "noise on R2" ? NoiseR2 : NoMic;
  if (mic_mode == HostMic) { if (!open_mic()) mic_mode = NoMic; } else close_mic();
}

std::vector<u8> slurp(const std::string& path) {
  std::vector<u8> out;
  if (FILE* f = std::fopen(path.c_str(), "rb")) {
    u8 buf[65536];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) out.insert(out.end(), buf, buf + n);
    std::fclose(f);
  }
  return out;
}

void write_save() {
  if (!nds || !nds->cart || save_path.empty() || nds->cart->sram().empty()) return;
  const std::string tmp = save_path + ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) { log(RETRO_LOG_ERROR, "save: cannot write %s", tmp.c_str()); return; }
  const bool ok = std::fwrite(nds->cart->sram().data(), 1, nds->cart->sram().size(), f) == nds->cart->sram().size();
  std::fclose(f);
  if (!ok || std::rename(tmp.c_str(), save_path.c_str()) != 0) { log(RETRO_LOG_ERROR, "save: cannot write %s", save_path.c_str()); return; }
  nds->cart->clear_sram_dirty();
  save_writes_seen = nds->cart->sram_writes();
}

std::string dir_env(unsigned cmd) {
  const char* d = nullptr;
  return env_cb(cmd, &d) && d ? d : "";
}

void blit(int screen, u32 x0, u32 y0) {
  u32 w, h; frame_size(w, h);
  const u32* src = nds->gpu.framebuffer(screen);
  for (u32 y = 0; y < SCREEN_H; ++y) std::memcpy(&frame[(y0 + y) * w + x0], src + y * SCREEN_W, SCREEN_W * 4);
}

// The game's CPAD analog patch from the built-in database (cheat/analog_db),
// enabled as the SDL frontend's Options > Analog would: the fuller "Nub"
// variant when the game has one.
void setup_analog() {
  analog_codes.clear();
  nds->analog_patch = false;
  u8 header[512] = {};
  nds->cart->rom_read(0, header, sizeof header);
  cheat::GameCheats cheats, analog; std::string err;
  if (!cheat::load_analog_for_header("", header, analog, err) || !cheat::merge_analog(cheats, std::move(analog), cheat::header_checksum(header))) {
    log(RETRO_LOG_INFO, "analog: no CPAD patch for this game%s%s", err.empty() ? "" : ": ", err.c_str());
    return;
  }
  cheat::Code* pick = nullptr;
  for (cheat::Code& c : cheats.codes) if (c.analog && !c.is_note() && (!pick || (c.analog_nub() && !pick->analog_nub()))) pick = &c;
  if (!pick) return;
  pick->enabled = true;
  cheat::enforce_exclusive(cheats.codes, cheats.groups);
  analog_codes = cheats.codes;
  const cheat::AnalogRequest req = cheat::analog_request(analog_codes);
  nds->analog_patch = req.schemes > 0;
  nds->analog_full = req.nub;
  nds->analog_layout = req.extended ? NDS::AnalogLayout::Extended : NDS::AnalogLayout::Packed;
  log(RETRO_LOG_INFO, "analog: \"%s\" enabled%s", pick->name.c_str(), req.nub ? " (nub + ZL/ZR)" : "");
}

void rebuild_cheats() {
  if (!nds) return;
  nds->cheats.codes = analog_codes;
  nds->cheats.codes.insert(nds->cheats.codes.end(), user_codes.begin(), user_codes.end());
}

} // namespace

RETRO_API void retro_set_environment(retro_environment_t cb) {
  env_cb = cb;
  set_options(cb);
  bool no_game = true; cb(RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME, &no_game);
  retro_log_callback lc{};
  if (cb(RETRO_ENVIRONMENT_GET_LOG_INTERFACE, &lc)) log_cb = lc.log;
}
RETRO_API void retro_set_video_refresh(retro_video_refresh_t cb) { video_cb = cb; }
RETRO_API void retro_set_audio_sample(retro_audio_sample_t) {}
RETRO_API void retro_set_audio_sample_batch(retro_audio_sample_batch_t cb) { audio_batch_cb = cb; }
RETRO_API void retro_set_input_poll(retro_input_poll_t cb) { input_poll_cb = cb; }
RETRO_API void retro_set_input_state(retro_input_state_t cb) { input_state_cb = cb; }

RETRO_API void retro_init() {}
RETRO_API void retro_deinit() {}
RETRO_API unsigned retro_api_version() { return RETRO_API_VERSION; }

RETRO_API void retro_get_system_info(retro_system_info* info) {
  *info = {};
  info->library_name = "DSperate";
  info->library_version = "3.3.0";
  info->valid_extensions = "nds|dsi|srl|cia|zip";
  info->need_fullpath = true;    // load_rom takes a path (zips are unpacked to a disk cache)
  info->block_extract = true;    // the core reads zips itself; the frontend must not unpack them
}

RETRO_API void retro_get_system_av_info(retro_system_av_info* info) {
  info->geometry = geometry();
  info->timing = {FPS, SAMPLE_RATE};
}

RETRO_API void retro_set_controller_port_device(unsigned, unsigned) {}

RETRO_API bool retro_load_game(const retro_game_info* game) {
  retro_pixel_format fmt = RETRO_PIXEL_FORMAT_XRGB8888;   // the core's framebuffers are 0xAARRGGBB
  if (!env_cb(RETRO_ENVIRONMENT_SET_PIXEL_FORMAT, &fmt)) { log(RETRO_LOG_ERROR, "XRGB8888 not supported"); return false; }
  env_cb(RETRO_ENVIRONMENT_SET_INPUT_DESCRIPTORS, const_cast<retro_input_descriptor*>(DESCS));
  if (!env_cb(RETRO_ENVIRONMENT_GET_RUMBLE_INTERFACE, &rumble_if)) rumble_if = {};
  mic_if.interface_version = RETRO_MICROPHONE_INTERFACE_VERSION;
  if (!env_cb(RETRO_ENVIRONMENT_GET_MICROPHONE_INTERFACE, &mic_if)) mic_if = {};

  nds = std::make_unique<NDS>();
  user_codes.clear(); analog_codes.clear();
  std::string sys = dir_env(RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY);
  const std::string saves = dir_env(RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY);
  if (FILE* f = std::fopen((sys + "/nds/bios7.bin").c_str(), "rb")) { std::fclose(f); sys += "/nds"; }   // Recalbox keeps the DS dumps in bios/nds
  std::string err;
  // Console settings for a generated firmware; a dump keeps its own unless a language is picked.
  static const char* const LANGS[] = {"japanese", "english", "french", "german", "italian", "spanish"};
  const std::string lang = var("dsperate_language");
  bios::UserSettings user;
  int lang_idx = -1;
  for (int i = 0; i < 6; ++i) if (lang == LANGS[i]) lang_idx = i;
  if (lang_idx >= 0) user.language = static_cast<u8>(lang_idx);
  // Dumps are optional: the FreeBIOS and a generated firmware direct-boot any game.
  if (!nds->load_bios(sys + "/bios9.bin", sys + "/bios7.bin", sys + "/firmware.bin", user, &err)) { log(RETRO_LOG_ERROR, "bios: %s", err.c_str()); return false; }
  if (!nds->bios_native) log(RETRO_LOG_INFO, "no bios9.bin/bios7.bin in %s; using the built-in FreeBIOS", sys.c_str());
  if (lang_idx >= 0 && !nds->firmware_synthetic) nds->write_user_settings(NDS::UserField::Language, user);   // in memory only; the dump is never written
  if (!nds->load_dsi_bios(sys + "/bios9i.bin", sys + "/bios7i.bin", &err)) log(RETRO_LOG_WARN, "dsi bios: %s", err.c_str());
  nds->rom_cache_dir = saves;
  nds->reset();
  const std::string boot = var("dsperate_boot"), console = var("dsperate_console");
  bool direct = true;
  if (game && game->path) {
    if (!nds->load_rom(game->path)) { log(RETRO_LOG_ERROR, "%s: %s", game->path, nds->rom_error.c_str()); return false; }
    const std::string p = game->path;
    const size_t slash = p.find_last_of('/'), dot = p.find_last_of('.');
    const std::string stem = p.substr(slash == std::string::npos ? 0 : slash + 1, dot == std::string::npos || (slash != std::string::npos && dot < slash) ? std::string::npos : dot - (slash == std::string::npos ? 0 : slash + 1));
    save_path = (saves.empty() ? p.substr(0, slash == std::string::npos ? 0 : slash) : saves) + "/" + stem + ".sav";
    if (const std::vector<u8> sav = slurp(save_path); !sav.empty()) {
      const cart::Cart::SaveLoad r = nds->cart->load_save(sav.data(), sav.size());
      log(RETRO_LOG_INFO, "save: loaded %zu bytes from %s%s", sav.size(), save_path.c_str(), r.fitted ? "" : " (not the save chip's size)");
    }
    save_writes_seen = nds->cart->sram_writes();
    // DSi: a DSiWare title is handed over as the launcher would (no NAND
    // needed); a DSi-enhanced cart runs the DSi machine in card mode.
    const bool dsiware = io::file_is_dsiware(p);
    const bool want_dsi = console == "dsi" || (console != "ds" && nds->bios_native_dsi && (dsiware || nds->cart->dsi_capable()));
    if (want_dsi && !nds->bios_native_dsi) log(RETRO_LOG_WARN, "DSi mode needs bios9i.bin and bios7i.bin in %s; running as a DS", sys.c_str());
    else if (want_dsi) {
      nds->set_dsi(true);
      std::string why; std::vector<std::string> made;
      if (dsiware && !nds->prepare_dsi_hle(user, &why, &made)) { log(RETRO_LOG_ERROR, "dsi: %s", why.c_str()); return false; }
      for (const std::string& m : made) log(RETRO_LOG_INFO, "dsi: %s", m.c_str());
      nds->reset();
      log(RETRO_LOG_INFO, "console: DSi%s", dsiware ? " (DSiWare, launcher hand-off)" : " (card mode)");
    }
    if (var("dsperate_rumble") == "enabled") nds->slot2 = std::make_unique<slot2::RumblePak>();
    if (var("dsperate_analog") == "enabled") setup_analog();
    direct = boot != "firmware" || !nds->can_boot_firmware() || nds->dsi;
    if (boot == "firmware" && !nds->can_boot_firmware()) log(RETRO_LOG_WARN, "firmware boot needs real BIOS and firmware dumps; direct-booting the ROM");
    if (direct) nds->setup_direct_boot();
  } else if (!nds->can_boot_firmware()) {
    log(RETRO_LOG_WARN, "no ROM and no real dumps: nothing boots");
  }
  rebuild_cheats();
#if DSPERATE_JIT
  if (var("dsperate_jit") != "disabled" && !jit::attach(*nds, true, true)) log(RETRO_LOG_WARN, "jit: could not attach, interpreting");
#endif
  apply_vars(true);
  return true;
}

RETRO_API bool retro_load_game_special(unsigned, const retro_game_info*, size_t) { return false; }

RETRO_API void retro_unload_game() {
  write_save();
  close_mic();
  nds.reset();
  save_path.clear();
}

RETRO_API void retro_reset() {
  if (!nds) return;
  nds->reset();
  if (nds->cart) nds->setup_direct_boot();
}

RETRO_API void retro_run() {
  if (!nds) return;
  bool updated = false;
  if (env_cb(RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE, &updated) && updated) apply_vars(false);

  input_poll_cb();
  auto btn = [](unsigned id) { return input_state_cb(0, RETRO_DEVICE_JOYPAD, 0, id) != 0; };
  auto axis = [](unsigned idx, unsigned id) { return input_state_cb(0, RETRO_DEVICE_ANALOG, idx, id); };
  input::Frame in;
  static const struct { unsigned retro; io::Io::Button ds; } MAP[] = {
    {RETRO_DEVICE_ID_JOYPAD_A, io::Io::BTN_A}, {RETRO_DEVICE_ID_JOYPAD_B, io::Io::BTN_B},
    {RETRO_DEVICE_ID_JOYPAD_X, io::Io::BTN_X}, {RETRO_DEVICE_ID_JOYPAD_Y, io::Io::BTN_Y},
    {RETRO_DEVICE_ID_JOYPAD_L, io::Io::BTN_L}, {RETRO_DEVICE_ID_JOYPAD_R, io::Io::BTN_R},
    {RETRO_DEVICE_ID_JOYPAD_START, io::Io::BTN_START}, {RETRO_DEVICE_ID_JOYPAD_SELECT, io::Io::BTN_SELECT},
    {RETRO_DEVICE_ID_JOYPAD_UP, io::Io::BTN_UP}, {RETRO_DEVICE_ID_JOYPAD_DOWN, io::Io::BTN_DOWN},
    {RETRO_DEVICE_ID_JOYPAD_LEFT, io::Io::BTN_LEFT}, {RETRO_DEVICE_ID_JOYPAD_RIGHT, io::Io::BTN_RIGHT},
  };
  for (const auto& m : MAP) if (btn(m.retro)) in.buttons |= static_cast<u16>(1u << m.ds);
  // L3 toggles the lid (sleep mode).
  const bool lid_btn = btn(RETRO_DEVICE_ID_JOYPAD_L3);
  if (lid_btn && !lid_btn_was) lid_closed = !lid_closed;
  lid_btn_was = lid_btn; in.lid = lid_closed;
  // Analog patch: sticks in SDL range (libretro's is the same), ZL/ZR on the triggers.
  if (nds->analog_patch) {
    in.cpad_x = axis(RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_X); in.cpad_y = axis(RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_Y);
    in.cstick_x = axis(RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X); in.cstick_y = axis(RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y);
    in.zl = btn(RETRO_DEVICE_ID_JOYPAD_L2); in.zr = btn(RETRO_DEVICE_ID_JOYPAD_R2);
  }
  // Stylus: the pointer spans the whole output frame and only the bottom
  // screen takes a touch; or a stick moves a cursor and R3 taps.
  u32 bx, by, w, h; frame_size(w, h);
  const bool bottom_shown = screen_origin(1, bx, by);
  if (touch_stick != NoStick) {
    const unsigned idx = touch_stick == Right ? RETRO_DEVICE_INDEX_ANALOG_RIGHT : RETRO_DEVICE_INDEX_ANALOG_LEFT;
    constexpr float SPEED = 4.0f;   // ponytail: pixels per frame at full deflection; an option if someone asks
    cursor_x = std::clamp(cursor_x + axis(idx, RETRO_DEVICE_ID_ANALOG_X) / 32767.0f * SPEED, 0.0f, SCREEN_W - 1.0f);
    cursor_y = std::clamp(cursor_y + axis(idx, RETRO_DEVICE_ID_ANALOG_Y) / 32767.0f * SPEED, 0.0f, SCREEN_H - 1.0f);
    if (btn(RETRO_DEVICE_ID_JOYPAD_R3)) { in.down = true; in.x = static_cast<u8>(cursor_x); in.y = static_cast<u8>(cursor_y); }
  }
  if (bottom_shown && input_state_cb(0, RETRO_DEVICE_POINTER, 0, RETRO_DEVICE_ID_POINTER_PRESSED)) {
    const int px = (input_state_cb(0, RETRO_DEVICE_POINTER, 0, RETRO_DEVICE_ID_POINTER_X) + 0x8000) * static_cast<int>(w) / 0x10000 - static_cast<int>(bx);
    const int py = (input_state_cb(0, RETRO_DEVICE_POINTER, 0, RETRO_DEVICE_ID_POINTER_Y) + 0x8000) * static_cast<int>(h) / 0x10000 - static_cast<int>(by);
    if (px >= 0 && px < static_cast<int>(SCREEN_W) && py >= 0 && py < static_cast<int>(SCREEN_H)) { in.down = true; in.x = static_cast<u8>(px); in.y = static_cast<u8>(py); }
  }
  // Microphone: the host's, or white noise while a trigger is held (blowing).
  static std::vector<s16> mic_buf;
  size_t mic_n = 0;
  if (mic_mode == HostMic && mic) {
    mic_buf.resize(mic_rate / 59 + 1);
    const int got = mic_if.read_mic(mic, mic_buf.data(), mic_buf.size());
    if (got > 0) mic_n = static_cast<size_t>(got);
  } else if ((mic_mode == NoiseL2 && btn(RETRO_DEVICE_ID_JOYPAD_L2)) || (mic_mode == NoiseR2 && btn(RETRO_DEVICE_ID_JOYPAD_R2))) {
    for (s8& v : in.mic) { noise ^= noise << 13; noise ^= noise >> 7; noise ^= noise << 17; v = static_cast<s8>((static_cast<int>(noise & 0xFF) - 128) * 4 / 5); }
  }
  input::apply(*nds, in, mic_n ? mic_buf.data() : nullptr, mic_n);

  // Frameskip as the headless harness does it: whole display periods, a fixed pattern, decided a frame ahead.
  if (frameskip > 0) {
    const int period = nds->gpu.display_phase_period();
    const int skip = frameskip * period, cycle = skip + period;
    nds->gpu.set_frame_skip(static_cast<int>((nds->frame_count + 1) % cycle) < skip);
  } else nds->gpu.set_frame_skip(false);

  nds->run_frame();

  for (int s = 0; s < 2; ++s) { u32 x, y; if (screen_origin(s, x, y)) blit(s, x, y); }
  if (touch_stick != NoStick && bottom_shown) {   // a small white cross where the stylus is
    const int cx = static_cast<int>(cursor_x), cy = static_cast<int>(cursor_y);
    for (int d = -3; d <= 3; ++d) {
      if (cx + d >= 0 && cx + d < static_cast<int>(SCREEN_W)) frame[(by + cy) * w + bx + cx + d] = 0xFFFFFFFFu;
      if (cy + d >= 0 && cy + d < static_cast<int>(SCREEN_H)) frame[(by + cy + d) * w + bx + cx] = 0xFFFFFFFFu;
    }
  }
  video_cb(frame.data(), w, h, w * 4);

  s16 buf[2048 * 2];
  for (size_t n; (n = nds->spu.take(buf, 2048)) != 0;) audio_batch_cb(buf, n);

  // Rumble Pak: each motor event is a short pulse on the controller.
  if (nds->slot2 && rumble_if.set_rumble_state) {
    if (nds->slot2->rumble_events() != rumble_seen) { rumble_seen = nds->slot2->rumble_events(); rumble_frames = 3; }
    if (rumble_frames > 0 && --rumble_frames == 0) rumble_if.set_rumble_state(0, RETRO_RUMBLE_STRONG, 0);
    else if (rumble_frames == 2) rumble_if.set_rumble_state(0, RETRO_RUMBLE_STRONG, 0xFFFF);
  }

  // Battery save: written once the chip has been quiet for a second, as the SDL frontend does.
  if (nds->cart && nds->cart->sram_dirty()) {
    if (nds->cart->sram_writes() != save_writes_seen) { save_writes_seen = nds->cart->sram_writes(); save_quiet_frames = 0; }
    else if (++save_quiet_frames >= 60) write_save();
  }
}

// Save states: the core's own stream behind an 8-byte length, so a frontend
// buffer sized with headroom reads back exactly what was written.
RETRO_API size_t retro_serialize_size() {
  if (!nds) return 0;
  state::Writer w; std::string err;
  if (!nds->save_state(w, err)) return 0;
  return w.data().size() + w.data().size() / 8 + 8;   // ponytail: +12.5% headroom for frame-to-frame growth
}
RETRO_API bool retro_serialize(void* data, size_t size) {
  if (!nds) return false;
  state::Writer w; std::string err;
  if (!nds->save_state(w, err)) { log(RETRO_LOG_ERROR, "state: %s", err.c_str()); return false; }
  const u64 n = w.data().size();
  if (n + 8 > size) return false;
  std::memcpy(data, &n, 8);
  std::memcpy(static_cast<u8*>(data) + 8, w.data().data(), n);
  return true;
}
RETRO_API bool retro_unserialize(const void* data, size_t size) {
  if (!nds || size < 8) return false;
  u64 n; std::memcpy(&n, data, 8);
  if (n + 8 > size) return false;
  state::Reader r(static_cast<const u8*>(data) + 8, n); std::string err;
  if (!nds->load_state(r, err)) { log(RETRO_LOG_ERROR, "state: %s", err.c_str()); return false; }
  return true;
}

// Action Replay codes from the frontend: "XXXXXXXX YYYYYYYY" words, any
// separators. They run from the ARM7's VBlank like the database's.
RETRO_API void retro_cheat_reset() { user_codes.clear(); rebuild_cheats(); }
RETRO_API void retro_cheat_set(unsigned index, bool enabled, const char* code) {
  if (index >= user_codes.size()) user_codes.resize(index + 1);
  cheat::Code& c = user_codes[index];
  c.name = "retro " + std::to_string(index);
  c.enabled = enabled;
  c.words.clear();
  for (const char* p = code ? code : ""; *p;) {
    while (*p && !std::isxdigit(static_cast<unsigned char>(*p))) ++p;
    if (!*p) break;
    char* end; const unsigned long v = std::strtoul(p, &end, 16);
    if (end - p != 8) { log(RETRO_LOG_WARN, "cheat %u: not an Action Replay code", index); c.words.clear(); break; }
    c.words.push_back(static_cast<u32>(v)); p = end;
  }
  rebuild_cheats();
}

RETRO_API unsigned retro_get_region() { return RETRO_REGION_NTSC; }
// Main RAM, for RetroArch's achievements and cheat search.
RETRO_API void* retro_get_memory_data(unsigned id) { return nds && id == RETRO_MEMORY_SYSTEM_RAM ? nds->bus.main_ram.get() : nullptr; }
RETRO_API size_t retro_get_memory_size(unsigned id) { return nds && id == RETRO_MEMORY_SYSTEM_RAM ? nds->bus.main_ram_size() : 0; }
