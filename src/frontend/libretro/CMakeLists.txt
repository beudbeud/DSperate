# SPDX-License-Identifier: GPL-3.0-or-later
# The libretro core (dsperate_libretro.so). libretro.h is the upstream header
# (MIT); the core only dlopens EGL/GLES for the GPU 3D raster, like every
# other frontend, so it links nothing a RetroArch host lacks.
add_library(dsperate_libretro SHARED libretro.cpp)
set_target_properties(dsperate_libretro PROPERTIES PREFIX "" C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
target_link_libraries(dsperate_libretro PRIVATE dsperate_core)
