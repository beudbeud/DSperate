// SPDX-License-Identifier: GPL-3.0-or-later
// DSperate - Nintendo DS emulator. Copyright (C) 2026 DSperate contributors.
#pragma once
#include "core/types.h"

#include <cstddef>

namespace ds::frontend {

// One screen at the internal resolution `n` (2-4) into `dst` (`pitch` in
// pixels), its 2D smoothed by xBRZ (xbrz/) from the 1x screen `fb`
// (256x192). Lines with `hrow[y]` set come from `hfb` (256n x 192n, the
// core's internal-resolution frame): there a block that is the 1x pixel
// repeated (2D layers) takes the smoothed one, any other (3D samples) is
// kept as drawn. `strength` (1-100): how much of xBRZ's sample is mixed
// over the pixel's own colour (100: xBRZ as is).
void smooth2d_screen(const u32* fb, const u32* hfb, const bool* hrow, u32 n, u32* dst, size_t pitch, u32 strength = 100);

} // namespace ds::frontend
