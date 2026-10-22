#pragma once

#include "fluid/particle_types.h"
#include <filesystem>
#include <vector>

namespace fluid {
// Binary little-endian PLY 1.0, vertex-only xyz IEEE-754 float32.
// Creates parent directories; throws on invalid coordinates or any write/close failure.
void write_ply(const std::filesystem::path& path, const std::vector<Vec3>& positions);
} // namespace fluid
