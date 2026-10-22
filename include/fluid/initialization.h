#pragma once

#include "fluid/particle_state.h"
#include "fluid/sim_params.h"
#include <cstddef>
#include <cstdint>

namespace fluid {
// Small CPU reference guard, not a GPU-scale preset. Recommend 128-512 particles.
inline constexpr std::size_t max_cpu_particles = 4096;

// A tall block toward the lower-x wall, centered in z. Jitter is a fraction of spacing.
// Uses explicitly mapped mt19937 integer output for deterministic seeded positions.
ParticleState make_dam_break(std::size_t count, const SimParams& params,
                             std::uint32_t seed = 42, float jitter = 0.02f);
} // namespace fluid
