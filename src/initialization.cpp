#include "fluid/initialization.h"
#include <cmath>
#include <random>
#include <stdexcept>

namespace fluid {
ParticleState make_dam_break(std::size_t count, const SimParams& params, std::uint32_t seed, float jitter) {
    params.validate();
    if (count == 0 || count > max_cpu_particles)
        throw std::invalid_argument("CPU particle count must be in [1, 4096]; recommend 128-512");
    if (!std::isfinite(jitter) || jitter < 0.0f || jitter > 0.1f)
        throw std::invalid_argument("jitter must be a finite spacing fraction in [0, 0.1]");

    // Integer dimensions avoid platform-dependent cbrt rounding near perfect cubes.
    std::size_t nx = 1;
    while (2 * nx * nx * nx < count) ++nx;
    const std::size_t ny = 2 * nx;
    const std::size_t nz = (count + nx * ny - 1) / (nx * ny);
    const float s = params.spacing;
    const Vec3 extent{static_cast<float>(nx - 1) * s, static_cast<float>(ny - 1) * s,
                      static_cast<float>(nz - 1) * s};
    const auto low = params.lower_bound();
    const auto high = params.upper_bound();
    const Vec3 origin{low.x + s, low.y + s, low.z + ((high.z - low.z) - extent.z) * 0.5f};
    const float margin = jitter * s;
    if (!inside_bounds(origin - Vec3{margin, margin, margin}, params) ||
        !inside_bounds(origin + extent + Vec3{margin, margin, margin}, params))
        throw std::invalid_argument("dam-break lattice does not fit padded box; reduce spacing or count");

    ParticleState state;
    state.positions.reserve(count);
    state.velocities.resize(count);
    std::mt19937 random(seed);
    const auto offset = [&]() {
        const float unit = static_cast<float>(random() >> 8) * (1.0f / 16777216.0f);
        return (2.0f * unit - 1.0f) * margin;
    };
    for (std::size_t i = 0; i < count; ++i) {
        const auto x = i % nx;
        const auto y = (i / nx) % ny;
        const auto z = i / (nx * ny);
        // Sequenced list initialization fixes the order of random draws.
        const Vec3 jitter_offset{offset(), offset(), offset()};
        state.positions.push_back(origin + Vec3{static_cast<float>(x) * s, static_cast<float>(y) * s,
                                                static_cast<float>(z) * s} + jitter_offset);
    }
    return state;
}
} // namespace fluid
