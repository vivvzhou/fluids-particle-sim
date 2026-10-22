#pragma once

#include "fluid/particle_types.h"
#include <vector>

namespace fluid {
struct ParticleState {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
};

struct SphFields {
    std::vector<float> densities;
    std::vector<float> pressures;
    std::vector<Vec3> accelerations;
};
} // namespace fluid
