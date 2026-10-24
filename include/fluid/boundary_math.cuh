#pragma once

#include "fluid/particle_types.h"

namespace fluid {
FLUID_HD inline void collide_axis(float& position, float& velocity, float low, float high, float restitution) {
    if (position <= low) {
        position = low;
        if (velocity < 0.0f) velocity = -velocity * restitution;
    } else if (position >= high) {
        position = high;
        if (velocity > 0.0f) velocity = -velocity * restitution;
    }
}

FLUID_HD inline void collide_bounds(Vec3& position, Vec3& velocity, Vec3 low, Vec3 high, float restitution) {
    collide_axis(position.x, velocity.x, low.x, high.x, restitution);
    collide_axis(position.y, velocity.y, low.y, high.y, restitution);
    collide_axis(position.z, velocity.z, low.z, high.z, restitution);
}
} // namespace fluid
