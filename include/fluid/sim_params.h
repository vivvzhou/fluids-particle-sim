#pragma once

#include "fluid/particle_types.h"

namespace fluid {

struct SimParams {
    Vec3 box_min{0.0f, 0.0f, 0.0f};
    Vec3 box_max{1.0f, 1.0f, 1.0f};
    Vec3 gravity{0.0f, -9.81f, 0.0f};
    float spacing = 0.05f;
    float smoothing_radius = 0.1f;
    float rest_density = 1000.0f;
    float stiffness = 100000.0f;      // Tait coefficient B, pressure units; exponent = 7.
    float viscosity = 0.01f;          // Kinematic viscosity nu, length^2 / time.
    float dt = 0.0005f;              // Fixed; changing parameters may require a smaller value.
    float boundary_padding = 0.025f;
    float restitution = 0.3f;        // Normal wall bounce; tangential velocity is unchanged.

    float particle_mass() const { return rest_density * spacing * spacing * spacing; }
    float density_floor() const { return rest_density * 1.0e-6f; }
    Vec3 lower_bound() const { return box_min + Vec3{boundary_padding, boundary_padding, boundary_padding}; }
    Vec3 upper_bound() const { return box_max - Vec3{boundary_padding, boundary_padding, boundary_padding}; }
    void validate() const;
};

bool inside_bounds(Vec3 position, const SimParams& params);

} // namespace fluid
