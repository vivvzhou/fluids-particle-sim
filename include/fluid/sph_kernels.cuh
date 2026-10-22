#pragma once

#include "fluid/particle_types.h"

namespace fluid {

// 3D kernels, h is the support radius (not half the support radius).
// Dimensionless evaluation avoids forming h^9. Construct only with validated h.
struct SphKernels {
    float h;
    float h_squared;
    float inverse_h;
    float poly6_scale;
    float spiky_scale;
    float spiky_gradient_scale;
    float viscosity_scale;

    FLUID_HD explicit SphKernels(float radius)
        : h(radius), h_squared(radius * radius), inverse_h(1.0f / radius) {
        constexpr float pi = 3.14159265358979323846f;
        const float inverse_h3 = inverse_h * inverse_h * inverse_h;
        poly6_scale = 315.0f / (64.0f * pi) * inverse_h3;
        spiky_scale = 15.0f / pi * inverse_h3;
        spiky_gradient_scale = -45.0f / pi * inverse_h3 * inverse_h;
        viscosity_scale = 45.0f / pi * inverse_h3 * inverse_h * inverse_h;
    }

    FLUID_HD float poly6(float radius_squared) const {
        if (radius_squared < 0.0f || radius_squared >= h_squared) return 0.0f;
        const float q = 1.0f - radius_squared / h_squared;
        return poly6_scale * q * q * q;
    }

    FLUID_HD float spiky(float radius) const {
        if (radius < 0.0f || radius >= h) return 0.0f;
        const float q = 1.0f - radius * inverse_h;
        return spiky_scale * q * q * q;
    }

    // Gradient with respect to x_i for displacement x_i - x_j.
    // Direction is undefined at overlap; suppress pressure below epsilon = 1e-5 h.
    FLUID_HD Vec3 spiky_gradient(Vec3 displacement) const {
        const float radius_squared = dot(displacement, displacement);
        if (radius_squared <= h_squared * 1.0e-10f || radius_squared >= h_squared) return {};
        const float radius = sqrtf(radius_squared);
        const float q = 1.0f - radius * inverse_h;
        return displacement * (spiky_gradient_scale * q * q / radius);
    }

    FLUID_HD float viscosity_laplacian(float radius) const {
        if (radius < 0.0f || radius >= h) return 0.0f;
        return viscosity_scale * (1.0f - radius * inverse_h);
    }
};

// Call with a finite positive density and validated parameters. The CPU layer
// rejects nonfinite results instead of masking overflow or silently clipping forces.
FLUID_HD inline float tait_pressure(float density, float rest_density, float stiffness) {
    if (density <= rest_density || stiffness == 0.0f) return 0.0f;
    const float ratio = density / rest_density;
    const float ratio2 = ratio * ratio;
    const float ratio4 = ratio2 * ratio2;
    return stiffness * (ratio4 * ratio2 * ratio - 1.0f);
}

} // namespace fluid
