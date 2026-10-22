#include "fluid/sim_params.h"
#include "fluid/sph_kernels.cuh"
#include <cmath>
#include <stdexcept>
#include <string>

namespace fluid {
namespace {
void positive(float value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0f)
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
}
void nonnegative(float value, const char* name) {
    if (!std::isfinite(value) || value < 0.0f)
        throw std::invalid_argument(std::string(name) + " must be finite and nonnegative");
}
} // namespace

void SimParams::validate() const {
    positive(spacing, "spacing");
    positive(smoothing_radius, "smoothing radius");
    positive(rest_density, "rest density");
    positive(dt, "dt");
    nonnegative(stiffness, "stiffness");
    nonnegative(viscosity, "viscosity");
    nonnegative(boundary_padding, "boundary padding");
    nonnegative(restitution, "restitution");
    if (restitution > 1.0f) throw std::invalid_argument("restitution must be in [0, 1]");
    if (!is_finite(box_min) || !is_finite(box_max) || !is_finite(gravity))
        throw std::invalid_argument("box and gravity must be finite");
    const auto low = lower_bound();
    const auto high = upper_bound();
    if (!is_finite(low) || !is_finite(high) || low.x >= high.x || low.y >= high.y || low.z >= high.z)
        throw std::invalid_argument("padded box must have positive extent on every axis");
    positive(dot(high - low, high - low), "squared box diagonal");
    positive(particle_mass(), "derived particle mass");
    positive(density_floor(), "density floor");
    const SphKernels kernels(smoothing_radius);
    positive(kernels.h_squared, "squared smoothing radius");
    positive(kernels.poly6_scale, "Poly6 coefficient");
    positive(-kernels.spiky_gradient_scale, "Spiky gradient coefficient");
    positive(kernels.viscosity_scale, "viscosity coefficient");
    positive(particle_mass() * kernels.poly6(0.0f), "self density");
}

bool inside_bounds(Vec3 position, const SimParams& params) {
    const auto low = params.lower_bound();
    const auto high = params.upper_bound();
    return is_finite(position) && position.x >= low.x && position.x <= high.x &&
           position.y >= low.y && position.y <= high.y && position.z >= low.z && position.z <= high.z;
}
} // namespace fluid
