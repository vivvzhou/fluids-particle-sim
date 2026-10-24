#include "fluid/cpu_reference.h"
#include "fluid/boundary_math.cuh"
#include "fluid/initialization.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace fluid {
namespace {
SimParams validated_params(SimParams params) {
    params.validate();
    return params;
}

void validate_state(const ParticleState& state, const SimParams& params) {
    if (state.positions.empty() || state.positions.size() > max_cpu_particles ||
        state.positions.size() != state.velocities.size())
        throw std::invalid_argument("state needs matching position/velocity arrays with 1-4096 particles");
    for (std::size_t i = 0; i < state.positions.size(); ++i) {
        if (!inside_bounds(state.positions[i], params) || !is_finite(state.velocities[i]))
            throw std::invalid_argument("initial particle " + std::to_string(i) + " is invalid or outside padded box");
    }
}
} // namespace

void collide_with_box(Vec3& position, Vec3& velocity, const SimParams& params) {
    collide_bounds(position, velocity, params.lower_bound(), params.upper_bound(), params.restitution);
}

CpuSphSolver::CpuSphSolver(SimParams params, ParticleState state)
    : params_(validated_params(params)), kernels_(params_.smoothing_radius), state_(std::move(state)) {
    validate_state(state_, params_);
    const auto count = particle_count();
    next_.positions.resize(count);
    next_.velocities.resize(count);
    fields_.densities.resize(count);
    fields_.pressures.resize(count);
    fields_.accelerations.resize(count);
    evaluate_fields();
}

void CpuSphSolver::evaluate_fields() {
    const auto count = particle_count();
    const float mass = params_.particle_mass();
    for (std::size_t i = 0; i < count; ++i) {
        float density = 0.0f;
        for (std::size_t j = 0; j < count; ++j) {
            const auto displacement = state_.positions[i] - state_.positions[j];
            density += mass * kernels_.poly6(dot(displacement, displacement)); // Includes self.
        }
        if (!std::isfinite(density) || density <= 0.0f)
            throw std::runtime_error("invalid density for particle " + std::to_string(i));
        fields_.densities[i] = density;
        fields_.pressures[i] = tait_pressure(density, params_.rest_density, params_.stiffness);
        if (!std::isfinite(fields_.pressures[i]))
            throw std::runtime_error("pressure overflow; reduce dt/stiffness or correct parameters");
    }

    for (std::size_t i = 0; i < count; ++i) {
        Vec3 acceleration = params_.gravity;
        const float rho_i = std::max(fields_.densities[i], params_.density_floor());
        const float pressure_i = (fields_.pressures[i] / rho_i) / rho_i;
        for (std::size_t j = 0; j < count; ++j) {
            if (i == j) continue;
            const auto displacement = state_.positions[i] - state_.positions[j];
            const float distance_squared = dot(displacement, displacement);
            if (distance_squared >= kernels_.h_squared) continue;
            const float rho_j = std::max(fields_.densities[j], params_.density_floor());
            const float pressure_j = (fields_.pressures[j] / rho_j) / rho_j;
            acceleration = acceleration + kernels_.spiky_gradient(displacement) * (-mass * (pressure_i + pressure_j));
            const float viscous_weight = params_.viscosity * (mass / rho_j) *
                                          kernels_.viscosity_laplacian(std::sqrt(distance_squared));
            acceleration = acceleration + (state_.velocities[j] - state_.velocities[i]) * viscous_weight;
        }
        if (!is_finite(acceleration)) throw std::runtime_error("nonfinite acceleration for particle " + std::to_string(i));
        fields_.accelerations[i] = acceleration;
    }
}

void CpuSphSolver::step() {
    evaluate_fields();
    for (std::size_t i = 0; i < particle_count(); ++i) {
        auto velocity = state_.velocities[i] + fields_.accelerations[i] * params_.dt;
        auto position = state_.positions[i] + velocity * params_.dt; // Semi-implicit Euler.
        if (!is_finite(position) || !is_finite(velocity))
            throw std::runtime_error("integration overflow for particle " + std::to_string(i));
        collide_with_box(position, velocity, params_);
        next_.positions[i] = position;
        next_.velocities[i] = velocity;
    }
    state_.positions.swap(next_.positions);
    state_.velocities.swap(next_.velocities);
}
} // namespace fluid
