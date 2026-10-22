#pragma once

#include "fluid/particle_state.h"
#include "fluid/sim_params.h"
#include "fluid/sph_kernels.cuh"
#include <cstddef>

namespace fluid {

// Separate helper for direct boundary tests; params must already be validated.
void collide_with_box(Vec3& position, Vec3& velocity, const SimParams& params);

class CpuSphSolver {
public:
    CpuSphSolver(SimParams params, ParticleState state);
    void evaluate_fields(); // Density -> EOS -> acceleration for current positions/velocities.
    void step();            // Evaluate at t, integrate into scratch, then swap state to t + dt.

    const ParticleState& state() const { return state_; }
    const SimParams& params() const { return params_; }
    // Fields describe the last evaluation (t before the most recent step).
    // Call evaluate_fields() explicitly if fields at the updated positions are needed.
    const SphFields& fields() const { return fields_; }
    std::size_t particle_count() const { return state_.positions.size(); }

private:
    SimParams params_;
    SphKernels kernels_;
    ParticleState state_;
    ParticleState next_;
    SphFields fields_;
};
} // namespace fluid
