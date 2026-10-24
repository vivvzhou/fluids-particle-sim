#pragma once

#include "fluid/particle_state.h"
#include "fluid/sim_params.h"
#include <cstddef>
#include <memory>
#include <string>

namespace fluid {

enum class CudaNeighborSearch { all_pairs, uniform_grid };

struct CudaStepTimings {
    float grid_ms = 0; // Key generation, sort and range reset/build; zero for all-pairs.
    float density_pressure_ms = 0;
    float forces_ms = 0;
    float integrate_ms = 0;
    float total_ms = 0;
};

struct CudaDeviceMetadata {
    std::string name;
    int major = 0, minor = 0;
    int runtime_version = 0, driver_version = 0, compiled_runtime_version = 0;
};

// M2 all-pairs (1-4096) or M3 uniform-grid solver (1-50000). Public headers contain
// no CUDA types. Owns its stream and all buffers; not copyable/movable/thread-safe.
// Keep the construction device current throughout the solver's lifetime.
class CudaSphSolver {
public:
    static constexpr unsigned block_size = 128;
    CudaSphSolver(SimParams params, const ParticleState& initial,
                  CudaNeighborSearch search = CudaNeighborSearch::all_pairs);
    ~CudaSphSolver();
    CudaSphSolver(const CudaSphSolver&) = delete;
    CudaSphSolver& operator=(const CudaSphSolver&) = delete;

    void evaluate_fields(); // Enqueues evaluation for current state.
    void step();            // Enqueues fields(t), integration(t+dt), then swaps A/B handles.
    void synchronize();     // Surfaces async CUDA and sticky numerical failures; call before teardown.
    // Waits for preallocated timing events without state/status readback. Call
    // synchronize() after sampling to surface sticky numerical failures.
    CudaStepTimings step_timed();

    ParticleState download_state();
    SphFields download_fields(); // Last evaluation, same time convention as CPU fields().
    std::size_t particle_count() const;
    const SimParams& params() const;
    CudaNeighborSearch neighbor_search() const;
    CudaDeviceMetadata device_info() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace fluid
