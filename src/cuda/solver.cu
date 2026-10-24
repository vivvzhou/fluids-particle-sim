#include "fluid/cuda_solver.h"
#include "fluid/initialization.h"
#include "cuda_resources.cuh"
#include "sph_pipeline.cuh"
#include <stdexcept>
#include <utility>
#include <vector>

namespace fluid {
namespace {
std::size_t validate(const SimParams& params, const ParticleState& state) {
    params.validate();
    if (state.positions.empty() || state.positions.size() > max_cpu_particles ||
        state.velocities.size() != state.positions.size())
        throw std::invalid_argument("CUDA all-pairs state needs matching arrays with 1-4096 particles");
    for (std::size_t i = 0; i < state.positions.size(); ++i) {
        if (!inside_bounds(state.positions[i], params) || !is_finite(state.velocities[i]))
            throw std::invalid_argument("invalid CUDA initial state at particle " + std::to_string(i));
    }
    return state.positions.size();
}
std::vector<float4> pack_host(const std::vector<Vec3>& input) {
    std::vector<float4> packed;
    packed.reserve(input.size());
    for (auto v : input) packed.push_back(make_float4(v.x, v.y, v.z, 0.0f));
    return packed;
}
std::vector<Vec3> unpack_host(const std::vector<float4>& input) {
    std::vector<Vec3> unpacked;
    unpacked.reserve(input.size());
    for (auto v : input) unpacked.push_back({v.x, v.y, v.z});
    return unpacked;
}
} // namespace

struct CudaSphSolver::Impl {
    SimParams params;
    std::size_t count;
    SphKernels kernels;
    cuda_detail::DeviceParams device_params;
    cuda_detail::DeviceBuffer<float4> positions_a, positions_b, velocities_a, velocities_b, accelerations;
    cuda_detail::DeviceBuffer<float> densities, pressures;
    cuda_detail::DeviceBuffer<int> status;
    cuda_detail::PinnedStatus host_status;
    cuda_detail::Event start, after_density, after_forces, finish;
    // Last owned CUDA resource: drained first, even if construction fails partway.
    cuda_detail::Stream stream;
    float4 *positions, *next_positions, *velocities, *next_velocities;
    int device = 0;
    bool failed = false;

    Impl(SimParams p, const ParticleState& initial)
        : params(p), count(validate(p, initial)), kernels(p.smoothing_radius), device_params(cuda_detail::device_params(p)),
          positions_a(count), positions_b(count), velocities_a(count), velocities_b(count), accelerations(count),
          densities(count), pressures(count), status(1),
          positions(positions_a.data), next_positions(positions_b.data),
          velocities(velocities_a.data), next_velocities(velocities_b.data) {
        FLUID_CUDA_CHECK(cudaGetDevice(&device));
        auto packed_positions = pack_host(initial.positions);
        auto packed_velocities = pack_host(initial.velocities);
        FLUID_CUDA_CHECK(cudaMemsetAsync(status.data, 0, sizeof(int), stream.value));
        // Blocking initialization uploads keep staging lifetimes safe even if a later copy throws.
        FLUID_CUDA_CHECK(cudaMemcpy(positions, packed_positions.data(), count * sizeof(float4), cudaMemcpyHostToDevice));
        FLUID_CUDA_CHECK(cudaMemcpy(velocities, packed_velocities.data(), count * sizeof(float4), cudaMemcpyHostToDevice));
        // Complete the status initialization before evaluating any fields.
        FLUID_CUDA_CHECK(cudaStreamSynchronize(stream.value));
    }

    void check() const {
        if (failed) throw std::runtime_error("CUDA solver is invalid after a numerical failure; construct a new solver");
        int active = 0;
        FLUID_CUDA_CHECK(cudaGetDevice(&active));
        if (active != device) throw std::runtime_error("CUDA solver must be used with its construction device current");
    }
    void density() {
        cuda_detail::launch_density(positions, densities.data, pressures.data, static_cast<int>(count), device_params, kernels, status.data, stream.value);
    }
    void force() {
        cuda_detail::launch_forces(positions, velocities, densities.data, pressures.data, accelerations.data,
                                  static_cast<int>(count), device_params, kernels, status.data, stream.value);
    }
    void integrate() {
        cuda_detail::launch_integrate(positions, velocities, accelerations.data, next_positions, next_velocities,
                                     static_cast<int>(count), device_params, status.data, stream.value);
        std::swap(positions, next_positions);
        std::swap(velocities, next_velocities);
    }
    void sync() {
        check();
        FLUID_CUDA_CHECK(cudaMemcpyAsync(host_status.data, status.data, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
        FLUID_CUDA_CHECK(cudaStreamSynchronize(stream.value));
        if (*host_status.data != 0) {
            failed = true;
            const char* stage = "unknown";
            switch (*host_status.data) {
                case cuda_detail::density_error: stage = "density"; break;
                case cuda_detail::pressure_error: stage = "pressure"; break;
                case cuda_detail::force_error: stage = "force"; break;
                case cuda_detail::integration_error: stage = "integration"; break;
            }
            throw std::runtime_error(std::string("CUDA numerical failure in ") + stage + "; discard this solver and check parameters/dt");
        }
    }
};

CudaSphSolver::CudaSphSolver(SimParams params, const ParticleState& initial)
    : impl_(std::make_unique<Impl>(params, initial)) {
    evaluate_fields();
    synchronize();
}
CudaSphSolver::~CudaSphSolver() = default;
std::size_t CudaSphSolver::particle_count() const { return impl_->count; }
const SimParams& CudaSphSolver::params() const { return impl_->params; }
void CudaSphSolver::evaluate_fields() { impl_->check(); impl_->density(); impl_->force(); }
void CudaSphSolver::step() { evaluate_fields(); impl_->integrate(); }
void CudaSphSolver::synchronize() { impl_->sync(); }

CudaStepTimings CudaSphSolver::step_timed() {
    auto& s = *impl_;
    s.check();
    FLUID_CUDA_CHECK(cudaEventRecord(s.start.value, s.stream.value));
    s.density();
    FLUID_CUDA_CHECK(cudaEventRecord(s.after_density.value, s.stream.value));
    s.force();
    FLUID_CUDA_CHECK(cudaEventRecord(s.after_forces.value, s.stream.value));
    s.integrate();
    FLUID_CUDA_CHECK(cudaEventRecord(s.finish.value, s.stream.value));
    FLUID_CUDA_CHECK(cudaEventSynchronize(s.finish.value));
    CudaStepTimings timings;
    FLUID_CUDA_CHECK(cudaEventElapsedTime(&timings.density_pressure_ms, s.start.value, s.after_density.value));
    FLUID_CUDA_CHECK(cudaEventElapsedTime(&timings.forces_ms, s.after_density.value, s.after_forces.value));
    FLUID_CUDA_CHECK(cudaEventElapsedTime(&timings.integrate_ms, s.after_forces.value, s.finish.value));
    FLUID_CUDA_CHECK(cudaEventElapsedTime(&timings.total_ms, s.start.value, s.finish.value));
    return timings; // Numerical status is read by explicit synchronize(), outside timing.
}

ParticleState CudaSphSolver::download_state() {
    auto& s = *impl_;
    s.sync();
    std::vector<float4> positions(s.count), velocities(s.count);
    // Blocking copies here are intentional: snapshots/readbacks are outside the hot path.
    FLUID_CUDA_CHECK(cudaMemcpy(positions.data(), s.positions, s.count * sizeof(float4), cudaMemcpyDeviceToHost));
    FLUID_CUDA_CHECK(cudaMemcpy(velocities.data(), s.velocities, s.count * sizeof(float4), cudaMemcpyDeviceToHost));
    return {unpack_host(positions), unpack_host(velocities)};
}
SphFields CudaSphSolver::download_fields() {
    auto& s = *impl_;
    s.sync();
    SphFields fields;
    fields.densities.resize(s.count); fields.pressures.resize(s.count);
    std::vector<float4> accelerations(s.count);
    FLUID_CUDA_CHECK(cudaMemcpy(fields.densities.data(), s.densities.data, s.count * sizeof(float), cudaMemcpyDeviceToHost));
    FLUID_CUDA_CHECK(cudaMemcpy(fields.pressures.data(), s.pressures.data, s.count * sizeof(float), cudaMemcpyDeviceToHost));
    FLUID_CUDA_CHECK(cudaMemcpy(accelerations.data(), s.accelerations.data, s.count * sizeof(float4), cudaMemcpyDeviceToHost));
    fields.accelerations = unpack_host(accelerations);
    return fields;
}
CudaDeviceMetadata CudaSphSolver::device_info() const {
    impl_->check();
    cudaDeviceProp properties{};
    FLUID_CUDA_CHECK(cudaGetDeviceProperties(&properties, impl_->device));
    CudaDeviceMetadata info;
    info.name = properties.name; info.major = properties.major; info.minor = properties.minor;
    FLUID_CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
    FLUID_CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
    info.compiled_runtime_version = CUDART_VERSION;
    return info;
}
} // namespace fluid
