#include "sph_pipeline.cuh"
#include "fluid/boundary_math.cuh"
#include "fluid/cuda_check.h"
#include "fluid/cuda_solver.h"

namespace fluid::cuda_detail {
namespace {
__global__ void integrate(const float4* positions, const float4* velocities, const float4* accelerations,
                          float4* next_positions, float4* next_velocities, int count, DeviceParams p, int* status) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    auto velocity = xyz(velocities[i]) + xyz(accelerations[i]) * p.dt;
    auto position = xyz(positions[i]) + velocity * p.dt;
    if (!finite(position) || !finite(velocity)) {
        fail(status, integration_error);
        next_positions[i] = positions[i]; next_velocities[i] = velocities[i];
        return;
    }
    collide_bounds(position, velocity, p.lower, p.upper, p.restitution);
    next_positions[i] = pack(position);
    next_velocities[i] = pack(velocity);
}
} // namespace

void launch_integrate(const float4* positions, const float4* velocities, const float4* acceleration,
                      float4* next_positions, float4* next_velocities, int count, DeviceParams params,
                      int* status, cudaStream_t stream) {
    constexpr auto block = CudaSphSolver::block_size;
    integrate<<<(count + block - 1) / block, block, 0, stream>>>(positions, velocities, acceleration,
                                                             next_positions, next_velocities, count, params, status);
    FLUID_CUDA_CHECK_LAUNCH(stream);
}
} // namespace fluid::cuda_detail
