#include "sph_pipeline.cuh"
#include "fluid/cuda_check.h"
#include "fluid/cuda_solver.h"

namespace fluid::cuda_detail {
namespace {
__global__ void density_pressure(const float4* positions, float* densities, float* pressures,
                                int count, DeviceParams p, SphKernels kernels, int* status) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    const auto position = xyz(positions[i]);
    float density = 0.0f;
    for (int j = 0; j < count; ++j) {
        const auto d = position - xyz(positions[j]);
        density += p.mass * kernels.poly6(dot(d, d)); // Includes self.
    }
    if (!isfinite(density) || density <= 0.0f) {
        fail(status, density_error);
        densities[i] = 0.0f; pressures[i] = 0.0f;
        return;
    }
    const float pressure = tait_pressure(density, p.rho0, p.stiffness);
    densities[i] = density;
    pressures[i] = isfinite(pressure) ? pressure : 0.0f;
    if (!isfinite(pressure)) fail(status, pressure_error);
}
} // namespace

void launch_density(const float4* positions, float* density, float* pressure, int count,
                    DeviceParams params, SphKernels kernels, int* status, cudaStream_t stream) {
    constexpr auto block = CudaSphSolver::block_size;
    density_pressure<<<(count + block - 1) / block, block, 0, stream>>>(positions, density, pressure, count, params, kernels, status);
    FLUID_CUDA_CHECK_LAUNCH(stream);
}
} // namespace fluid::cuda_detail
