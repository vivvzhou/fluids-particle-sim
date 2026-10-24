#include "sph_pipeline.cuh"
#include "spatial_grid.cuh"
#include "fluid/cuda_check.h"
#include "fluid/cuda_solver.h"

namespace fluid::cuda_detail {
namespace {
template<class Search>
__global__ void forces(const float4* positions, const float4* velocities, const float* densities,
                       const float* pressures, float4* accelerations, int count,
                       DeviceParams p, SphKernels kernels, int* status, Search search) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    Vec3 acceleration = p.gravity;
    const auto position = xyz(positions[i]);
    const auto velocity = xyz(velocities[i]);
    const float rho_i = fmaxf(densities[i], p.density_floor);
    const float pressure_i = (pressures[i] / rho_i) / rho_i;
    auto neighbors = search.neighbors(position);
    for (int j = neighbors.next(); j >= 0; j = neighbors.next()) {
        if (i == static_cast<unsigned>(j)) continue;
        const auto d = position - xyz(positions[j]);
        const float r2 = dot(d, d);
        if (r2 >= kernels.h_squared) continue;
        const float rho_j = fmaxf(densities[j], p.density_floor);
        const float pressure_j = (pressures[j] / rho_j) / rho_j;
        acceleration = acceleration + kernels.spiky_gradient(d) * (-p.mass * (pressure_i + pressure_j));
        const float weight = p.viscosity * (p.mass / rho_j) * kernels.viscosity_laplacian(sqrtf(r2));
        acceleration = acceleration + (xyz(velocities[j]) - velocity) * weight;
    }
    if (!finite(acceleration)) { fail(status, force_error); acceleration = {}; }
    accelerations[i] = pack(acceleration);
}
} // namespace

void launch_forces(const float4* positions, const float4* velocities, const float* density,
                   const float* pressure, float4* acceleration, int count, DeviceParams params,
                   SphKernels kernels, int* status, cudaStream_t stream) {
    constexpr auto block = CudaSphSolver::block_size;
    forces<<<(count + block - 1) / block, block, 0, stream>>>(positions, velocities, density, pressure, acceleration,
                                                          count, params, kernels, status, AllPairsView{count});
    FLUID_CUDA_CHECK_LAUNCH(stream);
}
void launch_grid_forces(const float4* positions, const float4* velocities, const float* density,
                        const float* pressure, float4* acceleration, int count, DeviceParams params,
                        SphKernels kernels, int* status, GridView grid, cudaStream_t stream) {
    constexpr auto block = CudaSphSolver::block_size;
    forces<<<(count + block - 1) / block, block, 0, stream>>>(positions, velocities, density, pressure, acceleration,
                                                          count, params, kernels, status, grid);
    FLUID_CUDA_CHECK_LAUNCH(stream);
}
} // namespace fluid::cuda_detail
