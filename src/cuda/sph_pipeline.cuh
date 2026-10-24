#pragma once

#include "fluid/sim_params.h"
#include "fluid/sph_kernels.cuh"
#include <cuda_runtime.h>

namespace fluid::cuda_detail {
struct DeviceParams {
    float mass, rho0, stiffness, viscosity, dt, density_floor, restitution;
    Vec3 lower, upper, gravity;
};
inline DeviceParams device_params(const SimParams& p) {
    return {p.particle_mass(), p.rest_density, p.stiffness, p.viscosity, p.dt,
            p.density_floor(), p.restitution, p.lower_bound(), p.upper_bound(), p.gravity};
}
__device__ inline Vec3 xyz(float4 value) { return {value.x, value.y, value.z}; }
__device__ inline float4 pack(Vec3 value) { return make_float4(value.x, value.y, value.z, 0.0f); }
__device__ inline bool finite(Vec3 value) { return isfinite(value.x) && isfinite(value.y) && isfinite(value.z); }
// One sticky status per solver, not force accumulation. First failing stage wins.
enum NumericalError { density_error = 1, pressure_error = 2, force_error = 3, integration_error = 4, grid_error = 5 };
__device__ inline void fail(int* status, NumericalError error) { atomicCAS(status, 0, static_cast<int>(error)); }

void launch_density(const float4* positions, float* density, float* pressure, int count,
                    DeviceParams params, SphKernels kernels, int* status, cudaStream_t stream);
void launch_forces(const float4* positions, const float4* velocities, const float* density,
                   const float* pressure, float4* acceleration, int count, DeviceParams params,
                   SphKernels kernels, int* status, cudaStream_t stream);
void launch_integrate(const float4* positions, const float4* velocities, const float4* acceleration,
                      float4* next_positions, float4* next_velocities, int count, DeviceParams params,
                      int* status, cudaStream_t stream);
} // namespace fluid::cuda_detail
