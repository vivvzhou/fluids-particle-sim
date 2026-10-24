#include "spatial_grid.cuh"
#include "fluid/cuda_solver.h"
#include "fluid/initialization.h"
#include <cub/device/device_radix_sort.cuh>
#include <cmath>
#include <stdexcept>

namespace fluid::cuda_detail {
namespace {
int dimension(float low, float high, float width) {
    const double n = std::ceil((static_cast<double>(high) - low) / width);
    if (!std::isfinite(n) || n < 1 || n > static_cast<double>(max_grid_cells))
        throw std::invalid_argument("grid dimension exceeds the M3 dense-grid limit; increase h or reduce the box");
    return static_cast<int>(n);
}
int validate_count(int count) {
    if (count < 1 || count > static_cast<int>(max_grid_particles))
        throw std::invalid_argument("M3 grid requires 1-50000 particles");
    return count;
}
std::size_t sort_storage(const SpatialGrid& grid) {
    std::size_t bytes = 0;
    FLUID_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, grid.keys_in.data, grid.keys_out.data,
                                                   grid.ids_in.data, grid.ids_out.data, grid.count));
    return bytes;
}
__global__ void build_keys(const float4* positions, std::uint32_t* keys, int* ids, int count,
                           GridGeometry geometry, int* status) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    const auto p = xyz(positions[i]);
    ids[i] = static_cast<int>(i);
    if (!finite(p) || p.x < geometry.origin.x || p.y < geometry.origin.y || p.z < geometry.origin.z ||
        p.x > geometry.upper.x || p.y > geometry.upper.y || p.z > geometry.upper.z) {
        fail(status, grid_error); keys[i] = 0; return;
    }
    keys[i] = geometry.key(geometry.coordinates(p));
}
__global__ void build_ranges(const std::uint32_t* keys, int* start, int* end, int count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    const auto key = keys[i];
    // Exactly one writer per start/end. No block-local synchronization is needed.
    if (i == 0 || keys[i - 1] != key) start[key] = static_cast<int>(i);
    if (i + 1 == static_cast<unsigned>(count) || keys[i + 1] != key) end[key] = static_cast<int>(i + 1);
}
} // namespace

GridGeometry GridGeometry::make(const SimParams& p) {
    p.validate();
    GridGeometry g{p.box_min, p.box_max, p.smoothing_radius,
                   dimension(p.box_min.x, p.box_max.x, p.smoothing_radius),
                   dimension(p.box_min.y, p.box_max.y, p.smoothing_radius),
                   dimension(p.box_min.z, p.box_max.z, p.smoothing_radius), 0};
    // Division guards before multiplication avoid overflow even for hostile input.
    std::size_t cells = static_cast<std::size_t>(g.nx);
    if (cells > max_grid_cells / static_cast<std::size_t>(g.ny))
        throw std::invalid_argument("grid exceeds 4194304 cells; increase h or reduce the box");
    cells *= static_cast<std::size_t>(g.ny);
    if (cells > max_grid_cells / static_cast<std::size_t>(g.nz))
        throw std::invalid_argument("grid exceeds 4194304 cells; increase h or reduce the box");
    g.cell_count = cells * static_cast<std::size_t>(g.nz);
    return g;
}

SpatialGrid::SpatialGrid(const SimParams& params, int particle_count)
    : geometry(GridGeometry::make(params)), count(validate_count(particle_count)),
      keys_in(count), keys_out(count), ids_in(count), ids_out(count),
      cell_start(geometry.cell_count), cell_end(geometry.cell_count),
      scratch_bytes(sort_storage(*this)), scratch(scratch_bytes) {}

void SpatialGrid::build(const float4* positions, int* status, cudaStream_t stream) {
    constexpr auto block = CudaSphSolver::block_size;
    build_keys<<<(count + block - 1) / block, block, 0, stream>>>(positions, keys_in.data, ids_in.data, count, geometry, status);
    FLUID_CUDA_CHECK_LAUNCH(stream);
    FLUID_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(scratch.data, scratch_bytes, keys_in.data, keys_out.data,
                                                   ids_in.data, ids_out.data, count, 0, 32, stream));
    FLUID_CUDA_CHECK_LAUNCH(stream);
    FLUID_CUDA_CHECK(cudaMemsetAsync(cell_start.data, 0xff, geometry.cell_count * sizeof(int), stream));
    FLUID_CUDA_CHECK(cudaMemsetAsync(cell_end.data, 0xff, geometry.cell_count * sizeof(int), stream));
    build_ranges<<<(count + block - 1) / block, block, 0, stream>>>(keys_out.data, cell_start.data, cell_end.data, count);
    FLUID_CUDA_CHECK_LAUNCH(stream);
}
} // namespace fluid::cuda_detail
