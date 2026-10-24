#pragma once

#include "cuda_resources.cuh"
#include "sph_pipeline.cuh"
#include <cstdint>

namespace fluid::cuda_detail {
// Dense range table guard: two int arrays use at most 32 MiB.
inline constexpr std::size_t max_grid_cells = 4 * 1024 * 1024;
struct CellCoord { int x, y, z; };
struct GridGeometry {
    Vec3 origin, upper;
    float width;
    int nx, ny, nz;
    std::size_t cell_count;
    static GridGeometry make(const SimParams& params);

    // Double division avoids reciprocal rounding at exact cell boundaries. Inputs
    // must be finite; closed upper domain faces map to the final valid cell.
    __device__ int axis(float position, float minimum, int size) const {
        const double cell = floor((static_cast<double>(position) - minimum) / width);
        return cell < 0 ? 0 : (cell >= size ? size - 1 : static_cast<int>(cell));
    }
    __device__ CellCoord coordinates(Vec3 p) const {
        return {axis(p.x, origin.x, nx), axis(p.y, origin.y, ny), axis(p.z, origin.z, nz)};
    }
    __device__ bool contains(CellCoord c) const {
        return c.x >= 0 && c.x < nx && c.y >= 0 && c.y < ny && c.z >= 0 && c.z < nz;
    }
    __device__ std::uint32_t key(CellCoord c) const {
        return static_cast<std::uint32_t>(c.x + nx * (c.y + ny * c.z));
    }
};

struct GridNeighbors {
    GridGeometry geometry;
    const int *ids, *start, *end;
    CellCoord center;
    int offset = 0, cursor = 0, limit = 0;
    __device__ int next() {
        while (cursor >= limit) {
            if (offset == 27) return -1;
            const int n = offset++;
            const CellCoord cell{center.x + n % 3 - 1, center.y + (n / 3) % 3 - 1, center.z + n / 9 - 1};
            if (!geometry.contains(cell)) continue;
            const auto key = geometry.key(cell);
            if (start[key] < 0) continue;
            cursor = start[key]; limit = end[key];
        }
        return ids[cursor++];
    }
};
struct GridView {
    GridGeometry geometry;
    const int *ids, *start, *end;
    __device__ GridNeighbors neighbors(Vec3 position) const {
        return {geometry, ids, start, end, geometry.coordinates(position)};
    }
};
struct AllPairsNeighbors {
    int count, cursor = 0;
    __device__ int next() { return cursor < count ? cursor++ : -1; }
};
struct AllPairsView {
    int count;
    __device__ AllPairsNeighbors neighbors(Vec3) const { return {count}; }
};

// Internal module; owner must drain its stream before destroying these resources.
// Physical particle arrays are never reordered. Empty ranges are always [-1,-1).
struct SpatialGrid {
    GridGeometry geometry;
    int count;
    DeviceBuffer<std::uint32_t> keys_in, keys_out;
    DeviceBuffer<int> ids_in, ids_out, cell_start, cell_end;
    std::size_t scratch_bytes;
    DeviceBuffer<unsigned char> scratch;
    SpatialGrid(const SimParams& params, int particle_count);
    void build(const float4* positions, int* status, cudaStream_t stream);
    GridView view() const { return {geometry, ids_out.data, cell_start.data, cell_end.data}; }
};

void launch_grid_density(const float4* positions, float* density, float* pressure, int count,
                         DeviceParams params, SphKernels kernels, int* status, GridView grid, cudaStream_t stream);
void launch_grid_forces(const float4* positions, const float4* velocities, const float* density,
                        const float* pressure, float4* acceleration, int count, DeviceParams params,
                        SphKernels kernels, int* status, GridView grid, cudaStream_t stream);
} // namespace fluid::cuda_detail
