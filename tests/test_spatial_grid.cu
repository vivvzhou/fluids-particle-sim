#include "spatial_grid.cuh"
#include "test_support.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using namespace fluid;
using namespace fluid::cuda_detail;

namespace {
// Exercise the exact traversal used by both physics kernels, with an independent
// all-pairs support mask. Counts (not booleans) detect repeated visits as well.
__global__ void neighbor_masks(const float4* positions, int count, float h2, GridView grid, int* actual, int* expected) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    const auto p = xyz(positions[i]);
    auto neighbors = grid.neighbors(p);
    for (int j = neighbors.next(); j >= 0; j = neighbors.next()) {
        const auto d = p - xyz(positions[j]);
        if (dot(d, d) < h2) ++actual[i * count + j];
    }
    for (int j = 0; j < count; ++j) {
        const auto d = p - xyz(positions[j]);
        expected[i * count + j] = dot(d, d) < h2 ? 1 : 0;
    }
}
template<class T> std::vector<T> download(const DeviceBuffer<T>& buffer, std::size_t count) {
    std::vector<T> host(count);
    FLUID_CUDA_CHECK(cudaMemcpy(host.data(), buffer.data, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}
std::vector<float4> random_positions(std::size_t count, const SimParams& p, unsigned seed) {
    std::mt19937 random(seed);
    std::vector<float4> result;
    auto coordinate = [&](float lo, float hi) {
        const float u = static_cast<float>(random() >> 8) / 16777216.0f;
        return lo + (hi - lo) * u;
    };
    for (std::size_t i = 0; i < count; ++i) {
        const float x = coordinate(p.box_min.x, p.box_max.x);
        const float y = coordinate(p.box_min.y, p.box_max.y);
        const float z = coordinate(p.box_min.z, p.box_max.z);
        result.push_back(make_float4(x, y, z, 0));
    }
    return result;
}
void verify(SpatialGrid& grid, DeviceBuffer<float4>& positions, DeviceBuffer<int>& status,
            DeviceBuffer<int>& actual, DeviceBuffer<int>& expected, Stream& stream, const std::vector<float4>& host) {
    const auto n = host.size();
    FLUID_CUDA_CHECK(cudaMemcpy(positions.data, host.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
    FLUID_CUDA_CHECK(cudaMemsetAsync(status.data, 0, sizeof(int), stream.value));
    grid.build(positions.data, status.data, stream.value);
    FLUID_CUDA_CHECK(cudaMemsetAsync(actual.data, 0, n * n * sizeof(int), stream.value));
    neighbor_masks<<<(n + 127) / 128, 128, 0, stream.value>>>(positions.data, static_cast<int>(n),
        grid.geometry.width * grid.geometry.width, grid.view(), actual.data, expected.data);
    FLUID_CUDA_CHECK_LAUNCH(stream.value);
    FLUID_CUDA_CHECK(cudaStreamSynchronize(stream.value));
    CHECK(download(status, 1)[0] == 0);
    const auto hits = download(actual, n * n), brute = download(expected, n * n);
    for (std::size_t i = 0; i < hits.size(); ++i) {
        if (hits[i] != brute[i]) throw std::runtime_error("neighbor mismatch at i=" + std::to_string(i / n) + " j=" + std::to_string(i % n));
    }
    const auto keys = download(grid.keys_out, n);
    const auto ids = download(grid.ids_out, n);
    CHECK(std::is_sorted(keys.begin(), keys.end()));
    auto sorted_ids = ids; std::sort(sorted_ids.begin(), sorted_ids.end());
    for (std::size_t i = 0; i < n; ++i) CHECK(sorted_ids[i] == static_cast<int>(i));
    const auto starts = download(grid.cell_start, grid.geometry.cell_count);
    const auto ends = download(grid.cell_end, grid.geometry.cell_count);
    std::vector<int> expected_start(starts.size(), -1), expected_end(ends.size(), -1);
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(keys[i] < starts.size());
        if (i == 0 || keys[i - 1] != keys[i]) expected_start[keys[i]] = static_cast<int>(i);
        else CHECK(ids[i - 1] < ids[i]); // Stable sort of initially ascending IDs.
        expected_end[keys[i]] = static_cast<int>(i + 1);
        // Independent host cell mapping, including closed upper faces.
        const auto value = host[ids[i]];
        const auto& g = grid.geometry;
        auto axis = [&](float x, float origin, int size) {
            return std::clamp(static_cast<int>(std::floor((static_cast<double>(x) - origin) / g.width)), 0, size - 1);
        };
        const int x = axis(value.x, g.origin.x, g.nx), y = axis(value.y, g.origin.y, g.ny), z = axis(value.z, g.origin.z, g.nz);
        CHECK(keys[i] == static_cast<unsigned>(x + g.nx * (y + g.ny * z)));
    }
    CHECK(starts == expected_start); CHECK(ends == expected_end);
}
void fixture(const SimParams& p, std::vector<float4> host, bool move = false) {
    const auto n = host.size();
    SpatialGrid grid(p, static_cast<int>(n));
    CHECK(grid.scratch_bytes > 0);
    DeviceBuffer<float4> positions(n);
    DeviceBuffer<int> status(1), actual(n * n), expected(n * n);
    Stream stream; // Destroy/drain before all buffers.
    verify(grid, positions, status, actual, expected, stream, host);
    if (move) {
        // All particles leave previously occupied cells, then cross the entire box.
        for (auto& x : host) x = make_float4(p.box_min.x, p.box_min.y, p.box_min.z, 0);
        verify(grid, positions, status, actual, expected, stream, host);
        for (auto& x : host) x = make_float4(p.box_max.x, p.box_max.y, p.box_max.z, 0);
        verify(grid, positions, status, actual, expected, stream, host);
    }
}
void random_cases() {
    SimParams p; p.boundary_padding = 0; p.smoothing_radius = 0.125f;
    for (int n : {128, 257, 512}) for (unsigned seed : {0u, 42u, 991u})
        fixture(p, random_positions(n, p, seed), n == 257 && seed == 42);
}
void edges() {
    SimParams p; p.boundary_padding = 0; p.smoothing_radius = 0.125f;
    fixture(p, {make_float4(0, 0, 0, 0)});
    auto host = random_positions(129, p, 42);
    const float edge = 0.5f;
    host[0] = make_float4(edge, edge, edge, 0);
    host[1] = host[0]; // Overlap.
    host[2] = make_float4(std::nextafter(edge, 0.0f), edge, edge, 0);
    host[3] = make_float4(std::nextafter(edge, 1.0f), edge, edge, 0);
    host[4] = make_float4(edge + p.smoothing_radius, edge, edge, 0); // Exactly h, excluded.
    host[5] = make_float4(std::nextafter(edge + p.smoothing_radius, edge), edge, edge, 0);
    host[6] = make_float4(edge, std::nextafter(edge, 0.0f), std::nextafter(edge, 1.0f), 0);
    for (int corner = 0; corner < 8; ++corner)
        host[7 + corner] = make_float4(float(corner & 1), float((corner >> 1) & 1), float((corner >> 2) & 1), 0);
    fixture(p, host, true);
    p.box_min = {-0.35f, 0.2f, -0.5f}; p.box_max = {0.55f, 1.27f, 0.13f}; p.smoothing_radius = 0.13f;
    auto shifted = random_positions(257, p, 991);
    shifted[0] = make_float4(p.box_min.x, p.box_min.y, p.box_min.z, 0);
    shifted[1] = make_float4(p.box_max.x, p.box_max.y, p.box_max.z, 0);
    fixture(p, shifted, true); // Noninteger domain / cell ratio; negative world origin.
    p = {}; p.smoothing_radius = 2; p.boundary_padding = 0;
    fixture(p, random_positions(257, p, 0), true); // One cell on all three axes.
    p.box_max = {0.1f, 1.0f, 1.0f}; p.smoothing_radius = 0.25f;
    fixture(p, random_positions(128, p, 0)); // One thin axis, adjacent cells on the others.
}
void invalid() {
    SimParams p;
    throws<std::invalid_argument>([&] { SpatialGrid grid(p, 0); });
    throws<std::invalid_argument>([&] { SpatialGrid grid(p, 50001); });
    p.smoothing_radius = 0.001f;
    throws<std::invalid_argument>([&] { SpatialGrid grid(p, 1); }); // 1e9-cell dense table rejected before allocation.
    p.box_max.x = 10000;
    throws<std::invalid_argument>([&] { GridGeometry::make(p); }); // Single-axis guard.
    p = {}; p.boundary_padding = 0;
    SpatialGrid grid(p, 1);
    DeviceBuffer<float4> positions(1); DeviceBuffer<int> status(1);
    Stream stream;
    for (float x : {-0.01f, 1.01f, std::numeric_limits<float>::quiet_NaN()}) {
        const auto bad = make_float4(x, 0.5f, 0.5f, 0);
        FLUID_CUDA_CHECK(cudaMemcpy(positions.data, &bad, sizeof(float4), cudaMemcpyHostToDevice));
        FLUID_CUDA_CHECK(cudaMemsetAsync(status.data, 0, sizeof(int), stream.value));
        grid.build(positions.data, status.data, stream.value);
        FLUID_CUDA_CHECK(cudaStreamSynchronize(stream.value));
        CHECK(download(status, 1)[0] == grid_error);
        CHECK(download(grid.keys_out, 1)[0] == 0); // Safe fallback; never an invalid range address.
    }
}
} // namespace

int main(int argc, char** argv) {
    int count = 0;
    const auto result = cudaGetDeviceCount(&count);
    if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver || (result == cudaSuccess && count == 0)) return 77;
    return run_test([&] {
        FLUID_CUDA_CHECK(result); CHECK(argc == 2);
        const std::string name = argv[1];
        if (name == "random") random_cases();
        else if (name == "edges") edges();
        else if (name == "invalid") invalid();
        else if (name == "sanitizer") { random_cases(); edges(); invalid(); }
        else throw std::invalid_argument("unknown grid test");
        std::cout << "spatial grid " << name << ": exact neighbor masks, ranges and IDs passed\n";
    });
}
