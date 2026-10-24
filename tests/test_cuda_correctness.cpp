#include "fluid/cpu_reference.h"
#include "fluid/cuda_solver.h"
#include "fluid/cuda_check.h"
#include "fluid/initialization.h"
#include "test_support.h"
#include <array>
#include <limits>
#include <memory>
#include <string>

namespace {
fluid::CudaNeighborSearch search = fluid::CudaNeighborSearch::all_pairs;
struct ErrorStats {
    const char* name;
    double absolute_tolerance;
    double maximum_error = 0, maximum_budget_fraction = 0;
    void compare(float gpu, float cpu, std::size_t i) {
        const double error = std::abs(static_cast<double>(gpu) - cpu);
        const double budget = absolute_tolerance + 1.0e-4 * std::abs(cpu);
        if (!std::isfinite(gpu) || !std::isfinite(cpu) || error > budget)
            throw std::runtime_error(std::string(name) + " particle " + std::to_string(i) +
                " CPU=" + std::to_string(cpu) + " GPU=" + std::to_string(gpu) + " error=" +
                std::to_string(error) + " tolerance=" + std::to_string(budget));
        maximum_error = std::max(maximum_error, error);
        maximum_budget_fraction = std::max(maximum_budget_fraction, error / budget);
    }
    void compare(fluid::Vec3 gpu, fluid::Vec3 cpu, std::size_t i) {
        compare(gpu.x, cpu.x, i); compare(gpu.y, cpu.y, i); compare(gpu.z, cpu.z, i);
    }
};
struct Comparison {
    std::array<ErrorStats, 5> errors{{{"position", 1.0e-5}, {"velocity", 1.0e-4},
                                   {"density", 1.0e-2}, {"pressure", 1.0}, {"acceleration", 1.0e-2}}};
    void compare(fluid::CudaSphSolver& gpu, const fluid::ParticleState& expected, const fluid::SphFields& expected_fields) {
        auto state = gpu.download_state();
        auto fields = gpu.download_fields();
        CHECK(state.positions.size() == expected.positions.size());
        for (std::size_t i = 0; i < expected.positions.size(); ++i) {
            errors[0].compare(state.positions[i], expected.positions[i], i);
            errors[1].compare(state.velocities[i], expected.velocities[i], i);
            errors[2].compare(fields.densities[i], expected_fields.densities[i], i);
            errors[3].compare(fields.pressures[i], expected_fields.pressures[i], i);
            errors[4].compare(fields.accelerations[i], expected_fields.accelerations[i], i);
            CHECK(fluid::inside_bounds(state.positions[i], gpu.params()));
        }
    }
    void compare(fluid::CudaSphSolver& gpu, const fluid::CpuSphSolver& cpu) { compare(gpu, cpu.state(), cpu.fields()); }
    void report() const {
        for (const auto& e : errors)
            std::cout << e.name << " max_abs_error=" << e.maximum_error << " max_tolerance_fraction=" << e.maximum_budget_fraction << '\n';
    }
};

void compare_steps(const fluid::SimParams& p, const fluid::ParticleState& initial, Comparison& comparison) {
    fluid::CpuSphSolver cpu(p, initial);
    fluid::CudaSphSolver gpu(p, initial, search);
    std::unique_ptr<fluid::CudaSphSolver> naive;
    if (search == fluid::CudaNeighborSearch::uniform_grid) naive = std::make_unique<fluid::CudaSphSolver>(p, initial);
    comparison.compare(gpu, cpu);
    if (naive) comparison.compare(gpu, naive->download_state(), naive->download_fields());
    for (int step = 0; step < 5; ++step) {
        cpu.step(); gpu.step();
        if (naive) naive->step();
        comparison.compare(gpu, cpu); // Fields are still from pre-integration state.
        if (naive) comparison.compare(gpu, naive->download_state(), naive->download_fields());
        cpu.evaluate_fields(); gpu.evaluate_fields();
        if (naive) naive->evaluate_fields();
        comparison.compare(gpu, cpu); // Explicit current-state field evaluation.
        if (naive) comparison.compare(gpu, naive->download_state(), naive->download_fields());
    }
}

void dam_break(std::size_t count) {
    Comparison comparison;
    fluid::SimParams p;
    for (std::uint32_t seed : {0u, 42u, 991u}) compare_steps(p, fluid::make_dam_break(count, p, seed), comparison);
    comparison.report();
}

void edges() {
    Comparison comparison;
    fluid::SimParams p;
    // Single particle and counts not divisible by the 128-thread launch block.
    for (std::size_t count : {1u, 129u, 257u}) compare_steps(p, fluid::make_dam_break(count, p), comparison);
    p.gravity = {}; p.spacing = 0.09f; p.stiffness = 100;
    fluid::ParticleState cluster{{{0.4f, 0.5f, 0.5f}, {0.4f, 0.5f, 0.5f}, {0.4000001f, 0.5f, 0.5f},
                                 {0.45f, 0.5f, 0.5f}, {0.47f, 0.52f, 0.51f}, {0.8f, 0.8f, 0.8f}},
                                {{1, 0, 0}, {-1, 0, 0}, {0, 0.2f, 0}, {0, 0, 1}, {0, 1, 0}, {1, 1, 1}}};
    compare_steps(p, cluster, comparison); // Overlaps, near overlaps, uneven density, pressure and viscosity.
    p.stiffness = 0;
    compare_steps(p, cluster, comparison);
    p = {}; p.gravity = {}; p.stiffness = 0; p.viscosity = 0;
    const float lo = p.lower_bound().x, hi = p.upper_bound().x;
    fluid::ParticleState walls{{{lo + 0.0001f, 0.3f, 0.3f}, {hi - 0.0001f, 0.4f, 0.4f},
                               {0.5f, lo + 0.0001f, 0.5f}, {0.6f, hi - 0.0001f, 0.6f},
                               {0.7f, 0.7f, lo + 0.0001f}, {0.8f, 0.8f, hi - 0.0001f}, {lo, hi, lo}},
                              {{-2, 0.1f, 0.2f}, {2, 0.1f, 0.2f}, {0.1f, -2, 0.2f}, {0.1f, 2, 0.2f},
                               {0.1f, 0.2f, -2}, {0.1f, 0.2f, 2}, {-1, 1, -1}}};
    compare_steps(p, walls, comparison);
    // Density below the denominator floor remains positive, finite, and unclipped in output.
    p = {}; p.spacing = 0.001f; p.smoothing_radius = 1.0f;
    compare_steps(p, {{{0.4f, 0.5f, 0.5f}, {0.45f, 0.5f, 0.5f}}, {{1, 0, 0}, {-1, 0, 0}}}, comparison);
    comparison.report();
}

void asynchronous() {
    fluid::SimParams p;
    auto initial = fluid::make_dam_break(129, p);
    fluid::CpuSphSolver cpu(p, initial);
    fluid::CudaSphSolver a(p, initial, search), b(p, initial, search);
    // Independent streams, preallocated scratch, and odd A/B swap parity without readbacks.
    for (int step = 0; step < 5; ++step) { a.step(); cpu.step(); b.step(); }
    Comparison comparison;
    comparison.compare(a, cpu); comparison.compare(b, cpu);
    const auto sa = a.download_state(), sb = b.download_state();
    for (std::size_t i = 0; i < sa.positions.size(); ++i) {
        near(sa.positions[i], sb.positions[i], 0, 0); near(sa.velocities[i], sb.velocities[i], 0, 0);
    }
    const auto timing = a.step_timed();
    b.step(); cpu.step(); a.synchronize(); b.synchronize();
    CHECK(std::isfinite(timing.total_ms) && timing.total_ms > 0);
    CHECK(timing.density_pressure_ms >= 0 && timing.forces_ms >= 0 && timing.integrate_ms >= 0);
    CHECK(timing.grid_ms >= 0);
    near(timing.total_ms, timing.grid_ms + timing.density_pressure_ms + timing.forces_ms + timing.integrate_ms, 0.005, 0.001);
    comparison.compare(a, cpu); comparison.compare(b, cpu);
}

void invalid() {
    fluid::SimParams p;
    const auto initial = fluid::make_dam_break(128, p);
    throws<std::invalid_argument>([&] { fluid::CudaSphSolver gpu(p, {}, search); });
    auto bad = initial; bad.velocities.pop_back();
    throws<std::invalid_argument>([&] { fluid::CudaSphSolver gpu(p, bad, search); });
    bad = initial; bad.positions[0].x = std::numeric_limits<float>::quiet_NaN();
    throws<std::invalid_argument>([&] { fluid::CudaSphSolver gpu(p, bad, search); });
    bad = initial; bad.positions[0].x = -1;
    throws<std::invalid_argument>([&] { fluid::CudaSphSolver gpu(p, bad, search); });
    const auto too_many = (search == fluid::CudaNeighborSearch::uniform_grid ? fluid::max_grid_particles : fluid::max_cpu_particles) + 1;
    bad.positions.resize(too_many); bad.velocities.resize(too_many);
    throws<std::invalid_argument>([&] { fluid::CudaSphSolver gpu(p, bad, search); });
    p.dt = 0;
    throws<std::invalid_argument>([&] { fluid::CudaSphSolver gpu(p, initial, search); });
    p = {}; p.spacing = 0.4f;
    p.smoothing_radius = search == fluid::CudaNeighborSearch::uniform_grid ? 0.01f : 0.001f;
    const fluid::ParticleState single{{{0.5f, 0.5f, 0.5f}}, {{}}};
    throws<std::runtime_error>([&] { fluid::CudaSphSolver gpu(p, single, search); }); // EOS overflow, constructor cleanup.
    p = {}; p.dt = 2; p.gravity.y = -std::numeric_limits<float>::max();
    fluid::CudaSphSolver gpu(p, single, search);
    gpu.step();
    throws<std::runtime_error>([&] { gpu.synchronize(); }); // Async integration error is reported.
    throws<std::runtime_error>([&] { gpu.step(); });        // Poisoned solver cannot silently continue.
}

void stability(std::size_t count = 512) {
    fluid::SimParams p;
    if (count > 512) {
        CHECK(search == fluid::CudaNeighborSearch::uniform_grid);
        p.spacing = 0.0125f; p.smoothing_radius = 0.025f; p.boundary_padding = 0.00625f; p.dt = 0.0001f;
    }
    fluid::CudaSphSolver gpu(p, fluid::make_dam_break(count, p), search);
    float peak_density = 0, peak_speed = 0;
    for (int step = 0; step < 1000; ++step) {
        gpu.step();
        auto state = gpu.download_state(); auto fields = gpu.download_fields();
        for (std::size_t i = 0; i < state.positions.size(); ++i) {
            CHECK(fluid::inside_bounds(state.positions[i], p));
            CHECK(fluid::is_finite(state.velocities[i]) && fluid::is_finite(fields.accelerations[i]));
            CHECK(std::isfinite(fields.densities[i]) && fields.densities[i] > 0);
            CHECK(std::isfinite(fields.pressures[i]) && fields.pressures[i] >= 0);
            peak_density = std::max(peak_density, fields.densities[i]);
            peak_speed = std::max(peak_speed, std::sqrt(fluid::dot(state.velocities[i], state.velocities[i])));
        }
    }
    gpu.evaluate_fields(); gpu.synchronize();
    CHECK(peak_density > p.rest_density && peak_density < 2 * p.rest_density);
    CHECK(peak_speed < 20);
    std::cout << "CUDA N=" << count << " steps=1000 dt=" << p.dt << " peak_density=" << peak_density << " peak_speed=" << peak_speed << '\n';
}
} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
        std::cout << "SKIP: no usable CUDA device/driver\n"; return 77;
    }
    return run_test([&] {
        FLUID_CUDA_CHECK(status);
        CHECK(argc == 2 || argc == 3);
        if (argc == 3) { CHECK(std::string(argv[2]) == "grid"); search = fluid::CudaNeighborSearch::uniform_grid; }
        const std::string name = argv[1];
        if (name == "compare128") dam_break(128);
        else if (name == "compare256") dam_break(256);
        else if (name == "compare512") dam_break(512);
        else if (name == "edges") edges();
        else if (name == "async") asynchronous();
        else if (name == "invalid") invalid();
        else if (name == "stability") stability();
        else if (name == "scale10000") stability(10000);
        else if (name == "scale25000") stability(25000);
        else if (name == "scale50000") stability(50000);
        else if (name == "sanitizer") { dam_break(128); edges(); asynchronous(); invalid(); }
        else throw std::invalid_argument("unknown CUDA test case");
    });
}
