#include "fluid/cuda_solver.h"
#include "fluid/initialization.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Options {
    std::size_t particles = 1024, warmup = 100, steps = 500;
    std::filesystem::path output = "results/m2";
};
std::size_t number(const std::string& text) {
    std::size_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::invalid_argument("expected an unsigned integer, got " + text);
    return result;
}
Options parse(int argc, char** argv) {
    Options options;
    std::vector<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (std::find(seen.begin(), seen.end(), flag) != seen.end()) throw std::invalid_argument("duplicate option: " + flag);
        seen.push_back(flag);
        if (i + 1 == argc) throw std::invalid_argument("missing value: " + flag);
        const std::string value = argv[++i];
        if (flag == "--particles") options.particles = number(value);
        else if (flag == "--warmup") options.warmup = number(value);
        else if (flag == "--steps") options.steps = number(value);
        else if (flag == "--output" && !value.empty()) options.output = value;
        else throw std::invalid_argument("unsupported option: " + flag);
    }
    if (options.particles == 0 || options.particles > fluid::max_cpu_particles)
        throw std::invalid_argument("M2 all-pairs profiling is limited to 1-4096 particles");
    if (options.steps == 0 || options.steps > 100000 || options.warmup > 100000)
        throw std::invalid_argument("steps must be in [1,100000]; warmup in [0,100000]");
    return options;
}
float percentile(std::vector<float> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(std::ceil(fraction * values.size())) - 1;
    return values[index]; // Nearest-rank percentile, documented in the report.
}
std::ofstream open_output(const std::filesystem::path& path) {
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(path);
    output.imbue(std::locale::classic());
    output << std::setprecision(9);
    return output;
}
std::string csv(const std::string& value) {
    std::string quoted = "\"";
    for (char c : value) { if (c == '"') quoted += '"'; quoted += c; }
    return quoted + '"';
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "M2 CUDA stage timing; use a Release build.\n"
                         "fluid_cuda_benchmark [--particles 1024] [--warmup 100] [--steps 500] [--output results/m2]\n"
                         "Fixed preset: spacing=0.025 h=0.05 dt=0.0001 seed=42, other SimParams defaults.\n"
                         "Writes samples.csv and summary.csv, overwriting matching filenames. Max N=4096.\n";
            return 0;
        }
        const std::string configuration = FLUID_BUILD_CONFIG;
        if (configuration != "Release" && configuration != "RelWithDebInfo" && configuration != "MinSizeRel")
            throw std::runtime_error("use Release for timings; Debug synchronizes after each kernel for error checks");
        const auto options = parse(argc, argv);
        fluid::SimParams p;
        p.spacing = 0.025f; p.smoothing_radius = 0.05f; p.boundary_padding = 0.0125f; p.dt = 0.0001f;
        fluid::CudaSphSolver solver(p, fluid::make_dam_break(options.particles, p, 42, 0.02f));
        const auto device = solver.device_info();
        // Allocate host sample storage before warming up; device buffers/events already exist.
        std::vector<fluid::CudaStepTimings> timings(options.steps);
        for (std::size_t i = 0; i < options.warmup; ++i) solver.step();
        solver.synchronize();
        const auto start = std::chrono::steady_clock::now();
        for (auto& sample : timings) sample = solver.step_timed();
        const auto end = std::chrono::steady_clock::now();
        solver.synchronize(); // Sticky numerical status checked outside measured event intervals.
        // Validate final current-state fields and bounds before writing any results.
        solver.evaluate_fields();
        const auto final_state = solver.download_state();
        const auto final_fields = solver.download_fields();
        for (std::size_t i = 0; i < options.particles; ++i) {
            if (!fluid::inside_bounds(final_state.positions[i], p) || !fluid::is_finite(final_state.velocities[i]) ||
                !std::isfinite(final_fields.densities[i]) || final_fields.densities[i] <= 0 ||
                !std::isfinite(final_fields.pressures[i]) || final_fields.pressures[i] < 0)
                throw std::runtime_error("invalid final benchmark state");
        }
        std::vector<float> density, forces, integrate, total;
        for (auto t : timings) {
            if (!std::isfinite(t.total_ms) || t.total_ms <= 0 || !std::isfinite(t.density_pressure_ms) ||
                !std::isfinite(t.forces_ms) || !std::isfinite(t.integrate_ms) ||
                t.density_pressure_ms < 0 || t.forces_ms < 0 || t.integrate_ms < 0)
                throw std::runtime_error("invalid CUDA event timing");
            density.push_back(t.density_pressure_ms); forces.push_back(t.forces_ms);
            integrate.push_back(t.integrate_ms); total.push_back(t.total_ms);
        }
        const double wall = std::chrono::duration<double>(end - start).count();
        std::filesystem::create_directories(options.output);
        auto samples = open_output(options.output / "samples.csv");
        samples << "sample,density_pressure_ms,forces_ms,integrate_ms,total_ms\n";
        for (std::size_t i = 0; i < timings.size(); ++i) {
            const auto t = timings[i];
            samples << i << ',' << t.density_pressure_ms << ',' << t.forces_ms << ',' << t.integrate_ms << ',' << t.total_ms << '\n';
        }
        samples.close();
        auto summary = open_output(options.output / "summary.csv");
        summary << "device,compute_capability,runtime_version,driver_api_version,compiled_runtime_version,build,host_compiler,os,"
                   "particles,warmup,steps,seed,jitter,spacing,h,mass,rho0,stiffness,viscosity,dt,padding,restitution,"
                   "box_min_x,box_min_y,box_min_z,box_max_x,box_max_y,box_max_z,gravity_x,gravity_y,gravity_z,block_size,"
                   "density_median_ms,density_p95_ms,forces_median_ms,forces_p95_ms,integrate_median_ms,integrate_p95_ms,"
                   "total_median_ms,total_p95_ms,host_wall_s,command\n";
        std::string command;
        for (int i = 0; i < argc; ++i) { if (i) command += ' '; command += '"' + std::string(argv[i]) + '"'; }
        summary << csv(device.name) << ',' << device.major << '.' << device.minor << ',' << device.runtime_version << ','
                << device.driver_version << ',' << device.compiled_runtime_version << ',' << FLUID_BUILD_CONFIG << ','
                << csv(FLUID_HOST_COMPILER) << ',' << FLUID_HOST_SYSTEM << ',' << options.particles << ',' << options.warmup << ','
                << options.steps << ",42,0.02," << p.spacing << ',' << p.smoothing_radius << ',' << p.particle_mass() << ','
                << p.rest_density << ',' << p.stiffness << ',' << p.viscosity << ',' << p.dt << ',' << p.boundary_padding << ','
                << p.restitution << ',' << p.box_min.x << ',' << p.box_min.y << ',' << p.box_min.z << ','
                << p.box_max.x << ',' << p.box_max.y << ',' << p.box_max.z << ','
                << p.gravity.x << ',' << p.gravity.y << ',' << p.gravity.z << ',' << fluid::CudaSphSolver::block_size << ','
                << percentile(density, 0.5) << ',' << percentile(density, 0.95) << ','
                << percentile(forces, 0.5) << ',' << percentile(forces, 0.95) << ','
                << percentile(integrate, 0.5) << ',' << percentile(integrate, 0.95) << ','
                << percentile(total, 0.5) << ',' << percentile(total, 0.95) << ',' << wall << ',' << csv(command) << '\n';
        summary.close();
        std::cout << device.name << " N=" << options.particles << " warmup=" << options.warmup << " steps=" << options.steps
                  << " block=" << fluid::CudaSphSolver::block_size << "\nmedian ms: density=" << percentile(density, 0.5)
                  << " force=" << percentile(forces, 0.5) << " integrate=" << percentile(integrate, 0.5)
                  << " total=" << percentile(total, 0.5) << " total p95=" << percentile(total, 0.95)
                  << "\nhost sample-loop wall time=" << wall << " s; CSV directory=" << options.output.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fluid_cuda_benchmark: " << error.what() << '\n'; return 1;
    }
}
