#include "cli.h"
#include "fluid/cpu_reference.h"
#include "fluid/device_info.h"
#include "fluid/export.h"
#include "fluid/initialization.h"
#if FLUID_HAS_CUDA
#include "fluid/cuda_solver.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace {
const fluid::ParticleState& snapshot(fluid::CpuSphSolver& solver) { return solver.state(); }
const fluid::SphFields& fields(fluid::CpuSphSolver& solver) { return solver.fields(); }
#if FLUID_HAS_CUDA
fluid::ParticleState snapshot(fluid::CudaSphSolver& solver) { return solver.download_state(); }
fluid::SphFields fields(fluid::CudaSphSolver& solver) { return solver.download_fields(); }
#endif

template<class Solver>
void run_simulation(Solver& solver, const fluid::CliOptions& options) {
        const auto dump = [&](std::size_t step) {
            std::ostringstream name;
            name << "frame_" << std::setfill('0') << std::setw(6) << step << ".ply";
            fluid::write_ply(options.output / name.str(), snapshot(solver).positions);
        };
        const auto start = std::chrono::steady_clock::now();
        if (options.dump_every != 0) dump(0);
        for (std::size_t step = 0; step < options.steps; ++step) {
            try { solver.step(); }
            catch (const std::exception& error) {
                throw std::runtime_error("step " + std::to_string(step + 1) + ": " + error.what());
            }
            const auto completed = step + 1;
            if (options.dump_every != 0 && (completed % options.dump_every == 0 || completed == options.steps))
                dump(completed);
        }
        solver.evaluate_fields(); // Final diagnostic densities correspond to final positions.
        const auto final_state = snapshot(solver);
        const auto final_fields = fields(solver);
        const auto finish = std::chrono::steady_clock::now();
        const auto density_range = std::minmax_element(final_fields.densities.begin(), final_fields.densities.end());
        double max_speed = 0;
        for (const auto velocity : final_state.velocities) {
            max_speed = std::max(max_speed, std::hypot(static_cast<double>(velocity.x),
                                                      static_cast<double>(velocity.y), static_cast<double>(velocity.z)));
        }
        std::cout << "Completed " << options.steps << " steps; simulated_time="
                  << static_cast<double>(options.steps) * options.params.dt
                  << " density_range=[" << *density_range.first << ',' << *density_range.second
                  << "] max_speed=" << max_speed
                  << '\n' << options.backend << " run wall time (includes exports/final diagnostics/readback): "
                  << std::chrono::duration<double>(finish - start).count() << " s\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = fluid::parse_cli(argc, argv);
        if (options.help) { fluid::print_help(std::cout); return 0; }
        if (options.device_info) return fluid::print_device_info(std::cout) ? 0 : 2;
#if !FLUID_HAS_CUDA
        if (options.backend == "cuda") throw std::invalid_argument("CUDA backend unavailable: configure with a CUDA Toolkit and FLUID_ENABLE_CUDA=ON");
#endif
        fluid::print_settings(std::cout, options);
        auto initial = fluid::make_dam_break(options.particles, options.params, options.seed, options.jitter);
        if (options.backend == "cpu") {
            fluid::CpuSphSolver solver(options.params, std::move(initial));
            run_simulation(solver, options);
        }
#if FLUID_HAS_CUDA
        else {
            fluid::CudaSphSolver solver(options.params, initial);
            run_simulation(solver, options);
        }
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fluid_sim: " << error.what() << '\n';
        return 1;
    }
}
