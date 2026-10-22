#include "cli.h"
#include "fluid/cpu_reference.h"
#include "fluid/device_info.h"
#include "fluid/export.h"
#include "fluid/initialization.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

int main(int argc, char** argv) {
    try {
        const auto options = fluid::parse_cli(argc, argv);
        if (options.help) {
            fluid::print_help(std::cout);
            return 0;
        }
        if (options.device_info) {
            return fluid::print_device_info(std::cout) ? 0 : 2;
        }
        fluid::print_settings(std::cout, options);
        auto initial = fluid::make_dam_break(options.particles, options.params, options.seed, options.jitter);
        fluid::CpuSphSolver solver(options.params, std::move(initial));
        const auto dump = [&](std::size_t step) {
            std::ostringstream name;
            name << "frame_" << std::setfill('0') << std::setw(6) << step << ".ply";
            fluid::write_ply(options.output / name.str(), solver.state().positions);
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
        const auto finish = std::chrono::steady_clock::now();
        const auto density_range = std::minmax_element(solver.fields().densities.begin(), solver.fields().densities.end());
        double max_speed = 0;
        for (const auto velocity : solver.state().velocities) {
            max_speed = std::max(max_speed, std::hypot(static_cast<double>(velocity.x),
                                                      static_cast<double>(velocity.y), static_cast<double>(velocity.z)));
        }
        std::cout << "Completed " << options.steps << " steps; simulated_time="
                  << static_cast<double>(options.steps) * options.params.dt
                  << " density_range=[" << *density_range.first << ',' << *density_range.second
                  << "] max_speed=" << max_speed
                  << "\nCPU run wall time (includes exports/final diagnostics): "
                  << std::chrono::duration<double>(finish - start).count() << " s\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fluid_sim: " << error.what() << '\n';
        return 1;
    }
}
