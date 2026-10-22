#pragma once

#include "fluid/sim_params.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>

namespace fluid {
struct CliOptions {
    SimParams params;
    std::size_t particles = 256;
    std::size_t steps = 1000;
    std::uint32_t seed = 42;
    float jitter = 0.02f;
    std::size_t dump_every = 0;
    std::filesystem::path output = "results/dam_break";
    bool help = false;
    bool device_info = false;
};
CliOptions parse_cli(int argc, char** argv);
void print_help(std::ostream& output);
void print_settings(std::ostream& output, const CliOptions& options);
} // namespace fluid
