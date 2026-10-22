#include "cli.h"
#include "fluid/initialization.h"
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>

namespace fluid {
namespace {
std::uint32_t integer(const std::string& text, const std::string& name) {
    std::uint32_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::invalid_argument(name + " requires an unsigned 32-bit integer");
    return value;
}
float real(const std::string& text, const std::string& name) {
    // Reject whitespace and trailing characters; reject NaN/Inf explicitly.
    if (text.empty() || text.find_first_of(" \t\r\n") != std::string::npos)
        throw std::invalid_argument(name + " requires a finite number");
    std::size_t consumed = 0;
    float value = 0;
    try { value = std::stof(text, &consumed); }
    catch (const std::exception&) { throw std::invalid_argument(name + " requires a finite number"); }
    if (consumed != text.size() || !std::isfinite(value))
        throw std::invalid_argument(name + " requires a finite number");
    return value;
}
} // namespace

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (!seen.insert(flag).second) throw std::invalid_argument("duplicate option: " + flag);
        if (flag == "--help" || flag == "--device-info") {
            if (argc != 2) throw std::invalid_argument(flag + " must be used alone");
            options.help = flag == "--help";
            options.device_info = flag == "--device-info";
            return options;
        }
        if (i + 1 == argc) throw std::invalid_argument("missing value or unsupported option: " + flag);
        const std::string value = argv[++i];
        if (flag == "--scene") {
            if (value != "dam-break") throw std::invalid_argument("only --scene dam-break is implemented");
        } else if (flag == "--backend") {
            if (value != "cpu") throw std::invalid_argument("only --backend cpu is implemented (M1)");
        } else if (flag == "--particles") options.particles = integer(value, flag);
        else if (flag == "--steps") options.steps = integer(value, flag);
        else if (flag == "--seed") options.seed = integer(value, flag);
        else if (flag == "--dump-every") options.dump_every = integer(value, flag);
        else if (flag == "--dt") options.params.dt = real(value, flag);
        else if (flag == "--spacing") options.params.spacing = real(value, flag);
        else if (flag == "--h") options.params.smoothing_radius = real(value, flag);
        else if (flag == "--rest-density") options.params.rest_density = real(value, flag);
        else if (flag == "--stiffness") options.params.stiffness = real(value, flag);
        else if (flag == "--viscosity") options.params.viscosity = real(value, flag);
        else if (flag == "--padding") options.params.boundary_padding = real(value, flag);
        else if (flag == "--restitution") options.params.restitution = real(value, flag);
        else if (flag == "--jitter") options.jitter = real(value, flag);
        else if (flag == "--output") {
            if (value.empty()) throw std::invalid_argument("output directory cannot be empty");
            options.output = value;
        } else throw std::invalid_argument("unsupported option: " + flag + "; use --help");
    }
    if (!seen.count("--h")) options.params.smoothing_radius = 2.0f * options.params.spacing;
    if (!seen.count("--padding")) options.params.boundary_padding = 0.5f * options.params.spacing;
    options.params.validate();
    if (options.particles == 0 || options.particles > max_cpu_particles)
        throw std::invalid_argument("CPU particle count must be in [1, 4096]; recommend 128-512");
    if (options.jitter < 0.0f || options.jitter > 0.1f)
        throw std::invalid_argument("jitter must be in [0, 0.1]");
    return options;
}

void print_help(std::ostream& output) {
    output << "fluid_sim: 3D WCSPH CPU reference (M0/M1 only)\n"
              "Usage: fluid_sim [options]\n"
              "  --help                 Show help (use alone)\n"
              "  --device-info          CUDA runtime/device query (use alone; unavailable = exit 2)\n"
              "  --scene dam-break      Deterministic tall fluid block\n"
              "  --backend cpu          Only implemented solver\n"
              "  --particles N          Default 256; allowed 1-4096, recommend 128-512; O(N^2)\n"
              "  --steps N              Default 1000; 0 exports/evaluates initial state only\n"
              "  --dt FLOAT             Fixed timestep, default 0.0005\n"
              "  --seed N               uint32 seed, default 42\n"
              "  --spacing FLOAT        Lattice spacing, default 0.05; mass = rho0 * spacing^3\n"
              "  --h FLOAT              Support radius, default 2 * spacing\n"
              "  --rest-density FLOAT   Default 1000\n"
              "  --stiffness FLOAT      Tait B, default 100000; negative pressure clamped to zero\n"
              "  --viscosity FLOAT      Kinematic nu, default 0.01\n"
              "  --padding FLOAT        Wall padding, default spacing / 2\n"
              "  --restitution FLOAT    Wall normal bounce in [0,1], default 0.3\n"
              "  --jitter FLOAT         Fraction of spacing in [0,0.1], default 0.02\n"
              "  --dump-every N         Default 0 (disabled); otherwise initial, intervals, final\n"
              "  --output PATH          PLY directory, default results/dam_break\n"
              "All count/step/seed options accept unsigned 32-bit integers.\n"
              "Custom parameters may need a smaller dt; there is no adaptive timestep.\n";
}

void print_settings(std::ostream& output, const CliOptions& options) {
    const auto& p = options.params;
    output << std::setprecision(std::numeric_limits<float>::max_digits10)
           << "backend=cpu scene=dam-break particles=" << options.particles << " steps=" << options.steps
           << " dt=" << p.dt << " seed=" << options.seed << " jitter=" << options.jitter
           << "\nspacing=" << p.spacing << " h=" << p.smoothing_radius << " mass=" << p.particle_mass()
           << " rho0=" << p.rest_density << " stiffness=" << p.stiffness << " viscosity_nu=" << p.viscosity
           << "\nbox_min=(" << p.box_min.x << ',' << p.box_min.y << ',' << p.box_min.z
           << ") box_max=(" << p.box_max.x << ',' << p.box_max.y << ',' << p.box_max.z
           << ") gravity=(" << p.gravity.x << ',' << p.gravity.y << ',' << p.gravity.z
           << ") padding=" << p.boundary_padding << " restitution=" << p.restitution
           << "\ndump_every=" << options.dump_every << " output=" << options.output.string() << '\n';
}
} // namespace fluid
