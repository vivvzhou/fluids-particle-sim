#include "fluid/cpu_reference.h"
#include "fluid/initialization.h"
#include "test_support.h"
#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace {
constexpr double pi = 3.14159265358979323846;

fluid::ParticleState particles(std::initializer_list<fluid::Vec3> positions) {
    return {positions, std::vector<fluid::Vec3>(positions.size())};
}

double analytic_density(const fluid::ParticleState& state, const fluid::SimParams& p, std::size_t i) {
    double sum = 0;
    const double h = p.smoothing_radius;
    for (auto x : state.positions) {
        const double dx = static_cast<double>(state.positions[i].x) - x.x;
        const double dy = static_cast<double>(state.positions[i].y) - x.y;
        const double dz = static_cast<double>(state.positions[i].z) - x.z;
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < h * h) sum += p.particle_mass() * 315.0 / (64 * pi * std::pow(h, 9)) * std::pow(h * h - r2, 3);
    }
    return sum;
}

void initialization() {
    fluid::SimParams p;
    near(p.particle_mass(), 0.125, 1.0e-7, 0);
    const auto a = fluid::make_dam_break(256, p, 42);
    const auto b = fluid::make_dam_break(256, p, 42);
    const auto c = fluid::make_dam_break(256, p, 43);
    CHECK(a.positions.size() == 256);
    bool changed = false;
    for (std::size_t i = 0; i < a.positions.size(); ++i) {
        near(a.positions[i], b.positions[i], 0, 0);
        near(a.velocities[i], {}, 0, 0);
        CHECK(fluid::inside_bounds(a.positions[i], p));
        changed = changed || a.positions[i].x != c.positions[i].x;
    }
    CHECK(changed);
    const auto lattice = fluid::make_dam_break(128, p, 42, 0);
    near(lattice.positions[1].x - lattice.positions[0].x, p.spacing);
    near(lattice.positions[4].y - lattice.positions[0].y, p.spacing);
    near(lattice.positions[32].z - lattice.positions[0].z, p.spacing);
    auto shifted = p;
    shifted.box_min = {-2, -3, -4};
    shifted.box_max = {-1, -2, -3};
    for (auto x : fluid::make_dam_break(128, shifted).positions) CHECK(fluid::inside_bounds(x, shifted));
}

void density() {
    fluid::SimParams p;
    fluid::CpuSphSolver isolated(p, particles({{0.5f, 0.5f, 0.5f}}));
    near(isolated.fields().densities[0], p.particle_mass() * 315.0 / (64 * pi * std::pow(p.smoothing_radius, 3)), 1.0e-4);
    CHECK(isolated.fields().pressures[0] == 0);
    auto state = particles({{0.4f, 0.5f, 0.5f}, {0.45f, 0.5f, 0.5f}, {0.8f, 0.5f, 0.5f}});
    fluid::CpuSphSolver solver(p, state);
    for (std::size_t i = 0; i < state.positions.size(); ++i)
        near(solver.fields().densities[i], analytic_density(state, p, i), 1.0e-4, 1.0e-6);
    fluid::CpuSphSolver overlap(p, particles({{0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}}));
    near(overlap.fields().densities[0], isolated.fields().densities[0] * 2);
    near(overlap.fields().accelerations[0], p.gravity);
    overlap.step();
    CHECK(fluid::is_finite(overlap.state().positions[0]));
}

void pressure() {
    fluid::SimParams p;
    p.spacing = 0.09f; // Raises mass so this two-particle fixture has positive pressure.
    p.stiffness = 100;
    p.gravity = {};
    p.viscosity = 0;
    auto state = particles({{0.4f, 0.5f, 0.5f}, {0.45f, 0.5f, 0.5f}});
    fluid::CpuSphSolver solver(p, state);
    const double rho = analytic_density(state, p, 0);
    const double pressure_value = p.stiffness * (std::pow(rho / p.rest_density, 7) - 1);
    const double r = static_cast<double>(state.positions[1].x) - state.positions[0].x;
    const double gradient = 45 / (pi * std::pow(p.smoothing_radius, 6)) * std::pow(p.smoothing_radius - r, 2);
    const double acceleration = -p.particle_mass() * 2 * pressure_value / (rho * rho) * gradient;
    near(solver.fields().pressures[0], pressure_value, 0.01, 5.0e-6);
    near(solver.fields().accelerations[0].x, acceleration, 1.0e-4, 5.0e-6);
    CHECK(solver.fields().accelerations[0].x < 0);
    CHECK(solver.fields().accelerations[1].x > 0);
    near(solver.fields().accelerations[0] + solver.fields().accelerations[1], {}, 1.0e-5, 0);
    solver.step();
    CHECK(solver.state().positions[0].x < state.positions[0].x);
    CHECK(solver.state().positions[1].x > state.positions[1].x);

    // Unequal local densities still conserve total internal pressure momentum.
    state.positions.push_back({0.47f, 0.52f, 0.51f});
    state.velocities.push_back({});
    fluid::CpuSphSolver asymmetric(p, state);
    CHECK(asymmetric.fields().densities[0] != asymmetric.fields().densities[2]);
    fluid::Vec3 total{};
    for (auto a : asymmetric.fields().accelerations) total = total + a;
    near(total, {}, 0.005, 0);
    fluid::CpuSphSolver overlap(p, particles({{0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}}));
    CHECK(overlap.fields().pressures[0] > 0);
    near(overlap.fields().accelerations[0], {});
}

void viscosity() {
    fluid::SimParams p;
    p.gravity = {};
    p.stiffness = 0;
    auto state = particles({{0.4f, 0.5f, 0.5f}, {0.45f, 0.5f, 0.5f}});
    state.velocities = {{1, 0, 0}, {-1, 0, 0}};
    fluid::CpuSphSolver solver(p, state);
    const double rho = analytic_density(state, p, 0);
    const double r = static_cast<double>(state.positions[1].x) - state.positions[0].x;
    const double expected = p.viscosity * p.particle_mass() / rho * -2.0 *
                            45 / (pi * std::pow(p.smoothing_radius, 6)) * (p.smoothing_radius - r);
    near(solver.fields().accelerations[0].x, expected, 1.0e-5, 2.0e-6);
    near(solver.fields().accelerations[0] + solver.fields().accelerations[1], {});
    solver.step();
    double energy = 0;
    for (auto v : solver.state().velocities) energy += fluid::dot(v, v);
    CHECK(energy < 2.0);
    state.positions.push_back({0.47f, 0.52f, 0.51f});
    state.velocities.assign(3, {2, -1, 3});
    fluid::CpuSphSolver uniform(p, state);
    for (auto a : uniform.fields().accelerations) near(a, {});

    // Unequal density weights: a destination uses each source's mass/rho_j.
    state.velocities = {{1, 0, 0}, {0, 0, 0}, {-1, 0, 0}};
    fluid::CpuSphSolver weighted(p, state);
    double weighted_expected = 0;
    for (std::size_t j = 1; j < 3; ++j) {
        const auto d = state.positions[0] - state.positions[j];
        const double distance = std::sqrt(fluid::dot(d, d));
        weighted_expected += p.viscosity * p.particle_mass() / analytic_density(state, p, j) *
                             (state.velocities[j].x - state.velocities[0].x) *
                             45 / (pi * std::pow(p.smoothing_radius, 6)) * (p.smoothing_radius - distance);
    }
    near(weighted.fields().accelerations[0].x, weighted_expected, 1.0e-5, 3.0e-6);
}

void integration() {
    fluid::SimParams p;
    auto state = particles({{0.5f, 0.7f, 0.5f}});
    state.velocities[0] = {0.25f, 0, -0.1f};
    fluid::CpuSphSolver solver(p, state);
    solver.step();
    near(solver.state().velocities[0], state.velocities[0] + p.gravity * p.dt);
    near(solver.state().positions[0], state.positions[0] + solver.state().velocities[0] * p.dt, 1.0e-7, 0);
    for (int i = 1; i < 20; ++i) solver.step();
    near(solver.state().positions[0].y, 0.7 + p.gravity.y * p.dt * p.dt * 20 * 21 / 2, 1.0e-6, 0);
    near(solver.state().velocities[0].y, p.gravity.y * p.dt * 20, 1.0e-6, 0);
    p.gravity = {};
    fluid::CpuSphSolver inertial(p, state);
    for (int i = 0; i < 20; ++i) inertial.step();
    near(inertial.state().velocities[0], state.velocities[0], 0, 0);
}

void boundaries() {
    fluid::SimParams p;
    const float low = p.lower_bound().x;
    const float high = p.upper_bound().x;
    for (int axis = 0; axis < 3; ++axis) {
        for (bool upper : {false, true}) {
            fluid::Vec3 x{0.5f, 0.5f, 0.5f}, v{0.2f, 0.3f, 0.4f};
            float* coords[] = {&x.x, &x.y, &x.z};
            float* speeds[] = {&v.x, &v.y, &v.z};
            *coords[axis] = upper ? 1.1f : -0.1f;
            *speeds[axis] = upper ? 2.0f : -2.0f;
            const auto original_velocity = v;
            fluid::collide_with_box(x, v, p);
            CHECK(fluid::inside_bounds(x, p));
            near(*coords[axis], upper ? high : low, 0, 0);
            near(*speeds[axis], upper ? -0.6 : 0.6);
            if (axis != 0) CHECK(v.x == original_velocity.x);
            if (axis != 1) CHECK(v.y == original_velocity.y);
            if (axis != 2) CHECK(v.z == original_velocity.z);
        }
    }
    fluid::Vec3 x{low, high, low}, v{-1, 2, -3};
    fluid::collide_with_box(x, v, p);
    near(v, {0.3f, -0.6f, 0.9f});
    x = {-1, 2, -1}; v = {1, -2, 3}; // Already moving inward: do not reflect again.
    fluid::collide_with_box(x, v, p);
    near(v, {1, -2, 3}, 0, 0);
    p.restitution = 0;
    x = {-1, -1, -1}; v = {-1, -1, -1};
    fluid::collide_with_box(x, v, p);
    near(v, {});

    // Full integration crosses a wall, using updated velocity before collision.
    p.gravity = {}; p.restitution = 0.3f;
    auto state = particles({{low + 0.0001f, 0.5f, 0.5f}});
    state.velocities[0] = {-1, 0.2f, 0};
    fluid::CpuSphSolver solver(p, state);
    solver.step();
    near(solver.state().positions[0].x, low, 0, 0);
    near(solver.state().velocities[0], {0.3f, 0.2f, 0});
}

void invalid() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (float value : {0.0f, -1.0f, nan, inf, 1.0e-30f, 1.0e30f}) {
        auto p = fluid::SimParams{}; p.smoothing_radius = value;
        throws<std::invalid_argument>([&] { p.validate(); });
    }
    for (float value : {0.0f, -1.0f, nan, inf}) {
        auto p = fluid::SimParams{}; p.dt = value;
        throws<std::invalid_argument>([&] { p.validate(); });
    }
    auto p = fluid::SimParams{}; p.boundary_padding = 0.5f;
    throws<std::invalid_argument>([&] { p.validate(); });
    p = {}; p.restitution = 1.1f;
    throws<std::invalid_argument>([&] { p.validate(); });
    p = {}; p.viscosity = -1;
    throws<std::invalid_argument>([&] { p.validate(); });
    p = {}; p.rest_density = 0;
    throws<std::invalid_argument>([&] { p.validate(); });
    p = {}; p.gravity.x = inf;
    throws<std::invalid_argument>([&] { p.validate(); });
    p = {};
    throws<std::invalid_argument>([&] { fluid::make_dam_break(0, p); });
    throws<std::invalid_argument>([&] { fluid::make_dam_break(fluid::max_grid_particles + 1, p); });
    auto large_params = p; large_params.spacing = 0.01f;
    const auto large = fluid::make_dam_break(4097, large_params);
    throws<std::invalid_argument>([&] { fluid::CpuSphSolver solver(large_params, large); });
    throws<std::invalid_argument>([&] { fluid::make_dam_break(128, p, 42, nan); });
    p.spacing = 1;
    throws<std::invalid_argument>([&] { fluid::make_dam_break(128, p); });
    p = {};
    throws<std::invalid_argument>([&] { fluid::CpuSphSolver solver(p, {}); });
    auto state = particles({{0.5f, 0.5f, 0.5f}}); state.velocities.clear();
    throws<std::invalid_argument>([&] { fluid::CpuSphSolver solver(p, state); });
    state = particles({{nan, 0.5f, 0.5f}});
    throws<std::invalid_argument>([&] { fluid::CpuSphSolver solver(p, state); });
    state = particles({{0, 0.5f, 0.5f}});
    throws<std::invalid_argument>([&] { fluid::CpuSphSolver solver(p, state); });
    state = particles({{0.5f, 0.5f, 0.5f}}); state.velocities[0].x = inf;
    throws<std::invalid_argument>([&] { fluid::CpuSphSolver solver(p, state); });
    state = particles({{0.5f, 0.5f, 0.5f}});
    p.gravity = {0, -std::numeric_limits<float>::max(), 0}; p.dt = 2;
    fluid::CpuSphSolver overflow(p, state);
    throws<std::runtime_error>([&] { overflow.step(); });
    near(overflow.state().positions[0], state.positions[0], 0, 0); // Failed step did not publish scratch.
    p = {}; p.spacing = 0.4f; p.smoothing_radius = 0.001f;
    throws<std::runtime_error>([&] { fluid::CpuSphSolver solver(p, state); });
}

void reproducibility() {
    fluid::SimParams p;
    fluid::CpuSphSolver a(p, fluid::make_dam_break(128, p, 123));
    fluid::CpuSphSolver b(p, fluid::make_dam_break(128, p, 123));
    for (int step = 0; step < 100; ++step) { a.step(); b.step(); }
    for (std::size_t i = 0; i < a.particle_count(); ++i) {
        near(a.state().positions[i], b.state().positions[i], 0, 0);
        near(a.state().velocities[i], b.state().velocities[i], 0, 0);
        near(a.fields().densities[i], b.fields().densities[i], 0, 0);
        near(a.fields().pressures[i], b.fields().pressures[i], 0, 0);
        near(a.fields().accelerations[i], b.fields().accelerations[i], 0, 0);
    }
}

void stability(std::size_t count) {
    fluid::SimParams p;
    fluid::CpuSphSolver solver(p, fluid::make_dam_break(count, p, 42));
    float max_density = 0, min_density = std::numeric_limits<float>::max(), max_speed = 0, max_pressure = 0;
    const auto inspect = [&] {
        for (std::size_t i = 0; i < count; ++i) {
            CHECK(fluid::inside_bounds(solver.state().positions[i], p));
            CHECK(fluid::is_finite(solver.state().velocities[i]));
            CHECK(fluid::is_finite(solver.fields().accelerations[i]));
            const float rho = solver.fields().densities[i];
            const float pressure_value = solver.fields().pressures[i];
            CHECK(std::isfinite(rho) && rho > 0);
            CHECK(std::isfinite(pressure_value) && pressure_value >= 0);
            min_density = std::min(min_density, rho); max_density = std::max(max_density, rho);
            max_pressure = std::max(max_pressure, pressure_value);
            const auto v = solver.state().velocities[i];
            max_speed = std::max(max_speed, std::sqrt(fluid::dot(v, v)));
        }
    };
    inspect();
    for (int step = 0; step < 1000; ++step) { solver.step(); inspect(); }
    solver.evaluate_fields(); inspect();
    CHECK(max_pressure > 0); // The fixture actually exercised pressure, not just falling particles.
    CHECK(max_density < 2 * p.rest_density);
    CHECK(max_speed < 20);
    std::cout << "N=" << count << " steps=1000 dt=" << p.dt << " seed=42 density=["
              << min_density << ',' << max_density << "] max_speed=" << max_speed
              << " max_pressure=" << max_pressure << '\n';
}
} // namespace

int main(int argc, char** argv) {
    return run_test([&] {
        CHECK(argc == 2);
        const std::string name = argv[1];
        if (name == "initialization") initialization();
        else if (name == "density") density();
        else if (name == "pressure") pressure();
        else if (name == "viscosity") viscosity();
        else if (name == "integration") integration();
        else if (name == "boundaries") boundaries();
        else if (name == "invalid") invalid();
        else if (name == "reproducibility") reproducibility();
        else if (name == "stability128") stability(128);
        else if (name == "stability256") stability(256);
        else if (name == "stability512") stability(512);
        else throw std::invalid_argument("unknown test case");
    });
}
