#pragma once

#include "fluid/particle_types.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

inline void check(bool condition, const char* expression, int line) {
    if (!condition) throw std::runtime_error("line " + std::to_string(line) + ": " + expression);
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

inline void near(double actual, double expected, double absolute = 1.0e-6, double relative = 1.0e-5) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > absolute + relative * std::abs(expected))
        throw std::runtime_error("expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}
inline void near(fluid::Vec3 actual, fluid::Vec3 expected, double absolute = 1.0e-6, double relative = 1.0e-5) {
    near(actual.x, expected.x, absolute, relative);
    near(actual.y, expected.y, absolute, relative);
    near(actual.z, expected.z, absolute, relative);
}
template<class Exception, class Function>
void throws(Function function) {
    try { function(); }
    catch (const Exception&) { return; }
    throw std::runtime_error("expected exception was not thrown");
}
template<class Function>
int run_test(Function function) {
    try { function(); return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
