#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define FLUID_HD __host__ __device__
#else
#define FLUID_HD
#endif

namespace fluid {

// Plain renderer-independent xyz value. No CUDA or graphics types in the CPU API.
struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

FLUID_HD inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
FLUID_HD inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
FLUID_HD inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
FLUID_HD inline Vec3 operator/(Vec3 a, float s) { return a * (1.0f / s); }
FLUID_HD inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline bool is_finite(Vec3 a) { return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z); }

} // namespace fluid
