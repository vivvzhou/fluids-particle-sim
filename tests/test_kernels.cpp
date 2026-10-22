#include "fluid/sph_kernels.cuh"
#include "test_support.h"

int main() {
    return run_test([] {
        constexpr double pi = 3.14159265358979323846;
        for (float h : {0.1f, 0.4f, 1.0f}) {
            const fluid::SphKernels kernels(h);
            near(kernels.poly6(0), 315.0 / (64.0 * pi * std::pow(h, 3)), 1.0e-5, 5.0e-7);
            CHECK(kernels.poly6(h * h) == 0);
            CHECK(kernels.poly6(4 * h * h) == 0);
            CHECK(kernels.spiky(h) == 0);
            CHECK(kernels.spiky(2 * h) == 0);
            CHECK(kernels.viscosity_laplacian(h) == 0);
            CHECK(kernels.viscosity_laplacian(2 * h) == 0);
            near(kernels.spiky_gradient({h, 0, 0}), {});
            near(kernels.spiky_gradient({2 * h, 0, 0}), {});
            near(kernels.spiky_gradient({}), {});
            near(kernels.spiky_gradient({h * 1.0e-7f, 0, 0}), {});
            near(kernels.viscosity_laplacian(0), 45.0 / (pi * std::pow(h, 5)), 1.0e-5, 5.0e-7);

            // Numerical volume integral tests the 3D normalization independently.
            double poly6_volume = 0, spiky_volume = 0;
            constexpr int bins = 20000;
            const double dr = static_cast<double>(h) / bins;
            for (int bin = 0; bin < bins; ++bin) {
                const float radius = static_cast<float>((bin + 0.5) * dr);
                const double shell = 4 * pi * radius * radius * dr;
                poly6_volume += kernels.poly6(radius * radius) * shell;
                spiky_volume += kernels.spiky(radius) * shell;
                CHECK(kernels.viscosity_laplacian(radius) >= 0);
            }
            near(poly6_volume, 1.0, 2.0e-6, 0);
            near(spiky_volume, 1.0, 2.0e-6, 0);

            const fluid::Vec3 displacement{0.3f * h, 0.4f * h, 0};
            const auto gradient = kernels.spiky_gradient(displacement);
            const double r = std::sqrt(fluid::dot(displacement, displacement));
            const double derivative = -45.0 / (pi * std::pow(h, 6)) * std::pow(h - r, 2);
            near(gradient.x, derivative * displacement.x / r, 1.0e-5, 2.0e-6);
            near(gradient.y, derivative * displacement.y / r, 1.0e-5, 2.0e-6);
            near(kernels.spiky_gradient(displacement * -1), gradient * -1);
            const float dx = h * 0.001f;
            const float x = h * 0.4f;
            near(kernels.spiky_gradient({x, 0, 0}).x,
                 (kernels.spiky(x + dx) - kernels.spiky(x - dx)) / (2 * dx), 1.0e-5, 2.0e-4);
        }
        CHECK(fluid::tait_pressure(500, 1000, 20000) == 0);
        CHECK(fluid::tait_pressure(1000, 1000, 20000) == 0);
        near(fluid::tait_pressure(1100, 1000, 20000), 20000 * (std::pow(1.1, 7) - 1), 0.02, 1.0e-6);
        CHECK(fluid::tait_pressure(2000, 1000, 0) == 0);
    });
}
