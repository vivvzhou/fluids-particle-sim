#include "fluid/particle_types.h"
#include <iostream>
#include <limits>

int main() {
    const fluid::Vec3 a{1.0f, 2.0f, 3.0f};
    const auto b = (a + a) / 2.0f;
    if (b.x != a.x || b.y != a.y || b.z != a.z || fluid::dot(a, a) != 14.0f ||
        !fluid::is_finite(a) || fluid::is_finite({std::numeric_limits<float>::infinity(), 0, 0})) {
        std::cerr << "Core vector/finite checks failed\n";
        return 1;
    }
    return 0;
}
