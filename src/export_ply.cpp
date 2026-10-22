#include "fluid/export.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <stdexcept>

namespace fluid {
namespace {
void write_float(std::ostream& output, float value) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "PLY requires IEEE-754 float32");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const unsigned char bytes[4] = {
        static_cast<unsigned char>(bits), static_cast<unsigned char>(bits >> 8),
        static_cast<unsigned char>(bits >> 16), static_cast<unsigned char>(bits >> 24)};
    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}
} // namespace

void write_ply(const std::filesystem::path& path, const std::vector<Vec3>& positions) {
    for (const auto position : positions) {
        if (!is_finite(position)) throw std::invalid_argument("cannot export nonfinite particle position");
    }
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    try {
        std::ofstream output;
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output.open(path, std::ios::binary | std::ios::trunc);
        output.imbue(std::locale::classic());
        output << "ply\nformat binary_little_endian 1.0\nelement vertex " << positions.size()
               << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
        for (const auto position : positions) {
            write_float(output, position.x);
            write_float(output, position.y);
            write_float(output, position.z);
        }
        output.close();
    } catch (const std::ios_base::failure& error) {
        throw std::runtime_error("PLY write failed for '" + path.string() + "': " + error.what());
    }
}
} // namespace fluid
