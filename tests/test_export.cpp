#include "fluid/export.h"
#include "test_support.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    return run_test([&] {
        CHECK(argc == 2);
        const std::filesystem::path directory = argv[1];
        const auto path = directory / "nested" / "positions.ply";
        fluid::write_ply(path, {{1.0f, -2.5f, 0.125f}, {0, 0.5f, 2.0f}});
        std::ifstream input(path, std::ios::binary);
        const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        const std::string header = "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
                                   "property float x\nproperty float y\nproperty float z\nend_header\n";
        CHECK(bytes.size() == header.size() + 24);
        CHECK(std::equal(header.begin(), header.end(), bytes.begin()));
        // Known IEEE-754 bytes independently verify byte order, float width, order,
        // negative coordinates, and absence of struct padding or binary-mode CRLFs.
        const std::array<unsigned char, 24> expected{{
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0xc0, 0x00, 0x00, 0x00, 0x3e,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x40}};
        CHECK(std::equal(expected.begin(), expected.end(), bytes.begin() + static_cast<std::ptrdiff_t>(header.size())));
        input.close();

        throws<std::invalid_argument>([&] {
            fluid::write_ply(directory / "invalid.ply", {{std::numeric_limits<float>::quiet_NaN(), 0, 0}});
        });
        CHECK(!std::filesystem::exists(directory / "invalid.ply"));
        throws<std::runtime_error>([&] { fluid::write_ply(directory, {{0, 0, 0}}); });
        const auto empty_path = directory / "empty.ply";
        fluid::write_ply(empty_path, {});
        std::ifstream empty(empty_path, std::ios::binary);
        const std::string empty_text{std::istreambuf_iterator<char>(empty), std::istreambuf_iterator<char>()};
        CHECK(empty_text.find("element vertex 0\n") != std::string::npos);
        CHECK(empty_text.size() == header.size());
    });
}
