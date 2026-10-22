#pragma once

#include <iosfwd>

namespace fluid {
// Returns false when CUDA was disabled at build time or no device is available.
// Runtime API failures throw, with operation, source location, and CUDA error.
bool print_device_info(std::ostream& output);
} // namespace fluid
