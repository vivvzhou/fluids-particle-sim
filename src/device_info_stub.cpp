#include "fluid/device_info.h"
#include <ostream>

namespace fluid {
bool print_device_info(std::ostream& output) {
    output << "CUDA runtime diagnostics unavailable: built without a CUDA Toolkit/compiler.\n"
              "CPU reference remains available. Install a compatible Toolkit and configure a fresh\n"
              "build with -DFLUID_ENABLE_CUDA=ON to query the GPU and CUDA runtime.\n";
    return false;
}
} // namespace fluid
