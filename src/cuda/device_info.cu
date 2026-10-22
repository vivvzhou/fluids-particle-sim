#include "fluid/cuda_check.h"
#include "fluid/device_info.h"
#include <ostream>

namespace fluid {
bool print_device_info(std::ostream& output) {
    int count = 0;
    const auto result = cudaGetDeviceCount(&count);
    if (result == cudaErrorNoDevice || (result == cudaSuccess && count == 0)) {
        output << "No CUDA-capable device is available.\n";
        return false;
    }
    FLUID_CUDA_CHECK(result);
    int runtime = 0;
    int driver = 0;
    FLUID_CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    FLUID_CUDA_CHECK(cudaDriverGetVersion(&driver));
    output << "CUDA runtime: " << runtime / 1000 << '.' << (runtime % 1000) / 10
           << " (" << runtime << ")\nCUDA driver API: " << driver
           << "\nCompiled CUDA runtime: " << CUDART_VERSION << '\n';
    int original_device = 0;
    FLUID_CUDA_CHECK(cudaGetDevice(&original_device));
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp properties{};
        FLUID_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        FLUID_CUDA_CHECK(cudaSetDevice(device));
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        FLUID_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        output << "Device " << device << ": " << properties.name
               << "\n  Compute capability: " << properties.major << '.' << properties.minor
               << "\n  Available memory: " << free_bytes << " bytes"
               << "\n  Total memory: " << total_bytes << " bytes\n";
    }
    FLUID_CUDA_CHECK(cudaSetDevice(original_device));
    return true;
}
} // namespace fluid
