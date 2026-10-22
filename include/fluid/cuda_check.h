#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace fluid {
inline void cuda_check(cudaError_t result, const char* expression, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " +
                                 expression + ": " + cudaGetErrorName(result) + " (" +
                                 cudaGetErrorString(result) + ")");
    }
}
} // namespace fluid

#define FLUID_CUDA_CHECK(expression) ::fluid::cuda_check((expression), #expression, __FILE__, __LINE__)

// Future kernel launches must check submission and, in Debug, asynchronous errors.
// Release code must also check its explicit stream synchronization/readback calls.
#ifndef NDEBUG
#define FLUID_CUDA_CHECK_LAUNCH(stream) do { \
    FLUID_CUDA_CHECK(cudaGetLastError()); \
    FLUID_CUDA_CHECK(cudaStreamSynchronize(stream)); \
} while (false)
#else
#define FLUID_CUDA_CHECK_LAUNCH(stream) do { FLUID_CUDA_CHECK(cudaGetLastError()); } while (false)
#endif
