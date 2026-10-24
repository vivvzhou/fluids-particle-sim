#pragma once

#include "fluid/cuda_check.h"
#include <cstddef>

namespace fluid::cuda_detail {
// Destructors cannot throw. The solver synchronizes explicitly to report errors;
// teardown also drains work before its buffers are released, including on exceptions.
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { FLUID_CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { FLUID_CUDA_CHECK(cudaEventCreate(&value)); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(std::size_t count) { FLUID_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct PinnedStatus {
    int* data = nullptr;
    PinnedStatus() { FLUID_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), sizeof(int))); }
    ~PinnedStatus() { if (data) cudaFreeHost(data); }
    PinnedStatus(const PinnedStatus&) = delete;
    PinnedStatus& operator=(const PinnedStatus&) = delete;
};
} // namespace fluid::cuda_detail
