cmake_minimum_required(VERSION 3.24)
project(fluid_sim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(FLUID_ENABLE_CUDA "Build CUDA device diagnostics and M2 all-pairs solver when nvcc is available" ON)

add_library(fluid_core STATIC src/sim_params.cpp src/initialization.cpp src/cpu_reference.cpp)
target_include_directories(fluid_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(fluid_core PUBLIC cxx_std_17)

add_library(fluid_io STATIC src/export_ply.cpp)
target_link_libraries(fluid_io PUBLIC fluid_core)

set(FLUID_HAS_CUDA OFF)
if(FLUID_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(FLUID_HAS_CUDA ON)
    else()
        message(WARNING "CUDA compiler not found: building CPU reference only. Device runtime queries are unavailable.")
    endif()
endif()

if(FLUID_HAS_CUDA)
    add_library(fluid_device STATIC src/cuda/device_info.cu)
    target_link_libraries(fluid_device PRIVATE CUDA::cudart)
    set_target_properties(fluid_device PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_library(fluid_cuda STATIC src/cuda/solver.cu src/cuda/density_pressure.cu
        src/cuda/forces.cu src/cuda/integrate.cu)
    target_link_libraries(fluid_cuda PUBLIC fluid_core PRIVATE CUDA::cudart)
    set_target_properties(fluid_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(fluid_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
else()
    add_library(fluid_device STATIC src/device_info_stub.cpp)
endif()
target_link_libraries(fluid_device PUBLIC fluid_core)

function(fluid_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
    else()
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
    endif()
endfunction()

add_executable(fluid_sim src/main.cpp src/cli.cpp)
target_link_libraries(fluid_sim PRIVATE fluid_core fluid_device fluid_io)
target_compile_definitions(fluid_sim PRIVATE FLUID_HAS_CUDA=$<BOOL:${FLUID_HAS_CUDA}>)
if(FLUID_HAS_CUDA)
    target_link_libraries(fluid_sim PRIVATE fluid_cuda)
    add_executable(fluid_cuda_benchmark src/cuda_benchmark.cpp)
    target_link_libraries(fluid_cuda_benchmark PRIVATE fluid_cuda)
    target_compile_definitions(fluid_cuda_benchmark PRIVATE
        FLUID_BUILD_CONFIG="$<CONFIG>"
        FLUID_HOST_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        FLUID_HOST_SYSTEM="${CMAKE_SYSTEM_NAME}")
    fluid_warnings(fluid_cuda_benchmark)
endif()
fluid_warnings(fluid_sim)
fluid_warnings(fluid_device)
fluid_warnings(fluid_core)
fluid_warnings(fluid_io)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_core tests/test_core.cpp)
    target_link_libraries(test_core PRIVATE fluid_core)
    fluid_warnings(test_core)
    add_test(NAME core COMMAND test_core)
    add_test(NAME cli_help COMMAND fluid_sim --help)
    if(FLUID_HAS_CUDA)
        add_test(NAME cuda_device_info COMMAND fluid_sim --device-info)
        set_tests_properties(cuda_device_info PROPERTIES SKIP_RETURN_CODE 2 LABELS cuda)
        add_executable(test_cuda_correctness tests/test_cuda_correctness.cpp)
        target_link_libraries(test_cuda_correctness PRIVATE fluid_cuda CUDA::cudart)
        fluid_warnings(test_cuda_correctness)
        foreach(case IN ITEMS compare128 compare256 compare512 edges async invalid stability)
            add_test(NAME cuda_${case} COMMAND test_cuda_correctness ${case})
            set_tests_properties(cuda_${case} PROPERTIES SKIP_RETURN_CODE 77 LABELS cuda TIMEOUT 120)
        endforeach()
        add_test(NAME cuda_cli COMMAND fluid_sim --backend cuda --particles 129 --steps 5
            --dump-every 2 --output ${CMAKE_CURRENT_BINARY_DIR}/test_output/cuda_cli)
        set_tests_properties(cuda_cli PROPERTIES LABELS cuda TIMEOUT 120)
        add_test(NAME cuda_benchmark_contract COMMAND ${CMAKE_COMMAND}
            -DPROGRAM=$<TARGET_FILE:fluid_cuda_benchmark>
            -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_output/benchmark
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_benchmark.cmake
            CONFIGURATIONS Release RelWithDebInfo MinSizeRel)
        set_tests_properties(cuda_benchmark_contract PROPERTIES LABELS cuda TIMEOUT 120)
    endif()

    add_executable(test_kernels tests/test_kernels.cpp)
    target_link_libraries(test_kernels PRIVATE fluid_core)
    fluid_warnings(test_kernels)
    add_test(NAME kernels COMMAND test_kernels)

    add_executable(test_cpu_reference tests/test_cpu_reference.cpp)
    target_link_libraries(test_cpu_reference PRIVATE fluid_core)
    fluid_warnings(test_cpu_reference)
    foreach(case IN ITEMS initialization density pressure viscosity integration boundaries invalid
                          reproducibility stability128 stability256 stability512)
        add_test(NAME cpu_${case} COMMAND test_cpu_reference ${case})
    endforeach()
    set_tests_properties(cpu_stability128 cpu_stability256 cpu_stability512 PROPERTIES LABELS stability TIMEOUT 120)

    add_executable(test_export tests/test_export.cpp)
    target_link_libraries(test_export PRIVATE fluid_io)
    fluid_warnings(test_export)
    add_test(NAME binary_ply COMMAND test_export ${CMAKE_CURRENT_BINARY_DIR}/test_output/ply)
    add_test(NAME cli_contract COMMAND ${CMAKE_COMMAND}
        -DPROGRAM=$<TARGET_FILE:fluid_sim>
        -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_output/cli
        -DHAS_CUDA=${FLUID_HAS_CUDA} -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.cmake)
endif()

message(STATUS "fluid_sim: C++17; CUDA diagnostics/solver = ${FLUID_HAS_CUDA}; scope = M0/M1/M2")
