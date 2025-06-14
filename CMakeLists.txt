cmake_minimum_required(VERSION 3.16)
project(TACS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detect architecture and set appropriate SIMD flags
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")

# Base optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops -fvectorize -fno-signed-zeros -fno-trapping-math -finline-functions -fomit-frame-pointer -flto -fuse-linker-plugin")

# Architecture-specific SIMD optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
    if(COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2")
        add_compile_definitions(USE_AVX2)
    endif()
    if(COMPILER_SUPPORTS_FMA)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mfma")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm64|aarch64)")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=native")
    add_compile_definitions(USE_NEON)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(include)

# Core Phase 2 sources
set(PHASE2_SOURCES
    src/core/tensor.cpp
    src/core/memory_manager.cpp
    src/core/tensor_ops.cpp
    src/layers/conv2d.cpp
    src/layers/batch_norm.cpp
    src/layers/leaky_relu.cpp
    src/layers/depthwise_conv2d.cpp
    src/models/tacsnet.cpp
    src/training/loss.cpp
    src/utils/matrix_ops.cpp
    src/utils/nms.cpp
    src/utils/quantization.cpp
)

set(SOURCES
    src/core/tensor.cpp
    src/core/memory_manager.cpp
    src/core/tensor_ops.cpp
    src/layers/conv2d.cpp
    src/layers/batch_norm.cpp
    src/layers/leaky_relu.cpp
    src/layers/depthwise_conv2d.cpp
    src/models/tacsnet.cpp
    src/models/accidentnet.cpp
    src/models/weathernet.cpp
    src/models/tacs_pipeline.cpp
    src/training/optimizer.cpp
    src/training/loss.cpp
    src/data/data_loader.cpp
    src/utils/matrix_ops.cpp
    src/utils/serialization.cpp
    src/utils/onnx_exporter.cpp
    src/utils/metrics.cpp
    src/utils/nms.cpp
    src/utils/quantization.cpp
    src/utils/batch_inference.cpp
    src/utils/image_decoder.cpp
    src/tracking/kalman_filter.cpp
    src/tracking/hungarian_algorithm.cpp
    src/tracking/memory_tracker.cpp
)

set(HEADERS
    include/core/tensor.h
    include/core/memory_manager.h
    include/layers/conv2d.h
    include/layers/batch_norm.h
    include/layers/leaky_relu.h
    include/models/tacsnet.h
    include/training/optimizer.h
    include/training/loss.h
    include/data/data_loader.h
    include/utils/matrix_ops.h
    include/utils/serialization.h
    include/utils/onnx_exporter.h
    include/utils/metrics.h
    include/tracking/kalman_filter.h
    include/tracking/hungarian_algorithm.h
    include/tracking/memory_tracker.h
)

add_executable(phase1_validation phase1_validation.cpp ${SOURCES})
add_executable(phase2_validation phase2_validation.cpp ${PHASE2_SOURCES})
add_executable(phase3_validation phase3_validation.cpp ${SOURCES})
add_executable(phase4_validation phase4_validation.cpp ${SOURCES})

find_package(GTest QUIET)
if(GTest_FOUND)
    enable_testing()
    add_subdirectory(tests)
endif()