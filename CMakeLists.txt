cmake_minimum_required(VERSION 3.16)
project(TACS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found. Plugin system features will be limited.")
endif()

# Find OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found. Parallel processing enabled.")
endif()

# Find SDL2
find_package(SDL2 QUIET)
if(NOT SDL2_FOUND)
    # Try pkg-config if find_package fails
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(SDL2 sdl2)
    endif()
endif()

if(SDL2_FOUND)
    message(STATUS "SDL2 found. Simulation frontend will be built.")
else()
    message(WARNING "SDL2 not found. Simulation frontend will not be built.")
endif()

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
    src/models/tacsnet_backward.cpp
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
    src/models/tacsnet_backward.cpp
    src/models/accidentnet.cpp
    src/models/weathernet.cpp
    src/models/tacs_pipeline.cpp
    src/training/optimizer.cpp
    src/training/adaptive_optimizer.cpp
    src/training/loss.cpp
    src/training/augmentation.cpp
    src/training/gradient_converter.cpp
    src/data/data_loader.cpp
    src/utils/matrix_ops.cpp
    src/utils/serialization.cpp
    src/utils/onnx_exporter.cpp
    src/utils/metrics.cpp
    src/utils/nms.cpp
    src/utils/quantization.cpp
    src/utils/batch_inference.cpp
    src/utils/image_decoder.cpp
    src/utils/image_processing.cpp
    src/utils/json_parser.cpp
    src/tracking/kalman_filter.cpp
    src/tracking/hungarian_algorithm.cpp
    src/tracking/memory_tracker.cpp
    src/rl/rl_policy_net.cpp
    src/drone/drone_swarm.cpp
    src/drone/aerial_inference.cpp
    src/federated/federated_learning.cpp
    src/federated/v2x_protocol.cpp
    src/federated/sha256.cpp
    src/plugin/plugin_system.cpp
)

# Phase 10 sources for simulation frontend
if(SDL2_FOUND)
    set(PHASE10_SOURCES
        ${SOURCES}
        src/simulation/simulation_frontend.cpp
        src/simulation/text_renderer.cpp
    )
endif()

set(HEADERS
    include/core/tensor.h
    include/core/memory_manager.h
    include/layers/conv2d.h
    include/layers/batch_norm.h
    include/layers/leaky_relu.h
    include/models/tacsnet.h
    include/training/optimizer.h
    include/training/adaptive_optimizer.h
    include/training/loss.h
    include/data/data_loader.h
    include/utils/matrix_ops.h
    include/utils/serialization.h
    include/utils/onnx_exporter.h
    include/utils/metrics.h
    include/tracking/kalman_filter.h
    include/tracking/hungarian_algorithm.h
    include/tracking/memory_tracker.h
    include/rl/rl_policy_net.h
    include/drone/drone_swarm.h
    include/drone/aerial_inference.h
)

# Phase 6 sources for ONNX runtime
set(PHASE6_SOURCES
    ${SOURCES}
    src/utils/onnx_runtime.cpp
    src/utils/onnx_simd_kernels.cpp
)

add_executable(phase1_validation phase1_validation.cpp ${SOURCES})
add_executable(phase2_validation phase2_validation.cpp ${PHASE2_SOURCES})
add_executable(phase3_validation phase3_validation.cpp ${SOURCES})
add_executable(phase4_validation phase4_validation.cpp ${SOURCES})
add_executable(phase5_validation phase5_validation.cpp ${SOURCES})
add_executable(phase6_validation phase6_validation.cpp ${PHASE6_SOURCES})
add_executable(phase7_validation phase7_validation.cpp ${SOURCES})
add_executable(phase8_validation phase8_validation.cpp ${SOURCES})
target_link_libraries(phase8_validation z pthread)
add_executable(phase9_validation phase9_validation.cpp ${SOURCES})
target_link_libraries(phase9_validation pthread)
if(OpenCV_FOUND)
    target_link_libraries(phase9_validation ${OpenCV_LIBS})
    target_include_directories(phase9_validation PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

# Test training adaptation with adaptive optimizer - commented out (file doesn't exist)
# add_executable(test_training_adaptation test_training_adaptation.cpp ${SOURCES})
# target_link_libraries(test_training_adaptation pthread)

# Test adaptive optimizer functionality - commented out (file doesn't exist)
# add_executable(test_adaptive_optimizer test_adaptive_optimizer.cpp ${SOURCES})
# target_link_libraries(test_adaptive_optimizer pthread)

# Phase 10 validation - always build but with different features
add_executable(phase10_validation phase10_validation.cpp ${SOURCES})
target_link_libraries(phase10_validation pthread)

if(SDL2_FOUND)
    target_compile_definitions(phase10_validation PRIVATE SDL2_FOUND)
    set(PHASE10_SOURCES_SDL ${SOURCES} src/simulation/simulation_frontend.cpp)
    target_sources(phase10_validation PRIVATE src/simulation/simulation_frontend.cpp)
    
    if(SDL2_INCLUDE_DIRS)
        target_include_directories(phase10_validation PRIVATE ${SDL2_INCLUDE_DIRS})
        target_link_libraries(phase10_validation ${SDL2_LIBRARIES})
    else()
        # Modern CMake SDL2 handling
        target_link_libraries(phase10_validation SDL2::SDL2)
    endif()
    
    # Simulation standalone executable
    add_executable(tacs_simulation src/simulation/simulation_main.cpp ${PHASE10_SOURCES_SDL})
    target_link_libraries(tacs_simulation pthread)
    target_compile_definitions(tacs_simulation PRIVATE SDL2_FOUND)
    if(SDL2_INCLUDE_DIRS)
        target_include_directories(tacs_simulation PRIVATE ${SDL2_INCLUDE_DIRS})
        target_link_libraries(tacs_simulation ${SDL2_LIBRARIES})
    else()
        target_link_libraries(tacs_simulation SDL2::SDL2)
    endif()
else()
    message(STATUS "Phase 10 will use console-only mode (no SDL2)")
endif()

add_executable(tacs_edge_runtime src/edge_runtime.cpp ${PHASE6_SOURCES})

# Phase 10 enhanced console demo
add_executable(phase10_console_demo phase10_console_demo.cpp ${SOURCES})
target_link_libraries(phase10_console_demo pthread)


# Main training executable - commented out (file doesn't exist)
# add_executable(train_model train_model.cpp ${SOURCES})
# target_link_libraries(train_model pthread)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(train_model OpenMP::OpenMP_CXX)
# endif()

# Create pre-trained weights executable
add_executable(create_pretrained_weights create_pretrained_weights.cpp ${SOURCES})
target_link_libraries(create_pretrained_weights pthread)

# Adaptive training executable for 99% accuracy - commented out (file doesn't exist)
# add_executable(train_adaptive train_adaptive.cpp ${SOURCES})
# target_link_libraries(train_adaptive pthread)

# Test model forward pass - commented out (file doesn't exist)
# add_executable(test_model_forward test_model_forward.cpp ${SOURCES})
# target_link_libraries(test_model_forward pthread)

# TACSNet training executable
add_executable(train_tacsnet train_tacsnet.cpp ${SOURCES})
target_link_libraries(train_tacsnet pthread)
if(OpenMP_CXX_FOUND)
    target_link_libraries(train_tacsnet OpenMP::OpenMP_CXX)
endif()

# TACSNet training with dataset support
add_executable(train_tacsnet_dataset train_tacsnet_dataset.cpp ${SOURCES})
target_link_libraries(train_tacsnet_dataset pthread)
if(OpenCV_FOUND)
    target_link_libraries(train_tacsnet_dataset ${OpenCV_LIBS})
    target_include_directories(train_tacsnet_dataset PRIVATE ${OpenCV_INCLUDE_DIRS})
else()
    message(WARNING "OpenCV not found. train_tacsnet_dataset will not be built.")
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(train_tacsnet_dataset OpenMP::OpenMP_CXX)
endif()

# Test TACSNet training functionality - commented out (file doesn't exist)
# add_executable(test_tacsnet_training test_tacsnet_training.cpp ${SOURCES})
# target_link_libraries(test_tacsnet_training pthread)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(test_tacsnet_training OpenMP::OpenMP_CXX)
# endif()

# TACSNet 99% accuracy training - commented out (file doesn't exist)
# add_executable(train_tacsnet_99 train_tacsnet_99.cpp ${SOURCES})
# target_link_libraries(train_tacsnet_99 pthread)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(train_tacsnet_99 OpenMP::OpenMP_CXX)
# endif()

# Simple TACSNet test - commented out (file doesn't exist)
# add_executable(test_tacsnet_simple test_tacsnet_simple.cpp ${SOURCES})
# target_link_libraries(test_tacsnet_simple pthread)

# TACSNet training with adaptive optimizer - commented out (file doesn't exist)
# add_executable(train_tacsnet_adaptive train_tacsnet_adaptive.cpp ${SOURCES})
# target_link_libraries(train_tacsnet_adaptive pthread)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(train_tacsnet_adaptive OpenMP::OpenMP_CXX)
# endif()


find_package(GTest QUIET)
if(GTest_FOUND)
    enable_testing()
    add_subdirectory(tests)
endif()