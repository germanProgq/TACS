cmake_minimum_required(VERSION 3.16)
project(TACS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops -fvectorize -fno-signed-zeros -fno-trapping-math")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(include)

set(SOURCES
    src/core/tensor.cpp
    src/core/memory_manager.cpp
    src/layers/conv2d.cpp
    src/layers/batch_norm.cpp
    src/layers/leaky_relu.cpp
    src/models/tacsnet.cpp
    src/training/optimizer.cpp
    src/training/loss.cpp
    src/data/data_loader.cpp
    src/utils/matrix_ops.cpp
    src/utils/serialization.cpp
    src/utils/onnx_exporter.cpp
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
)

add_executable(tacs src/main.cpp ${SOURCES})
add_executable(phase1_validation phase1_validation.cpp ${SOURCES})

find_package(GTest QUIET)
if(GTest_FOUND)
    enable_testing()
    add_subdirectory(tests)
endif()

target_compile_options(tacs PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
    $<$<CONFIG:Debug>:-fsanitize=undefined>
)

target_link_options(tacs PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
    $<$<CONFIG:Debug>:-fsanitize=undefined>
)