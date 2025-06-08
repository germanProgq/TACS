# Use static GoogleTest libraries directly to avoid dynamic linking issues
if(APPLE)
    set(GTEST_INCLUDE_DIR "/opt/homebrew/include")
    set(GTEST_LIBRARY "/opt/homebrew/lib/libgtest.a")
    set(GTEST_MAIN_LIBRARY "/opt/homebrew/lib/libgtest_main.a")
else()
    find_package(GTest REQUIRED)
endif()

set(TEST_SOURCES
    test_tensor.cpp
    test_layers.cpp
    test_tacsnet.cpp
    test_serialization.cpp
)

set(LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/src/core/tensor.cpp
    ${CMAKE_SOURCE_DIR}/src/core/memory_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/layers/conv2d.cpp
    ${CMAKE_SOURCE_DIR}/src/layers/batch_norm.cpp
    ${CMAKE_SOURCE_DIR}/src/layers/leaky_relu.cpp
    ${CMAKE_SOURCE_DIR}/src/models/tacsnet.cpp
    ${CMAKE_SOURCE_DIR}/src/training/optimizer.cpp
    ${CMAKE_SOURCE_DIR}/src/training/loss.cpp
    ${CMAKE_SOURCE_DIR}/src/data/data_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/matrix_ops.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/serialization.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/onnx_exporter.cpp
)

add_executable(run_tests ${TEST_SOURCES} ${LIB_SOURCES})

if(APPLE)
    target_link_libraries(run_tests
        ${GTEST_LIBRARY}
        ${GTEST_MAIN_LIBRARY}
    )
    target_include_directories(run_tests PRIVATE ${GTEST_INCLUDE_DIR})
else()
    target_link_libraries(run_tests
        GTest::gtest
        GTest::gtest_main
    )
endif()

target_include_directories(run_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_test(NAME TensorTests COMMAND run_tests)