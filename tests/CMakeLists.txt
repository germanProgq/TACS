find_package(GTest REQUIRED)

set(TEST_SOURCES
    test_tensor.cpp
    test_layers.cpp
    test_tacsnet.cpp
    test_serialization.cpp
    ../src/core/tensor.cpp
    ../src/core/memory_manager.cpp
    ../src/layers/conv2d.cpp
    ../src/layers/batch_norm.cpp
    ../src/layers/leaky_relu.cpp
    ../src/models/tacsnet.cpp
    ../src/training/optimizer.cpp
    ../src/training/loss.cpp
    ../src/data/data_loader.cpp
    ../src/utils/matrix_ops.cpp
    ../src/utils/serialization.cpp
    ../src/utils/onnx_exporter.cpp
)

add_executable(run_tests ${TEST_SOURCES})

target_link_libraries(run_tests 
    GTest::gtest 
    GTest::gtest_main
)

target_include_directories(run_tests PRIVATE ../include)

add_test(NAME AllTests COMMAND run_tests)