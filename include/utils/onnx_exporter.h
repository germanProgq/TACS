/**
 * @file onnx_exporter.h
 * @brief ONNX export functionality for TACSNet deployment
 * 
 * Manual ONNX graph construction for edge deployment without paid frameworks.
 * Exports trained models to standard ONNX format for TensorRT optimization
 * and cross-platform inference compatibility.
 */
#pragma once

#include "core/tensor.h"
#include "models/tacsnet.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace tacs {
namespace utils {

struct ONNXNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, std::vector<int64_t>> int_attributes;
    std::unordered_map<std::string, std::vector<float>> float_attributes;
    std::unordered_map<std::string, std::string> string_attributes;
};

struct ONNXTensor {
    std::string name;
    std::vector<int64_t> dims;
    int32_t data_type;
    std::vector<float> float_data;
    std::vector<int64_t> int64_data;
};

struct ONNXValueInfo {
    std::string name;
    std::vector<int64_t> dims;
    int32_t data_type;
};

class ONNXExporter {
public:
    ONNXExporter();
    ~ONNXExporter() = default;
    
    bool export_model(const models::TACSNet& model, const std::string& filepath);
    
private:
    std::vector<ONNXNode> nodes_;
    std::vector<ONNXTensor> initializers_;
    std::vector<ONNXValueInfo> inputs_;
    std::vector<ONNXValueInfo> outputs_;
    
    void add_conv_node(const std::string& input_name, const std::string& weight_name,
                       const std::string& bias_name, const std::string& output_name,
                       const std::vector<int64_t>& kernel_shape, 
                       const std::vector<int64_t>& strides,
                       const std::vector<int64_t>& pads);
    
    void add_batchnorm_node(const std::string& input_name, const std::string& scale_name,
                            const std::string& bias_name, const std::string& mean_name,
                            const std::string& var_name, const std::string& output_name);
    
    void add_leakyrelu_node(const std::string& input_name, const std::string& output_name,
                            float alpha);
    
    void add_maxpool_node(const std::string& input_name, const std::string& output_name,
                          const std::vector<int64_t>& kernel_shape,
                          const std::vector<int64_t>& strides);
    
    void add_upsample_node(const std::string& input_name, const std::string& output_name,
                           const std::vector<float>& scales);
    
    void add_concat_node(const std::vector<std::string>& input_names, 
                         const std::string& output_name, int64_t axis);
    
    void add_tensor_initializer(const std::string& name, const core::Tensor& tensor);
    
    void add_input_info(const std::string& name, const std::vector<int64_t>& dims,
                        int32_t data_type);
    
    void add_output_info(const std::string& name, const std::vector<int64_t>& dims,
                         int32_t data_type);
    
    bool write_onnx_file(const std::string& filepath);
    
    void write_protobuf_varint(std::vector<uint8_t>& buffer, uint64_t value);
    void write_protobuf_string(std::vector<uint8_t>& buffer, const std::string& str);
    void write_protobuf_bytes(std::vector<uint8_t>& buffer, const std::vector<uint8_t>& data);
    
    int32_t get_onnx_data_type(core::DataType dtype);
};

}
}