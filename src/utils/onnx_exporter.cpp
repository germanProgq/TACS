#include "utils/onnx_exporter.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

namespace tacs {
namespace utils {

ONNXExporter::ONNXExporter() {
    nodes_.clear();
    initializers_.clear();
    inputs_.clear();
    outputs_.clear();
}

bool ONNXExporter::export_model(const models::TACSNet& model, const std::string& filepath) {
    nodes_.clear();
    initializers_.clear();
    inputs_.clear();
    outputs_.clear();
    
    add_input_info("input", {1, 3, 416, 416}, get_onnx_data_type(core::DataType::FLOAT32));
    
    std::string current_output = "input";
    
    for (int i = 0; i < 7; ++i) {
        std::string conv_name = "conv" + std::to_string(i);
        std::string bn_name = "bn" + std::to_string(i);
        std::string relu_name = "relu" + std::to_string(i);
        std::string pool_name = "pool" + std::to_string(i);
        
        std::string conv_weight = conv_name + "_weight";
        std::string conv_bias = conv_name + "_bias";
        std::string bn_scale = bn_name + "_scale";
        std::string bn_bias = bn_name + "_bias";
        std::string bn_mean = bn_name + "_mean";
        std::string bn_var = bn_name + "_var";
        
        add_conv_node(current_output, conv_weight, conv_bias, conv_name,
                      {3, 3}, {1, 1}, {1, 1, 1, 1});
        
        add_batchnorm_node(conv_name, bn_scale, bn_bias, bn_mean, bn_var, bn_name);
        
        add_leakyrelu_node(bn_name, relu_name, 0.1f);
        
        current_output = relu_name;
        
        if (i < 5) {
            add_maxpool_node(current_output, pool_name, {2, 2}, {2, 2});
            current_output = pool_name;
        }
    }
    
    for (int head = 0; head < 3; ++head) {
        std::string head_name = "detection_head_" + std::to_string(head);
        std::string head_weight = head_name + "_weight";
        std::string head_bias = head_name + "_bias";
        
        add_conv_node(current_output, head_weight, head_bias, head_name,
                      {1, 1}, {1, 1}, {0, 0, 0, 0});
        
        std::string bbox_output = head_name + "_bbox";
        std::string obj_output = head_name + "_objectness";
        std::string cls_output = head_name + "_classes";
        
        add_output_info(bbox_output, {1, 3, -1, -1, 4}, get_onnx_data_type(core::DataType::FLOAT32));
        add_output_info(obj_output, {1, 3, -1, -1, 1}, get_onnx_data_type(core::DataType::FLOAT32));
        add_output_info(cls_output, {1, 3, -1, -1, 3}, get_onnx_data_type(core::DataType::FLOAT32));
    }
    
    return write_onnx_file(filepath);
}

void ONNXExporter::add_conv_node(const std::string& input_name, const std::string& weight_name,
                                 const std::string& bias_name, const std::string& output_name,
                                 const std::vector<int64_t>& kernel_shape, 
                                 const std::vector<int64_t>& strides,
                                 const std::vector<int64_t>& pads) {
    ONNXNode node;
    node.name = output_name;
    node.op_type = "Conv";
    node.inputs = {input_name, weight_name};
    if (!bias_name.empty()) {
        node.inputs.push_back(bias_name);
    }
    node.outputs = {output_name};
    node.int_attributes["kernel_shape"] = kernel_shape;
    node.int_attributes["strides"] = strides;
    node.int_attributes["pads"] = pads;
    
    nodes_.push_back(node);
}

void ONNXExporter::add_batchnorm_node(const std::string& input_name, const std::string& scale_name,
                                      const std::string& bias_name, const std::string& mean_name,
                                      const std::string& var_name, const std::string& output_name) {
    ONNXNode node;
    node.name = output_name;
    node.op_type = "BatchNormalization";
    node.inputs = {input_name, scale_name, bias_name, mean_name, var_name};
    node.outputs = {output_name};
    node.float_attributes["epsilon"] = {1e-5f};
    
    nodes_.push_back(node);
}

void ONNXExporter::add_leakyrelu_node(const std::string& input_name, const std::string& output_name,
                                      float alpha) {
    ONNXNode node;
    node.name = output_name;
    node.op_type = "LeakyRelu";
    node.inputs = {input_name};
    node.outputs = {output_name};
    node.float_attributes["alpha"] = {alpha};
    
    nodes_.push_back(node);
}

void ONNXExporter::add_maxpool_node(const std::string& input_name, const std::string& output_name,
                                    const std::vector<int64_t>& kernel_shape,
                                    const std::vector<int64_t>& strides) {
    ONNXNode node;
    node.name = output_name;
    node.op_type = "MaxPool";
    node.inputs = {input_name};
    node.outputs = {output_name};
    node.int_attributes["kernel_shape"] = kernel_shape;
    node.int_attributes["strides"] = strides;
    
    nodes_.push_back(node);
}

void ONNXExporter::add_upsample_node(const std::string& input_name, const std::string& output_name,
                                     const std::vector<float>& scales) {
    ONNXNode node;
    node.name = output_name;
    node.op_type = "Upsample";
    node.inputs = {input_name};
    node.outputs = {output_name};
    node.float_attributes["scales"] = scales;
    
    nodes_.push_back(node);
}

void ONNXExporter::add_concat_node(const std::vector<std::string>& input_names, 
                                   const std::string& output_name, int64_t axis) {
    ONNXNode node;
    node.name = output_name;
    node.op_type = "Concat";
    node.inputs = input_names;
    node.outputs = {output_name};
    node.int_attributes["axis"] = {axis};
    
    nodes_.push_back(node);
}

void ONNXExporter::add_tensor_initializer(const std::string& name, const core::Tensor& tensor) {
    ONNXTensor onnx_tensor;
    onnx_tensor.name = name;
    
    const auto& shape = tensor.shape();
    onnx_tensor.dims.assign(shape.begin(), shape.end());
    onnx_tensor.data_type = get_onnx_data_type(tensor.dtype());
    
    if (tensor.dtype() == core::DataType::FLOAT32) {
        const float* data = tensor.data_float();
        onnx_tensor.float_data.assign(data, data + tensor.size());
    }
    
    initializers_.push_back(onnx_tensor);
}

void ONNXExporter::add_input_info(const std::string& name, const std::vector<int64_t>& dims,
                                  int32_t data_type) {
    ONNXValueInfo info;
    info.name = name;
    info.dims = dims;
    info.data_type = data_type;
    inputs_.push_back(info);
}

void ONNXExporter::add_output_info(const std::string& name, const std::vector<int64_t>& dims,
                                   int32_t data_type) {
    ONNXValueInfo info;
    info.name = name;
    info.dims = dims;
    info.data_type = data_type;
    outputs_.push_back(info);
}

bool ONNXExporter::write_onnx_file(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open ONNX file for writing: " << filepath << std::endl;
        return false;
    }
    
    std::vector<uint8_t> model_proto;
    
    // Build GraphProto
    std::vector<uint8_t> graph_proto;
    
    // Graph name (field 1)
    write_protobuf_field(graph_proto, 1, WireType::LENGTH_DELIMITED);
    write_protobuf_string(graph_proto, "TACSNet");
    
    // Nodes (field 1 in GraphProto)
    for (const auto& node : nodes_) {
        std::vector<uint8_t> node_proto;
        
        // Input names (field 1)
        for (const auto& input : node.inputs) {
            write_protobuf_field(node_proto, 1, WireType::LENGTH_DELIMITED);
            write_protobuf_string(node_proto, input);
        }
        
        // Output names (field 2)
        for (const auto& output : node.outputs) {
            write_protobuf_field(node_proto, 2, WireType::LENGTH_DELIMITED);
            write_protobuf_string(node_proto, output);
        }
        
        // Op type (field 4)
        write_protobuf_field(node_proto, 4, WireType::LENGTH_DELIMITED);
        write_protobuf_string(node_proto, node.op_type);
        
        // Attributes
        for (const auto& attr : node.int_attributes) {
            std::vector<uint8_t> attr_proto;
            
            // Name (field 1)
            write_protobuf_field(attr_proto, 1, WireType::LENGTH_DELIMITED);
            write_protobuf_string(attr_proto, attr.first);
            
            // Type (field 20) = INTS
            write_protobuf_field(attr_proto, 20, WireType::VARINT);
            write_protobuf_varint(attr_proto, 7);
            
            // Ints (field 7)
            for (int64_t val : attr.second) {
                write_protobuf_field(attr_proto, 7, WireType::VARINT);
                write_protobuf_varint(attr_proto, val);
            }
            
            // Add attribute to node
            write_protobuf_field(node_proto, 5, WireType::LENGTH_DELIMITED);
            write_protobuf_bytes(node_proto, attr_proto);
        }
        
        for (const auto& attr : node.float_attributes) {
            std::vector<uint8_t> attr_proto;
            
            // Name (field 1)
            write_protobuf_field(attr_proto, 1, WireType::LENGTH_DELIMITED);
            write_protobuf_string(attr_proto, attr.first);
            
            // Type (field 20) = FLOATS
            write_protobuf_field(attr_proto, 20, WireType::VARINT);
            write_protobuf_varint(attr_proto, 6);
            
            // Floats (field 6)
            for (float val : attr.second) {
                write_protobuf_field(attr_proto, 6, WireType::FIXED32);
                write_protobuf_fixed32(attr_proto, *reinterpret_cast<const uint32_t*>(&val));
            }
            
            // Add attribute to node
            write_protobuf_field(node_proto, 5, WireType::LENGTH_DELIMITED);
            write_protobuf_bytes(node_proto, attr_proto);
        }
        
        // Add node to graph
        write_protobuf_field(graph_proto, 1, WireType::LENGTH_DELIMITED);
        write_protobuf_bytes(graph_proto, node_proto);
    }
    
    // Initializers (field 5 in GraphProto)
    for (const auto& init : initializers_) {
        std::vector<uint8_t> tensor_proto;
        
        // Name (field 8)
        write_protobuf_field(tensor_proto, 8, WireType::LENGTH_DELIMITED);
        write_protobuf_string(tensor_proto, init.name);
        
        // Data type (field 2)
        write_protobuf_field(tensor_proto, 2, WireType::VARINT);
        write_protobuf_varint(tensor_proto, init.data_type);
        
        // Dims (field 1)
        for (int64_t dim : init.dims) {
            write_protobuf_field(tensor_proto, 1, WireType::VARINT);
            write_protobuf_varint(tensor_proto, dim);
        }
        
        // Float data (field 4)
        for (float val : init.float_data) {
            write_protobuf_field(tensor_proto, 4, WireType::FIXED32);
            write_protobuf_fixed32(tensor_proto, *reinterpret_cast<const uint32_t*>(&val));
        }
        
        // Add initializer to graph
        write_protobuf_field(graph_proto, 5, WireType::LENGTH_DELIMITED);
        write_protobuf_bytes(graph_proto, tensor_proto);
    }
    
    // Value infos for inputs (field 11)
    for (const auto& input : inputs_) {
        std::vector<uint8_t> value_info_proto;
        
        // Name (field 1)
        write_protobuf_field(value_info_proto, 1, WireType::LENGTH_DELIMITED);
        write_protobuf_string(value_info_proto, input.name);
        
        // Type (field 2)
        std::vector<uint8_t> type_proto;
        std::vector<uint8_t> tensor_type_proto;
        
        // Elem type (field 1)
        write_protobuf_field(tensor_type_proto, 1, WireType::VARINT);
        write_protobuf_varint(tensor_type_proto, input.data_type);
        
        // Shape (field 2)
        std::vector<uint8_t> shape_proto;
        for (int64_t dim : input.dims) {
            std::vector<uint8_t> dim_proto;
            
            // Dim value (field 1)
            write_protobuf_field(dim_proto, 1, WireType::VARINT);
            write_protobuf_varint(dim_proto, dim);
            
            // Add dim to shape
            write_protobuf_field(shape_proto, 1, WireType::LENGTH_DELIMITED);
            write_protobuf_bytes(shape_proto, dim_proto);
        }
        
        write_protobuf_field(tensor_type_proto, 2, WireType::LENGTH_DELIMITED);
        write_protobuf_bytes(tensor_type_proto, shape_proto);
        
        // Tensor type (field 1)
        write_protobuf_field(type_proto, 1, WireType::LENGTH_DELIMITED);
        write_protobuf_bytes(type_proto, tensor_type_proto);
        
        write_protobuf_field(value_info_proto, 2, WireType::LENGTH_DELIMITED);
        write_protobuf_bytes(value_info_proto, type_proto);
        
        // Add value info to graph
        write_protobuf_field(graph_proto, 11, WireType::LENGTH_DELIMITED);
        write_protobuf_bytes(graph_proto, value_info_proto);
    }
    
    // Value infos for outputs (field 12)
    for (const auto& output : outputs_) {
        std::vector<uint8_t> value_info_proto;
        
        // Name (field 1)
        write_protobuf_field(value_info_proto, 1, WireType::LENGTH_DELIMITED);
        write_protobuf_string(value_info_proto, output.name);
        
        // Type (field 2)
        std::vector<uint8_t> type_proto;
        std::vector<uint8_t> tensor_type_proto;
        
        // Elem type (field 1)
        write_protobuf_field(tensor_type_proto, 1, WireType::VARINT);
        write_protobuf_varint(tensor_type_proto, output.data_type);
        
        // Shape (field 2)
        std::vector<uint8_t> shape_proto;
        for (int64_t dim : output.dims) {
            std::vector<uint8_t> dim_proto;
            
            if (dim == -1) {
                // Dim param (field 2) for dynamic dimension
                write_protobuf_field(dim_proto, 2, WireType::LENGTH_DELIMITED);
                write_protobuf_string(dim_proto, "dynamic");
            } else {
                // Dim value (field 1)
                write_protobuf_field(dim_proto, 1, WireType::VARINT);
                write_protobuf_varint(dim_proto, dim);
            }
            
            // Add dim to shape
            write_protobuf_field(shape_proto, 1, WireType::LENGTH_DELIMITED);
            write_protobuf_bytes(shape_proto, dim_proto);
        }
        
        write_protobuf_field(tensor_type_proto, 2, WireType::LENGTH_DELIMITED);
        write_protobuf_bytes(tensor_type_proto, shape_proto);
        
        // Tensor type (field 1)
        write_protobuf_field(type_proto, 1, WireType::LENGTH_DELIMITED);
        write_protobuf_bytes(type_proto, tensor_type_proto);
        
        write_protobuf_field(value_info_proto, 2, WireType::LENGTH_DELIMITED);
        write_protobuf_bytes(value_info_proto, type_proto);
        
        // Add value info to graph
        write_protobuf_field(graph_proto, 12, WireType::LENGTH_DELIMITED);
        write_protobuf_bytes(graph_proto, value_info_proto);
    }
    
    // Build ModelProto
    // IR version (field 1)
    write_protobuf_field(model_proto, 1, WireType::VARINT);
    write_protobuf_varint(model_proto, 7); // ONNX IR version 7
    
    // Opset imports (field 8)
    std::vector<uint8_t> opset_proto;
    write_protobuf_field(opset_proto, 1, WireType::LENGTH_DELIMITED);
    write_protobuf_string(opset_proto, ""); // default domain
    write_protobuf_field(opset_proto, 2, WireType::VARINT);
    write_protobuf_varint(opset_proto, 14); // opset version 14
    
    write_protobuf_field(model_proto, 8, WireType::LENGTH_DELIMITED);
    write_protobuf_bytes(model_proto, opset_proto);
    
    // Producer name (field 2)
    write_protobuf_field(model_proto, 2, WireType::LENGTH_DELIMITED);
    write_protobuf_string(model_proto, "TACS");
    
    // Producer version (field 3)
    write_protobuf_field(model_proto, 3, WireType::LENGTH_DELIMITED);
    write_protobuf_string(model_proto, "1.0");
    
    // Model version (field 5)
    write_protobuf_field(model_proto, 5, WireType::VARINT);
    write_protobuf_varint(model_proto, 1);
    
    // Doc string (field 6)
    write_protobuf_field(model_proto, 6, WireType::LENGTH_DELIMITED);
    write_protobuf_string(model_proto, "TACSNet YOLOv3-lite model for traffic object detection");
    
    // Graph (field 7)
    write_protobuf_field(model_proto, 7, WireType::LENGTH_DELIMITED);
    write_protobuf_bytes(model_proto, graph_proto);
    
    // Write the complete model
    file.write(reinterpret_cast<const char*>(model_proto.data()), model_proto.size());
    file.close();
    
    std::cout << "ONNX model exported to: " << filepath << std::endl;
    return true;
}

void ONNXExporter::write_protobuf_varint(std::vector<uint8_t>& buffer, uint64_t value) {
    while (value >= 0x80) {
        buffer.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    buffer.push_back(static_cast<uint8_t>(value & 0x7F));
}

void ONNXExporter::write_protobuf_string(std::vector<uint8_t>& buffer, const std::string& str) {
    write_protobuf_varint(buffer, str.length());
    buffer.insert(buffer.end(), str.begin(), str.end());
}

void ONNXExporter::write_protobuf_bytes(std::vector<uint8_t>& buffer, const std::vector<uint8_t>& data) {
    write_protobuf_varint(buffer, data.size());
    buffer.insert(buffer.end(), data.begin(), data.end());
}

void ONNXExporter::write_protobuf_field(std::vector<uint8_t>& buffer, uint32_t field_number, WireType wire_type) {
    uint32_t tag = (field_number << 3) | static_cast<uint32_t>(wire_type);
    write_protobuf_varint(buffer, tag);
}

void ONNXExporter::write_protobuf_fixed32(std::vector<uint8_t>& buffer, uint32_t value) {
    buffer.push_back(value & 0xFF);
    buffer.push_back((value >> 8) & 0xFF);
    buffer.push_back((value >> 16) & 0xFF);
    buffer.push_back((value >> 24) & 0xFF);
}

int32_t ONNXExporter::get_onnx_data_type(core::DataType dtype) {
    switch (dtype) {
        case core::DataType::FLOAT32: return 1;
        case core::DataType::FLOAT16: return 10;
        case core::DataType::INT8: return 3;
        case core::DataType::INT32: return 6;
        default: return 1;
    }
}

}
}