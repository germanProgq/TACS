#include "utils/onnx_exporter.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <random>
#include <cmath>

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
    
    // Extract model configuration from actual TACSNet instance
    const auto& input_shape = model.get_input_shape();
    add_input_info("input", {input_shape[0], input_shape[1], input_shape[2], input_shape[3]}, 
                   get_onnx_data_type(core::DataType::FLOAT32));
    
    std::string current_output = "input";
    std::vector<std::string> scale_outputs;
    
    // Export actual backbone architecture from model instance
    const auto& backbone_config = model.get_backbone_config();
    std::vector<std::pair<int, int>> layer_channels;
    
    // Extract layer configuration from model
    for (size_t i = 0; i < backbone_config.num_layers; ++i) {
        int in_channels = (i == 0) ? input_shape[1] : backbone_config.channels[i-1];
        int out_channels = backbone_config.channels[i];
        layer_channels.push_back({in_channels, out_channels});
    }
    
    for (size_t i = 0; i < layer_channels.size(); ++i) {
        std::string conv_name = "backbone_conv" + std::to_string(i);
        std::string bn_name = "backbone_bn" + std::to_string(i);
        std::string relu_name = "backbone_relu" + std::to_string(i);
        std::string pool_name = "backbone_pool" + std::to_string(i);
        
        std::string conv_weight = conv_name + "_weight";
        std::string conv_bias = conv_name + "_bias";
        std::string bn_scale = bn_name + "_scale";
        std::string bn_bias = bn_name + "_bias";
        std::string bn_mean = bn_name + "_mean";
        std::string bn_var = bn_name + "_var";
        
        // Export actual weights from model if available
        int in_ch = layer_channels[i].first;
        int out_ch = layer_channels[i].second;
        
        // Check if model has trained weights
        if (model.has_trained_weights()) {
            // Export actual weights from model layers
            auto conv_layer = model.get_backbone_layer(i);
            add_tensor_initializer(conv_weight, conv_layer->get_weight_tensor());
            add_tensor_initializer(conv_bias, conv_layer->get_bias_tensor());
            
            auto bn_layer = model.get_batchnorm_layer(i);
            add_tensor_initializer(bn_scale, bn_layer->weight());
            add_tensor_initializer(bn_bias, bn_layer->bias());
            add_tensor_initializer(bn_mean, bn_layer->running_mean());
            add_tensor_initializer(bn_var, bn_layer->running_var());
        } else {
            // Initialize with proper dimensions for untrained model
            add_weight_initializer(conv_weight, {out_ch, in_ch, 3, 3});
            add_bias_initializer(conv_bias, {out_ch});
            add_weight_initializer(bn_scale, {out_ch});
            add_bias_initializer(bn_bias, {out_ch});
            add_weight_initializer(bn_mean, {out_ch});
            add_weight_initializer(bn_var, {out_ch});
        }
        
        // Conv + BN + LeakyReLU
        add_conv_node(current_output, conv_weight, conv_bias, conv_name,
                      {3, 3}, {1, 1}, {1, 1, 1, 1});
        
        add_batchnorm_node(conv_name, bn_scale, bn_bias, bn_mean, bn_var, bn_name);
        
        // Get activation parameters from model configuration
        float leaky_alpha = backbone_config.leaky_relu_alpha;
        add_leakyrelu_node(bn_name, relu_name, leaky_alpha);
        
        current_output = relu_name;
        
        // Save outputs for multi-scale detection
        if (i == 2) scale_outputs.push_back(current_output); // 52x52 scale
        if (i == 3) scale_outputs.push_back(current_output); // 26x26 scale
        if (i == 5) scale_outputs.push_back(current_output); // 13x13 scale
        
        // Add pooling for downsampling (except last layer)
        if (i < 5) {
            add_maxpool_node(current_output, pool_name, {2, 2}, {2, 2});
            current_output = pool_name;
        }
    }
    
    // Export detection heads at multiple scales
    // Head 0: 13x13 (small objects) - connects to layer 5 (1024 channels)
    // Head 1: 26x26 (medium objects) - connects to layer 3 (256 channels)
    // Head 2: 52x52 (large objects) - connects to layer 2 (128 channels)
    std::vector<std::pair<int, int>> head_configs = {
        {1024, 2}, // 13x13 from layer 5
        {256, 1},  // 26x26 from layer 3
        {128, 0}   // 52x52 from layer 2
    };
    
    for (int head = 0; head < 3; ++head) {
        std::string head_name = "detection_head_" + std::to_string(head);
        std::string head_weight = head_name + "_weight";
        std::string head_bias = head_name + "_bias";
        
        int in_channels = head_configs[head].first;
        int scale_idx = head_configs[head].second;
        int num_anchors = model.get_num_anchors();
        int num_classes = model.get_num_classes();
        int output_channels = num_anchors * (5 + num_classes); // 5 = x,y,w,h,objectness
        
        // Add weight initializers for detection head
        add_weight_initializer(head_weight, {output_channels, in_channels, 1, 1});
        add_bias_initializer(head_bias, {output_channels});
        
        // Add 1x1 conv for detection
        add_conv_node(scale_outputs[scale_idx], head_weight, head_bias, head_name,
                      {1, 1}, {1, 1}, {0, 0, 0, 0});
        
        // Add split node to separate bbox, objectness, and class predictions
        std::string reshape_name = head_name + "_reshape";
        std::string split_name = head_name + "_split";
        
        add_reshape_node(head_name, reshape_name, {0, num_anchors, -1, 0, 5 + num_classes});
        
        std::string bbox_output = head_name + "_bbox";
        std::string obj_output = head_name + "_objectness";
        std::string cls_output = head_name + "_classes";
        
        add_split_node(reshape_name, {bbox_output, obj_output, cls_output}, 4, {4, 1, num_classes});
        
        // Define output tensors with dynamic spatial dimensions
        add_output_info(bbox_output, {1, num_anchors, -1, -1, 4}, get_onnx_data_type(core::DataType::FLOAT32));
        add_output_info(obj_output, {1, num_anchors, -1, -1, 1}, get_onnx_data_type(core::DataType::FLOAT32));
        add_output_info(cls_output, {1, num_anchors, -1, -1, num_classes}, get_onnx_data_type(core::DataType::FLOAT32));
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

void ONNXExporter::add_weight_initializer(const std::string& name, const std::vector<int64_t>& dims) {
    // Create weight tensor with Xavier initialization
    size_t total_size = 1;
    for (auto dim : dims) {
        total_size *= dim;
    }
    
    std::vector<float> weights(total_size);
    float fan_in = dims.size() >= 2 ? dims[1] * (dims.size() > 2 ? dims[2] * dims[3] : 1) : 1;
    float fan_out = dims[0] * (dims.size() > 2 ? dims[2] * dims[3] : 1);
    float scale = std::sqrt(2.0f / (fan_in + fan_out));
    
    // Use configurable seed or random initialization
    unsigned int seed = 42; // Default seed
    const char* seed_env = std::getenv("TACS_RANDOM_SEED");
    if (seed_env != nullptr) {
        seed = static_cast<unsigned int>(std::stoul(seed_env));
    } else {
        // Use hardware random if available
        std::random_device rd;
        if (rd.entropy() > 0) {
            seed = rd();
        }
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (size_t i = 0; i < total_size; ++i) {
        weights[i] = dist(rng);
    }
    
    ONNXTensor tensor;
    tensor.name = name;
    tensor.dims = dims;
    tensor.data_type = get_onnx_data_type(core::DataType::FLOAT32);
    tensor.float_data = weights;
    
    initializers_.push_back(tensor);
}

void ONNXExporter::add_bias_initializer(const std::string& name, const std::vector<int64_t>& dims) {
    // Create bias tensor initialized to zero
    size_t total_size = 1;
    for (auto dim : dims) {
        total_size *= dim;
    }
    
    ONNXTensor tensor;
    tensor.name = name;
    tensor.dims = dims;
    tensor.data_type = get_onnx_data_type(core::DataType::FLOAT32);
    tensor.float_data.resize(total_size, 0.0f);
    
    initializers_.push_back(tensor);
}

void ONNXExporter::add_reshape_node(const std::string& input_name, const std::string& output_name,
                                    const std::vector<int64_t>& shape) {
    ONNXNode node;
    node.name = output_name;
    node.op_type = "Reshape";
    node.inputs = {input_name, output_name + "_shape"};
    node.outputs = {output_name};
    
    // Add shape as initializer
    ONNXTensor shape_tensor;
    shape_tensor.name = output_name + "_shape";
    shape_tensor.dims = {static_cast<int64_t>(shape.size())};
    shape_tensor.data_type = get_onnx_data_type(core::DataType::INT64);
    shape_tensor.int64_data = shape;
    initializers_.push_back(shape_tensor);
    
    nodes_.push_back(node);
}

void ONNXExporter::add_split_node(const std::string& input_name, const std::vector<std::string>& output_names,
                                  int64_t axis, const std::vector<int64_t>& split_sizes) {
    ONNXNode node;
    node.name = output_names[0] + "_split";
    node.op_type = "Split";
    node.inputs = {input_name};
    node.outputs = output_names;
    node.int_attributes["axis"] = {axis};
    if (!split_sizes.empty()) {
        node.int_attributes["split"] = split_sizes;
    }
    
    nodes_.push_back(node);
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
        
        // Int64 data (field 7)
        for (int64_t val : init.int64_data) {
            write_protobuf_field(tensor_proto, 7, WireType::VARINT);
            write_protobuf_varint(tensor_proto, val);
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
        case core::DataType::INT64: return 7;
        default: return 1;
    }
}

}
}