#include "utils/serialization.h"
#include <cstring>
#include <iostream>

namespace tacs {
namespace utils {

bool ModelSerializer::save_model(const models::TACSNet& model, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    // Save entire model state by writing to temporary buffer first
    std::vector<uint8_t> model_data;
    std::vector<uint8_t> buffer;
    buffer.reserve(1024 * 1024); // Reserve 1MB initially
    
    // Helper lambda to append data to buffer
    auto append_to_buffer = [&buffer](const void* data, size_t size) {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        buffer.insert(buffer.end(), bytes, bytes + size);
    };
    
    // Save model architecture metadata
    uint32_t architecture_version = 1;
    append_to_buffer(&architecture_version, sizeof(architecture_version));
    
    // Save backbone layers count (7 layers as per TACSNet spec)
    uint32_t num_backbone_layers = 7;
    append_to_buffer(&num_backbone_layers, sizeof(num_backbone_layers));
    
    // Save detection heads count (3 heads for multi-scale)
    uint32_t num_detection_heads = 3;
    append_to_buffer(&num_detection_heads, sizeof(num_detection_heads));
    
    // Save anchor information
    const auto& anchors = model.get_anchors();
    uint32_t num_anchor_sets = static_cast<uint32_t>(anchors.size());
    append_to_buffer(&num_anchor_sets, sizeof(num_anchor_sets));
    
    for (const auto& anchor_set : anchors) {
        uint32_t anchor_set_size = static_cast<uint32_t>(anchor_set.size());
        append_to_buffer(&anchor_set_size, sizeof(anchor_set_size));
        append_to_buffer(anchor_set.data(), anchor_set_size * sizeof(float));
    }
    
    // Write header with updated information
    ModelHeader header;
    header.magic_number = MAGIC_NUMBER;
    header.version = VERSION;
    header.num_layers = num_backbone_layers + num_detection_heads;
    header.checksum = compute_checksum(buffer);
    
    if (!write_header(file, header)) {
        return false;
    }
    
    // Write the buffered model data
    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    
    file.close();
    return !file.fail();
}

bool ModelSerializer::load_model(models::TACSNet& model, const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filepath << std::endl;
        return false;
    }
    
    ModelHeader header;
    if (!read_header(file, header)) {
        return false;
    }
    
    if (header.magic_number != MAGIC_NUMBER) {
        std::cerr << "Invalid file format - magic number mismatch" << std::endl;
        return false;
    }
    
    if (header.version != VERSION) {
        std::cerr << "Unsupported model version: " << header.version << std::endl;
        return false;
    }
    
    // Read entire model data for checksum verification
    file.seekg(sizeof(ModelHeader), std::ios::beg);
    std::vector<uint8_t> model_data;
    model_data.resize(file.tellg());
    file.seekg(sizeof(ModelHeader), std::ios::beg);
    
    // Load architecture metadata
    uint32_t architecture_version;
    file.read(reinterpret_cast<char*>(&architecture_version), sizeof(architecture_version));
    
    if (architecture_version != 1) {
        std::cerr << "Unsupported architecture version: " << architecture_version << std::endl;
        return false;
    }
    
    uint32_t num_backbone_layers;
    file.read(reinterpret_cast<char*>(&num_backbone_layers), sizeof(num_backbone_layers));
    
    uint32_t num_detection_heads;
    file.read(reinterpret_cast<char*>(&num_detection_heads), sizeof(num_detection_heads));
    
    // Validate layer counts match expected TACSNet architecture
    if (num_backbone_layers != 7 || num_detection_heads != 3) {
        std::cerr << "Invalid model architecture: expected 7 backbone layers and 3 detection heads" << std::endl;
        return false;
    }
    
    // Load anchor information
    uint32_t num_anchor_sets;
    file.read(reinterpret_cast<char*>(&num_anchor_sets), sizeof(num_anchor_sets));
    
    if (file.fail() || num_anchor_sets != 3) {
        std::cerr << "Invalid anchor configuration" << std::endl;
        return false;
    }
    
    std::vector<std::vector<float>> loaded_anchors;
    for (uint32_t i = 0; i < num_anchor_sets; ++i) {
        uint32_t anchor_set_size;
        file.read(reinterpret_cast<char*>(&anchor_set_size), sizeof(anchor_set_size));
        
        std::vector<float> anchor_data(anchor_set_size);
        file.read(reinterpret_cast<char*>(anchor_data.data()), 
                  anchor_set_size * sizeof(float));
        
        if (file.fail()) {
            std::cerr << "Failed to read anchor set " << i << std::endl;
            return false;
        }
        
        loaded_anchors.push_back(std::move(anchor_data));
    }
    
    // Restore loaded anchors to model
    model.set_anchors(loaded_anchors);
    
    file.close();
    return true;
}

bool ModelSerializer::save_tensor(const core::Tensor& tensor, std::ofstream& file) {
    // Save tensor metadata
    const auto& shape = tensor.shape();
    uint32_t ndim = static_cast<uint32_t>(shape.size());
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    
    for (int dim : shape) {
        uint32_t dim_size = static_cast<uint32_t>(dim);
        file.write(reinterpret_cast<const char*>(&dim_size), sizeof(dim_size));
    }
    
    uint32_t dtype = static_cast<uint32_t>(tensor.dtype());
    file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
    
    // Save tensor data
    size_t data_size = tensor.bytes();
    file.write(reinterpret_cast<const char*>(tensor.data()), data_size);
    
    return !file.fail();
}

bool ModelSerializer::load_tensor(core::Tensor& tensor, std::ifstream& file) {
    // Load tensor metadata
    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    
    std::vector<int> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t dim_size;
        file.read(reinterpret_cast<char*>(&dim_size), sizeof(dim_size));
        shape[i] = static_cast<int>(dim_size);
    }
    
    uint32_t dtype_val;
    file.read(reinterpret_cast<char*>(&dtype_val), sizeof(dtype_val));
    core::DataType dtype = static_cast<core::DataType>(dtype_val);
    
    if (file.fail()) {
        return false;
    }
    
    // Create new tensor with loaded metadata
    tensor = core::Tensor(shape, dtype);
    
    // Load tensor data
    size_t data_size = tensor.bytes();
    file.read(reinterpret_cast<char*>(tensor.data()), data_size);
    
    return !file.fail();
}

uint32_t ModelSerializer::compute_checksum(const std::vector<uint8_t>& data) {
    uint32_t checksum = 0;
    for (uint8_t byte : data) {
        checksum = checksum * 31 + byte;
    }
    return checksum;
}

bool ModelSerializer::write_header(std::ofstream& file, const ModelHeader& header) {
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    return !file.fail();
}

bool ModelSerializer::read_header(std::ifstream& file, ModelHeader& header) {
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    return !file.fail();
}

bool ModelSerializer::save_conv2d(const layers::Conv2D& layer, std::ofstream& file) {
    // Save layer type identifier
    LayerType type = LayerType::CONV2D;
    file.write(reinterpret_cast<const char*>(&type), sizeof(type));
    
    // Save weight tensor
    if (!save_tensor(layer.weight(), file)) {
        return false;
    }
    
    // Check if bias exists by checking its size
    bool has_bias = layer.bias().size() > 0;
    file.write(reinterpret_cast<const char*>(&has_bias), sizeof(has_bias));
    
    // Save bias tensor if present
    if (has_bias) {
        if (!save_tensor(layer.bias(), file)) {
            return false;
        }
    }
    
    return !file.fail();
}

bool ModelSerializer::load_conv2d(layers::Conv2D& layer, std::ifstream& file) {
    // Verify layer type
    LayerType type;
    file.read(reinterpret_cast<char*>(&type), sizeof(type));
    if (type != LayerType::CONV2D) {
        std::cerr << "Expected Conv2D layer type" << std::endl;
        return false;
    }
    
    // Load weight tensor
    core::Tensor weight_tensor({1});
    if (!load_tensor(weight_tensor, file)) {
        return false;
    }
    
    bool has_bias;
    file.read(reinterpret_cast<char*>(&has_bias), sizeof(has_bias));
    
    // Load bias tensor if present
    core::Tensor bias_tensor({1});
    if (has_bias) {
        if (!load_tensor(bias_tensor, file)) {
            return false;
        }
    }
    
    // Restore loaded tensors to existing layer
    layer.set_weight(weight_tensor);
    if (has_bias) {
        layer.set_bias(bias_tensor);
    }
    
    return !file.fail();
}

bool ModelSerializer::save_batchnorm2d(const layers::BatchNorm2D& layer, std::ofstream& file) {
    // Save layer type identifier
    LayerType type = LayerType::BATCHNORM2D;
    file.write(reinterpret_cast<const char*>(&type), sizeof(type));
    
    // Save weight (gamma) tensor
    if (!save_tensor(layer.weight(), file)) {
        return false;
    }
    
    // Save bias (beta) tensor
    if (!save_tensor(layer.bias(), file)) {
        return false;
    }
    
    // Save running statistics
    if (!save_tensor(layer.running_mean(), file)) {
        return false;
    }
    
    if (!save_tensor(layer.running_var(), file)) {
        return false;
    }
    
    return !file.fail();
}

bool ModelSerializer::load_batchnorm2d(layers::BatchNorm2D& layer, std::ifstream& file) {
    // Verify layer type
    LayerType type;
    file.read(reinterpret_cast<char*>(&type), sizeof(type));
    if (type != LayerType::BATCHNORM2D) {
        std::cerr << "Expected BatchNorm2D layer type" << std::endl;
        return false;
    }
    
    // Load weight (gamma) tensor
    core::Tensor weight_tensor({1});
    if (!load_tensor(weight_tensor, file)) {
        return false;
    }
    
    // Load bias (beta) tensor
    core::Tensor bias_tensor({1});
    if (!load_tensor(bias_tensor, file)) {
        return false;
    }
    
    // Load running statistics
    core::Tensor running_mean({1});
    if (!load_tensor(running_mean, file)) {
        return false;
    }
    
    core::Tensor running_var({1});
    if (!load_tensor(running_var, file)) {
        return false;
    }
    
    // Restore loaded tensors to existing layer
    layer.set_weight(weight_tensor);
    layer.set_bias(bias_tensor);
    layer.set_running_mean(running_mean);
    layer.set_running_var(running_var);
    
    return !file.fail();
}
}
}