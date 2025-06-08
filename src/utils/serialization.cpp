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
    
    ModelHeader header;
    header.magic_number = MAGIC_NUMBER;
    header.version = VERSION;
    header.num_layers = 0; // Will be updated after counting layers
    header.checksum = 0;   // Will be computed later
    
    if (!write_header(file, header)) {
        return false;
    }
    
    // Note: This is a basic implementation. In a full implementation,
    // we would iterate through all model layers and save their weights.
    // For now, we save a placeholder to demonstrate the serialization framework.
    
    // Save anchor information
    const auto& anchors = model.get_anchors();
    uint32_t num_anchor_sets = static_cast<uint32_t>(anchors.size());
    file.write(reinterpret_cast<const char*>(&num_anchor_sets), sizeof(num_anchor_sets));
    
    for (const auto& anchor_set : anchors) {
        uint32_t anchor_set_size = static_cast<uint32_t>(anchor_set.size());
        file.write(reinterpret_cast<const char*>(&anchor_set_size), sizeof(anchor_set_size));
        file.write(reinterpret_cast<const char*>(anchor_set.data()), 
                   anchor_set_size * sizeof(float));
    }
    
    file.close();
    return true;
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
    
    // Load anchor information
    uint32_t num_anchor_sets;
    file.read(reinterpret_cast<char*>(&num_anchor_sets), sizeof(num_anchor_sets));
    
    if (file.fail()) {
        std::cerr << "Failed to read anchor set count" << std::endl;
        return false;
    }
    
    // Note: In a full implementation, we would restore the loaded anchors
    // to the model. For now, we just validate the file structure.
    
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
    }
    
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

}
}