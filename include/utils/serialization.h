/**
 * @file serialization.h
 * @brief Model serialization utilities for TACS neural networks
 * 
 * Provides binary serialization for neural network weights and architecture.
 * Supports save/load operations for checkpointing and deployment scenarios
 * with version control and integrity verification.
 */
#pragma once

#include "core/tensor.h"
#include "models/tacsnet.h"
#include <string>
#include <fstream>
#include <vector>

namespace tacs {
namespace utils {

struct ModelHeader {
    uint32_t magic_number;
    uint32_t version;
    uint32_t num_layers;
    uint32_t checksum;
};

class ModelSerializer {
public:
    static bool save_model(const models::TACSNet& model, const std::string& filepath);
    static bool load_model(models::TACSNet& model, const std::string& filepath);
    
    static bool save_tensor(const core::Tensor& tensor, std::ofstream& file);
    static bool load_tensor(core::Tensor& tensor, std::ifstream& file);
    
private:
    static const uint32_t MAGIC_NUMBER = 0x54414353; // "TACS"
    static const uint32_t VERSION = 1;
    
    static uint32_t compute_checksum(const std::vector<uint8_t>& data);
    static bool write_header(std::ofstream& file, const ModelHeader& header);
    static bool read_header(std::ifstream& file, ModelHeader& header);
};

}
}