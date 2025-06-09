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
#include "layers/conv2d.h"
#include "layers/batch_norm.h"
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
    
    // Layer-specific serialization
    static bool save_conv2d(const layers::Conv2D& layer, std::ofstream& file);
    static bool load_conv2d(layers::Conv2D& layer, std::ifstream& file);
    
    static bool save_batchnorm2d(const layers::BatchNorm2D& layer, std::ofstream& file);
    static bool load_batchnorm2d(layers::BatchNorm2D& layer, std::ifstream& file);
    
private:
    static const uint32_t MAGIC_NUMBER = 0x54414353; // "TACS"
    static const uint32_t VERSION = 1;
    
    static uint32_t compute_checksum(const std::vector<uint8_t>& data);
    static bool write_header(std::ofstream& file, const ModelHeader& header);
    static bool read_header(std::ifstream& file, ModelHeader& header);
    
    enum class LayerType : uint32_t {
        CONV2D = 1,
        BATCHNORM2D = 2,
        LEAKY_RELU = 3,
        DETECTION_HEAD = 4
    };
};

}
}