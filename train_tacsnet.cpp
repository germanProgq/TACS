/*
 * TACSNet Training Program
 * Trains the traffic detection model on synthetic dataset
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <algorithm>

#include "models/tacsnet_ultra.h"
#include "data/data_loader.h"
#include "training/optimizer.h"
#include "training/loss.h"
#include "utils/metrics.h"
#include "utils/serialization.h"
#include "core/memory_manager.h"

using namespace tacs;

struct TrainingConfig {
    std::string dataset_path;
    std::string output_dir;
    std::string config_file;
    int batch_size = 16;
    int epochs = 100;
    float learning_rate = 0.001f;
    float momentum = 0.9f;
    float weight_decay = 0.0005f;
    int validation_interval = 5;
    int checkpoint_interval = 10;
    bool use_adam = false;
};

TrainingConfig parse_args(int argc, char** argv) {
    TrainingConfig config;
    
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path> <output_dir> <config_file>" << std::endl;
        std::exit(1);
    }
    
    config.dataset_path = argv[1];
    config.output_dir = argv[2];
    config.config_file = argv[3];
    
    // Load config from JSON file
    std::ifstream config_stream(config.config_file);
    if (config_stream.is_open()) {
        // Parse JSON config (simplified - in production use a proper JSON parser)
        std::string line;
        while (std::getline(config_stream, line)) {
            if (line.find("batch_size") != std::string::npos) {
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    config.batch_size = std::stoi(line.substr(pos + 1));
                }
            }
            if (line.find("epochs") != std::string::npos) {
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    config.epochs = std::stoi(line.substr(pos + 1));
                }
            }
            if (line.find("learning_rate") != std::string::npos) {
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    config.learning_rate = std::stof(line.substr(pos + 1));
                }
            }
            if (line.find("\"adam\"") != std::string::npos) {
                config.use_adam = true;
            }
        }
    }
    
    return config;
}

void train_epoch(models::TACSNetUltra& model,
                data::DataLoader& train_loader,
                training::Optimizer& optimizer,
                training::YOLOLoss& loss_fn,
                int epoch) {
    
    float total_loss = 0.0f;
    float total_obj_loss = 0.0f;
    float total_bbox_loss = 0.0f;
    float total_cls_loss = 0.0f;
    int num_batches = 0;
    
    train_loader.reset();
    
    while (train_loader.has_next()) {
        auto batch = train_loader.next_batch();
        
        // Prepare batch tensors
        std::vector<core::Tensor> images;
        std::vector<std::vector<data::BoundingBox>> targets;
        
        for (const auto& sample : batch) {
            images.push_back(sample.image);
            targets.push_back(sample.boxes);
        }
        
        // Forward pass
        std::vector<core::Tensor> predictions;
        for (const auto& image : images) {
            predictions.push_back(model.forward(image));
        }
        
        // Compute loss
        auto [loss, obj_loss, bbox_loss, cls_loss] = loss_fn.compute_loss(predictions, targets);
        
        // Backward pass
        auto gradients = loss_fn.compute_gradients(predictions, targets);
        model.backward(gradients);
        
        // Update weights
        optimizer.step();
        
        // Accumulate losses
        total_loss += loss;
        total_obj_loss += obj_loss;
        total_bbox_loss += bbox_loss;
        total_cls_loss += cls_loss;
        num_batches++;
        
        // Progress
        if (num_batches % 10 == 0) {
            std::cout << "\rEpoch " << epoch << " - Batch " << num_batches 
                     << "/" << train_loader.num_batches()
                     << " - Loss: " << std::fixed << std::setprecision(4) 
                     << loss << std::flush;
        }
    }
    
    // Print epoch summary
    std::cout << "\nEpoch " << epoch << " Summary:"
              << " Total Loss: " << total_loss / num_batches
              << " (Obj: " << total_obj_loss / num_batches
              << ", BBox: " << total_bbox_loss / num_batches
              << ", Cls: " << total_cls_loss / num_batches << ")\n";
}

void validate(models::TACSNetUltra& model,
              data::DataLoader& val_loader,
              int epoch) {
    
    utils::MetricsCalculator metrics(3, 0.5f);  // 3 classes, 0.5 IoU threshold
    val_loader.reset();
    
    while (val_loader.has_next()) {
        auto batch = val_loader.next_batch();
        
        for (const auto& sample : batch) {
            // Forward pass
            auto predictions = model.forward(sample.image);
            
            // Convert predictions to bounding boxes
            auto pred_boxes = model.decode_predictions(predictions, 0.25f);
            
            // Update metrics
            metrics.update(pred_boxes, sample.boxes);
        }
    }
    
    // Calculate mAP
    auto map_results = metrics.calculate_map();
    
    std::cout << "Validation - Epoch " << epoch << ":\n";
    std::cout << "  mAP@0.5: " << map_results.map << "\n";
    std::cout << "  Per-class AP: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << map_results.per_class_ap[i] << " ";
    }
    std::cout << "\n\n";
}

int main(int argc, char** argv) {
    // Initialize memory manager
    core::MemoryManager::getInstance().initialize(1024 * 1024 * 1024);  // 1GB pool
    
    // Parse arguments
    auto config = parse_args(argc, argv);
    
    std::cout << "TACSNet Training Program\n";
    std::cout << "Dataset: " << config.dataset_path << "\n";
    std::cout << "Output: " << config.output_dir << "\n";
    std::cout << "Batch size: " << config.batch_size << "\n";
    std::cout << "Epochs: " << config.epochs << "\n\n";
    
    // Create model
    models::TACSNetUltra model;
    
    // Create data loaders
    std::string train_path = config.dataset_path + "/train";
    std::string val_path = config.dataset_path + "/val";
    
    data::DataLoader train_loader(train_path, config.batch_size, true, true);
    data::DataLoader val_loader(val_path, config.batch_size, false, true);
    
    std::cout << "Training samples: " << train_loader.num_samples() << "\n";
    std::cout << "Validation samples: " << val_loader.num_samples() << "\n\n";
    
    // Create optimizer
    std::unique_ptr<training::Optimizer> optimizer;
    if (config.use_adam) {
        optimizer = std::make_unique<training::AdamOptimizer>(
            model.parameters(), config.learning_rate, 0.9f, 0.999f, config.weight_decay);
    } else {
        optimizer = std::make_unique<training::SGDOptimizer>(
            model.parameters(), config.learning_rate, config.momentum, config.weight_decay);
    }
    
    // Create loss function
    training::YOLOLoss loss_fn(1.0f, 5.0f, 1.0f);  // lambda_obj, lambda_bbox, lambda_cls
    
    // Training loop
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        // Train
        train_epoch(model, train_loader, *optimizer, loss_fn, epoch);
        
        // Validate
        if (epoch % config.validation_interval == 0) {
            validate(model, val_loader, epoch);
        }
        
        // Save checkpoint
        if (epoch % config.checkpoint_interval == 0) {
            std::string checkpoint_path = config.output_dir + "/checkpoint_epoch_" + 
                                        std::to_string(epoch) + ".bin";
            utils::Serialization::save_model(model, checkpoint_path);
            std::cout << "Saved checkpoint: " << checkpoint_path << "\n";
        }
        
        // Learning rate decay
        if (epoch % 30 == 0) {
            float new_lr = optimizer->get_learning_rate() * 0.1f;
            optimizer->set_learning_rate(new_lr);
            std::cout << "Learning rate decayed to: " << new_lr << "\n";
        }
    }
    
    // Save final model
    std::string final_model_path = config.output_dir + "/tacsnet_final.bin";
    utils::Serialization::save_model(model, final_model_path);
    std::cout << "\nTraining complete! Final model saved to: " << final_model_path << "\n";
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
    std::cout << "Total training time: " << duration.count() << " minutes\n";
    
    return 0;
}
