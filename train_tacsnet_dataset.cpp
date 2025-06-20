/**
 * @file train_tacsnet_dataset.cpp
 * @brief TACSNet training with custom dataset support
 * 
 * Supports loading datasets in YOLO format with configuration
 * from environment variables or command line arguments.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>
#include <thread>
#include <sstream>
#include <csignal>
#include <numeric>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "models/tacsnet.h"
#include "training/optimizer.h"
#include "training/loss.h"
#include "training/augmentation.h"
#include "training/gradient_converter.h"
#include "utils/metrics.h"
#include "utils/nms.h"
#include "core/tensor.h"
#include "utils/json_parser.h"

namespace fs = std::filesystem;
using namespace tacs;
using namespace std::chrono;

// Training configuration with dataset support
struct TrainingConfig {
    // Dataset parameters
    std::string dataset_path = "";
    std::string train_path = "";
    std::string val_path = "";
    
    // Model parameters
    int input_size = 416;
    int num_classes = 3; // car, pedestrian, cyclist
    
    // Training parameters
    int num_epochs = 100;
    int batch_size = 8;
    int validation_interval = 5;
    
    // Learning rate schedule
    float initial_lr = 0.001f;
    float lr_decay = 0.95f;
    int lr_decay_epochs = 10;
    float min_lr = 1e-7f;
    
    // Loss weights
    float lambda_obj = 0.5f;
    float lambda_bbox = 1.0f;
    float lambda_cls = 0.5f;
    
    // Validation parameters
    float conf_threshold = 0.5f;
    float nms_threshold = 0.45f;
    float target_map = 0.92f;
    
    // Early stopping
    int patience = 20;
    float min_improvement = 0.001f;
    
    // Checkpointing
    std::string output_dir = "./output";
    std::string checkpoint_dir = "./output/checkpoints/";
    std::string model_dir = "./output/models/";
    std::string log_dir = "./output/logs/";
    int checkpoint_interval = 10;
    
    // Data augmentation
    bool use_augmentation = true;
    float aug_probability = 0.8f;
    
    // Load from JSON config files
    bool load_from_env() {
        // Check for dataset config
        const char* dataset_config_path = std::getenv("TACS_DATASET_CONFIG");
        if (dataset_config_path) {
            std::ifstream file(dataset_config_path);
            if (file.is_open()) {
                utils::JSONParser parser;
                auto config = parser.parse(file);
                
                dataset_path = config["dataset_path"].as_string();
                train_path = config["train_path"].as_string();
                val_path = config["val_path"].as_string();
                num_classes = config["num_classes"].as_int();
                input_size = config["input_size"].as_int();
                batch_size = config["batch_size"].as_int();
                use_augmentation = config["augmentation"].as_bool();
                aug_probability = config["augmentation_prob"].as_float();
                
                file.close();
            }
        }
        
        // Check for training config
        const char* training_config_path = std::getenv("TACS_TRAINING_CONFIG");
        if (training_config_path) {
            std::ifstream file(training_config_path);
            if (file.is_open()) {
                utils::JSONParser parser;
                auto config = parser.parse(file);
                
                num_epochs = config["num_epochs"].as_int();
                initial_lr = config["learning_rate"].as_float();
                lr_decay = config["lr_decay"].as_float();
                lr_decay_epochs = config["lr_decay_epochs"].as_int();
                validation_interval = config["validation_interval"].as_int();
                checkpoint_interval = config["checkpoint_interval"].as_int();
                target_map = config["target_map"].as_float();
                patience = config["early_stopping_patience"].as_int();
                output_dir = config["output_dir"].as_string();
                model_dir = config["model_dir"].as_string();
                checkpoint_dir = config["checkpoint_dir"].as_string();
                log_dir = config["log_dir"].as_string();
                
                file.close();
            }
        }
        
        return !dataset_path.empty() && !train_path.empty() && !val_path.empty();
    }
};

// Dataset sample structure
struct DatasetSample {
    cv::Mat image;
    std::vector<utils::Detection> targets;
    std::string image_path;
};

// Custom dataset loader for YOLO format
class YOLODatasetLoader {
public:
    YOLODatasetLoader(const std::string& data_path, const TrainingConfig& config) 
        : data_path_(data_path), config_(config) {
        load_image_paths();
    }
    
    size_t size() const { return image_paths_.size(); }
    
    DatasetSample get_sample(size_t idx) {
        if (idx >= image_paths_.size()) {
            throw std::out_of_range("Sample index out of range");
        }
        
        DatasetSample sample;
        sample.image_path = image_paths_[idx];
        
        // Load image
        sample.image = cv::imread(sample.image_path);
        if (sample.image.empty()) {
            throw std::runtime_error("Failed to load image: " + sample.image_path);
        }
        
        // Resize to input size
        cv::resize(sample.image, sample.image, cv::Size(config_.input_size, config_.input_size));
        
        // Load labels
        std::string label_path = fs::path(sample.image_path).replace_extension(".txt").string();
        sample.targets = load_labels(label_path);
        
        return sample;
    }
    
    std::vector<DatasetSample> get_batch(size_t start_idx, size_t batch_size) {
        std::vector<DatasetSample> batch;
        batch.reserve(batch_size);
        
        for (size_t i = 0; i < batch_size && start_idx + i < size(); ++i) {
            batch.push_back(get_sample(start_idx + i));
        }
        
        return batch;
    }
    
    void shuffle() {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(image_paths_.begin(), image_paths_.end(), g);
    }
    
private:
    void load_image_paths() {
        for (const auto& entry : fs::directory_iterator(data_path_)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
                    image_paths_.push_back(entry.path().string());
                }
            }
        }
        
        std::cout << "Loaded " << image_paths_.size() << " images from " << data_path_ << std::endl;
    }
    
    std::vector<utils::Detection> load_labels(const std::string& label_path) {
        std::vector<utils::Detection> targets;
        
        std::ifstream file(label_path);
        if (!file.is_open()) {
            // No labels for this image
            return targets;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            int class_id;
            float x, y, w, h;
            
            if (iss >> class_id >> x >> y >> w >> h) {
                // Convert from normalized coordinates to pixel coordinates
                utils::Detection det(
                    x * config_.input_size,
                    y * config_.input_size,
                    w * config_.input_size,
                    h * config_.input_size,
                    1.0f,  // confidence (ground truth)
                    class_id
                );
                targets.push_back(det);
            }
        }
        
        return targets;
    }
    
    std::string data_path_;
    TrainingConfig config_;
    std::vector<std::string> image_paths_;
};

// Convert OpenCV Mat to Tensor
core::Tensor mat_to_tensor(const cv::Mat& mat) {
    core::Tensor tensor({1, 3, mat.rows, mat.cols});
    float* data = tensor.data_float();
    
    // Convert BGR to RGB and normalize to [0, 1]
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < mat.rows; ++y) {
            for (int x = 0; x < mat.cols; ++x) {
                int idx = c * mat.rows * mat.cols + y * mat.cols + x;
                // OpenCV uses BGR, we want RGB
                data[idx] = mat.at<cv::Vec3b>(y, x)[2 - c] / 255.0f;
            }
        }
    }
    
    return tensor;
}

// Apply data augmentation
void apply_augmentation(cv::Mat& image, std::vector<utils::Detection>& targets, 
                       float aug_prob, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    if (prob_dist(rng) > aug_prob) {
        return;  // No augmentation
    }
    
    // Random horizontal flip
    if (prob_dist(rng) < 0.5f) {
        cv::flip(image, image, 1);
        for (auto& target : targets) {
            target.x = image.cols - target.x;
        }
    }
    
    // Random brightness adjustment
    if (prob_dist(rng) < 0.5f) {
        float brightness = std::uniform_real_distribution<float>(0.7f, 1.3f)(rng);
        image.convertTo(image, -1, brightness, 0);
    }
    
    // Random contrast adjustment
    if (prob_dist(rng) < 0.5f) {
        float contrast = std::uniform_real_distribution<float>(0.8f, 1.2f)(rng);
        image.convertTo(image, -1, contrast, 0);
    }
}

// Training function with dataset support
void train_tacsnet_with_dataset(const TrainingConfig& config, bool use_pretrained = true) {
    std::cout << "\n=== TACSNet Training with Custom Dataset ===\n\n";
    std::cout << "Dataset: " << config.dataset_path << "\n";
    std::cout << "Training samples: " << config.train_path << "\n";
    std::cout << "Validation samples: " << config.val_path << "\n\n";
    
    // Create directories
    fs::create_directories(config.checkpoint_dir);
    fs::create_directories(config.model_dir);
    fs::create_directories(config.log_dir);
    
    // Initialize model
    std::cout << "Initializing TACSNetUltra model...\n";
    auto model = std::make_unique<models::TACSNetUltra>(use_pretrained);
    model->set_training(true);
    
    // Initialize optimizer
    std::cout << "Setting up SGD optimizer with momentum...\n";
    training::SGDOptimizer optimizer(config.initial_lr, 0.9f, 0.0005f);
    
    // Initialize loss function
    training::LossWeights loss_weights;
    loss_weights.objectness = config.lambda_obj;
    loss_weights.bbox = config.lambda_bbox;
    loss_weights.classification = config.lambda_cls;
    training::YOLOLoss loss_fn(loss_weights);
    
    // Create data loaders
    YOLODatasetLoader train_loader(config.train_path, config);
    YOLODatasetLoader val_loader(config.val_path, config);
    
    if (train_loader.size() == 0) {
        throw std::runtime_error("No training samples found!");
    }
    if (val_loader.size() == 0) {
        throw std::runtime_error("No validation samples found!");
    }
    
    std::cout << "Training samples: " << train_loader.size() << "\n";
    std::cout << "Validation samples: " << val_loader.size() << "\n\n";
    
    // Initialize optimizer parameters
    auto weights = model->get_weights();
    for (auto& [name, weight] : weights) {
        optimizer.add_parameter_group(name, weight);
    }
    
    // Training state
    float best_map = 0.0f;
    int best_epoch = 0;
    int patience_counter = 0;
    
    // Augmentation RNG
    std::mt19937 aug_rng(42);
    
    // Main training loop
    std::cout << "Starting training loop...\n";
    std::cout << std::string(100, '=') << "\n";
    
    for (int epoch = 0; epoch < config.num_epochs; ++epoch) {
        auto epoch_start = high_resolution_clock::now();
        
        // Update learning rate
        if (epoch > 0 && epoch % config.lr_decay_epochs == 0) {
            float new_lr = std::max(optimizer.get_learning_rate() * config.lr_decay, config.min_lr);
            optimizer.set_learning_rate(new_lr);
        }
        
        // Shuffle training data
        train_loader.shuffle();
        
        // Training phase
        float epoch_loss = 0.0f;
        size_t num_batches = (train_loader.size() + config.batch_size - 1) / config.batch_size;
        
        model->set_training(true);
        
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            // Get batch
            auto batch = train_loader.get_batch(batch_idx * config.batch_size, config.batch_size);
            
            // Zero gradients
            model->zero_gradients();
            
            float batch_loss = 0.0f;
            
            // Process each sample in batch
            for (const auto& sample : batch) {
                // Apply augmentation
                cv::Mat aug_image = sample.image.clone();
                std::vector<utils::Detection> aug_targets = sample.targets;
                
                if (config.use_augmentation) {
                    apply_augmentation(aug_image, aug_targets, config.aug_probability, aug_rng);
                }
                
                // Convert to tensor
                auto input = mat_to_tensor(aug_image);
                
                // Convert targets to tensor format
                core::Tensor target_tensor({1, 50, 5});  // Max 50 objects
                target_tensor.zero();
                float* target_data = target_tensor.data_float();
                
                for (size_t i = 0; i < std::min(aug_targets.size(), size_t(50)); ++i) {
                    const auto& t = aug_targets[i];
                    target_data[i * 5 + 0] = t.class_id;
                    target_data[i * 5 + 1] = t.x / config.input_size;
                    target_data[i * 5 + 2] = t.y / config.input_size;
                    target_data[i * 5 + 3] = t.w / config.input_size;
                    target_data[i * 5 + 4] = t.h / config.input_size;
                }
                
                // Forward pass
                auto predictions = model->forward(input);
                
                // Compute loss
                float loss = loss_fn.compute_loss(predictions, target_tensor, model->get_anchors());
                batch_loss += loss;
                
                // Backward pass
                auto gradients = loss_fn.backward(predictions, target_tensor, model->get_anchors());
                
                // Apply gradients
                for (size_t scale = 0; scale < gradients.size() && scale < predictions.size(); ++scale) {
                    const auto& grad = gradients[scale];
                    const auto& pred = predictions[scale];
                    
                    int num_anchors = pred.bbox_predictions.shape()[1];
                    int num_classes = pred.class_predictions.shape()[4];
                    
                    auto [bbox_grad, obj_grad, cls_grad] = 
                        training::GradientConverter::split_combined_gradient(
                            grad, num_anchors, num_classes);
                    
                    // Scale by batch size
                    float scale_factor = 1.0f / config.batch_size;
                    bbox_grad.scale(scale_factor);
                    obj_grad.scale(scale_factor);
                    cls_grad.scale(scale_factor);
                    
                    model->backward(bbox_grad, obj_grad, cls_grad);
                }
            }
            
            // Average losses
            batch_loss /= batch.size();
            epoch_loss += batch_loss;
            
            // Update weights
            auto weight_grads = model->get_weight_gradients();
            for (const auto& [name, grad] : weight_grads) {
                optimizer.set_gradient(name, grad);
            }
            optimizer.step();
            
            // Progress update
            if ((batch_idx + 1) % 10 == 0) {
                std::cout << "\rEpoch " << epoch + 1 << "/" << config.num_epochs 
                          << " - Batch " << batch_idx + 1 << "/" << num_batches
                          << " - Loss: " << std::fixed << std::setprecision(4) << batch_loss
                          << std::flush;
            }
        }
        
        // Validation phase
        if ((epoch + 1) % config.validation_interval == 0) {
            std::cout << "\r" << std::string(80, ' ') << "\r";  // Clear line
            
            model->set_training(false);
            
            // Compute validation mAP
            float val_map = 0.0f;
            // ... validation code would go here ...
            
            std::cout << "Epoch " << std::setw(3) << epoch + 1 
                      << " | Loss: " << std::setw(8) << epoch_loss / num_batches
                      << " | mAP: " << std::setw(6) << val_map
                      << " | LR: " << std::scientific << std::setprecision(2) 
                      << optimizer.get_learning_rate() << "\n";
            
            // Check for improvement
            if (val_map > best_map + config.min_improvement) {
                best_map = val_map;
                best_epoch = epoch + 1;
                patience_counter = 0;
                
                // Save best model
                std::string best_path = config.model_dir + "tacsnet_best.bin";
                model->saveModel(best_path);
                std::cout << "  ✓ New best model saved (mAP: " << best_map << ")\n";
            } else {
                patience_counter++;
            }
            
            // Early stopping
            if (patience_counter >= config.patience) {
                std::cout << "\nEarly stopping triggered\n";
                break;
            }
            
            // Check if target reached
            if (val_map >= config.target_map) {
                std::cout << "\n✓ Target mAP reached: " << val_map << " >= " << config.target_map << "\n";
                break;
            }
        }
        
        // Checkpoint
        if ((epoch + 1) % config.checkpoint_interval == 0) {
            std::string checkpoint_path = config.checkpoint_dir + "checkpoint_epoch_" + 
                                         std::to_string(epoch + 1) + ".bin";
            model->saveModel(checkpoint_path);
        }
        
        auto epoch_end = high_resolution_clock::now();
        double time_seconds = duration_cast<milliseconds>(epoch_end - epoch_start).count() / 1000.0;
    }
    
    // Save final model
    std::string final_path = config.model_dir + "tacsnet_final.bin";
    model->saveModel(final_path);
    
    // Training summary
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "Training Complete!\n";
    std::cout << "Best mAP: " << std::fixed << std::setprecision(4) << best_map 
              << " at epoch " << best_epoch << "\n";
    std::cout << "Final model saved to: " << final_path << "\n";
}

// Main entry point
int main(int argc, char* argv[]) {
    try {
        // Set up signal handlers
        std::signal(SIGINT, [](int) {
            std::cout << "\n\nTraining interrupted by user.\n";
            std::exit(0);
        });
        
        // Load configuration
        TrainingConfig config;
        
        // Try to load from environment variables first
        if (!config.load_from_env()) {
            std::cerr << "Error: Dataset configuration not found.\n";
            std::cerr << "Please set TACS_DATASET_CONFIG and TACS_TRAINING_CONFIG environment variables.\n";
            return 1;
        }
        
        // Check for command line flags
        bool use_pretrained = true;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--no-pretrained") {
                use_pretrained = false;
                std::cout << "Training from scratch (no pretrained weights)\n";
            }
        }
        
        // Run training
        train_tacsnet_with_dataset(config, use_pretrained);
        
    } catch (const std::exception& e) {
        std::cerr << "\nError during training: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}