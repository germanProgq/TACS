#!/usr/bin/env python3
"""
Train TACSNet model using the generated synthetic dataset.
This script interfaces with the C++ training executable.
"""

import os
import sys
import json
import subprocess
import argparse
import time
from datetime import datetime
import shutil

class TACSNetTrainer:
    def __init__(self, dataset_path, output_dir, config=None):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.config = config or self.get_default_config()
        
        # Paths
        self.executable_path = './build/train_tacsnet'
        self.config_file = os.path.join(output_dir, 'training_config.json')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def get_default_config(self):
        """Default training configuration"""
        return {
            'model': {
                'type': 'TACSNetUltra',  # Use the optimized version
                'num_classes': 3,
                'input_size': 416,
                'anchors': [
                    [10, 13], [16, 30], [33, 23],    # Small objects
                    [30, 61], [62, 45], [59, 119],   # Medium objects
                    [116, 90], [156, 198], [373, 326] # Large objects
                ]
            },
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'learning_rate': 0.001,
                'momentum': 0.9,
                'weight_decay': 0.0005,
                'optimizer': 'sgd',  # 'sgd' or 'adam'
                'lr_schedule': {
                    'type': 'step',
                    'step_size': 30,
                    'gamma': 0.1
                }
            },
            'loss': {
                'lambda_obj': 1.0,
                'lambda_bbox': 5.0,
                'lambda_cls': 1.0,
                'obj_pos_weight': 1.0,
                'obj_neg_weight': 0.5
            },
            'augmentation': {
                'enabled': True,
                'random_flip': 0.5,
                'random_scale': [0.8, 1.2],
                'random_brightness': 0.2,
                'random_contrast': 0.2
            },
            'validation': {
                'interval': 5,  # Validate every N epochs
                'iou_threshold': 0.5,
                'conf_threshold': 0.25
            },
            'checkpoint': {
                'save_interval': 10,
                'keep_last_n': 5
            }
        }
    
    def check_dataset(self):
        """Verify dataset exists and is valid"""
        required_dirs = [
            os.path.join(self.dataset_path, 'train', 'images'),
            os.path.join(self.dataset_path, 'train', 'labels'),
            os.path.join(self.dataset_path, 'val', 'images'),
            os.path.join(self.dataset_path, 'val', 'labels')
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise ValueError(f"Required directory not found: {dir_path}")
        
        # Check dataset info
        info_path = os.path.join(self.dataset_path, 'dataset_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                dataset_info = json.load(f)
                print(f"Dataset info:")
                print(f"  Classes: {', '.join(dataset_info['classes'])}")
                print(f"  Training samples: {dataset_info['train_samples']}")
                print(f"  Validation samples: {dataset_info['val_samples']}")
                print(f"  Image size: {dataset_info['image_size']}")
                return dataset_info
        else:
            print("Warning: dataset_info.json not found")
            return None
    
    def build_training_executable(self):
        """Build the C++ training executable if needed"""
        print("\nChecking training executable...")
        
        # Check if build directory exists
        if not os.path.exists('build'):
            print("Build directory not found. Creating and building...")
            os.makedirs('build')
        
        # Check if we need to create the training program
        train_cpp_path = 'train_tacsnet.cpp'
        if not os.path.exists(train_cpp_path):
            print("Creating training program...")
            self.create_training_program(train_cpp_path)
        
        # Build the executable
        print("Building C++ training executable...")
        build_commands = [
            'cd build',
            'cmake ..',
            'make train_tacsnet -j4'
        ]
        
        try:
            for cmd in build_commands:
                result = subprocess.run(cmd, shell=True, check=True, 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Build error: {result.stderr}")
                    return False
            print("Build successful!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Build failed: {e}")
            return False
    
    def create_training_program(self, output_path):
        """Create the C++ training program"""
        training_code = '''/*
 * TACSNet Training Program
 * Trains the traffic detection model on synthetic dataset
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <algorithm>

#include "models/tacsnet.h"
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
            if (line.find("\\"adam\\"") != std::string::npos) {
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
            std::cout << "\\rEpoch " << epoch << " - Batch " << num_batches 
                     << "/" << train_loader.num_batches()
                     << " - Loss: " << std::fixed << std::setprecision(4) 
                     << loss << std::flush;
        }
    }
    
    // Print epoch summary
    std::cout << "\\nEpoch " << epoch << " Summary:"
              << " Total Loss: " << total_loss / num_batches
              << " (Obj: " << total_obj_loss / num_batches
              << ", BBox: " << total_bbox_loss / num_batches
              << ", Cls: " << total_cls_loss / num_batches << ")\\n";
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
    
    std::cout << "Validation - Epoch " << epoch << ":\\n";
    std::cout << "  mAP@0.5: " << map_results.map << "\\n";
    std::cout << "  Per-class AP: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << map_results.per_class_ap[i] << " ";
    }
    std::cout << "\\n\\n";
}

int main(int argc, char** argv) {
    // Initialize memory manager
    core::MemoryManager::getInstance().initialize(1024 * 1024 * 1024);  // 1GB pool
    
    // Parse arguments
    auto config = parse_args(argc, argv);
    
    std::cout << "TACSNet Training Program\\n";
    std::cout << "Dataset: " << config.dataset_path << "\\n";
    std::cout << "Output: " << config.output_dir << "\\n";
    std::cout << "Batch size: " << config.batch_size << "\\n";
    std::cout << "Epochs: " << config.epochs << "\\n\\n";
    
    // Create model
    models::TACSNetUltra model;
    
    // Create data loaders
    std::string train_path = config.dataset_path + "/train";
    std::string val_path = config.dataset_path + "/val";
    
    data::DataLoader train_loader(train_path, config.batch_size, true, true);
    data::DataLoader val_loader(val_path, config.batch_size, false, true);
    
    std::cout << "Training samples: " << train_loader.num_samples() << "\\n";
    std::cout << "Validation samples: " << val_loader.num_samples() << "\\n\\n";
    
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
            std::cout << "Saved checkpoint: " << checkpoint_path << "\\n";
        }
        
        // Learning rate decay
        if (epoch % 30 == 0) {
            float new_lr = optimizer->get_learning_rate() * 0.1f;
            optimizer->set_learning_rate(new_lr);
            std::cout << "Learning rate decayed to: " << new_lr << "\\n";
        }
    }
    
    // Save final model
    std::string final_model_path = config.output_dir + "/tacsnet_final.bin";
    utils::Serialization::save_model(model, final_model_path);
    std::cout << "\\nTraining complete! Final model saved to: " << final_model_path << "\\n";
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
    std::cout << "Total training time: " << duration.count() << " minutes\\n";
    
    return 0;
}
'''
        with open(output_path, 'w') as f:
            f.write(training_code)
        print(f"Created {output_path}")
    
    def prepare_config(self):
        """Save configuration to JSON file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Saved training config to {self.config_file}")
    
    def train(self):
        """Run the training process"""
        print("\n" + "="*60)
        print("TACSNet Training Script")
        print("="*60)
        
        # Check dataset
        print("\n1. Checking dataset...")
        dataset_info = self.check_dataset()
        
        # Build executable
        print("\n2. Building training executable...")
        if not self.build_training_executable():
            print("Failed to build training executable")
            return False
        
        # Prepare configuration
        print("\n3. Preparing configuration...")
        self.prepare_config()
        
        # Run training
        print("\n4. Starting training...")
        print("-"*60)
        
        cmd = [
            self.executable_path,
            self.dataset_path,
            self.output_dir,
            self.config_file
        ]
        
        try:
            # Run training with real-time output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, 
                                     universal_newlines=True,
                                     bufsize=1)
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
            
            process.wait()
            
            if process.returncode == 0:
                print("\n" + "="*60)
                print("Training completed successfully!")
                print(f"Model saved to: {self.output_dir}")
                print("="*60)
                
                # Create inference script
                self.create_inference_script()
                return True
            else:
                print(f"\nTraining failed with return code: {process.returncode}")
                return False
                
        except Exception as e:
            print(f"Error during training: {e}")
            return False
    
    def create_inference_script(self):
        """Create a simple inference script"""
        inference_script = f'''#!/usr/bin/env python3
"""
Run inference using the trained TACSNet model
"""

import subprocess
import sys

def run_inference(image_path, model_path="{self.output_dir}/tacsnet_final.bin"):
    """Run inference on an image"""
    cmd = ["./build/inference_tacsnet", model_path, image_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {{result.stderr}}")
    except Exception as e:
        print(f"Failed to run inference: {{e}}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    run_inference(sys.argv[1])
'''
        
        script_path = os.path.join(self.output_dir, 'inference.py')
        with open(script_path, 'w') as f:
            f.write(inference_script)
        os.chmod(script_path, 0o755)
        print(f"\nCreated inference script: {script_path}")

def main():
    parser = argparse.ArgumentParser(description='Train TACSNet on synthetic dataset')
    parser.add_argument('--dataset', default='./datasets/traffic_detection/',
                       help='Path to dataset directory')
    parser.add_argument('--output', default='./models/trained/',
                       help='Output directory for trained model')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd',
                       help='Optimizer type')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Update config based on arguments
    trainer = TACSNetTrainer(args.dataset, args.output)
    trainer.config['training']['batch_size'] = args.batch_size
    trainer.config['training']['epochs'] = args.epochs
    trainer.config['training']['learning_rate'] = args.lr
    trainer.config['training']['optimizer'] = args.optimizer
    
    # Run training
    success = trainer.train()
    
    if success:
        print("\nNext steps:")
        print("1. Test the model on new images:")
        print(f"   python {args.output}/inference.py <image_path>")
        print("\n2. Evaluate on validation set:")
        print(f"   ./build/evaluate_tacsnet {args.output}/tacsnet_final.bin {args.dataset}/val")
        print("\n3. Run with tracking:")
        print(f"   ./build/demo_tracking {args.output}/tacsnet_final.bin <video_path>")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())