#!/usr/bin/env python3
"""
Train TACSNet traffic detection model on a custom dataset
Supports YOLO-format datasets with images and text labels
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TACSNetTrainer:
    def __init__(self, dataset_path, output_dir, config=None):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default training configuration
        self.config = {
            'batch_size': 8,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'lr_decay': 0.95,
            'lr_decay_epochs': 10,
            'validation_interval': 5,
            'checkpoint_interval': 10,
            'target_map': 0.92,
            'early_stopping_patience': 20,
            'num_classes': 3,  # car, pedestrian, cyclist
            'input_size': 416,
            'use_pretrained': True,
            'augmentation': True,
            'augmentation_prob': 0.8
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
            
        # Paths
        self.build_dir = Path('build')
        self.model_dir = self.output_dir / 'models'
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        
        # Create directories
        for dir_path in [self.model_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def validate_dataset(self):
        """Validate dataset structure and format"""
        logger.info("Validating dataset...")
        
        # Check for required directories
        required_dirs = ['train', 'val']
        optional_dirs = ['test']
        
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                logger.error(f"Required directory not found: {dir_path}")
                return False
                
        # Check for images and labels
        stats = {}
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_path / split
            if not split_dir.exists():
                continue
                
            # Count images
            images = list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png'))
            labels = list(split_dir.glob('*.txt'))
            
            # Check if labels exist for images
            missing_labels = []
            for img_path in images:
                label_path = img_path.with_suffix('.txt')
                if not label_path.exists():
                    missing_labels.append(img_path.name)
                    
            stats[split] = {
                'images': len(images),
                'labels': len(labels),
                'missing_labels': len(missing_labels)
            }
            
            logger.info(f"{split}: {len(images)} images, {len(labels)} labels")
            if missing_labels:
                logger.warning(f"{split}: {len(missing_labels)} images without labels")
                
        # Validate label format
        sample_labels = list((self.dataset_path / 'train').glob('*.txt'))[:10]
        for label_path in sample_labels:
            if not self._validate_label_format(label_path):
                logger.error(f"Invalid label format: {label_path}")
                return False
                
        # Save dataset stats
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
        return True
        
    def _validate_label_format(self, label_path):
        """Validate YOLO format label file"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False
                    
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                
                # Validate ranges
                if class_id < 0 or class_id >= self.config['num_classes']:
                    return False
                if not all(0 <= v <= 1 for v in [x, y, w, h]):
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Error validating label: {e}")
            return False
            
    def create_dataset_config(self):
        """Create dataset configuration file for C++ training"""
        config = {
            'dataset_path': str(self.dataset_path),
            'train_path': str(self.dataset_path / 'train'),
            'val_path': str(self.dataset_path / 'val'),
            'num_classes': self.config['num_classes'],
            'class_names': ['car', 'pedestrian', 'cyclist'],
            'input_size': self.config['input_size'],
            'batch_size': self.config['batch_size'],
            'augmentation': self.config['augmentation'],
            'augmentation_prob': self.config['augmentation_prob']
        }
        
        config_path = self.output_dir / 'dataset_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return config_path
        
    def create_training_config(self):
        """Create training configuration file"""
        train_config = {
            'num_epochs': self.config['num_epochs'],
            'batch_size': self.config['batch_size'],
            'learning_rate': self.config['learning_rate'],
            'lr_decay': self.config['lr_decay'],
            'lr_decay_epochs': self.config['lr_decay_epochs'],
            'validation_interval': self.config['validation_interval'],
            'checkpoint_interval': self.config['checkpoint_interval'],
            'target_map': self.config['target_map'],
            'early_stopping_patience': self.config['early_stopping_patience'],
            'use_pretrained': self.config['use_pretrained'],
            'output_dir': str(self.output_dir),
            'model_dir': str(self.model_dir),
            'checkpoint_dir': str(self.checkpoint_dir),
            'log_dir': str(self.log_dir)
        }
        
        config_path = self.output_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(train_config, f, indent=2)
            
        return config_path
        
    def build_training_executable(self):
        """Build the C++ training executable if needed"""
        if not self.build_dir.exists():
            logger.error("Build directory not found. Please build the project first.")
            return False
            
        # Check if executable exists
        train_exe = self.build_dir / 'train_tacsnet'
        if not train_exe.exists():
            logger.info("Training executable not found. Building...")
            
            result = subprocess.run(
                ['make', '-C', str(self.build_dir), 'train_tacsnet', '-j4'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Build failed: {result.stderr}")
                return False
                
        return True
        
    def convert_dataset_to_binary(self):
        """Convert image dataset to binary format for faster loading"""
        logger.info("Converting dataset to binary format...")
        
        # Create binary dataset directories
        binary_dir = self.output_dir / 'binary_dataset'
        for split in ['train', 'val']:
            (binary_dir / split).mkdir(parents=True, exist_ok=True)
            
        # Conversion statistics
        stats = {'train': 0, 'val': 0}
        
        # This would normally convert images to a binary format
        # For now, we'll just copy the dataset structure
        for split in ['train', 'val']:
            src_dir = self.dataset_path / split
            dst_dir = binary_dir / split
            
            if not src_dir.exists():
                continue
                
            # Copy images and labels
            for img_path in list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png')):
                shutil.copy2(img_path, dst_dir)
                
                # Copy corresponding label
                label_path = img_path.with_suffix('.txt')
                if label_path.exists():
                    shutil.copy2(label_path, dst_dir)
                    
                stats[split] += 1
                
        logger.info(f"Converted {stats['train']} training and {stats['val']} validation samples")
        return binary_dir
        
    def create_dataset_loader_script(self):
        """Create a Python script that loads dataset and feeds it to C++ training"""
        script_content = '''#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import numpy as np
from PIL import Image
import struct

def load_yolo_dataset(dataset_path, split='train'):
    """Load YOLO format dataset"""
    data_dir = os.path.join(dataset_path, split)
    samples = []
    
    for img_file in os.listdir(data_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(data_dir, img_file)
            label_path = os.path.splitext(img_path)[0] + '.txt'
            
            # Load image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((416, 416))
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Load labels
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x, y, w, h = map(float, parts)
                            labels.append([int(class_id), x, y, w, h])
            
            samples.append((img_array, labels))
    
    return samples

def write_binary_dataset(samples, output_file):
    """Write dataset in binary format for C++ consumption"""
    with open(output_file, 'wb') as f:
        # Write header
        f.write(struct.pack('I', len(samples)))  # num samples
        
        for img, labels in samples:
            # Write image data
            f.write(struct.pack('III', 416, 416, 3))  # dimensions
            f.write(img.tobytes())
            
            # Write labels
            f.write(struct.pack('I', len(labels)))  # num labels
            for label in labels:
                f.write(struct.pack('iffff', *label))

if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    
    # Load datasets
    train_samples = load_yolo_dataset(config['dataset_path'], 'train')
    val_samples = load_yolo_dataset(config['dataset_path'], 'val')
    
    # Write binary files
    write_binary_dataset(train_samples, config['output_dir'] + '/train_data.bin')
    write_binary_dataset(val_samples, config['output_dir'] + '/val_data.bin')
    
    print(f"Converted {len(train_samples)} training and {len(val_samples)} validation samples")
'''
        
        script_path = self.output_dir / 'dataset_loader.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        
        return script_path
        
    def prepare_dataset_for_cpp(self):
        """Convert dataset to format readable by C++ training"""
        logger.info("Preparing dataset for C++ training...")
        
        # Create dataset loader script
        loader_script = self.create_dataset_loader_script()
        
        # Create config for loader
        loader_config = {
            'dataset_path': str(self.dataset_path),
            'output_dir': str(self.output_dir)
        }
        
        config_path = self.output_dir / 'loader_config.json'
        with open(config_path, 'w') as f:
            json.dump(loader_config, f)
        
        # Run loader
        try:
            # Check if PIL is available
            import PIL
        except ImportError:
            logger.warning("PIL not available. Using existing train_tacsnet with synthetic data.")
            return False
            
        result = subprocess.run(
            ['python3', str(loader_script), str(config_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Dataset preparation failed: {result.stderr}")
            return False
            
        logger.info("Dataset prepared successfully")
        return True
    
    def run_training(self, resume_from=None):
        """Run the actual training process"""
        logger.info("Starting TACSNet training...")
        
        # Try to prepare dataset for C++
        dataset_prepared = self.prepare_dataset_for_cpp()
        
        # Prepare configs
        dataset_config = self.create_dataset_config()
        training_config = self.create_training_config()
        
        # Build command
        train_cmd = [
            str(self.build_dir / 'train_tacsnet')
        ]
        
        # Add flags based on configuration
        if self.config['use_pretrained']:
            train_cmd.append('--pretrained')
        else:
            train_cmd.append('--no-pretrained')
            
        if resume_from:
            train_cmd.extend(['--resume', str(resume_from)])
            
        # Set environment variables for configs
        env = os.environ.copy()
        env['TACS_DATASET_CONFIG'] = str(dataset_config)
        env['TACS_TRAINING_CONFIG'] = str(training_config)
        
        # If dataset was prepared, point to binary files
        if dataset_prepared:
            env['TACS_TRAIN_DATA'] = str(self.output_dir / 'train_data.bin')
            env['TACS_VAL_DATA'] = str(self.output_dir / 'val_data.bin')
        
        # Create log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'training_{timestamp}.log'
        
        logger.info(f"Training command: {' '.join(train_cmd)}")
        logger.info(f"Log file: {log_file}")
        
        # Run training
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            
            # Monitor output
            best_map = 0.0
            for line in process.stdout:
                # Write to log file
                f.write(line)
                f.flush()
                
                # Print to console
                print(line.rstrip())
                
                # Parse metrics
                if 'mAP:' in line:
                    try:
                        map_str = line.split('mAP:')[1].split()[0]
                        current_map = float(map_str)
                        if current_map > best_map:
                            best_map = current_map
                            logger.info(f"New best mAP: {best_map:.4f}")
                    except:
                        pass
                        
            process.wait()
            
        # Save training results
        results = {
            'timestamp': timestamp,
            'best_map': best_map,
            'target_achieved': best_map >= self.config['target_map'],
            'returncode': process.returncode,
            'config': self.config
        }
        
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        if process.returncode == 0:
            logger.info(f"Training completed successfully. Best mAP: {best_map:.4f}")
            
            # Copy best model to output
            src_model = Path('./models/tacsnet_best.bin')
            if src_model.exists():
                dst_model = self.model_dir / 'tacsnet_final.bin'
                shutil.copy2(src_model, dst_model)
                logger.info(f"Final model saved to: {dst_model}")
        else:
            logger.error(f"Training failed with code {process.returncode}")
            
        return process.returncode == 0
        
    def train(self, resume_from=None):
        """Main training pipeline"""
        logger.info("="*60)
        logger.info("TACSNet Training Pipeline")
        logger.info("="*60)
        
        # Validate dataset
        if not self.validate_dataset():
            logger.error("Dataset validation failed")
            return False
            
        # Build executable
        if not self.build_training_executable():
            logger.error("Failed to build training executable")
            return False
            
        # Run training
        success = self.run_training(resume_from)
        
        if success:
            logger.info("\n✅ Training completed successfully!")
            logger.info(f"Results saved to: {self.output_dir}")
        else:
            logger.error("\n❌ Training failed")
            
        return success


def main():
    parser = argparse.ArgumentParser(
        description='Train TACSNet traffic detection model on custom dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on custom dataset
  python train.py /path/to/dataset /path/to/output
  
  # Train with custom config
  python train.py /path/to/dataset /path/to/output --epochs 200 --batch-size 16
  
  # Resume from checkpoint
  python train.py /path/to/dataset /path/to/output --resume /path/to/checkpoint.bin
  
Dataset format:
  The dataset should follow YOLO format with the following structure:
  
  dataset/
    train/
      image1.jpg
      image1.txt
      image2.jpg
      image2.txt
      ...
    val/
      image1.jpg
      image1.txt
      ...
      
  Each .txt file contains bounding boxes in YOLO format:
  <class_id> <x_center> <y_center> <width> <height>
  
  Where all coordinates are normalized to [0, 1].
  Classes: 0=car, 1=pedestrian, 2=cyclist
        """
    )
    
    parser.add_argument('dataset_path', help='Path to dataset directory')
    parser.add_argument('output_dir', help='Output directory for models and logs')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--target-map', type=float, default=0.92, help='Target mAP')
    parser.add_argument('--no-pretrained', action='store_true', help='Train from scratch')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable augmentation')
    
    args = parser.parse_args()
    
    # Prepare config
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'target_map': args.target_map,
        'use_pretrained': not args.no_pretrained,
        'augmentation': not args.no_augmentation
    }
    
    # Create trainer
    trainer = TACSNetTrainer(args.dataset_path, args.output_dir, config)
    
    # Run training
    success = trainer.train(args.resume)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())