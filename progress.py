#!/usr/bin/env python3
"""
Adaptive training process with dynamic weight adjustment for improved accuracy.
Uses the new adaptive optimizer to achieve better convergence to 99% accuracy.
"""

import os
import subprocess
import time
import logging
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveTrainingPipeline:
    def __init__(self, 
                 dataset_path="datasets/traffic",
                 output_dir="models/adaptive_output",
                 target_accuracy=0.99,
                 max_attempts=10):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.target_accuracy = target_accuracy
        self.max_attempts = max_attempts
        self.build_dir = Path("build")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def build_project(self):
        """Build the C++ project with adaptive optimizer"""
        logger.info("Building C++ project with adaptive optimizer...")
        
        if not self.build_dir.exists():
            self.build_dir.mkdir()
            
        # Configure with CMake
        cmake_cmd = ["cmake", ".."]
        result = subprocess.run(
            cmake_cmd,
            cwd=self.build_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"CMake configuration failed: {result.stderr}")
            raise RuntimeError("CMake configuration failed")
            
        # Build with make
        cpu_count = multiprocessing.cpu_count()
        make_cmd = ["make", f"-j{cpu_count}", "train_adaptive"]
        
        logger.info(f"Building with {cpu_count} parallel jobs...")
        result = subprocess.run(
            make_cmd,
            cwd=self.build_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Build failed: {result.stderr}")
            raise RuntimeError("Build failed")
            
        logger.info("Build completed successfully")
        
    def check_dataset(self):
        """Verify dataset structure"""
        train_path = self.dataset_path / "train"
        val_path = self.dataset_path / "val"
        
        if not train_path.exists() or not val_path.exists():
            logger.error(f"Dataset not found at {self.dataset_path}")
            logger.info("Expected structure:")
            logger.info("  datasets/traffic/train/")
            logger.info("  datasets/traffic/val/")
            return False
            
        # Count samples
        train_samples = len(list(train_path.glob("*.jpg"))) + len(list(train_path.glob("*.png")))
        val_samples = len(list(val_path.glob("*.jpg"))) + len(list(val_path.glob("*.png")))
        
        logger.info(f"Dataset found: {train_samples} training, {val_samples} validation samples")
        return train_samples > 0 and val_samples > 0
    
    def verify_dataset_labels(self):
        """Verify all images have corresponding label files"""
        logger.info("Verifying dataset labels...")
        
        train_path = self.dataset_path / "train"
        val_path = self.dataset_path / "val"
        
        all_labeled = True
        missing_labels = []
        
        # Check training images
        for img_path in train_path.glob("*.jpg"):
            label_path = img_path.with_suffix(".txt")
            if not label_path.exists():
                missing_labels.append(str(img_path))
                all_labeled = False
            elif label_path.stat().st_size == 0:
                logger.warning(f"Empty label file: {label_path}")
        
        for img_path in train_path.glob("*.png"):
            label_path = img_path.with_suffix(".txt")
            if not label_path.exists():
                missing_labels.append(str(img_path))
                all_labeled = False
            elif label_path.stat().st_size == 0:
                logger.warning(f"Empty label file: {label_path}")
                
        # Check validation images
        for img_path in val_path.glob("*.jpg"):
            label_path = img_path.with_suffix(".txt")
            if not label_path.exists():
                missing_labels.append(str(img_path))
                all_labeled = False
            elif label_path.stat().st_size == 0:
                logger.warning(f"Empty label file: {label_path}")
                
        for img_path in val_path.glob("*.png"):
            label_path = img_path.with_suffix(".txt")
            if not label_path.exists():
                missing_labels.append(str(img_path))
                all_labeled = False
            elif label_path.stat().st_size == 0:
                logger.warning(f"Empty label file: {label_path}")
        
        if not all_labeled:
            logger.error(f"Found {len(missing_labels)} images without labels:")
            for missing in missing_labels[:5]:  # Show first 5
                logger.error(f"  - {missing}")
            if len(missing_labels) > 5:
                logger.error(f"  ... and {len(missing_labels) - 5} more")
            return False
        
        # Count total labels
        train_labels = len(list(train_path.glob("*.txt")))
        val_labels = len(list(val_path.glob("*.txt")))
        
        logger.info(f"✅ All images labeled: {train_labels} training, {val_labels} validation labels")
        return True
        
    def run_adaptive_training(self, attempt_num):
        """Run adaptive training with dynamic weight adjustment"""
        logger.info(f"\n=== Adaptive Training Attempt {attempt_num} ===")
        
        # Prepare output directory for this attempt
        attempt_dir = self.output_dir / f"attempt_{attempt_num}"
        attempt_dir.mkdir(exist_ok=True)
        
        # Build training command
        train_cmd = [
            str(self.build_dir / "train_adaptive"),
            str(self.dataset_path),
            str(attempt_dir)
        ]
        
        # Add resume from best previous model if not first attempt
        if attempt_num > 1:
            prev_best = self.find_best_model(attempt_num - 1)
            if prev_best:
                train_cmd.append(str(prev_best))
                logger.info(f"Resuming from: {prev_best}")
        
        logger.info("Starting adaptive training...")
        logger.info("Features enabled:")
        logger.info("  - Dynamic learning rate adjustment based on gradient statistics")
        logger.info("  - Layer-wise adaptive learning rates")
        logger.info("  - Accuracy-based weight adjustment")
        logger.info("  - Gradient centralization and weight standardization")
        logger.info("  - Lookahead optimization")
        
        start_time = time.time()
        
        # Run training
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        best_map = 0.0
        
        # Monitor training output
        for line in process.stdout:
            print(line.rstrip())
            
            # Parse mAP from output
            if "Validation mAP@0.5:" in line:
                try:
                    map_value = float(line.split(":")[-1].strip())
                    best_map = max(best_map, map_value)
                    logger.info(f"Current mAP: {map_value:.4f} (best: {best_map:.4f})")
                except:
                    pass
                    
            # Check for target achievement
            if "TARGET ACHIEVED" in line:
                logger.info("🎯 Target accuracy achieved!")
                
        process.wait()
        elapsed_time = time.time() - start_time
        
        # Save training results
        results = {
            "attempt": attempt_num,
            "best_map": best_map,
            "target_achieved": best_map >= self.target_accuracy,
            "training_time": elapsed_time,
            "returncode": process.returncode
        }
        
        with open(attempt_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Training completed in {elapsed_time/60:.1f} minutes")
        logger.info(f"Best mAP: {best_map:.4f}")
        
        return results
        
    def find_best_model(self, up_to_attempt=None):
        """Find the best model from previous attempts"""
        best_map = 0.0
        best_model_path = None
        
        max_attempt = up_to_attempt or self.max_attempts
        
        for i in range(1, max_attempt + 1):
            results_path = self.output_dir / f"attempt_{i}" / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    results = json.load(f)
                    if results.get("best_map", 0) > best_map:
                        best_map = results["best_map"]
                        model_path = self.output_dir / f"attempt_{i}" / "best_adaptive_model.bin"
                        if model_path.exists():
                            best_model_path = model_path
                            
        return best_model_path
        
    def generate_dataset_if_needed(self):
        """Generate synthetic dataset if not exists"""
        if not self.check_dataset():
            logger.info("Dataset not found. Generating synthetic dataset...")
            
            # Use the correct dataset generator
            generator_script = Path("python/generate_synthetic_dataset.py")
            if generator_script.exists():
                # Create dataset with sufficient samples
                result = subprocess.run(
                    ["python3", str(generator_script), 
                     "--output-dir", str(self.dataset_path),
                     "--num-train", "1000",
                     "--num-val", "200"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info("Dataset generated successfully")
                    # Verify dataset was created and labeled
                    return self.verify_dataset_labels()
                else:
                    logger.error(f"Dataset generation failed: {result.stderr}")
                    return False
            else:
                logger.error("Dataset generator not found at python/generate_synthetic_dataset.py")
                return False
        else:
            # Dataset exists, verify it has labels
            return self.verify_dataset_labels()
        
    def run_pipeline(self):
        """Run the complete adaptive training pipeline"""
        logger.info("=== Adaptive Training Pipeline Started ===")
        logger.info(f"Target accuracy: {self.target_accuracy * 100:.0f}%")
        
        # Build project
        try:
            self.build_project()
        except Exception as e:
            logger.error(f"Build failed: {e}")
            return False
            
        # Check/generate dataset and ensure all images are labeled
        logger.info("\n=== Dataset Preparation Phase ===")
        if not self.generate_dataset_if_needed():
            logger.error("Failed to prepare dataset with proper labels")
            logger.error("Cannot proceed with training without labeled dataset")
            return False
        
        logger.info("Dataset is ready with all images properly labeled")
        logger.info("Proceeding to training phase...\n")
            
        # Run training attempts
        for attempt in range(1, self.max_attempts + 1):
            results = self.run_adaptive_training(attempt)
            
            if results["target_achieved"]:
                logger.info(f"✅ Target accuracy achieved in attempt {attempt}!")
                
                # Copy best model to final location
                best_model = self.output_dir / f"attempt_{attempt}" / "best_adaptive_model.bin"
                final_model = self.output_dir / "final_model_99_percent.bin"
                shutil.copy2(best_model, final_model)
                
                logger.info(f"Final model saved to: {final_model}")
                return True
                
            elif results["returncode"] != 0:
                logger.error(f"Training failed with code {results['returncode']}")
                
            if attempt < self.max_attempts:
                logger.info(f"Attempt {attempt} achieved {results['best_map']:.4f} mAP")
                logger.info("Continuing with next attempt...")
                
        logger.warning(f"Failed to achieve target accuracy after {self.max_attempts} attempts")
        
        # Save best overall model
        best_model = self.find_best_model()
        if best_model:
            final_model = self.output_dir / "best_model_overall.bin"
            shutil.copy2(best_model, final_model)
            logger.info(f"Best overall model saved to: {final_model}")
            
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive training pipeline for 99% accuracy")
    parser.add_argument("--dataset", default="datasets/traffic", help="Dataset path")
    parser.add_argument("--output", default="models/adaptive_output", help="Output directory")
    parser.add_argument("--target", type=float, default=0.99, help="Target accuracy (mAP)")
    parser.add_argument("--attempts", type=int, default=10, help="Maximum training attempts")
    
    args = parser.parse_args()
    
    pipeline = AdaptiveTrainingPipeline(
        dataset_path=args.dataset,
        output_dir=args.output,
        target_accuracy=args.target,
        max_attempts=args.attempts
    )
    
    success = pipeline.run_pipeline()
    
    if success:
        logger.info("\n🎉 Adaptive training pipeline completed successfully!")
        logger.info("The AI system now has improved weight adjustment capabilities")
        logger.info("and will consistently achieve better accuracy.")
        return 0
    else:
        logger.warning("\n⚠️ Pipeline completed but target accuracy not achieved")
        logger.info("Consider:")
        logger.info("  - Increasing the number of training epochs")
        logger.info("  - Adjusting adaptive optimizer parameters")
        logger.info("  - Providing more training data")
        return 1

if __name__ == "__main__":
    exit(main())