#!/usr/bin/env python3
"""
Train TACS traffic detection model - Python wrapper for C++ training
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train TACS traffic detection model')
    parser.add_argument('dataset_path', help='Path to dataset directory')
    parser.add_argument('output_dir', help='Output directory for models')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Build if needed
    if not os.path.exists('./build/train_model'):
        print("Building training executable...")
        subprocess.run(['make', '-C', 'build', 'train_model', '-j4'])
    
    # Run training
    cmd = ['./build/train_model', args.dataset_path, args.output_dir]
    if args.resume:
        cmd.append(args.resume)
    
    print(f"Starting training: {' '.join(cmd)}")
    return subprocess.call(cmd)

if __name__ == '__main__':
    sys.exit(main())