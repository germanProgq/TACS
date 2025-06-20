#!/usr/bin/env python3
"""Visualize dataset with bounding boxes"""

import os
from PIL import Image, ImageDraw
import sys

def visualize_dataset_sample(dataset_dir, output_dir="visualization"):
    """Visualize a few samples from the dataset with bounding boxes"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define class names and colors
    class_names = ['car', 'pedestrian', 'cyclist']
    class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    
    # Process training samples
    train_images_dir = os.path.join(dataset_dir, "train", "images")
    train_labels_dir = os.path.join(dataset_dir, "train", "labels")
    
    # Get first 3 images
    image_files = sorted([f for f in os.listdir(train_images_dir) if f.endswith('.jpg')])[:3]
    
    for img_file in image_files:
        # Load image
        img_path = os.path.join(train_images_dir, img_file)
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Load corresponding labels
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(train_labels_dir, label_file)
        
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        
                        # Convert from YOLO format to pixel coordinates
                        x1 = int((cx - w/2) * width)
                        y1 = int((cy - h/2) * height)
                        x2 = int((cx + w/2) * width)
                        y2 = int((cy + h/2) * height)
                        
                        # Draw bounding box
                        color = class_colors[class_id]
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        
                        # Draw label
                        label = f"{class_names[class_id]}"
                        draw.text((x1, y1-10), label, fill=color)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"vis_{img_file}")
        img.save(output_path)
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "test_dataset"
    visualize_dataset_sample(dataset_dir)