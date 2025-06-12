import os
import json
import argparse
from PIL import Image
import numpy as np

def validate_annotations(annotation_file, img_dir):
    """Validate annotation file and image files."""
    print("Validating annotations and images...")
    
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Check each image
    for img_name, data in annotations.items():
        # Check if image exists
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} not found")
            continue
            
        # Load image to get dimensions
        img = Image.open(img_path)
        img_w, img_h = img.size
        
        # Check bounding boxes
        boxes = data['box_examples_coordinates']
        if len(boxes) != 3:
            print(f"Warning: {img_name} has {len(boxes)} boxes, expected 3")
            
        for box in boxes:
            x1, y1, x2, y2 = box
            # Check if coordinates are within image bounds
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                print(f"Warning: Box coordinates out of bounds in {img_name}: {box}")
            # Check if box is valid (x2 > x1 and y2 > y1)
            if x2 <= x1 or y2 <= y1:
                print(f"Warning: Invalid box coordinates in {img_name}: {box}")
                
        # Check if image dimensions are reasonable
        if img_w > 2000 or img_h > 2000:
            print(f"Warning: Large image dimensions in {img_name}: {img_w}x{img_h}")
            
    print("Validation complete!")

def validate_splits(split_file, annotation_file):
    """Validate split file against annotation file."""
    print("Validating splits...")
    
    # Load splits and annotations
    with open(split_file, 'r') as f:
        splits = json.load(f)
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Check if all images in splits have annotations
    for split_name, img_list in splits.items():
        print(f"\nChecking {split_name} split...")
        for img_name in img_list:
            if img_name not in annotations:
                print(f"Warning: {img_name} in {split_name} split has no annotation")
                
    print("Split validation complete!")

def main():
    parser = argparse.ArgumentParser(description='Validate dataset')
    parser.add_argument('--annotation', type=str, required=True,
                        help='Path to annotation file')
    parser.add_argument('--splits', type=str, required=True,
                        help='Path to splits file')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Path to image directory')
    args = parser.parse_args()
    
    # Validate annotations and images
    validate_annotations(args.annotation, args.img_dir)
    
    # Validate splits
    validate_splits(args.splits, args.annotation)

if __name__ == '__main__':
    main() 