import os
import json
import argparse
from PIL import Image
import numpy as np

def preprocess_annotations(annotation_file, img_dir, target_size=840):
    """Preprocess annotations and images to target size."""
    print(f"Preprocessing data to size {target_size}x{target_size}...")
    
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(img_dir), f"preprocessed_{target_size}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    new_annotations = {}
    for img_name, data in annotations.items():
        # Load image
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} not found")
            continue
            
        img = Image.open(img_path)
        orig_w, orig_h = img.size
        
        # Calculate scaling factors
        scale_w = target_size / orig_w
        scale_h = target_size / orig_h
        
        # Resize image
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Save resized image
        new_img_path = os.path.join(output_dir, img_name)
        img.save(new_img_path)
        
        # Scale bounding boxes
        new_boxes = []
        for box in data['box_examples_coordinates']:
            x1, y1, x2, y2 = box
            new_box = [
                x1 * scale_w,
                y1 * scale_h,
                x2 * scale_w,
                y2 * scale_h
            ]
            new_boxes.append(new_box)
            
        # Store new annotation
        new_annotations[img_name] = {
            'box_examples_coordinates': new_boxes
        }
    
    # Save new annotations
    new_annotation_file = os.path.join(output_dir, 'annotation.json')
    with open(new_annotation_file, 'w') as f:
        json.dump(new_annotations, f, indent=2)
        
    # Copy splits file
    splits_file = os.path.join(os.path.dirname(annotation_file), 'test_split.json')
    if os.path.exists(splits_file):
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        new_splits_file = os.path.join(output_dir, 'test_split.json')
        with open(new_splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
    
    print(f"Preprocessing complete! Output saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--annotation', type=str, required=True,
                        help='Path to annotation file')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Path to image directory')
    parser.add_argument('--target_size', type=int, default=840,
                        help='Target size for preprocessing')
    args = parser.parse_args()
    
    preprocess_annotations(args.annotation, args.img_dir, args.target_size)

if __name__ == '__main__':
    main() 