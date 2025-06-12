import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.ops as ops

import numpy as np
import re
import timm
import pandas as pd
import math
from datetime import datetime

from src.model import VisualBackbone
from src.utils import (
    convert_4corners_to_x1y1x2y2, 
    get_counting_metrics, 
    log_results,
    add_dummy_row,
    exist_match_df,
    exist_and_delete_match_df,
    load_json, 
    get_features, 
    bboxes_tointeger, 
    compute_avg_conv_filter, 
    rescale_tensor,
    resize_conv_maps,
    rescale_bbox,
    str2bool,
    ellipse_coverage
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def process_example(
    idx, img_filename, entry, model, transform, map_keys, img_dir, config, return_maps=False
):
    img = Image.open(os.path.join(img_dir, img_filename)).convert('RGB')
    w, h = img.size    

    with torch.no_grad():
        feats = get_features(
            model, img, transform, map_keys,
            divide_et_impera=config.divide_et_impera,
            divide_et_impera_twice=config.divide_et_impera_twice
        )
        if config.cosine_similarity or config.normalize_features:
            feats = feats / feats.norm(dim=1, keepdim=True)

    # Process exemplars
    ex_bboxes = [convert_4corners_to_x1y1x2y2(b) for b in entry['box_examples_coordinates']]
    if config.num_exemplars is not None:
        assert config.num_exemplars > 0, "num_exemplars must be greater than 0. config.num_exemplars = " + config.num_exemplars
        ex_bboxes = ex_bboxes[:config.num_exemplars]
    bboxes = np.array([(x1 / w, y1 / h, x2 / h, y2 / h) for x1, y1, x2, y2 in ex_bboxes]) * feats.shape[-1]
    bboxes = bboxes_tointeger(bboxes, config.remove_bbox_intersection)

    conv_maps = []
    pooled_features_list = []
    output_sizes = []
    rescaled_bboxes = []

    for bbox in bboxes:
        bbox_tensor = torch.tensor(bbox)
        output_size = (
            int(bbox_tensor[3] - bbox_tensor[1]), 
            int(bbox_tensor[2] - bbox_tensor[0])
        )

        pooled = ops.roi_align(
            feats, [bbox_tensor.unsqueeze(0).float().to(device)],
            output_size=output_size, spatial_scale=1.0
        )
        if config.ellipse_kernel_cleaning:
            ellipse = ellipse_coverage(pooled.shape[-2], pooled.shape[-1]).unsqueeze(0).unsqueeze(0).to(device)
            pooled *= ellipse
            
        pooled_features_list.append(pooled)

        if config.exemplar_avg:
            continue

        conv_weights = pooled.view(feats.shape[1], 1, *output_size)
        conv_layer = nn.Conv2d(
            in_channels=feats.shape[1],
            out_channels=1 if config.cosine_similarity else feats.shape[1],
            kernel_size=output_size,
            padding=0,
            groups=1 if config.cosine_similarity else feats.shape[1],
            bias=False
        )
        conv_layer.weight = nn.Parameter(pooled if config.cosine_similarity else conv_weights)

        with torch.no_grad():
            output = conv_layer(feats[0])

        if config.correct_bbox_resize:
            rescaled_bbox = rescale_bbox(bbox_tensor, output, feats)
        else:
            rescaled_bbox = bbox_tensor

        rescaled_bboxes.append(rescaled_bbox)

        if config.use_roi_norm and not config.roi_norm_after_mean:
            if config.cosine_similarity:
                output += 1.0
            pooled_output = ops.roi_align(
                output.unsqueeze(0), [rescaled_bbox.unsqueeze(0).float().to(device)],
                output_size=output_size, spatial_scale=1.0
            )
            output = output / pooled_output.sum()

        conv_maps.append(output)
        output_sizes.append(output_size)

    if config.exemplar_avg:
        pooled = compute_avg_conv_filter(pooled_features_list)
        output_size = pooled.shape[1:]
        conv_weights = pooled.view(pooled.shape[0], 1, *output_size)

        conv_layer = nn.Conv2d(
            in_channels=feats.shape[1],
            out_channels=1 if config.cosine_similarity else feats.shape[1],
            kernel_size=output_size,
            padding=0,
            groups=1 if config.cosine_similarity else feats.shape[1],
            bias=False
        )
        conv_layer.weight = nn.Parameter(pooled.unsqueeze(0) if config.cosine_similarity else conv_weights)

        with torch.no_grad():
            output = conv_layer(feats[0])

        if config.use_roi_norm and not config.roi_norm_after_mean:
            raise NotImplementedError("ROI norm after conv_mean is not implemented for average-based filter.")

        conv_maps.append(output)
        output_sizes.append(output_size)

    output = post_process_density_map(
        conv_maps, pooled_features_list, rescaled_bboxes, output_sizes, config
    )
    if return_maps:
        return None, output
    return None, output.sum().item()


def post_process_density_map(conv_maps, pooled_features_list, rescaled_bboxes, output_sizes, config):
    if config.exemplar_avg:
        output = conv_maps[0]
    else:
        # Resize all conv_maps to the same size
        output, resize_ratios = resize_conv_maps(conv_maps)
        output = output.mean(dim=0)

        if config.use_roi_norm and config.roi_norm_after_mean:
            if config.cosine_similarity:
                output += 1.0
            pooled_vals = []
            for bbox, ratio in zip(rescaled_bboxes, resize_ratios):
                scaled_bbox = torch.tensor([
                    bbox[0] * ratio[1], bbox[1] * ratio[0],
                    bbox[2] * ratio[1], bbox[3] * ratio[0]
                ]).int()
                output_size = (
                    int(scaled_bbox[3] - scaled_bbox[1]),
                    int(scaled_bbox[2] - scaled_bbox[0])
                )
                pooled = ops.roi_align(
                    output.unsqueeze(0).unsqueeze(0),
                    [scaled_bbox.unsqueeze(0).float().to(device)],
                    output_size=output_size, spatial_scale=1.0
                )
                pooled_vals.append(pooled)

            if config.ellipse_normalization:
                norm_coeff = sum([(p[0, 0] * ellipse_coverage(p.shape[-2], p.shape[-1]).to(device)).sum() for p in pooled_vals]) / (len(pooled_vals) * config.scaling_coeff)
            else:
                norm_coeff = sum([p.sum() for p in pooled_vals]) / (len(pooled_vals) * config.scaling_coeff)
            if config.fixed_norm_coeff is not None:
                norm_coeff = config.fixed_norm_coeff

            output = output / norm_coeff

    if config.use_minmax_norm:
        output = (output - output.min()) / (output.max() - output.min())

    if config.use_threshold:
        output = (output > config.threshold).float()

    if config.filter_background:
        output = filter_background(output, pooled_features_list, config)

    if config.ellipse_normalization:
        output = ellipse_normalization(output)

    return output


def filter_background(density_map, pooled_features_list, config):
    """Filter out background noise from the density map."""
    if not pooled_features_list:
        return density_map
        
    # Calculate threshold based on the size of the largest exemplar
    max_size = max([f.shape[-2] * f.shape[-1] for f in pooled_features_list])
    threshold = (1.0 / max_size) * config.scaling_coeff
    
    # Apply threshold
    density_map[density_map < threshold] = 0
    
    return density_map


def ellipse_normalization(density_map):
    """Apply ellipse normalization to the density map."""
    h, w = density_map.shape[-2:]
    ellipse = ellipse_coverage(h, w).to(device)
    return density_map * ellipse


def parse_args():
    parser = argparse.ArgumentParser(description='Convolutional Counting')
    parser.add_argument('--model_name', type=str, default='dinov2_vits14_reg',
                        help='Name of the model to use')
    parser.add_argument('--custom_weights', type=str, default=None,
                        help='Path to custom model weights (e.g., LoRA fine-tuned model)')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--annotation', type=str, required=True,
                        help='Path to annotation file')
    parser.add_argument('--splits', type=str, required=True,
                        help='Path to splits file')
    parser.add_argument('--divide_et_impera', action='store_true',
                        help='Use divide et impera strategy')
    parser.add_argument('--divide_et_impera_twice', action='store_true',
                        help='Use divide et impera strategy twice')
    parser.add_argument('--exemplar_avg', action='store_true',
                        help='Average exemplar features')
    parser.add_argument('--cosine_similarity', action='store_true',
                        help='Use cosine similarity')
    parser.add_argument('--normalize_features', action='store_true',
                        help='Normalize features')
    parser.add_argument('--normalize_only_biggest_bbox', action='store_true',
                        help='Normalize only the biggest bounding box')
    parser.add_argument('--use_threshold', action='store_true',
                        help='Use threshold for counting')
    parser.add_argument('--use_roi_norm', action='store_true',
                        help='Use ROI normalization')
    parser.add_argument('--roi_norm_after_mean', action='store_true',
                        help='Apply ROI normalization after mean')
    parser.add_argument('--use_minmax_norm', action='store_true',
                        help='Use min-max normalization')
    parser.add_argument('--remove_bbox_intersection', action='store_true',
                        help='Remove bounding box intersections')
    parser.add_argument('--correct_bbox_resize', action='store_true',
                        help='Correct bounding box resize')
    parser.add_argument('--scaling_coeff', type=float, default=1.0,
                        help='Scaling coefficient')
    parser.add_argument('--fixed_norm_coeff', type=float, default=None,
                        help='Fixed normalization coefficient')
    parser.add_argument('--filter_background', action='store_true',
                        help='Filter background')
    parser.add_argument('--ellipse_normalization', action='store_true',
                        help='Use ellipse normalization')
    parser.add_argument('--ellipse_kernel_cleaning', action='store_true',
                        help='Use ellipse kernel cleaning')
    parser.add_argument('--split', type=str, default='test',
                        help='Split to use')
    parser.add_argument('--num_exemplars', type=int, default=3,
                        help='Number of exemplars to use')
    parser.add_argument('--save_preds_to_file', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--log_results', action='store_true',
                        help='Log results')
    parser.add_argument('--no_skip', action='store_true',
                        help='Do not skip any images')
    parser.add_argument('--resize_dim', type=int, default=840,
                        help='Resize dimension')
    parser.add_argument('--log_file', type=str, default='results/results.json',
                        help='Path to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment for tracking')
    return parser.parse_args()


def main():
    args = parse_args()
    print("Parameters Recap:")
    print(json.dumps(vars(args), indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create experiment directory
    if args.experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join("results", f"{args.experiment_name}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        args.log_file = os.path.join(experiment_dir, "results.json")
        
        # Save configuration
        with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
            json.dump(vars(args), f, indent=2)

    # Load model
    model = VisualBackbone(
        args.model_name, 
        img_size=args.resize_dim,
        custom_weights=args.custom_weights
    ).to(device).eval()

    # Use model's transform
    transform = model.get_transform()

    # Load annotations and splits
    with open(args.annotation, 'r') as f:
        annotations = json.load(f)
    with open(args.splits, 'r') as f:
        splits = json.load(f)

    # Process images
    results = []
    total_mae = 0
    total_mse = 0
    count = 0

    for idx, img_filename in enumerate(tqdm(splits[args.split])):
        if img_filename not in annotations:
            continue

        entry = {
            'filename': img_filename,
            'box_examples_coordinates': annotations[img_filename]['box_examples_coordinates']
        }

        _, pred = process_example(idx, img_filename, entry, model, transform, map_keys, args.img_dir, args)
        gt = len(annotations[img_filename]['box_examples_coordinates'])

        # Calculate metrics
        mae = abs(pred - gt)
        mse = (pred - gt) ** 2
        total_mae += mae
        total_mse += mse
        count += 1

        # Store detailed results
        result = {
            'filename': img_filename,
            'predicted_count': float(pred),
            'ground_truth': gt,
            'mae': float(mae),
            'mse': float(mse),
            'exemplar_boxes': entry['box_examples_coordinates']
        }
        results.append(result)

    # Calculate final metrics
    final_mae = total_mae / count
    final_mse = total_mse / count
    final_rmse = math.sqrt(final_mse)

    # Create summary
    summary = {
        'model_name': args.model_name,
        'dataset': 'screws',
        'split': args.split,
        'total_images': count,
        'final_metrics': {
            'mae': float(final_mae),
            'mse': float(final_mse),
            'rmse': float(final_rmse)
        },
        'parameters': vars(args),
        'results': results
    }

    # Save results
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    with open(args.log_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation completed:")
    print(f"Total images processed: {count}")
    print(f"MAE: {final_mae:.2f}")
    print(f"MSE: {final_mse:.2f}")
    print(f"RMSE: {final_rmse:.2f}")
    print(f"Results saved to: {args.log_file}")

if __name__ == "__main__":
    main()
