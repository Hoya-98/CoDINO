#!/bin/bash

# Set default values
MODEL_NAME="dinov2_vits14_reg"
RESIZE_DIM=840
IMG_DIR="data/screws/images"
ANNOTATION="data/screws/annotations/annotation.json"
SPLITS="data/screws/annotations/test_split.json"
LORA_WEIGHTS=""  # Default to empty (use pretrained weights)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lora_weights)
            LORA_WEIGHTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run evaluation
python evaluate.py \
    --model_name $MODEL_NAME \
    --resize_dim $RESIZE_DIM \
    --img_dir $IMG_DIR \
    --annotation $ANNOTATION \
    --splits $SPLITS \
    ${LORA_WEIGHTS:+--lora_weights $LORA_WEIGHTS}