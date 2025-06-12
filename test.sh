#!/bin/bash

# Set default values
MODEL_NAME="dinov2_vits14_reg"
RESIZE_DIM=840
IMG_DIR="data/screws/images"
ANNOTATION="data/screws/annotations/annotation.json"
SPLITS="data/screws/annotations/test_split.json"
CUSTOM="dinov2_vits14_merged.pth"
# Run evaluation
python convolutional_counting.py \
    --model_name $MODEL_NAME \
    --custom_weights $CUSTOM \
    --resize_dim $RESIZE_DIM \
    --img_dir $IMG_DIR \
    --annotation $ANNOTATION \
    --splits $SPLITS \
    --divide_et_impera \
    --divide_et_impera_twice \
    --filter_background \
    --ellipse_normalization \
    --ellipse_kernel_cleaning