#!/usr/bin/env bash

# Example runner commands for custom_gaussiansplat/train.py
# Required env:
#   export BASE_PATH="/path/to/scene_root"
#   # expects ${BASE_PATH}/sparse/0 and ${BASE_PATH}/images

python custom_gaussiansplat/train.py \
  --colmap-path "$BASE_PATH/sparse/0" \
  --images-path "$BASE_PATH/images" \
  --output-dir "$BASE_PATH/cust_gs_outputs" \
  --verbosity 1 \
  --log-interval 100 \
  --densify-from-iter 1000 \
  --densify-interval 100 \
  --viewer \
  --iterations 30000

# Depth objective can be selected via --depth-objective {pearson,silog}
python custom_gaussiansplat/train.py \
  --colmap-path "$BASE_PATH/sparse/0" \
  --images-path "$BASE_PATH/images" \
  --output-dir "$BASE_PATH/cust_gs_outputs_depth" \
  --iterations 30000 \
  --enable-depth-loss \
  --depth-loss-weight 0.01 \
  --depth-loss-start-iter 0 \
  --depth-objective pearson

# Advanced regularization + semantics setup
python custom_gaussiansplat/train.py \
  --colmap-path "$BASE_PATH/sparse/0" \
  --images-path "$BASE_PATH/images" \
  --output-dir "$BASE_PATH/cust_gs_outputs_semantics" \
  --log-interval 10 \
  --densify-from-iter 0 \
  --densify-interval 500 \
  --densify-until-iter 15000 \
  --tb-image-interval 100 \
  --viewer \
  --iterations 30000 \
  --sh-degree 3 \
  --lr-means 0.00002 \
  --lr-quats 0.00002 \
  --lr-scales 0.001 \
  --lr-sh 0.002 \
  --grad-threshold 0.0002 \
  --enable-scale-reg --scale-reg-weight 0.01 \
  --enable-opacity-entropy-reg \
  --sam-loss-weight 0.02 \
  --train-semantics \
  --semantics-dim 4 \
  --semantic-start-iter 15000 \
  --semantics-path "${BASE_PATH}/language_features_dim3" \
  --verbosity 3 \
  --enable-opacity-reg --opacity-reg-weight 0.0005 \
