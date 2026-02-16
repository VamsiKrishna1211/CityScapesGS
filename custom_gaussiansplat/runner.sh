python train.py --colmap-path $BASE_PATH/sparse/0 --images-path $BASE_PATH/images --output-dir $BASE_PATH/cust_gs_outputs --verbosity 1 --log-interval 100 --densify-from-iter 1000 --densify-interval 100 --viewer --iterations 30000


python custom_gaussiansplat/train.py \
    --colmap-path data/boston/sparse/0 \
    --images-path data/boston/images \
    --enable-scale-reg --scale-reg-weight 0.05 \
    --enable-depth-culling --near-plane-threshold 0.1 \
    --enable-aggressive-pruning --max-world-scale 0.08 \
    --enable-visibility-tracking --min-view-count 3

python custom_gaussiansplat/train.py \
    --colmap-path $BASE_PATH/sparse/0 \
    --images-path $BASE_PATH/images \
    --output-dir $BASE_PATH/cust_gs_outputs \
    --log-interval 10 \
    --densify-from-iter 10000 \
    --densify-interval 100 \
    --viewer \
    --iterations 30000 \
    --sh-degree 3 \
    --enable-scale-reg --scale-reg-weight 0.01 \
    --enable-visibility-tracking --min-view-count 3 \
    --enable-scale-reg \
    --enable-depth-culling

python custom_gaussiansplat/train.py \
    --colmap-path $BASE_PATH/sparse/0 \
    --images-path $BASE_PATH/images \
    --output-dir $BASE_PATH/cust_gs_outputs \
    --log-interval 100 \
    --densify-from-iter 100 \
    --densify-interval 10 \
    --viewer \
    --iterations 30000 \
    --sh-degree 3


python custom_gaussiansplat/train.py \
    --colmap-path $BASE_PATH/sparse/0 \
    --images-path $BASE_PATH/images \
    --output-dir $BASE_PATH/cust_gs_outputs \
    --log-interval 10 \
    --densify-from-iter 10000 \
    --densify-interval 100 \
    --viewer \
    --iterations 30000 \
    --sh-degree 3 \
    --lr-means 0.0002 \
    --enable-scale-reg --scale-reg-weight 0.01 \
    --enable-visibility-tracking --min-view-count 3 \
    --enable-scale-reg \
    --enable-depth-culling

python custom_gaussiansplat/train.py \
    --colmap-path $BASE_PATH/sparse/0 \
    --images-path $BASE_PATH/images \
    --output-dir $BASE_PATH/cust_gs_outputs \
    --log-interval 10 \
    --densify-from-iter 0 \
    --densify-interval 1000 \
    --densify-until-iter 10000 \
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
    --enable-opacity-reg --opacity-reg-weight 0.0005 \
    --verbosity 3 \
    --enable-depth-loss \
    --depth-loss-weight 0.01 \
    --depth-loss-start-iter 0
    # --phase-starts 500 3000 15000 \
    # --phase-ends 3000 15000 30000 \
    # --phase-densify-intervals 100 100 0 \
    # --phase-opacity-resets 3000 0 0

python custom_gaussiansplat/train.py \
    --colmap-path $BASE_PATH/sparse/0 \
    --images-path $BASE_PATH/images \
    --iterations 30000 \

python custom_gaussiansplat/train.py \
    --colmap-path $BASE_PATH/sparse/0 \
    --images-path $BASE_PATH/images \
    --output-dir $BASE_PATH/cust_gs_outputs \
    --log-interval 10 \
    --densify-from-iter 0 \
    --densify-interval 1000 \
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
    --enable-opacity-reg --opacity-reg-weight 0.0005 \
    --verbosity 3
