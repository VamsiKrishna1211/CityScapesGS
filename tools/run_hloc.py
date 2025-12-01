#!/usr/bin/env python3
"""
HLOC Structure-from-Motion Pipeline
Runs feature extraction, matching, and 3D reconstruction using HLOC.
"""

import argparse
import sys
from pathlib import Path

import torch

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

from hloc import (
    extract_features,
    match_features,
    match_dense,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_exhaustive,
)


def parse_args():
    """Parse command line arguments with dynamically inferred config options."""
    
    # Get available configurations dynamically
    retrieval_options = list(extract_features.confs.keys())
    feature_options = list(extract_features.confs.keys())
    matcher_options = list(match_features.confs.keys())
    dense_matcher_options = list(match_dense.confs.keys())
    
    parser = argparse.ArgumentParser(
        description="Run HLOC Structure-from-Motion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Configurations:
  Retrieval configs: {', '.join(retrieval_options)}
  Feature configs: {', '.join(feature_options)}
  Matcher configs: {', '.join(matcher_options)}
  Dense matcher configs: {', '.join(dense_matcher_options)}
        """
    )
    
    # Path arguments
    parser.add_argument(
        "--base_path",
        type=Path,
        default=Path("/home/vamsik1211/Data/Projects/3D-Reconstructions/CityScapeGS/data/boston"),
        help="Base path for the dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--images_suffix",
        type=str,
        default="images",
        help="Images directory suffix (default: %(default)s)"
    )
    parser.add_argument(
        "--masks_suffix",
        type=str,
        default="masks_filtered_colmap",
        help="Masks directory suffix (default: %(default)s)"
    )
    parser.add_argument(
        "--outputs_suffix",
        type=str,
        default="hloc_outputs",
        help="Outputs directory suffix (default: %(default)s)"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--retrieval_config",
        type=str,
        default="netvlad",
        choices=retrieval_options,
        help=f"Retrieval configuration (default: %(default)s)"
    )
    parser.add_argument(
        "--feature_config",
        type=str,
        default="superpoint_max",
        choices=feature_options,
        help=f"Feature extraction configuration (default: %(default)s)"
    )
    parser.add_argument(
        "--matcher_config",
        type=str,
        default="superglue",
        choices=matcher_options,
        help=f"Feature matching configuration (default: %(default)s)"
    )
    parser.add_argument(
        "--dense_matcher_config",
        type=str,
        default="loftr",
        choices=dense_matcher_options,
        help=f"Dense matching configuration (not used by default) (default: %(default)s)"
    )
    parser.add_argument(
        "--use_dense_matcher",
        action="store_true",
        help="Use dense matcher instead of sparse matcher"
    )
    
    # Retrieval arguments
    parser.add_argument(
        "--num_matched",
        type=int,
        default=300,
        help="Number of image pairs to retrieve (default: %(default)s)"
    )
    
    # Camera arguments
    parser.add_argument(
        "--camera_mode",
        type=str,
        default="PER_IMAGE",
        choices=["SINGLE", "PER_FOLDER", "PER_IMAGE"],
        help="Camera mode for reconstruction (default: %(default)s)"
    )
    parser.add_argument(
        "--image_options",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=None,
        help="Image options as key-value pairs (can be used multiple times). "
             "Example: --image_options camera_model PINHOLE --image_options camera_params '1800,1800,1920,1080'. "
             "Default: camera_model=PINHOLE, camera_params=1800.161857181376,1800.161857181376,1920.0,1080.0"
    )
    
    # Mapper options
    parser.add_argument(
        "--mapper_options",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=None,
        help="COLMAP Mapper options as key-value pairs (can be used multiple times). "
             "Example: --mapper_options ba_local_num_images 20 --mapper_options filter_max_reproj_error 1.5\n\n"
             
             "INITIALIZATION:\n"
             "  init_min_num_inliers (default: 100) - Min inlier matches for initial reconstruction\n"
             "  init_max_error (default: 4.0) - Max reprojection error (px) for inliers during init\n"
             "  init_max_forward_motion (default: 0.95) - Max forward motion ratio (0.0-1.0)\n"
             "  init_min_tri_angle (default: 16.0) - Min triangulation angle (degrees) for init\n"
             "  init_max_reg_trials (default: 2) - Max attempts to find initial image pair\n\n"
             
             "CAMERA REGISTRATION (Absolute Pose):\n"
             "  abs_pose_max_error (default: 12.0) - Max reprojection error (px) for P3P RANSAC\n"
             "  abs_pose_min_num_inliers (default: 30) - Min 2D-3D correspondences for registration\n"
             "  abs_pose_min_inlier_ratio (default: 0.25) - Min ratio of inliers to total matches\n"
             "  abs_pose_refine_focal_length (default: True) - Refine focal length during pose estimation\n"
             "  abs_pose_refine_extra_params (default: True) - Refine distortion params during pose estimation\n\n"
             
             "BUNDLE ADJUSTMENT:\n"
             "  ba_local_num_images (default: 6) - Images to optimize in local BA after registration\n"
             "  ba_local_min_tri_angle (default: 6.0) - Min triangulation angle (degrees) for local BA\n"
             "  ba_global_max_num_iterations (default: 10) - Max iterations for global BA\n"
             "  ba_global_ignore_redundant_points3D (default: False) - Skip points with many observations\n"
             "  ba_global_prune_points_min_coverage_gain (default: 0.05) - Threshold for removing points\n\n"
             
             "FILTERING & QUALITY:\n"
             "  filter_max_reproj_error (default: 4.0) - Max reprojection error (px) to keep 3D point\n"
             "  filter_min_tri_angle (default: 1.5) - Min triangulation angle (degrees) to keep point\n"
             "  min_focal_length_ratio (default: 0.1) - Min focal_length / max(width,height)\n"
             "  max_focal_length_ratio (default: 10.0) - Max focal_length / max(width,height)\n"
             "  max_extra_param (default: 1.0) - Max absolute value for distortion coefficients\n\n"
             
             "STRATEGY & OPTIMIZATION:\n"
             "  max_reg_trials (default: 3) - Max attempts to register an image before skipping\n"
             "  fix_existing_frames (default: False) - Don't optimize already-registered cameras\n"
             "  image_selection_method (default: MIN_UNCERTAINTY) - Next image selection strategy:\n"
             "    MIN_UNCERTAINTY, MAX_VISIBLE_POINTS_NUM, MAX_VISIBLE_POINTS_RATIO, MIN_UNCERTAINTY_MIN_NUM_INLIERS\n\n"
             
             "PERFORMANCE:\n"
             "  num_threads (default: -1) - Number of threads (-1 = use all cores)\n"
             "  random_seed (default: -1) - RANSAC seed for reproducibility (-1 = random)"
    )
    
    # Visualization
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the SfM reconstruction"
    )
    parser.add_argument(
        "--skip_reconstruction",
        action="store_true",
        help="Skip reconstruction step (only extract features and match)"
    )
    
    return parser.parse_args()


def parse_mapper_value(value_str):
    """Parse mapper option value to appropriate type."""
    # Handle boolean strings
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    
    # Try int first
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string
    return value_str


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    # Setup paths using base_path and suffixes
    base_path = args.base_path
    
    images = base_path / args.images_suffix
    masks_dir = base_path / args.masks_suffix
    outputs = base_path / args.outputs_suffix
    outputs.mkdir(exist_ok=True, parents=True)
    
    sfm_pairs = outputs / "pairs-sfm.txt"
    sfm_dir = outputs / "sfm"
    sfm_dir.mkdir(exist_ok=True, parents=True)
    
    # Validate paths
    if not images.exists():
        print(f"Error: Images directory not found: {images}")
        sys.exit(1)
    
    # Get configurations
    retrieval_config = extract_features.confs[args.retrieval_config]
    feature_config = extract_features.confs[args.feature_config]
    matcher_config = match_features.confs[args.matcher_config]
    dense_matcher_config = match_dense.confs[args.dense_matcher_config]
    
    print(f"Base path: {base_path}")
    print(f"Images: {images}")
    print(f"Masks: {masks_dir}")
    print(f"Outputs: {outputs}")
    print(f"\nRetrieval config: {args.retrieval_config}")
    print(f"Feature config: {args.feature_config}")
    print(f"Matcher config: {args.matcher_config}")
    if args.use_dense_matcher:
        print(f"Dense matcher config: {args.dense_matcher_config}")
    
    # Step 1: Extract retrieval features and generate pairs
    print("\n[1/4] Extracting retrieval features...")
    retrieval_path = extract_features.main(retrieval_config, images, outputs)
    
    print(f"[2/4] Generating image pairs (num_matched={args.num_matched})...")
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=args.num_matched)
    # pairs_from_exhaustive.main(sfm_pairs, image_list=images)
    
    # Step 2: Extract features and match
    print("[3/4] Extracting features...")
    feature_path = extract_features.main(feature_config, images, outputs)
    
    print("[3/4] Matching features...")
    if args.use_dense_matcher:
        match_path = match_dense.main(
            dense_matcher_config, sfm_pairs, images, outputs
        )
    else:
        match_path = match_features.main(
            matcher_config, sfm_pairs, feature_config["output"], outputs
        )
    
    if args.skip_reconstruction:
        print("\nSkipping reconstruction as requested.")
        print(f"Features saved to: {feature_path}")
        print(f"Matches saved to: {match_path}")
        return
    
    # Step 3: Reconstruction
    print("[4/4] Running 3D reconstruction...")
    
    # Build image_options from CLI arguments or use defaults
    image_options = {
        "camera_model": "PINHOLE",
        "camera_params": "1800.161857181376,1800.161857181376,1920.0,1080.0",
    }
    
    # Override with user-provided image_options
    if args.image_options:
        for key, value in args.image_options:
            image_options[key] = value
    
    # Only add mask_path if it exists
    if masks_dir.exists():
        image_options["mask_path"] = masks_dir
    else:
        print(f"Warning: Masks directory not found: {masks_dir}")
    
    # Build mapper_options from CLI arguments or use defaults
    mapper_options = {
        "ba_local_num_images": 20,
        "ba_local_min_tri_angle": 2.0,
        "filter_max_reproj_error": 1.5,
        "init_min_tri_angle": 15.0,
        "init_max_reg_trials": 16,
        "abs_pose_max_error": 30,
        "abs_pose_min_num_inliers": 10,
        "ba_global_max_num_iterations": 10,
        "init_max_forward_motion": 1.0,
    }
    
    # Override with user-provided mapper_options
    if args.mapper_options:
        for key, value in args.mapper_options:
            mapper_options[key] = parse_mapper_value(value)
    
    print(f"\nImage options: {image_options}")
    print(f"Mapper options: {mapper_options}")
    
    model = reconstruction.main(
        sfm_dir, 
        images, 
        sfm_pairs,
        feature_path, 
        match_path,
        camera_mode=args.camera_mode,
        image_options=image_options,
        mapper_options=mapper_options
    )
    
    print(f"\nReconstruction complete! Output saved to: {sfm_dir}")
    
    # Step 4: Visualization (optional)
    if args.visualize:
        print("\nVisualizing reconstruction...")
        visualization.visualize_sfm_2d(model, images)


if __name__ == "__main__":
    main()


