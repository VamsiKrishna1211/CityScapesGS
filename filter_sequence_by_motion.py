import torch
import torchvision
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import numpy as np
from torch.nn import functional as T_F
import torchvision.transforms as T
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse
import gc
import os
from PIL import Image

# Set PyTorch memory allocator configuration to reduce fragmentation
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

class MotionSequenceFilter:
    def __init__(self, motion_threshold=5.0, device=None, mask_dir=None):
        """
        Initialize the motion sequence filter.
        
        Args:
            motion_threshold: Mean optical flow magnitude threshold to consider as "movement"
            device: Device to run the model on (cuda/cpu)
            mask_dir: Optional directory containing masks for each image (same naming as images)
        """
        self.motion_threshold = motion_threshold
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_dir = Path(mask_dir) if mask_dir else None
        
        # Load optical flow model
        print(f"Loading RAFT model on {self.device}...")
        self.model = raft_small(progress=True, weights=Raft_Small_Weights.DEFAULT).eval().to(self.device)
        self.model = torch.compile(self.model)
        print("Model loaded successfully!")
        
    def preprocess(self, img_tensor):
        """Preprocess image tensor for optical flow model."""
        transforms = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(1080, 1920)),  # Must be divisible by 8 for RAFT
        ])
        return transforms(img_tensor)
    
    def load_mask(self, img_path):
        """
        Load mask for a given image path.
        
        Args:
            img_path: Path to the image
            
        Returns:
            Binary mask tensor (1 = keep, 0 = discard) or None if no mask found
        """
        if not self.mask_dir:
            return None
        
        # Try to find mask with same name
        mask_path = self.mask_dir / img_path.name
        # Also try common mask extensions
        if not mask_path.exists():
            mask_path = self.mask_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            mask_path = self.mask_dir / (img_path.stem + ".jpg")
        
        if not mask_path.exists():
            return None
        
        try:
            # Load mask and convert to binary (0 or 1)
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            # Invert mask: 0 = object (discard), 255 = background (keep)
            # So we create: 1 = keep, 0 = discard
            mask_binary = (mask == 0).astype(np.float32)  # 1 where mask is 0 (object)
            # mask_binary = 1.0 - mask_binary  # Invert to 1=keep, 0=discard --- IGNORE ---
            return torch.from_numpy(mask_binary)
        except Exception as e:
            print(f"  Warning: Could not load mask from {mask_path}: {e}")
            return None
    
    def compute_optical_flow(self, img1_path, img2_path):
        """
        Compute optical flow between two images.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            Mean magnitude of optical flow (excluding masked regions if masks provided)
        """
        # Clear cache before loading new images
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Load images
        img1 = torchvision.io.read_image(str(img1_path))
        img2 = torchvision.io.read_image(str(img2_path))
        
        # Load masks if available
        mask1 = self.load_mask(img1_path)
        mask2 = self.load_mask(img2_path)
        
        # Preprocess
        img1_batch = self.preprocess(img1).unsqueeze(0)
        img2_batch = self.preprocess(img2).unsqueeze(0)
        
        # Compute optical flow
        with torch.no_grad():
            img1_device = img1_batch.to(self.device)
            img2_device = img2_batch.to(self.device)
            
            flow_predictions = self.model(
                img1_device, 
                img2_device, 
                num_flow_updates=128
            )
            predicted_flows = flow_predictions[-1]
            
            # Calculate flow magnitude on GPU
            flow_magnitude = torch.sqrt(
                predicted_flows[0, 0, :, :] ** 2 + 
                predicted_flows[0, 1, :, :] ** 2
            )
            
            # Apply masks if available
            if mask1 is not None and mask2 is not None:
                # Resize masks to match flow dimensions
                mask1_resized = T_F.interpolate(
                    mask1.unsqueeze(0).unsqueeze(0), 
                    size=flow_magnitude.shape,
                    mode='nearest'
                ).squeeze()
                mask2_resized = T_F.interpolate(
                    mask2.unsqueeze(0).unsqueeze(0), 
                    size=flow_magnitude.shape,
                    mode='nearest'
                ).squeeze()
                
                # Combine masks (only keep regions valid in both frames)
                combined_mask = (mask1_resized * mask2_resized).to(self.device)
                
                # Apply mask to flow
                masked_flow = flow_magnitude * combined_mask
                
                # Calculate mean only over valid (non-masked) regions
                valid_pixels = combined_mask.sum()
                if valid_pixels > 0:
                    mean_flow = (masked_flow.sum() / valid_pixels).item()
                else:
                    # If no valid pixels, return 0 (no motion)
                    mean_flow = 0.0
                    print("  Warning: No valid pixels after masking!")
            else:
                # No masks, use full image
                mean_flow = flow_magnitude.mean().item()
            
            # Move everything back to CPU before deletion
            predicted_flows = predicted_flows.cpu()
            flow_magnitude = flow_magnitude.cpu()
        
        # Clear GPU memory
        del img1, img2, img1_batch, img2_batch, img1_device, img2_device
        del flow_predictions, predicted_flows, flow_magnitude
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return mean_flow
    
    def filter_sequence(self, sequence_dir, output_dir=None, copy_files=True, auxiliary_dirs=None):
        """
        Filter image sequence based on motion detection.
        
        Args:
            sequence_dir: Directory containing the image sequence
            output_dir: Directory to save filtered images (optional)
            copy_files: If True, copy files to output_dir. If False, just return the list
            auxiliary_dirs: List of auxiliary data directories (masks, depths, etc.) to also filter
            
        Returns:
            List of selected image paths with motion statistics
        """
        sequence_path = Path(sequence_dir)
        auxiliary_dirs = auxiliary_dirs or []
        
        # Get all image files sorted by name
        image_files = sorted(list(sequence_path.glob("*.jpg")) + 
                           list(sequence_path.glob("*.png")))
        
        if len(image_files) < 2:
            print("Not enough images in the sequence!")
            return []
        
        print(f"Found {len(image_files)} images in sequence")
        print(f"Motion threshold: {self.motion_threshold}")
        
        # Create output directory if needed
        if output_dir and copy_files:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {output_path}")
            
            # Create auxiliary output directories
            auxiliary_output_dirs = []
            for aux_dir in auxiliary_dirs:
                aux_path = Path(aux_dir)
                aux_output_path = aux_path.parent / f"{aux_path.name}_filtered"
                aux_output_path.mkdir(parents=True, exist_ok=True)
                auxiliary_output_dirs.append((aux_path, aux_output_path))
                print(f"Auxiliary output directory: {aux_output_path}")
        else:
            auxiliary_output_dirs = []
        
        selected_images = []
        motion_stats = []
        
        # Start with the first image
        current_index = 0
        last_saved_index = None
        
        # Always add the first image
        selected_images.append({
            'path': image_files[current_index],
            'original_index': current_index,
            'name': image_files[current_index].name
        })
        last_saved_index = current_index
        print(f"\nAdding first image: {image_files[current_index].name}")
        
        # Process remaining images
        next_candidate_index = current_index + 1
        
        while next_candidate_index < len(image_files):
            img1_path = image_files[current_index]
            img2_path = image_files[next_candidate_index]
            
            # Compute optical flow
            print(f"\nProcessing pair: {img1_path.name} <-> {img2_path.name}")
            mean_flow = self.compute_optical_flow(img1_path, img2_path)
            mask_status = " (with masks)" if self.mask_dir else ""
            print(f"  Mean optical flow{mask_status}: {mean_flow:.4f}")
            
            # Check if masking resulted in no valid pixels (mean_flow == 0.0 with masks)
            if mean_flow == 0.0 and self.mask_dir:
                print(f"  ⚠ No valid pixels after masking. Using last selected image as reference.")
                # Keep this image and use it as the new reference
                if last_saved_index != next_candidate_index:
                    selected_images.append({
                        'path': img2_path,
                        'original_index': next_candidate_index,
                        'name': img2_path.name
                    })
                    last_saved_index = next_candidate_index
                
                motion_stats.append({
                    'pair': f"{img1_path.name} -> {img2_path.name}",
                    'mean_flow': mean_flow,
                    'status': 'kept (no valid pixels)'
                })
                
                # Move current to the newly saved image and check next candidate
                current_index = next_candidate_index
                next_candidate_index = current_index + 1
            elif mean_flow >= self.motion_threshold:
                # Movement detected - keep this image
                print(f"  ✓ Movement detected! Keeping {img2_path.name}")
                
                # Add the candidate image (only if not already added)
                if last_saved_index != next_candidate_index:
                    selected_images.append({
                        'path': img2_path,
                        'original_index': next_candidate_index,
                        'name': img2_path.name
                    })
                    last_saved_index = next_candidate_index
                
                motion_stats.append({
                    'pair': f"{img1_path.name} -> {img2_path.name}",
                    'mean_flow': mean_flow,
                    'status': 'kept'
                })
                
                # Move current to the newly saved image and check next candidate
                current_index = next_candidate_index
                next_candidate_index = current_index + 1
            else:
                # No significant movement - discard candidate, try next
                print(f"  ✗ No significant movement. Discarding {img2_path.name}")
                motion_stats.append({
                    'pair': f"{img1_path.name} -> {img2_path.name}",
                    'mean_flow': mean_flow,
                    'status': 'discarded'
                })
                
                # Keep current_index the same, skip to next candidate
                next_candidate_index += 1
        
        print(f"\n{'='*60}")
        print(f"Filtering complete!")
        print(f"Original images: {len(image_files)}")
        print(f"Selected images: {len(selected_images)}")
        print(f"Discarded images: {len(image_files) - len(selected_images)}")
        
        # Copy files if requested
        if output_dir and copy_files and selected_images:
            print(f"\nCopying {len(selected_images)} images to {output_dir}...")
            for idx, img_info in enumerate(tqdm(selected_images)):
                # Create new filename with sequential numbering
                new_name = f"{idx:06d}{img_info['path'].suffix}"
                dest_path = Path(output_dir) / new_name
                shutil.copy2(img_info['path'], dest_path)
            print("Copy complete!")
            
            # Copy auxiliary data files
            for aux_input_dir, aux_output_dir in auxiliary_output_dirs:
                print(f"\nCopying auxiliary data from {aux_input_dir.name}...")
                copied_count = 0
                for idx, img_info in enumerate(tqdm(selected_images)):
                    # Try to find matching auxiliary file
                    original_name = img_info['name']
                    aux_file = aux_input_dir / original_name
                    
                    # Try different extensions if exact match not found
                    if not aux_file.exists():
                        stem = img_info['path'].stem
                        for ext in ['.png', '.jpg', '.npy', '.npz', '.exr', '.tiff', '.tif']:
                            aux_file = aux_input_dir / f"{stem}{ext}"
                            if aux_file.exists():
                                break
                    
                    if aux_file.exists():
                        # Create new filename with same extension as auxiliary file
                        new_name = f"{idx:06d}{aux_file.suffix}"
                        dest_path = aux_output_dir / new_name
                        shutil.copy2(aux_file, dest_path)
                        copied_count += 1
                    else:
                        print(f"  Warning: Could not find auxiliary file for {original_name}")
                
                print(f"Copied {copied_count}/{len(selected_images)} auxiliary files from {aux_input_dir.name}")
        
        # Final memory cleanup
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Print summary
        print(f"\n{'='*60}")
        print("Motion Statistics Summary:")
        print(f"{'='*60}")
        for stat in motion_stats:
            status_symbol = "✓" if stat['status'] == 'kept' else "✗"
            print(f"{status_symbol} {stat['pair']}: {stat['mean_flow']:.4f} ({stat['status']})")
        
        return selected_images, motion_stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter image sequence based on optical flow motion detection"
    )
    parser.add_argument(
        "sequence_dir",
        type=str,
        help="Directory containing the image sequence"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for filtered images (default: sequence_dir + '_filtered')"
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=5.0,
        help="Mean optical flow threshold to consider as movement (default: 5.0)"
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help="Directory containing masks for images (to exclude segmented objects from flow calculation)"
    )
    parser.add_argument(
        "--auxiliary-dir",
        type=str,
        action='append',
        dest='auxiliary_dirs',
        default=[],
        help="Auxiliary data directory to filter alongside images (can be used multiple times for masks, depths, etc.)"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Don't copy files, just analyze and report"
    )
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"{args.sequence_dir}_filtered"
    
    # Initialize filter
    filter_obj = MotionSequenceFilter(
        motion_threshold=args.motion_threshold,
        mask_dir=args.mask_dir
    )
    
    # Process sequence
    selected_images, motion_stats = filter_obj.filter_sequence(
        sequence_dir=args.sequence_dir,
        output_dir=args.output_dir,
        copy_files=not args.no_copy,
        auxiliary_dirs=args.auxiliary_dirs
    )
    
    # Save statistics to file
    if selected_images and not args.no_copy:
        stats_file = Path(args.output_dir) / "motion_statistics.txt"
        with open(stats_file, 'w') as f:
            f.write(f"Motion Threshold: {args.motion_threshold}\n")
            f.write(f"Mask Directory: {args.mask_dir if args.mask_dir else 'None'}\n")
            f.write(f"Auxiliary Directories: {', '.join(args.auxiliary_dirs) if args.auxiliary_dirs else 'None'}\n")
            f.write(f"Original images: {len(list(Path(args.sequence_dir).glob('*.jpg')))}\n")
            f.write(f"Selected images: {len(selected_images)}\n\n")
            f.write("="*60 + "\n")
            f.write("Detailed Motion Statistics:\n")
            f.write("="*60 + "\n")
            for stat in motion_stats:
                f.write(f"{stat['pair']}: {stat['mean_flow']:.4f} ({stat['status']})\n")
            f.write("\n")
            f.write("="*60 + "\n")
            f.write("Selected Images:\n")
            f.write("="*60 + "\n")
            for idx, img_info in enumerate(selected_images):
                f.write(f"{idx:06d}: {img_info['name']} (original: {img_info['original_index']})\n")
        
        print(f"\nStatistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
