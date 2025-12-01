"""Advanced batch object removal script using optical flow for better object detection and removal quality.
Processes images directly in the specified folder.
"""

import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import torchvision.transforms as T
import numpy as np
import cv2
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pathlib import Path
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OpticalFlowObjectRemoval:
    """Advanced pipeline using optical flow for better object detection and removal."""
    
    def __init__(self, device="cuda", use_optical_flow=True, save_masks=True, combine_masks=False):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_optical_flow = use_optical_flow
        self.save_masks = save_masks
        self.combine_masks = combine_masks
        logger.info(f"Using device: {self.device}")
        
        # Load models
        logger.info("Loading models...")
        self.load_models()
        
        # Target objects to remove - use set for O(1) lookup
        self.target_objects = {"car", "person", "truck", "bicycle", 
                               "aeroplane", "bus", "train", "truck", "boat", "bird", 
                               "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                               "backpack", "umbrella", "handbag", "tie", "suitcase",
                               "kite"
                               }
        
        # Image preprocessing
        self.preprocess = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),
            T.Resize(size=(1080, 1920)),
        ])
        
        # Pre-allocate kernel for dilation
        self.dilation_kernel = np.ones((5, 5), np.uint8)
        
        # Cache for label to id mapping
        self.label_to_id = {v: k for k, v in self.ob_model.config.id2label.items()}
        self.target_label_ids = {self.label_to_id.get(obj) for obj in self.target_objects if self.label_to_id.get(obj) is not None}
        
    def load_models(self):
        """Load all required models."""
        # Object Detection model
        self.ob_image_processor = RTDetrImageProcessor.from_pretrained(
            "PekingU/rtdetr_r101vd_coco_o365"
        )
        self.ob_model = RTDetrV2ForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r101vd_coco_o365"
        ).to(self.device)
        self.ob_model.eval()
        
        # SAM2 Segmentation model
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        
        # Optical Flow model
        if self.use_optical_flow:
            self.flow_model = raft_small(
                progress=True, 
                weights=Raft_Small_Weights.DEFAULT
            ).eval().to(self.device)
        
        logger.info("Models loaded successfully")
    
    def detect_objects(self, image_np, confidence_threshold=0.25):
        """Detect target objects in an image - optimized."""
        inputs = self.ob_image_processor(images=[image_np], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.ob_model(**inputs)
        
        target_sizes = torch.tensor([[image_np.shape[0], image_np.shape[1]]], device=self.device)
        results = self.ob_image_processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )
        
        bboxes = []
        labels = []
        for result in results:
            # Vectorized filtering using cached label IDs
            label_ids = result["labels"].cpu().numpy()
            boxes = result["boxes"].cpu().numpy()
            
            # Filter using numpy for speed
            mask = np.isin(label_ids, list(self.target_label_ids))
            if mask.any():
                filtered_boxes = boxes[mask].astype(np.int32)
                bboxes.extend(filtered_boxes.tolist())
                labels.extend([self.ob_model.config.id2label[int(lid)] for lid in label_ids[mask]])
        
        return bboxes, labels
    
    def compute_optical_flow(self, img1_tensor, img2_tensor):
        """Compute optical flow between two images."""
        # Ensure tensors are on device
        if img1_tensor.device != self.device:
            img1_tensor = img1_tensor.to(self.device)
        if img2_tensor.device != self.device:
            img2_tensor = img2_tensor.to(self.device)
            
        img1_batch = self.preprocess(img1_tensor).unsqueeze(0)
        img2_batch = self.preprocess(img2_tensor).unsqueeze(0)
        
        with torch.no_grad():
            flow_predictions = self.flow_model(
                img1_batch, 
                img2_batch, 
                num_flow_updates=128
            )
            predicted_flows = flow_predictions[-1]
        
        return predicted_flows
    
    def get_motion_mask(self, flow, threshold=20.0):
        """
        Generate motion mask from optical flow using RANSAC.
        
        Args:
            flow: optical flow tensor
            threshold: motion threshold in pixels
            
        Returns:
            motion mask as numpy array
        """
        # Convert flow to numpy once
        flow_np = flow[0].cpu().permute(1, 2, 0).numpy()
        h, w = flow_np.shape[:2]
        
        # Create sparse grid for RANSAC
        n_steps = 10
        y_indices, x_indices = np.mgrid[n_steps//2:h:n_steps, n_steps//2:w:n_steps]
        src_pts = np.stack((x_indices.ravel(), y_indices.ravel()), axis=-1).astype(np.float32)
        
        flow_at_pts = flow_np[y_indices.ravel(), x_indices.ravel()]
        dst_pts = src_pts + flow_at_pts
        
        # Find homography with RANSAC
        H, ransac_mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, 
            ransacReprojThreshold=5.0, maxIters=100000
        )
        
        if H is None:
            # If RANSAC fails, use simple threshold (optimized)
            flow_magnitude = np.linalg.norm(flow_np, axis=-1)
            motion_mask = (flow_magnitude > threshold).astype(np.uint8) * 255
            return motion_mask
        
        # Create dense grid - optimized allocation
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        all_src_pts = np.stack((x_coords.ravel(), y_coords.ravel()), axis=-1)
        
        # Predict background motion
        all_dst_pts_pred = cv2.perspectiveTransform(all_src_pts.reshape(1, -1, 2), H)
        all_dst_pts_pred = all_dst_pts_pred.reshape(-1, 2)
        
        # Calculate predicted vs actual flow - vectorized
        flow_pred = (all_dst_pts_pred - all_src_pts).reshape(h, w, 2)
        flow_error = np.linalg.norm(flow_np - flow_pred, axis=-1)
        
        # Threshold to get motion mask
        motion_mask = (flow_error > threshold).astype(np.uint8) * 255
        
        return motion_mask
    
    def refine_masks_with_flow(self, masks, motion_mask, image_shape):
        """Refine segmentation masks using optical flow motion mask."""
        # Use CPU-based resize instead of GPU interpolation to save VRAM
        upsampled_motion_mask = cv2.resize(
            motion_mask,
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        refined_masks = []
        
        for mask in masks:
            mask_np = mask.squeeze()
            # Keep only parts of mask that are moving
            intersection = np.logical_and(upsampled_motion_mask > 0, mask_np > 0)
            refined_masks.append(intersection)
        
        return refined_masks
    
    def segment_objects(self, image_np, bboxes):
        """Segment objects using SAM2 - with batch processing for large bbox counts (memory constraint)."""
        if len(bboxes) == 0:
            return np.array([])
        
        max_boxes_per_batch = 10  # Keep this for memory constraints
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image_np)
            
            # If bboxes fit in one batch, process directly
            if len(bboxes) <= max_boxes_per_batch:
                input_boxes = np.array(bboxes, dtype=np.int32)
                masks, _, _ = self.predictor.predict(
                    box=input_boxes,
                    multimask_output=False
                )
                if masks.ndim == 3:
                    masks = masks[:, np.newaxis, :, :]
                return masks
            
            # Otherwise batch process
            all_masks = []
            for i in range(0, len(bboxes), max_boxes_per_batch):
                batch_bboxes = bboxes[i:i + max_boxes_per_batch]
                input_boxes = np.array(batch_bboxes, dtype=np.int32)
                masks, _, _ = self.predictor.predict(
                    box=input_boxes,
                    multimask_output=False
                )
                if masks.ndim == 3:
                    masks = masks[:, np.newaxis, :, :]
                all_masks.append(masks)
            
            return np.concatenate(all_masks, axis=0)
    
    def inpaint_masked_regions(self, image_np, masks):
        """Remove objects by zeroing out masked regions (no inpainting) - optimized."""
        if len(masks) == 0:
            return image_np
        
        # Fast mask combination
        if masks.ndim == 4 and masks.shape[1] == 1:
            combined_mask = np.max(masks[:, 0], axis=0)
        elif masks.ndim == 3:
            combined_mask = np.max(masks, axis=0)
        else:
            combined_mask = masks.squeeze()
        
        # Convert to uint8 and dilate in one step
        combined_mask = cv2.dilate(
            (combined_mask > 0.5).astype(np.uint8),
            self.dilation_kernel,
            iterations=1
        )
        
        # Zero out masked regions
        filtered_image = image_np.copy()
        filtered_image[combined_mask > 0] = 0
        
        return filtered_image
    
    def save_individual_masks(self, masks, labels, image_name, masks_base_path):
        """
        Save individual masks to a folder structure.
        
        Args:
            masks: Array of masks
            labels: List of label names for each mask
            image_name: Name of the image (without extension)
            masks_base_path: Base path for saving masks
        """
        if len(masks) == 0:
            return
        
        # Create folder for this image
        image_mask_folder = masks_base_path / image_name
        image_mask_folder.mkdir(parents=True, exist_ok=True)
        
        # Save each mask
        for idx, (mask, label) in enumerate(zip(masks, labels)):
            # Process mask - ensure 2D
            if isinstance(mask, np.ndarray):
                mask_2d = mask.squeeze() if mask.ndim > 2 else mask
            else:
                mask_2d = mask
            
            # Convert to uint8 (0-255)
            mask_uint8 = ((mask_2d > 0.5).astype(np.uint8) * 255)
            
            # Save mask with label and index
            mask_filename = f"{label}_{idx}.png"
            mask_path = image_mask_folder / mask_filename
            cv2.imwrite(str(mask_path), mask_uint8)
    
    def save_combined_mask(self, masks, image_name, masks_base_path):
        """
        Save all masks combined into a single file - optimized.
        
        Args:
            masks: Array of masks
            image_name: Name of the image (without extension)
            masks_base_path: Base path for saving masks
        """
        if len(masks) == 0:
            return
        
        # Fast mask combination
        if masks.ndim == 4 and masks.shape[1] == 1:
            combined_mask = np.max(masks[:, 0], axis=0)
        elif masks.ndim == 3:
            combined_mask = np.max(masks, axis=0)
        else:
            combined_mask = masks.squeeze()
        
        # Convert to uint8 (inverted: 0 where mask, 255 where no mask)
        mask_uint8 = ((1 - (combined_mask > 0.5).astype(np.uint8)) * 255).astype(np.uint8)
        
        # Save combined mask
        mask_path = masks_base_path / f"{image_name}.png"
        cv2.imwrite(str(mask_path), mask_uint8)

    
    def process_image(self, image_path, filtered_path, masks_path, prev_image_tensor=None):
        """
        Process a single image - optimized for speed.
        
        Args:
            image_path: Path to image file
            filtered_path: Output directory path for filtered images
            masks_path: Output directory path for masks (optional)
            prev_image_tensor: Previous image tensor for optical flow
            
        Returns:
            success: Boolean indicating success
            current_image_tensor: Current image tensor for next image
        """
        try:
            # Load image directly as BGR (no conversion needed for saving)
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                logger.warning(f"Failed to load image: {image_path}")
                return False, prev_image_tensor
            
            # Convert to RGB only for model inference
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            output_path = filtered_path / image_path.name
            image_name = image_path.stem
            
            # Detect objects
            bboxes, labels = self.detect_objects(image_rgb)
            
            # If no objects detected, copy file directly (fastest)
            if len(bboxes) == 0:
                cv2.imwrite(str(output_path), image_bgr)
                current_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1) if self.use_optical_flow else None
                return True, current_tensor
            
            # Segment objects (with SAM2 batching for memory)
            masks = self.segment_objects(image_rgb, bboxes)
            
            # Refine with optical flow if enabled
            if self.use_optical_flow and prev_image_tensor is not None:
                current_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
                flow = self.compute_optical_flow(prev_image_tensor, current_tensor)
                motion_mask = self.get_motion_mask(flow)
                masks = self.refine_masks_with_flow(masks, motion_mask, image_rgb.shape)
            else:
                current_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1) if self.use_optical_flow else None
            
            # Save masks if enabled
            if self.save_masks and masks_path is not None:
                if self.combine_masks:
                    self.save_combined_mask(masks, image_name, masks_path)
                else:
                    self.save_individual_masks(masks, labels, image_name, masks_path)
            
            # Remove objects
            filtered_rgb = self.inpaint_masked_regions(image_rgb, masks)
            
            # Convert back to BGR and save
            filtered_bgr = cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), filtered_bgr)
            
            return True, current_tensor
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return False, prev_image_tensor
    
    def process_folder(self, folder_path, filtered_output_path=None, masks_output_path=None):
        """Process all images in a folder sequentially - optimized for speed."""
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            return
        
        # Create output directories
        if filtered_output_path is None:
            filtered_path = folder_path.parent / "filtered"
        else:
            filtered_path = Path(filtered_output_path)
        filtered_path.mkdir(parents=True, exist_ok=True)
        
        masks_path = None
        if self.save_masks:
            if masks_output_path is None:
                masks_path = folder_path.parent / "masks"
            else:
                masks_path = Path(masks_output_path)
            masks_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = sorted(list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")))
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {folder_path}")
            return
        
        logger.info(f"Processing {folder_path}: {len(image_files)} images")
        
        success_count = 0
        prev_image_tensor = None
        
        # Process images sequentially
        for image_path in tqdm(image_files, desc=f"Processing {folder_path.name}"):
            success, current_tensor = self.process_image(
                image_path, filtered_path, masks_path, prev_image_tensor
            )
            if success:
                success_count += 1
            prev_image_tensor = current_tensor
        
        logger.info(f"Completed {folder_path.name}: {success_count}/{len(image_files)} images processed")


def main():
    """Main function to process images in folder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch object removal with optical flow - optimized sequential processing")
    parser.add_argument("--no-flow", action="store_true", help="Disable optical flow refinement")
    parser.add_argument('--save-masks', action='store_true', 
                        help='Enable saving masks')
    parser.add_argument('--combine-masks', action='store_true',
                        help='Save all masks of an image as a single combined file instead of individual files in a folder')
    parser.add_argument('--base-path', type=str,
                        default='/home/vamsik1211/Data/Projects/3D-Reconstructions/CityScapeGS/data/boston_cusfm_sequence',
                        help='Base path to folder containing images')
    parser.add_argument('--filtered-output', type=str, default=None,
                        help='Output path for filtered images (default: ../filtered relative to base-path)')
    parser.add_argument('--masks-output', type=str, default=None,
                        help='Output path for masks (default: ../masks relative to base-path)')
    args = parser.parse_args()
    
    # Setup paths
    base_path = Path(args.base_path)
    
    if not base_path.exists():
        logger.error(f"Base path does not exist: {base_path}")
        return
    
    # Determine output paths
    filtered_output = args.filtered_output if args.filtered_output else str(base_path.parent / "filtered")
    masks_output = args.masks_output if args.masks_output else str(base_path.parent / "masks")
    
    logger.info(f"Processing images in: {base_path}")
    logger.info(f"Filtered images will be saved to: {filtered_output}")
    logger.info(f"Optical flow: {'disabled' if args.no_flow else 'enabled'}")
    
    if args.save_masks:
        logger.info(f"Mask saving enabled (combine_masks={args.combine_masks})")
        logger.info(f"Masks will be saved to: {masks_output}")
    else:
        logger.info("Mask saving disabled")
    
    # Initialize pipeline
    pipeline = OpticalFlowObjectRemoval(
        use_optical_flow=not args.no_flow,
        save_masks=args.save_masks,
        combine_masks=args.combine_masks
    )
    
    # Process the folder
    try:
        pipeline.process_folder(
            base_path,
            filtered_output_path=filtered_output,
            masks_output_path=masks_output
        )
    except Exception as e:
        logger.error(f"Failed to process folder {base_path}: {str(e)}")
        return
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
