"""
Batch Depth Prediction Script
Predicts depth maps for images using Hugging Face depth estimation models.
Supports both standard Depth Anything V2 models and Depth Anything V3 models.
Saves both raw depth maps (.npy) and normalized visualization images (.png).
"""

import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation
from pathlib import Path
from tqdm import tqdm
import logging
import gc
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import argparse
import sys
import torch.nn.functional as F

# Try to import Depth Anything V3 (optional)
try:
    from depth_anything_3.api import DepthAnything3
    DA3_AVAILABLE = True
except ImportError:
    DA3_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """PyTorch Dataset for loading images."""
    
    def __init__(self, image_files):
        """
        Args:
            image_files: List of image file paths
        """
        self.image_files = image_files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with 'image_np', 'image_path', and 'idx'
        """
        img_path = self.image_files[idx]
        image_np = cv2.imread(str(img_path))
        
        if image_np is None:
            # Return dummy data if image fails to load
            return {
                'image_np': None,
                'image_path': str(img_path),
                'idx': idx
            }
        
        # Convert BGR to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        return {
            'image_np': image_np,
            'image_path': str(img_path),
            'idx': idx
        }


def custom_collate_fn(batch):
    """Custom collate function to handle None images."""
    images = [item['image_np'] for item in batch]
    paths = [item['image_path'] for item in batch]
    indices = [item['idx'] for item in batch]
    
    return {
        'image_np': images,
        'image_path': paths,
        'idx': indices
    }


class DepthPredictor:
    """Depth prediction pipeline using Hugging Face models."""
    
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf", device="cuda", use_da3=False):
        """
        Initialize depth predictor.
        
        Args:
            model_name: Hugging Face model name for depth estimation
            device: Device to use ('cuda' or 'cpu')
            use_da3: Whether to use Depth Anything V3 (default: False)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.is_da3 = use_da3
        
        if self.is_da3 and not DA3_AVAILABLE:
            logger.error("Depth Anything V3 requested but depth_anything_3 is not installed.")
            logger.error("Please install it from: https://github.com/DepthAnything/Depth-Anything-V3")
            raise ImportError("depth_anything_3 module not found")
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model type: {'Depth Anything V3' if self.is_da3 else 'Depth Anything V2'}")
        
        # Load model
        logger.info(f"Loading depth estimation model: {self.model_name}")
        self.load_model()
    
    def load_model(self):
        """Load the depth estimation model."""
        if self.is_da3:
            # Load Depth Anything V3 model
            self.model = DepthAnything3.from_pretrained(self.model_name)
            self.model = self.model.to(device=self.device)
            self.image_processor = None  # DA3 handles preprocessing internally
            logger.info("Depth Anything V3 model loaded successfully")
        else:
            # Load standard HuggingFace model
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(
                self.model_name,
                depth_estimation_type="metric",
                max_depth=300.0,
            ).to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
    
    def predict_depth(self, image_np):
        """
        Predict depth map for an image.
        
        Args:
            image_np: numpy array of shape (H, W, 3) in RGB format
            
        Returns:
            depth map as numpy array (float32)
        """
        if self.is_da3:
            # Use Depth Anything V3 inference
            # DA3 expects BGR format (OpenCV format)
            # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # DA3 inference returns a prediction object
            with torch.no_grad():
                prediction = self.model.inference([image_np])
            
            # Extract depth from prediction (shape: [1, H, W])
            depth = prediction.depth[0]  # Get first (and only) image depth
            depth = torch.tensor(depth)

            print("Prediction data", prediction)

            depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=image_np.shape[:2], mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
            print("Shape of depth map:", depth.shape)
            depth = depth.detach().cpu().numpy()

            # Clear CUDA cache after each prediction to prevent memory leaks
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return depth
        else:
            # Use standard HuggingFace model
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image_np)
            
            # Prepare image for the model
            inputs = self.image_processor(images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Interpolate to original size
            post_processed_output = self.image_processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(pil_image.height, pil_image.width)],
            )
            
            predicted_depth = post_processed_output[0]["predicted_depth"]
            predicted_depth = predicted_depth + (-1 * predicted_depth.min()) if predicted_depth.min() < 0 else predicted_depth
            predicted_depth[predicted_depth <= 0] = 1e-5  # Avoid zero or negative depths

            depth = predicted_depth.detach().cpu().numpy()
            
            # Clear CUDA cache to prevent memory leaks
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return depth
    
    def process_batch(self, batch_data, depth_npy_path, depth_png_path):
        """
        Process a batch of images to predict depth.
        
        Args:
            batch_data: Batch dictionary from DataLoader
            depth_npy_path: Output directory for raw depth .npy files
            depth_png_path: Output directory for normalized depth .png files
            
        Returns:
            success_count: Number of successfully processed images
        """
        images_np = batch_data['image_np']
        image_paths = batch_data['image_path']
        
        # Filter out None images (failed to load)
        valid_mask = [img is not None for img in images_np]
        images_np = [img for img, valid in zip(images_np, valid_mask) if valid]
        image_paths = [path for path, valid in zip(image_paths, valid_mask) if valid]
        
        if len(images_np) == 0:
            return 0
        
        success_count = 0
        
        # Process each image
        for image_np, img_path in zip(images_np, image_paths):
            try:
                image_name = Path(img_path).stem
                
                # Predict depth
                depth_array = self.predict_depth(image_np)
                
                # Save raw depth as .npy
                npy_output_path = depth_npy_path / f"{image_name}.npy"
                np.save(str(npy_output_path), depth_array)
                
                # Normalize depth to 0-255 and save as PNG
                depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
                depth_image = (depth_normalized * 255).astype(np.uint8)
                png_output_path = depth_png_path / f"{image_name}.png"
                cv2.imwrite(str(png_output_path), depth_image)
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue
            finally:
                # Force garbage collection after each image to prevent memory leaks
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        
        return success_count
    
    def process_folder(self, input_folder, output_npy_folder, output_png_folder, 
                       batch_size=1, num_workers=4, file_extensions=None):
        """
        Process all images in a folder.
        
        Args:
            input_folder: Path to folder containing input images
            output_npy_folder: Path to folder for raw depth .npy files
            output_png_folder: Path to folder for normalized depth .png files
            batch_size: Batch size for processing
            num_workers: Number of DataLoader workers
            file_extensions: List of file extensions to process (default: ['.jpg', '.png', '.jpeg'])
        """
        input_path = Path(input_folder)
        depth_npy_path = Path(output_npy_folder)
        depth_png_path = Path(output_png_folder)
        
        # Create output directories
        depth_npy_path.mkdir(parents=True, exist_ok=True)
        depth_png_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        if file_extensions is None:
            file_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
        
        image_files = []
        for ext in file_extensions:
            image_files.extend(list(input_path.glob(f"*{ext}")))
        image_files = sorted(image_files)
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {input_path}")
            return
        
        logger.info(f"Found {len(image_files)} images in {input_path}")
        logger.info(f"Output NPY folder: {depth_npy_path}")
        logger.info(f"Output PNG folder: {depth_png_path}")
        logger.info(f"Using batch_size={batch_size}, num_workers={num_workers}")
        
        # Create dataset and dataloader
        dataset = ImageDataset(image_files)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
            collate_fn=custom_collate_fn,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        success_count = 0
        
        # Process batches
        for batch_data in tqdm(dataloader, desc="Processing images"):
            batch_success = self.process_batch(batch_data, depth_npy_path, depth_png_path)
            success_count += batch_success
        
        logger.info(f"Completed: {success_count}/{len(image_files)} images processed")
        
        # Clear cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def main():
    """Main function to run depth prediction."""
    parser = argparse.ArgumentParser(
        description="Batch depth prediction for images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--input-folder', '-i',
        type=str,
        required=True,
        help='Path to folder containing input images'
    )
    parser.add_argument(
        '--output-npy-folder', '-on',
        type=str,
        required=True,
        help='Path to folder for output raw depth .npy files'
    )
    parser.add_argument(
        '--output-png-folder', '-op',
        type=str,
        required=True,
        help='Path to folder for output normalized depth .png files'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='depth-anything/Depth-Anything-V2-Small-hf',
        help='Hugging Face model name for depth estimation. '
             'Depth Anything V2 models: '
             'depth-anything/Depth-Anything-V2-Small-hf, '
             'depth-anything/Depth-Anything-V2-Base-hf, '
             'depth-anything/Depth-Anything-V2-Large-hf, '
             'LiheYoung/depth-anything-large-hf, '
             'Intel/dpt-large. '
             'Depth Anything V3 models (use with --use-da3 flag): '
             'depth-anything/DA3NESTED-GIANT-LARGE, '
             'depth-anything/DA3NESTED-GIANT-BASE, '
             'depth-anything/DA3NESTED-GIANT-SMALL'
    )
    parser.add_argument(
        '--use-da3',
        action='store_true',
        help='Use Depth Anything V3 API (requires depth_anything_3 installation). '
             'Default is to use Depth Anything V2.'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1,
        help='Batch size for processing images'
    )
    parser.add_argument(
        '--num-workers', '-w',
        type=int,
        default=4,
        help='Number of DataLoader workers for parallel image loading'
    )
    parser.add_argument(
        '--extensions', '-e',
        type=str,
        nargs='+',
        default=['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'],
        help='File extensions to process (e.g., .jpg .png)'
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    if not Path(args.input_folder).exists():
        logger.error(f"Input folder does not exist: {args.input_folder}")
        return
    
    # Initialize predictor
    predictor = DepthPredictor(model_name=args.model, device=args.device, use_da3=args.use_da3)
    
    # Process folder
    predictor.process_folder(
        input_folder=args.input_folder,
        output_npy_folder=args.output_npy_folder,
        output_png_folder=args.output_png_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        file_extensions=args.extensions
    )
    
    logger.info("Depth prediction completed!")


if __name__ == "__main__":
    main()
