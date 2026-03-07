"""
Batch Depth Prediction Script
Predicts depth maps for images using Hugging Face depth estimation models.
Saves raw depth maps (.npy) for 3DGS and robustly normalized heatmaps (.png) for visualization.
"""

import argparse
import gc
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Try to import Depth Anything V3 (optional)
try:
    from depth_anything_3.api import DepthAnything3

    DA3_AVAILABLE = True
except ImportError:
    DA3_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuning for better performance on fixed-size inputs
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True


class ImageDataset(Dataset):
    """PyTorch Dataset optimized for direct PIL loading."""

    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            # Load natively in PIL, skipping cv2 BGR/RGB conversion overhead
            image = Image.open(str(img_path)).convert("RGB")
            return {'image': image, 'image_path': str(img_path), 'idx': idx, 'valid': True}
        except Exception as e:
            logger.error(f"Failed to load {img_path}: {e}")
            return {'image': None, 'image_path': str(img_path), 'idx': idx, 'valid': False}


def custom_collate_fn(batch):
    # Filter out any images that failed to load
    valid_batch = [item for item in batch if item['valid']]
    return {
        'images': [item['image'] for item in valid_batch],
        'image_paths': [item['image_path'] for item in valid_batch],
    }


class DepthPredictor:
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf", device="cuda", use_da3=False, inference_size=518):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.is_da3 = use_da3
        self.inference_size = inference_size
        self.patch_size = 14

        if self.is_da3 and not DA3_AVAILABLE:
            raise ImportError("Depth Anything V3 requested but depth_anything_3 is not installed.")

        logger.info(f"Using device: {self.device} | Model: {'V3' if self.is_da3 else 'V2'}")
        self.load_model()

    def load_model(self):
        if self.is_da3:
            self.model = DepthAnything3.from_pretrained(self.model_name).to(self.device)
            self.image_processor = None
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
            # REMOVED: depth_estimation_type="metric", max_depth=300.0
            # We must let the HF config dictate its own head architecture.
            self.model = AutoModelForDepthEstimation.from_pretrained(
                self.model_name
            ).to(self.device)

            # EXPLICITLY force float32. ViT-L attention layers will overflow in float16,
            # resulting in NaNs and a flat output image.
            self.model = self.model.to(torch.float32)

        self.model.eval()

    def _get_aspect_preserving_size(self, orig_h, orig_w):
        """Calculates dimensions that preserve aspect ratio AND are multiples of patch_size."""
        scale = self.inference_size / max(orig_h, orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        # Snap to nearest multiple of patch_size (14)
        new_h = (new_h // self.patch_size) * self.patch_size
        new_w = (new_w // self.patch_size) * self.patch_size
        return max(new_h, self.patch_size), max(new_w, self.patch_size)

    def predict_depth_batch(self, pil_images):
        """Processes a batch of PIL images strictly maintaining aspect ratio."""
        if self.is_da3:
            # V3 API expects numpy arrays
            np_images = [np.array(img) for img in pil_images]
            return [self._predict_single_da3(img) for img in np_images]

        orig_sizes = [img.size[::-1] for img in pil_images]  # PIL is (W, H), we want (H, W)

        # Assuming all images in a 3DGS batch share the same resolution.
        # If they don't, HF processor will pad them automatically.
        target_h, target_w = self._get_aspect_preserving_size(orig_sizes[0][0], orig_sizes[0][1])

        inputs = self.image_processor(
            images=pil_images,
            return_tensors="pt",
            size={"height": target_h, "width": target_w}
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_depth = outputs.predicted_depth
        if predicted_depth.dim() == 2:
            predicted_depth = predicted_depth.unsqueeze(0)

        depths = []
        for i, (orig_h, orig_w) in enumerate(orig_sizes):
            # Upsample back to exact original resolution
            d = F.interpolate(
                predicted_depth[i].unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
            depths.append(d)

        return depths

    def _predict_single_da3(self, image_np):
        with torch.no_grad():
            prediction = self.model.inference([image_np])
            depth = torch.tensor(prediction.depth[0])
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=image_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return depth.cpu().numpy()

    def process_batch(self, batch_data, depth_npy_path, depth_png_path):
        if not batch_data['images']:
            return 0

        pil_images = batch_data['images']
        image_paths = batch_data['image_paths']
        success_count = 0

        try:
            depth_arrays = self.predict_depth_batch(pil_images)

            for depth_array, img_path in zip(depth_arrays, image_paths):
                image_name = Path(img_path).stem

                # 1. Save strictly raw data for the 3DGS loss function
                np.save(str(depth_npy_path / f"{image_name}.npy"), depth_array)

                # 2. Robust Percentile Normalization for visualization (DO NOT USE MIN/MAX)
                p_lower = np.percentile(depth_array, 2)
                p_upper = np.percentile(depth_array, 98)
                depth_clipped = np.clip(depth_array, p_lower, p_upper)
                depth_range = p_upper - p_lower if p_upper != p_lower else 1.0
                depth_normalized = (depth_clipped - p_lower) / depth_range

                # Convert to 8-bit and apply a heatmap colormap so you can actually see it
                depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
                cv2.imwrite(str(depth_png_path / f"{image_name}.png"), depth_color)
                success_count += 1

        except Exception as e:
            logger.error(f"Error during batch inference: {str(e)}")
        finally:
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return success_count

    def process_folder(self, input_folder, output_npy_folder, output_png_folder, batch_size=1, num_workers=4, file_extensions=None):
        input_path = Path(input_folder)
        depth_npy_path = Path(output_npy_folder)
        depth_png_path = Path(output_png_folder)
        depth_npy_path.mkdir(parents=True, exist_ok=True)
        depth_png_path.mkdir(parents=True, exist_ok=True)

        if file_extensions is None:
            file_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']

        image_files = []
        for ext in file_extensions:
            image_files.extend(list(input_path.glob(f"*{ext}")))
        image_files = sorted(image_files)

        if not image_files:
            logger.warning(f"No images found in {input_path}")
            return

        dataset = ImageDataset(image_files)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
            collate_fn=custom_collate_fn,
        )

        success_count = 0
        for batch_data in tqdm(dataloader, desc="Processing images"):
            success_count += self.process_batch(batch_data, depth_npy_path, depth_png_path)

        logger.info(f"Completed: {success_count}/{len(image_files)} images processed")


def main():
    parser = argparse.ArgumentParser(
        description="Batch depth prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input-folder', '-i', type=str, required=True)
    parser.add_argument('--output-npy-folder', '-on', type=str, required=True)
    parser.add_argument('--output-png-folder', '-op', type=str, required=True)
    parser.add_argument('--model', '-m', type=str, default='depth-anything/Depth-Anything-V2-Small-hf')
    parser.add_argument('--use-da3', action='store_true')
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    parser.add_argument('--num-workers', '-w', type=int, default=4)
    parser.add_argument('--extensions', '-e', type=str, nargs='+', default=['.jpg', '.png', '.jpeg', 'JPG', 'PNG', 'JPEG'])
    parser.add_argument('--inference-size', '-s', type=int, default=518)
    args = parser.parse_args()

    if not Path(args.input_folder).exists():
        logger.error(f"Input folder missing: {args.input_folder}")
        return

    predictor = DepthPredictor(
        model_name=args.model,
        device=args.device,
        use_da3=args.use_da3,
        inference_size=args.inference_size,
    )
    predictor.process_folder(
        args.input_folder,
        args.output_npy_folder,
        args.output_png_folder,
        args.batch_size,
        args.num_workers,
        args.extensions,
    )


if __name__ == "__main__":
    main()
