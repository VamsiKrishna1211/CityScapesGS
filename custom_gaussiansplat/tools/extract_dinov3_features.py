"""
Extracts dense DINO features using Hugging Face models (compatible with DINOv2 and DINOv3).
Saves feature maps and metadata as .pt files using torch.save.
"""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DINOExtractor:
    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", device="cuda", patch_size=None):
        self.device = device if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading DINO model ({model_name}) on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Determine feature dimension from the model config
        self.feature_dim = self.model.config.hidden_size

        # Determine patch size
        if patch_size is not None:
            self.patch_size = patch_size
            logger.info(f"Using user-specified patch size: {self.patch_size}")
        elif hasattr(self.model.config, "patch_size"):
            self.patch_size = self.model.config.patch_size
            logger.info(f"Detected patch size from config: {self.patch_size}")
        else:
            # Heuristic fallback
            if "dinov2" in model_name.lower():
                self.patch_size = 14
            elif "dinov3" in model_name.lower():
                self.patch_size = 16
            else:
                self.patch_size = 14 # Default
            logger.info(f"Inferred patch size from model name: {self.patch_size}")

        logger.info(f"Model loaded. Feature dimension: {self.feature_dim}, Patch size: {self.patch_size}")

    def _get_aspect_preserving_size(self, orig_h, orig_w):
        """Forces dimensions to be multiples of the DINO patch size."""
        new_h = (orig_h // self.patch_size) * self.patch_size
        new_w = (orig_w // self.patch_size) * self.patch_size
        return max(new_h, self.patch_size), max(new_w, self.patch_size)

    def extract_dense_features(self, pil_image):
        """Extracts dense feature grid from image."""
        orig_w, orig_h = pil_image.size
        target_h, target_w = self._get_aspect_preserving_size(orig_h, orig_w)

        # Resize image to be compatible with patch size
        img_resized = pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)

        inputs = self.processor(
            images=img_resized,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # last_hidden_state shape: [1, num_tokens, feature_dim]
            hidden_states = outputs.last_hidden_state

        # Calculate spatial dimensions
        h_feat, w_feat = target_h // self.patch_size, target_w // self.patch_size
        expected_tokens = h_feat * w_feat
        token_count = hidden_states.shape[1]

        # Handle tokens: include CLS token as the last patch if present
        if token_count == expected_tokens + 1:
            # patches: [1, expected_tokens, feature_dim], cls: [1, 1, feature_dim]
            patches = hidden_states[:, 1:, :]
            cls_token = hidden_states[:, 0:1, :]
            patch_features = torch.cat([patches, cls_token], dim=1)
        elif token_count == expected_tokens:
            patch_features = hidden_states
        else:
            raise ValueError(
                f"Unexpected token count: got {token_count}, expected {expected_tokens} "
                f"(or {expected_tokens + 1} with CLS) for image {target_h}x{target_w}. "
                f"Check if the patch size {self.patch_size} is correct for this model."
            )

        # Return features and the inference shape used
        dense_features = patch_features.squeeze(0)
        return dense_features, (target_h, target_w)

    def process_folder(self, input_folder, output_folder):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
        image_files = []
        for ext in extensions:
            image_files.extend(list(input_path.glob(f"*{ext}")))
        image_files = sorted(image_files)

        if not image_files:
            logger.warning(f"No images found in {input_path}")
            return

        logger.info(f"Processing {len(image_files)} images...")
        success_count = 0

        for img_path in tqdm(image_files, desc="Extracting DINO Features"):
            try:
                image = Image.open(str(img_path)).convert("RGB")
                features, inference_shape = self.extract_dense_features(image)

                # Save as a dictionary using torch.save
                data_to_save = {
                    "features": features.cpu(),
                    "patch_size": self.patch_size,
                    "inference_image_shape": inference_shape, # (H, W)
                }

                save_path = output_path / f"{img_path.stem}.pt"
                torch.save(data_to_save, save_path)
                success_count += 1

                # Periodic memory cleanup
                if success_count % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")

        logger.info(f"Completed: {success_count}/{len(image_files)} images processed.")

def main():
    parser = argparse.ArgumentParser(description="Extract dense DINO features and save as .pt files.")
    parser.add_argument('--input-folder', '-i', type=str, required=True, help="Folder containing images")
    parser.add_argument('--output-folder', '-o', type=str, required=True, help="Folder to save .pt features")
    parser.add_argument('--model-name', '-m', type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m", help="Hugging Face model name")
    parser.add_argument('--patch-size', '-p', type=int, default=None, help="Manually specify patch size (e.g. 14 or 16)")
    parser.add_argument('--device', '-d', type=str, default='cuda', help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    extractor = DINOExtractor(
        model_name=args.model_name,
        device=args.device,
        patch_size=args.patch_size
    )
    extractor.process_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
