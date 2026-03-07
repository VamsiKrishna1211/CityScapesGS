"""
Batch DINOv2 Feature & SAM Object Extraction Script
Extracts dense DINOv2 feature maps, segments objects using SAM (Automatic Mask Generation),
and pools the features per object. Saves data as PyTorch dictionaries (.pt).
"""

import argparse
import gc
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class ImageDataset(Dataset):
    """PyTorch Dataset optimized for direct PIL loading."""
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(str(img_path)).convert("RGB")
            return {'image': image, 'image_path': str(img_path), 'valid': True}
        except Exception as e:
            logger.error(f"Failed to load {img_path}: {e}")
            return {'image': None, 'image_path': str(img_path), 'valid': False}


def custom_collate_fn(batch):
    valid_batch = [item for item in batch if item['valid']]
    return {
        'images': [item['image'] for item in valid_batch],
        'image_paths': [item['image_path'] for item in valid_batch],
    }


class DINOv2SAMPipeline:
    def __init__(self, dinov2_model="facebook/dinov2-base", sam_model="facebook/sam-vit-base", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.patch_size = 14  # DINOv2 patch size
        self.feature_dim = 768 # ViT-Base dimension

        logger.info(f"Loading DINOv2 ({dinov2_model}) and SAM ({sam_model}) on {self.device}...")
        
        # Load DINOv2
        self.dino_processor = AutoImageProcessor.from_pretrained(dinov2_model)
        self.dino_model = AutoModel.from_pretrained(dinov2_model).to(self.device)
        self.dino_model.eval()

        # Load SAM (Automatic Mask Generation pipeline)
        # We use the HF pipeline for simplicity and robust batching
        self.sam_pipeline = pipeline(
            "mask-generation",
            model=sam_model,
            use_fast=False,
            device=0 if self.device == "cuda" else -1
        )

    def _get_aspect_preserving_size(self, orig_h, orig_w):
        """Forces dimensions to be multiples of the DINOv2 patch size (14)."""
        new_h = (orig_h // self.patch_size) * self.patch_size
        new_w = (orig_w // self.patch_size) * self.patch_size
        return max(new_h, self.patch_size), max(new_w, self.patch_size)

    def get_dense_dinov2_features(self, pil_image):
        """Extracts dense DINOv2 patch-grid features without upsampling to full resolution."""
        orig_w, orig_h = pil_image.size
        target_h, target_w = self._get_aspect_preserving_size(orig_h, orig_w)

        # Resize PIL image for DINOv2 processing
        img_resized = pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        inputs = self.dino_processor(
            images=img_resized,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            # Reshape back to spatial grid: (1, Feature_Dim, H/14, W/14)
            h_feat, w_feat = target_h // self.patch_size, target_w // self.patch_size
            expected_tokens = h_feat * w_feat

            hidden_states = outputs.last_hidden_state
            token_count = hidden_states.shape[1]

            # Handle both outputs with CLS token and pure patch-token outputs.
            if token_count == expected_tokens + 1:
                patch_features = hidden_states[:, 1:, :]
            elif token_count == expected_tokens:
                patch_features = hidden_states
            else:
                raise ValueError(
                    f"Unexpected DINO token count: got {token_count}, expected {expected_tokens} "
                    f"(or {expected_tokens + 1} with CLS) for resized image {target_h}x{target_w}."
                )

            dense_features = patch_features.permute(0, 2, 1).reshape(1, self.feature_dim, h_feat, w_feat)

        # Return low-resolution patch-grid features: (768, H/14, W/14)
        return dense_features.squeeze(0)

    def downsample_mask_to_feature_grid(self, mask, target_h, target_w):
        """Downsamples SAM mask to DINO feature grid using area-based pooling."""
        mask_4d = mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        return F.adaptive_avg_pool2d(mask_4d, (target_h, target_w)).squeeze(0).squeeze(0)

    def process_single_image(self, pil_image, image_path, output_folder):
        """Processes one image through both models and pools features."""
        try:
            # 1. Get dense DINOv2 features at native patch-grid resolution
            dense_features = self.get_dense_dinov2_features(pil_image)
            _, h_feat, w_feat = dense_features.shape
            dense_features_flat = dense_features.view(self.feature_dim, -1)

            # 2. Generate SAM Masks
            # SAM returns a list of dictionaries with 'mask' (boolean arrays)
            sam_outputs = self.sam_pipeline(pil_image)
            
            extracted_objects = []
            
            for obj in sam_outputs['masks']:
                # Convert mask to PyTorch tensor without redundant copy warnings
                mask = torch.as_tensor(obj, device=self.device, dtype=torch.float32) # Shape: (H, W)
                
                # Filter out extremely small masks that act as noise
                if mask.sum() < 50:
                    continue

                # 3. Area-pool the mask to DINO grid and pool features there (low memory path)
                mask_lowres = self.downsample_mask_to_feature_grid(mask, h_feat, w_feat)
                mask_weights = mask_lowres.reshape(-1)
                weight_sum = mask_weights.sum()
                if weight_sum <= 1e-6:
                    continue

                # Weighted average over DINO patch tokens; result shape: (768,)
                object_feature = (dense_features_flat * mask_weights.unsqueeze(0)).sum(dim=1) / weight_sum
                
                # Normalize the feature vector (crucial for cosine similarity matching later)
                object_feature = F.normalize(object_feature, p=2, dim=0)

                extracted_objects.append({
                    "mask": mask_lowres.cpu(),
                    "feature": object_feature.cpu(),
                    "area": mask.sum().item()
                })

            # Save the payload
            image_name = Path(image_path).stem
            save_path = output_folder / f"{image_name}.pt"
            
            payload = {
                "image_path": image_path,
                "objects": extracted_objects,
                "dense_features_shape": tuple(dense_features.shape)
            }
            
            torch.save(payload, save_path)
            return True

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return False

    def process_folder(self, input_folder, output_folder, file_extensions=None):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

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
        # Batch size MUST be 1 because SAM generates a variable number of masks per image.
        # Ragged tensor matching across batches is inefficient and error-prone here.
        dataloader = DataLoader(
            dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )

        success_count = 0
        for batch_data in tqdm(dataloader, desc="Extracting Features & Masks"):
            if not batch_data['images']:
                continue
            
            # Since batch_size=1, we just take the first item
            img = batch_data['images'][0]
            path = batch_data['image_paths'][0]
            
            if self.process_single_image(img, path, output_path):
                success_count += 1
            
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        logger.info(f"Completed: {success_count}/{len(image_files)} images processed")

def main():
    parser = argparse.ArgumentParser(description="Extract SAM masks and pooled DINOv2 features.")
    parser.add_argument('--input-folder', '-i', type=str, required=True)
    parser.add_argument('--output-folder', '-o', type=str, required=True)
    parser.add_argument('--device', '-d', type=str, default='cuda')
    args = parser.parse_args()

    extractor = DINOv2SAMPipeline(device=args.device)
    extractor.process_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
