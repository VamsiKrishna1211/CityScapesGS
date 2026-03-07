"""
Batch Sky Segmentation Script (GroundingDINO + SAM)
Segments sky regions from images using text-guided detection and mask refinement.
Saves binary masks as .npy (raw) and .png (visual) with optional overlay previews.
"""

import argparse
import gc
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('high')

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")


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
            return {"image": image, "image_path": str(img_path), "valid": True}
        except Exception as exc:
            logger.error(f"Failed to load {img_path}: {exc}")
            return {"image": None, "image_path": str(img_path), "valid": False}


def custom_collate_fn(batch):
    valid_batch = [item for item in batch if item["valid"]]
    return {
        "images": [item["image"] for item in valid_batch],
        "image_paths": [item["image_path"] for item in valid_batch],
    }


class GroundedSAMSkySegmenter:
    def __init__(
        self,
        grounding_model_name="IDEA-Research/grounding-dino-base",
        sam_model_name="facebook/sam-vit-base",
        device="cuda",
        box_threshold=0.28,
        text_threshold=0.25,
        sky_prompts=None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.grounding_model_name = grounding_model_name
        self.sam_model_name = sam_model_name
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        if sky_prompts is None:
            sky_prompts = [
                "sky.",
                "cloud.",
                "clouds.",
                "overcast sky.",
                "blue sky.",
                "clear sky.",
                "horizon.",
                "atmosphere.",
                "skyline.",
            ]
        self.sky_prompts = sky_prompts

        logger.info(
            "Using device: %s | Grounding model: %s | SAM model: %s",
            self.device,
            self.grounding_model_name,
            self.sam_model_name,
        )
        self.load_models()

    def load_models(self):
        self.grounding_processor = AutoProcessor.from_pretrained(self.grounding_model_name)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.grounding_model_name
        ).to(self.device, dtype=torch.float32)

        self.grounding_model = torch.compile(self.grounding_model)

        self.sam_processor = SamProcessor.from_pretrained(self.sam_model_name)
        self.sam_model = SamModel.from_pretrained(self.sam_model_name).to(
            self.device, dtype=torch.float32
        )
        self.sam_model = torch.compile(self.sam_model)

        self.grounding_model.eval()
        self.sam_model.eval()

    def _detect_sky_boxes(self, pil_image):
        h, w = pil_image.size[1], pil_image.size[0]

        all_boxes = []
        for sky_prompt in self.sky_prompts:
            inputs = self.grounding_processor(
                images=pil_image,
                text=sky_prompt,
                return_tensors="pt",
            ).to(self.device)

            # Keep all floating tensors in true FP32.
            for key, value in inputs.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    inputs[key] = value.to(torch.float32)

            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

            # transformers changed this API across versions; support both old/new signatures.
            try:
                results = self.grounding_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=[(h, w)],
                )
            except TypeError:
                try:
                    results = self.grounding_processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        target_sizes=[(h, w)],
                    )
                except TypeError:
                    results = self.grounding_processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        threshold=self.box_threshold,
                        target_sizes=[(h, w)],
                    )

            if results and len(results[0]["boxes"]) > 0:
                prompt_boxes = results[0]["boxes"].detach().cpu().numpy().astype(np.float32)
                all_boxes.append(prompt_boxes)

        if not all_boxes:
            return np.empty((0, 4), dtype=np.float32)

        boxes = np.concatenate(all_boxes, axis=0)
        boxes = np.unique(boxes.round(decimals=1), axis=0).astype(np.float32)
        return boxes

    def _segment_boxes_with_sam(self, pil_image, boxes):
        h, w = pil_image.size[1], pil_image.size[0]
        if boxes.shape[0] == 0:
            return np.zeros((h, w), dtype=bool)

        # SamProcessor expects a per-image list of input boxes.
        sam_inputs = self.sam_processor(
            pil_image,
            input_boxes=[boxes.tolist()],
            return_tensors="pt",
        ).to(self.device)

        # Keep all floating tensors in true FP32.
        for key, value in sam_inputs.items():
            if torch.is_tensor(value) and value.is_floating_point():
                sam_inputs[key] = value.to(torch.float32)

        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

        original_sizes = sam_inputs["original_sizes"]
        reshaped_input_sizes = sam_inputs["reshaped_input_sizes"]

        if hasattr(original_sizes, "detach"):
            original_sizes = original_sizes.detach().cpu()
        if hasattr(reshaped_input_sizes, "detach"):
            reshaped_input_sizes = reshaped_input_sizes.detach().cpu()

        post_process_masks = getattr(self.sam_processor, "post_process_masks", None)
        if post_process_masks is None:
            post_process_masks = self.sam_processor.image_processor.post_process_masks

        upscaled_masks = post_process_masks(
            masks=sam_outputs.pred_masks.detach().cpu(),
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
        )

        # Shape after post-processing: [num_boxes, num_candidates, H, W]
        masks = upscaled_masks[0].numpy()
        iou_scores = sam_outputs.iou_scores[0].detach().cpu().numpy()

        combined_mask = np.zeros((h, w), dtype=bool)
        for i in range(masks.shape[0]):
            best_idx = int(np.argmax(iou_scores[i]))
            box_mask = masks[i, best_idx] > 0
            combined_mask |= box_mask

        return self._refine_mask(combined_mask)

    @staticmethod
    def _refine_mask(mask):
        mask_uint8 = (mask.astype(np.uint8) * 255)
        close_kernel = np.ones((9, 9), np.uint8)
        open_kernel = np.ones((5, 5), np.uint8)

        closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=1)

        return opened > 0

    @staticmethod
    def _create_overlay(image_rgb, sky_mask):
        overlay = image_rgb.copy()
        # Tint sky in cyan for quick visual QA.
        overlay[sky_mask] = (
            0.35 * overlay[sky_mask] + 0.65 * np.array([0, 220, 255], dtype=np.float32)
        ).astype(np.uint8)
        return overlay

    def process_batch(self, batch_data, output_npy_path, output_png_path, output_overlay_path=None):
        if not batch_data["images"]:
            return 0

        pil_images = batch_data["images"]
        image_paths = batch_data["image_paths"]
        success_count = 0

        try:
            for pil_image, img_path in zip(pil_images, image_paths):
                image_name = Path(img_path).stem
                image_np = np.array(pil_image)

                sky_boxes = self._detect_sky_boxes(pil_image)
                sky_mask = self._segment_boxes_with_sam(pil_image, sky_boxes)

                np.save(str(output_npy_path / f"{image_name}.npy"), sky_mask.astype(np.uint8))
                cv2.imwrite(str(output_png_path / f"{image_name}.png"), (sky_mask.astype(np.uint8) * 255))

                if output_overlay_path is not None:
                    overlay = self._create_overlay(image_np, sky_mask)
                    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_overlay_path / f"{image_name}.png"), overlay_bgr)

                success_count += 1

        except Exception as exc:
            logger.error(f"Error during batch segmentation: {exc}")
        finally:
            # gc.collect()
            # if self.device == "cuda":
            #     torch.cuda.empty_cache()
            pass

        return success_count

    def process_folder(
        self,
        input_folder,
        output_npy_folder,
        output_png_folder,
        output_overlay_folder=None,
        batch_size=1,
        num_workers=4,
        file_extensions=None,
    ):
        input_path = Path(input_folder)
        output_npy_path = Path(output_npy_folder)
        output_png_path = Path(output_png_folder)

        output_npy_path.mkdir(parents=True, exist_ok=True)
        output_png_path.mkdir(parents=True, exist_ok=True)

        output_overlay_path = None
        if output_overlay_folder:
            output_overlay_path = Path(output_overlay_folder)
            output_overlay_path.mkdir(parents=True, exist_ok=True)

        if file_extensions is None:
            file_extensions = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]

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
        for batch_data in tqdm(dataloader, desc="Segmenting sky"):
            success_count += self.process_batch(
                batch_data,
                output_npy_path,
                output_png_path,
                output_overlay_path=output_overlay_path,
            )

        logger.info(f"Completed: {success_count}/{len(image_files)} images processed")


def main():
    parser = argparse.ArgumentParser(
        description="Batch sky segmentation using GroundingDINO + SAM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-folder", "-i", type=str, required=True)
    parser.add_argument("--output-npy-folder", "-on", type=str, required=True)
    parser.add_argument("--output-png-folder", "-op", type=str, required=True)
    parser.add_argument("--output-overlay-folder", "-oo", type=str, default=None)
    parser.add_argument("--grounding-model", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model", type=str, default="facebook/sam-vit-base")
    parser.add_argument(
        "--sky-prompts",
        type=str,
        nargs="+",
        default=[
            "sky.",
            "cloud.",
            "clouds.",
            "overcast sky.",
            "blue sky.",
            "clear sky.",
            "horizon.",
            "atmosphere.",
            "skyline.",
        ],
    )
    parser.add_argument("--box-threshold", type=float, default=0.5)
    parser.add_argument("--text-threshold", type=float, default=0.5)
    parser.add_argument("--device", "-d", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--num-workers", "-w", type=int, default=4)
    parser.add_argument(
        "--extensions",
        "-e",
        type=str,
        nargs="+",
        default=[".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"],
    )
    args = parser.parse_args()

    if not Path(args.input_folder).exists():
        logger.error(f"Input folder missing: {args.input_folder}")
        return

    segmenter = GroundedSAMSkySegmenter(
        grounding_model_name=args.grounding_model,
        sam_model_name=args.sam_model,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        sky_prompts=args.sky_prompts,
    )

    segmenter.process_folder(
        input_folder=args.input_folder,
        output_npy_folder=args.output_npy_folder,
        output_png_folder=args.output_png_folder,
        output_overlay_folder=args.output_overlay_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        file_extensions=args.extensions,
    )


if __name__ == "__main__":
    main()
