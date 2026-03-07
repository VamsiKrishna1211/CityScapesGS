"""
Batch resize utility for Gaussian Splatting-style image pyramids.

Given an input image folder (typically `<scene>/images`), this script creates
sibling folders like `images_2`, `images_4`, `images_8` where each image is
resized by the corresponding integer downscale factor using bicubic interpolation.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ResizeDataset(Dataset):
    def __init__(self, tasks: list[dict[str, Any]]) -> None:
        self.tasks = tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task = self.tasks[index]
        image_path: Path = task["image_path"]
        with Image.open(image_path) as img:
            # Keep grayscale as grayscale; convert complex/alpha formats to RGB.
            src = img.convert("RGB") if img.mode in {"P", "RGBA", "LA", "CMYK", "YCbCr", "HSV"} else img
            arr = np.array(src, copy=True)
            if arr.ndim == 2:
                tensor = torch.from_numpy(arr).unsqueeze(0).contiguous()
            else:
                tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        return {
            "tensor": tensor,
            "rel_path": task["rel_path"],
            "needed_scales": task["needed_scales"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create downscaled image folders (images_2, images_4, ...) with bicubic resize."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Input image directory (e.g., data/Mip-NeRF-360/bicycle/images).",
    )
    parser.add_argument(
        "-s",
        "--scales",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Integer downscale factors to generate (default: 2 4 8).",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=None,
        help="Output root where images_<scale> folders are created (default: input-dir parent).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively include images in nested folders and preserve structure.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing resized files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for torch DataLoader/interpolation (default: 16).",
    )
    return parser.parse_args()


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files)


def target_size(width: int, height: int, scale: int) -> tuple[int, int]:
    return max(1, width // scale), max(1, height // scale)


def save_tensor_image(tensor_chw_u8: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    channels = tensor_chw_u8.shape[0]

    if channels == 1:
        arr = tensor_chw_u8.squeeze(0).numpy()
        img = Image.fromarray(arr, mode="L")
    elif channels == 3:
        arr = tensor_chw_u8.permute(1, 2, 0).numpy()
        img = Image.fromarray(arr, mode="RGB")
    elif channels == 4:
        arr = tensor_chw_u8.permute(1, 2, 0).numpy()
        img = Image.fromarray(arr, mode="RGBA")
    else:
        raise ValueError(f"Unsupported number of channels for save: {channels}")

    img.save(out_path)


def build_tasks(
    image_paths: list[Path],
    input_dir: Path,
    scales: list[int],
    output_dirs: dict[int, Path],
    overwrite: bool,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    tasks: list[dict[str, Any]] = []
    skipped: dict[int, int] = {scale: 0 for scale in scales}

    for image_path in image_paths:
        rel_path = image_path.relative_to(input_dir)
        needed_scales: list[int] = []
        for scale in scales:
            out_path = output_dirs[scale] / rel_path
            if out_path.exists() and not overwrite:
                skipped[scale] += 1
            else:
                needed_scales.append(scale)

        if needed_scales:
            tasks.append(
                {
                    "image_path": image_path,
                    "rel_path": rel_path,
                    "needed_scales": needed_scales,
                }
            )

    return tasks, skipped


def process_batch(
    batch: list[dict[str, Any]],
    scales: list[int],
    output_dirs: dict[int, Path],
    written: dict[int, int],
    device: torch.device,
) -> None:
    shapes = [sample["tensor"].shape for sample in batch]
    can_stack = all(shape == shapes[0] for shape in shapes)
    stacked = None
    if can_stack:
        stacked = torch.stack([sample["tensor"] for sample in batch], dim=0)
        stacked = stacked.to(device=device, non_blocking=True, dtype=torch.float32).div_(255.0)

    for scale in scales:
        indices = [idx for idx, sample in enumerate(batch) if scale in sample["needed_scales"]]
        if not indices:
            continue

        if can_stack and stacked is not None:
            in_h, in_w = int(stacked.shape[-2]), int(stacked.shape[-1])
            out_w, out_h = target_size(in_w, in_h, scale)
            resized = F.interpolate(stacked[indices], size=(out_h, out_w), mode="bicubic", align_corners=False)
            resized_u8 = resized.mul(255.0).round_().clamp_(0, 255).to(torch.uint8).cpu()
            for j, sample_idx in enumerate(indices):
                out_path = output_dirs[scale] / batch[sample_idx]["rel_path"]
                save_tensor_image(resized_u8[j], out_path)
                written[scale] += 1
            continue

        for sample_idx in indices:
            sample = batch[sample_idx]
            tensor = sample["tensor"].unsqueeze(0).to(device=device, non_blocking=True, dtype=torch.float32).div_(255.0)
            in_h, in_w = int(tensor.shape[-2]), int(tensor.shape[-1])
            out_w, out_h = target_size(in_w, in_h, scale)
            resized = F.interpolate(tensor, size=(out_h, out_w), mode="bicubic", align_corners=False)
            resized_u8 = resized.squeeze(0).mul(255.0).round_().clamp_(0, 255).to(torch.uint8).cpu()
            out_path = output_dirs[scale] / sample["rel_path"]
            save_tensor_image(resized_u8, out_path)
            written[scale] += 1


def validate_scales(scales: list[int]) -> list[int]:
    unique = sorted(set(scales))
    invalid = [s for s in unique if s <= 1]
    if invalid:
        raise ValueError(f"All scales must be integers > 1. Invalid values: {invalid}")
    return unique


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    scales = validate_scales(args.scales)
    output_root = args.output_root.expanduser().resolve() if args.output_root else input_dir.parent
    batch_size = max(1, int(args.batch_size))

    image_paths = collect_images(input_dir, args.recursive)
    if not image_paths:
        logger.warning("No images found in %s", input_dir)
        return

    logger.info("Found %d images in %s", len(image_paths), input_dir)
    logger.info("Scales: %s", scales)
    logger.info("Batch size: %d", batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using torch device: %s", device)

    output_dirs: dict[int, Path] = {}
    for scale in scales:
        out_dir = output_root / f"{input_dir.name}_{scale}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[scale] = out_dir

    tasks, skipped = build_tasks(
        image_paths=image_paths,
        input_dir=input_dir,
        scales=scales,
        output_dirs=output_dirs,
        overwrite=args.overwrite,
    )

    if not tasks:
        for scale in scales:
            logger.info(
                "Scale %d complete | output: %s | written: %d | skipped: %d",
                scale,
                output_dirs[scale],
                0,
                skipped[scale],
            )
        return

    num_workers = min(8, os.cpu_count() or 1)
    loader = DataLoader(
        ResizeDataset(tasks),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=lambda batch: batch,
    )

    written = {scale: 0 for scale in scales}
    pbar = tqdm(total=len(tasks), desc="Resizing batches", unit="img")
    for batch in loader:
        process_batch(
            batch=batch,
            scales=scales,
            output_dirs=output_dirs,
            written=written,
            device=device,
        )
        pbar.update(len(batch))
    pbar.close()

    for scale in scales:
        logger.info(
            "Scale %d complete | output: %s | written: %d | skipped: %d",
            scale,
            output_dirs[scale],
            written[scale],
            skipped[scale],
        )


if __name__ == "__main__":
    main()
