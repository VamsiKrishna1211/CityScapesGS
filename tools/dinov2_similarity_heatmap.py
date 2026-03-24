#!/usr/bin/env python3
"""Run DINOv2 inference on an image and plot a similarity heatmap.

Example:
    python tools/dinov2_similarity_heatmap.py \
        --image /path/to/image.jpg \
        --output-dir /tmp/dino_out \
        --query-x 0.5 --query-y 0.5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 features and visualize a cosine-similarity heatmap."
    )
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument(
        "--model",
        default="facebook/dinov2-base",
        type=str,
        help="Hugging Face DINOv2 model id",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/dinov2_heatmap",
        type=str,
        help="Directory for feature and heatmap outputs",
    )
    parser.add_argument(
        "--query-x",
        default=0.5,
        type=float,
        help="Query x in [0,1], normalized image coordinate",
    )
    parser.add_argument(
        "--query-y",
        default=0.5,
        type=float,
        help="Query y in [0,1], normalized image coordinate",
    )
    parser.add_argument(
        "--alpha",
        default=0.5,
        type=float,
        help="Overlay strength for the heatmap",
    )
    parser.add_argument(
        "--inference-width",
        default=840,
        type=int,
        help="Inference width in pixels (will be aligned to patch size)",
    )
    parser.add_argument(
        "--inference-height",
        default=840,
        type=int,
        help="Inference height in pixels (will be aligned to patch size)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="cuda, cpu, or mps",
    )
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def align_to_patch(value: int, patch_size: int) -> int:
    return max(patch_size, (int(value) // patch_size) * patch_size)


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    requested_device = args.device.lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif requested_device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    else:
        device = requested_device

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)

    patch_size = 14
    infer_w = align_to_patch(args.inference_width, patch_size)
    infer_h = align_to_patch(args.inference_height, patch_size)
    infer_image = image.resize((infer_w, infer_h), Image.Resampling.BICUBIC)

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    # Disable processor-side resize/crop so we control inference resolution.
    inputs = processor(
        images=infer_image,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # DINOv2 returns [CLS] + patch tokens for ViT backbones.
    tokens = outputs.last_hidden_state[0]
    patch_tokens = tokens[1:] if tokens.shape[0] > 1 else tokens

    h_in = int(inputs["pixel_values"].shape[-2])
    w_in = int(inputs["pixel_values"].shape[-1])
    h_feat = h_in // patch_size
    w_feat = w_in // patch_size

    if patch_tokens.shape[0] != h_feat * w_feat:
        raise RuntimeError(
            "Token/grid mismatch: "
            f"tokens={patch_tokens.shape[0]} vs grid={h_feat}x{w_feat}."
        )

    features = patch_tokens.reshape(h_feat, w_feat, -1)
    features = F.normalize(features, p=2, dim=-1)

    qx = clamp01(args.query_x)
    qy = clamp01(args.query_y)
    qx_idx = min(w_feat - 1, int(round(qx * (w_feat - 1))))
    qy_idx = min(h_feat - 1, int(round(qy * (h_feat - 1))))

    query_vec = features[qy_idx, qx_idx]

    # Cosine similarity against every patch feature.
    sim = torch.einsum("hwc,c->hw", features, query_vec)
    sim_np = sim.detach().cpu().numpy()

    sim_min, sim_max = sim_np.min(), sim_np.max()
    sim_vis = (sim_np - sim_min) / (sim_max - sim_min + 1e-8)

    base_name = image_path.stem
    feat_out = output_dir / f"{base_name}_dinov2_feats.pt"
    heatmap_out = output_dir / f"{base_name}_similarity_heatmap.png"

    payload = {
        "image_path": str(image_path),
        "model": args.model,
        "feature_grid": features.detach().cpu(),  # [Hf, Wf, C]
        "similarity_map": sim.detach().cpu(),     # [Hf, Wf]
        "query_patch_xy": (qx_idx, qy_idx),
        "query_norm_xy": (qx, qy),
    }
    torch.save(payload, feat_out)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_np)
    ax.imshow(
        sim_vis,
        cmap="inferno",
        alpha=args.alpha,
        interpolation="bilinear",
        extent=(0, image_np.shape[1], image_np.shape[0], 0),
    )

    px = (qx_idx + 0.5) * image_np.shape[1] / w_feat
    py = (qy_idx + 0.5) * image_np.shape[0] / h_feat
    ax.scatter([px], [py], c="cyan", s=40)
    ax.set_title("DINOv2 Similarity Heatmap")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(heatmap_out, dpi=200)
    plt.close(fig)

    print(f"Saved features: {feat_out}")
    print(f"Saved heatmap:  {heatmap_out}")
    print(f"Inference size: ({infer_w}, {infer_h})")
    print(f"Feature grid shape: {tuple(features.shape)}")
    print(f"Query patch index (x,y): ({qx_idx},{qy_idx})")


if __name__ == "__main__":
    main()
