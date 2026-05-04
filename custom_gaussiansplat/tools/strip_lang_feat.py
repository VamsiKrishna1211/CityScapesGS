"""strip_lang_feat.py — Remove _anchor_lang_feat from a ScaffoldModel checkpoint.

Geometry-only checkpoints that were saved while a SemanticScaffoldModel was
active contain ``_anchor_lang_feat`` with shape [0] (the empty placeholder from
``__init__``).  This confuses the semantic fine-tuning script into treating the
checkpoint as an already-trained semantic model, causing an IndexError at
training time.

Usage
-----
    python tools/strip_lang_feat.py --checkpoint path/to/checkpoint_45000.pt
    # writes  path/to/checkpoint_45000_stripped.pt  by default

    python tools/strip_lang_feat.py \\
        --checkpoint path/to/checkpoint_45000.pt \\
        --output path/to/checkpoint_45000_clean.pt
    # or overwrite in-place (careful!):
    python tools/strip_lang_feat.py \\
        --checkpoint path/to/checkpoint_45000.pt \\
        --inplace
"""
import argparse
import shutil
from pathlib import Path

import torch


KEYS_TO_STRIP = {
    "_anchor_lang_feat",
    "language_codebook",
    "codebook_proj",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strip semantic placeholder keys from a geometry-only checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to the input checkpoint (.pt)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output path. Defaults to <stem>_stripped.pt next to input.")
    p.add_argument("--inplace", action="store_true",
                   help="Overwrite the input file instead of writing a new one.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.inplace:
        out_path = args.checkpoint
    elif args.output is not None:
        out_path = args.output
    else:
        out_path = args.checkpoint.with_stem(args.checkpoint.stem + "_stripped")

    print(f"Loading  : {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    state_dict = checkpoint.get("model_state_dict", {})
    removed = []
    for key in list(state_dict.keys()):
        if key in KEYS_TO_STRIP:
            del state_dict[key]
            removed.append(key)

    # Also strip top-level semantic metadata keys written by train_semantics.py
    # so this checkpoint is never mistaken for a completed semantic run.
    for meta_key in ("lang_feat_dim", "codebook_size", "clip_dim"):
        if meta_key in checkpoint:
            del checkpoint[meta_key]
            removed.append(f"[top-level] {meta_key}")

    if not removed:
        print("Nothing to strip — no semantic keys found in this checkpoint.")
    else:
        print("Stripped keys:")
        for k in removed:
            print(f"  {k}")

    if args.inplace and out_path == args.checkpoint:
        # Write to a temp file first to avoid corrupting the original on failure.
        tmp = out_path.with_suffix(".tmp.pt")
        torch.save(checkpoint, tmp)
        shutil.move(str(tmp), str(out_path))
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, out_path)

    print(f"Saved to : {out_path}")


if __name__ == "__main__":
    main()
