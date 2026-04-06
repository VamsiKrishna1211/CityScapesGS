#!/usr/bin/env python3
"""Populate/normalize Instant-NGP frame file_path entries.

This script updates transforms.json frames by:
- filling missing file_path from frame_index using common filename patterns
- normalizing file_path values to be relative to BASE_PATH

It is designed to match the lookup behavior used in custom_gaussiansplat.dataset
for Instant-NGP frame image resolution.

When requested, this script can also synthesize frame.transform_matrix from
MatrixCity-style frame.rot_mat. The conversion is explicitly one-way:
MatrixCity rot_mat (w2c-ish export) -> Instant-NGP transform_matrix field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add/normalize frame file_path entries in transforms.json")
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Scene root path (BASE_PATH), e.g. .../block_1",
    )
    parser.add_argument(
        "--transforms",
        type=Path,
        default=None,
        help="Path to transforms.json (default: BASE_PATH/transforms.json)",
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="images",
        help="Image directory under BASE_PATH used for frame_index mapping",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Also remap frames that already have a file_path if it cannot be resolved",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak backup before writing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show summary without writing changes",
    )
    parser.add_argument(
        "--add-transform-matrix",
        action="store_true",
        help=(
            "Add/overwrite frame.transform_matrix from MatrixCity frame.rot_mat using conversion: "
            "rot_mat[:3,:3] *= --rot-mat-scale-multiplier and rot_mat[:3,3] /= --translation-scale"
        ),
    )
    parser.add_argument(
        "--transform-overwrite",
        action="store_true",
        help="When --add-transform-matrix is set, overwrite existing transform_matrix values",
    )
    parser.add_argument(
        "--rot-mat-scale-multiplier",
        type=float,
        default=100.0,
        help="Multiplier applied to rot_mat[:3,:3] when creating transform_matrix (default: 100.0)",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help=(
            "Divisor applied to rot_mat[:3,3] when creating transform_matrix (default: 1.0). "
            "Use your dataset SCALE value here."
        ),
    )
    return parser.parse_args()


def to_posix_rel(path: Path, base_path: Path) -> str:
    try:
        rel = path.resolve().relative_to(base_path.resolve())
        return rel.as_posix()
    except ValueError:
        return Path(path).as_posix()


def build_frame_index_candidates(frame_index: int) -> list[str]:
    candidates: list[str] = []

    def add(value: str) -> None:
        if value not in candidates:
            candidates.append(value)

    for ext in IMAGE_EXTENSIONS:
        for width in (4, 5, 6):
            add(f"{frame_index:0{width}d}{ext}")
            if frame_index > 0:
                add(f"{frame_index - 1:0{width}d}{ext}")
        add(f"{frame_index}{ext}")
        if frame_index > 0:
            add(f"{frame_index - 1}{ext}")

    return candidates


def resolve_existing_file_path(
    file_path_str: str,
    base_path: Path,
    transforms_root: Path,
    images_root: Path,
) -> Optional[Path]:
    p = Path(str(file_path_str))

    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                transforms_root / p,
                images_root / p,
                base_path / p,
            ]
        )

    if p.suffix == "":
        extra: list[Path] = []
        for c in candidates:
            for ext in IMAGE_EXTENSIONS:
                extra.append(c.with_suffix(ext))
        candidates.extend(extra)

    seen = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_file():
            return c.resolve()

    return None


def collect_images(images_root: Path) -> dict[str, Path]:
    if not images_root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")

    image_map: dict[str, Path] = {}
    for p in sorted(images_root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Keep first match for a basename to avoid non-determinism.
        image_map.setdefault(p.name, p.resolve())
    return image_map


def build_transform_matrix_from_matrixcity_rot_mat(
    frame: dict,
    rot_mat_scale_multiplier: float,
    translation_scale: float,
) -> Optional[list[list[float]]]:
    """Build Instant-NGP transform_matrix from MatrixCity rot_mat.

    Conversion:
    - matrix[:3, :3] *= rot_mat_scale_multiplier
    - matrix[:3, 3] /= translation_scale
    """
    rot_mat = frame.get("rot_mat", None)
    if rot_mat is None:
        return None

    if not isinstance(rot_mat, list) or len(rot_mat) != 4:
        return None

    matrix: list[list[float]] = []
    for row in rot_mat:
        if not isinstance(row, list) or len(row) != 4:
            return None
        try:
            matrix.append([float(v) for v in row])
        except (TypeError, ValueError):
            return None

    for i in range(3):
        for j in range(3):
            matrix[i][j] *= float(rot_mat_scale_multiplier)
        matrix[i][3] /= float(translation_scale)

    return matrix


def update_transforms(
    base_path: Path,
    transforms_path: Path,
    images_root: Path,
    overwrite_existing: bool,
    add_transform_matrix: bool,
    transform_overwrite: bool,
    rot_mat_scale_multiplier: float,
    translation_scale: float,
) -> tuple[dict, dict[str, int]]:
    with transforms_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    frames = meta.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("transforms.json has invalid format: 'frames' must be a list")

    image_map = collect_images(images_root)

    stats = {
        "total_frames": len(frames),
        "updated": 0,
        "transform_updated": 0,
        "already_ok": 0,
        "missing_frame_index": 0,
        "unresolved": 0,
        "invalid_frame_objects": 0,
    }

    transforms_root = transforms_path.parent.resolve()
    base_path = base_path.resolve()

    for frame in frames:
        if not isinstance(frame, dict):
            stats["invalid_frame_objects"] += 1
            continue

        if add_transform_matrix:
            has_existing_transform = "transform_matrix" in frame and frame.get("transform_matrix") is not None
            if transform_overwrite or not has_existing_transform:
                transform_matrix = build_transform_matrix_from_matrixcity_rot_mat(
                    frame=frame,
                    rot_mat_scale_multiplier=rot_mat_scale_multiplier,
                    translation_scale=translation_scale,
                )
                if transform_matrix is not None and frame.get("transform_matrix") != transform_matrix:
                    frame["transform_matrix"] = transform_matrix
                    stats["transform_updated"] += 1

        existing_path = frame.get("file_path", None)
        frame_index = frame.get("frame_index", None)

        if isinstance(existing_path, str) and existing_path.strip():
            resolved_existing = resolve_existing_file_path(existing_path, base_path, transforms_root, images_root)
            if resolved_existing is not None:
                normalized = to_posix_rel(resolved_existing, base_path)
                if frame.get("file_path") != normalized:
                    frame["file_path"] = normalized
                    stats["updated"] += 1
                else:
                    stats["already_ok"] += 1
                continue
            if not overwrite_existing:
                stats["unresolved"] += 1
                continue

        if frame_index is None:
            stats["missing_frame_index"] += 1
            continue

        try:
            idx = int(frame_index)
        except (TypeError, ValueError):
            stats["unresolved"] += 1
            continue

        resolved = None
        for name in build_frame_index_candidates(idx):
            resolved = image_map.get(name)
            if resolved is not None:
                break

        if resolved is None:
            stats["unresolved"] += 1
            continue

        rel_path = to_posix_rel(resolved, base_path)
        if frame.get("file_path") != rel_path:
            frame["file_path"] = rel_path
            stats["updated"] += 1
        else:
            stats["already_ok"] += 1

    return meta, stats


def main() -> int:
    args = parse_args()

    base_path = args.base_path.expanduser().resolve()
    transforms_path = args.transforms.expanduser().resolve() if args.transforms else (base_path / "transforms.json")
    images_root = (base_path / args.images_subdir).resolve()

    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found: {transforms_path}")

    if args.translation_scale == 0:
        raise ValueError("--translation-scale must be non-zero")

    meta, stats = update_transforms(
        base_path=base_path,
        transforms_path=transforms_path,
        images_root=images_root,
        overwrite_existing=args.overwrite_existing,
        add_transform_matrix=args.add_transform_matrix,
        transform_overwrite=args.transform_overwrite,
        rot_mat_scale_multiplier=args.rot_mat_scale_multiplier,
        translation_scale=args.translation_scale,
    )

    print("Update summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if args.dry_run:
        print("Dry run enabled, no files written.")
        return 0

    if not args.no_backup:
        backup_path = transforms_path.with_suffix(transforms_path.suffix + ".bak")
        backup_path.write_text(transforms_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup written: {backup_path}")

    with transforms_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")

    print(f"Updated transforms written: {transforms_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
