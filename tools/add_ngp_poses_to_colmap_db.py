#!/usr/bin/env python3
"""Inject Instant-NGP camera poses into an existing COLMAP database.

This script updates the `images` table prior pose fields:
  - prior_qw, prior_qx, prior_qy, prior_qz
  - prior_tx, prior_ty, prior_tz

It assumes cameras, images, features, and matches are already present in
`database.db` and only writes pose priors for triangulation/reconstruction.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pycolmap
except ImportError as exc:
    raise ImportError(
        "pycolmap is required. Install it in your active environment."
    ) from exc


@dataclass(frozen=True)
class PosePrior:
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add camera pose priors from Instant-NGP transforms.json to an "
            "existing COLMAP database.db"
        )
    )
    parser.add_argument(
        "--transforms-path",
        required=True,
        type=Path,
        help="Path to Instant-NGP transforms.json",
    )
    parser.add_argument(
        "--database-path",
        required=True,
        type=Path,
        help="Path to COLMAP database.db",
    )
    parser.add_argument(
        "--images-path",
        default=None,
        type=Path,
        help=(
            "Optional image root path used for frame-name resolution when "
            "transforms file paths are relative"
        ),
    )
    parser.add_argument(
        "--default-ext",
        default=".png",
        help="Extension to append when a frame file_path has no suffix (default: .png)",
    )
    parser.add_argument(
        "--index-width",
        type=int,
        default=4,
        help="Zero-padding width for frame_index filename candidates (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report matches and updates without writing to the database",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any frame cannot be matched to a unique COLMAP image",
    )
    return parser.parse_args()


def sanitize_rotation(rotation: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3)."""
    u, _, vt = np.linalg.svd(rotation)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def rigidify_w2c_preserve_center(w2c_affine: np.ndarray) -> np.ndarray:
    """Convert an affine w2c matrix to rigid form while preserving camera center."""
    if w2c_affine.shape != (4, 4):
        raise ValueError(f"Expected 4x4 w2c matrix, got {w2c_affine.shape}")
    if not np.isfinite(w2c_affine).all():
        raise ValueError("w2c matrix contains non-finite values")

    affine_rot = w2c_affine[:3, :3].astype(np.float64)
    affine_t = w2c_affine[:3, 3].astype(np.float64)
    rot_rigid = sanitize_rotation(affine_rot)

    try:
        cam_center = -np.linalg.solve(affine_rot, affine_t)
    except np.linalg.LinAlgError:
        if np.linalg.matrix_rank(affine_rot) < 3:
            raise ValueError("w2c rotation block is singular or near-singular")
        cam_center = -np.linalg.pinv(affine_rot) @ affine_t

    t_rigid = -rot_rigid @ cam_center
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot_rigid
    out[:3, 3] = t_rigid
    return out


def matrixcity_rotmat_to_rigid_w2c(rot_mat_4x4: np.ndarray) -> np.ndarray:
    """Convert MatrixCity-style rot_mat (w2c with possible uniform scale) to rigid w2c."""
    w2c_affine = np.asarray(rot_mat_4x4, dtype=np.float64)
    if w2c_affine.shape != (4, 4):
        raise ValueError(f"rot_mat must be 4x4, got shape {w2c_affine.shape}")
    if not np.isfinite(w2c_affine).all():
        raise ValueError("rot_mat contains non-finite values")

    _, singular_values, _ = np.linalg.svd(w2c_affine[:3, :3].astype(np.float64))
    uniform_scale = float(np.mean(np.abs(singular_values)))
    if not np.isfinite(uniform_scale) or uniform_scale <= 1e-8:
        raise ValueError("Could not infer a valid uniform scale from rot_mat")

    w2c_affine = w2c_affine.copy()
    if abs(uniform_scale - 1.0) > 1e-6:
        w2c_affine[:3, 3] = w2c_affine[:3, 3] * uniform_scale

    return rigidify_w2c_preserve_center(w2c_affine)


def rotmat_to_colmap_qvec(rotation: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to COLMAP quaternion [qw, qx, qy, qz]."""
    r = sanitize_rotation(rotation)
    k = np.array(
        [
            [r[0, 0] - r[1, 1] - r[2, 2], 0.0, 0.0, 0.0],
            [r[1, 0] + r[0, 1], r[1, 1] - r[0, 0] - r[2, 2], 0.0, 0.0],
            [r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], r[2, 2] - r[0, 0] - r[1, 1], 0.0],
            [r[1, 2] - r[2, 1], r[2, 0] - r[0, 2], r[0, 1] - r[1, 0], r[0, 0] + r[1, 1] + r[2, 2]],
        ]
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec.astype(np.float64)


def load_ngp_frames(transforms_path: Path) -> list[dict]:
    with transforms_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {transforms_path}")
    return frames


def pose_from_frame(frame: dict) -> Optional[PosePrior]:
    rot_mat = frame.get("rot_mat")
    transform = frame.get("transform_matrix")

    if rot_mat is not None:
        try:
            w2c = matrixcity_rotmat_to_rigid_w2c(np.asarray(rot_mat, dtype=np.float64))
        except Exception as exc:
            if transform is not None:
                raise ValueError(
                    "Frame has both rot_mat and transform_matrix, but rot_mat is invalid. "
                    "MatrixCity-strict precedence refuses fallback to transform_matrix. "
                    f"rot_mat error: {exc}"
                ) from exc
            raise
    elif transform is not None:
        c2w = np.asarray(transform, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"transform_matrix must be 4x4, got shape {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("transform_matrix contains non-finite values")
        try:
            w2c = rigidify_w2c_preserve_center(np.linalg.inv(c2w))
        except np.linalg.LinAlgError as exc:
            raise ValueError("transform_matrix is singular") from exc
    else:
        return None
    qvec = rotmat_to_colmap_qvec(w2c[:3, :3])
    tvec = w2c[:3, 3]

    return PosePrior(
        qw=float(qvec[0]),
        qx=float(qvec[1]),
        qy=float(qvec[2]),
        qz=float(qvec[3]),
        tx=float(tvec[0]),
        ty=float(tvec[1]),
        tz=float(tvec[2]),
    )


def to_posix_str(path_like: Path | str) -> str:
    return str(Path(path_like)).replace("\\", "/")


def build_frame_name_candidates(
    file_path_str: str,
    transforms_root: Path,
    images_root: Optional[Path],
    default_ext: str,
) -> List[str]:
    p = Path(file_path_str)
    if p.suffix == "":
        p = p.with_suffix(default_ext)

    candidates: list[str] = []

    def add_candidate(value: str) -> None:
        value = value.replace("\\", "/")
        if value not in candidates:
            candidates.append(value)

    add_candidate(to_posix_str(p))
    add_candidate(p.name)

    roots: list[Path] = [transforms_root]
    if images_root is not None:
        roots.append(images_root)

    for root in roots:
        joined = (root / p).resolve()
        add_candidate(to_posix_str(joined))
        try:
            add_candidate(to_posix_str(joined.relative_to(root.resolve())))
        except ValueError:
            pass

    return candidates


def build_frame_index_candidates(frame_index: int, default_ext: str, index_width: int) -> List[str]:
    candidates: list[str] = []

    def add_candidate(value: str) -> None:
        if value not in candidates:
            candidates.append(value)

    widths = [max(1, index_width), 4, 5, 6]
    for w in widths:
        add_candidate(f"{frame_index:0{w}d}{default_ext}")
        if frame_index > 0:
            add_candidate(f"{frame_index - 1:0{w}d}{default_ext}")

    add_candidate(f"{frame_index}{default_ext}")
    if frame_index > 0:
        add_candidate(f"{frame_index - 1}{default_ext}")

    return candidates


def load_db_images_with_pycolmap(database_path: Path) -> Dict[str, int]:
    with pycolmap.Database.open(database_path) as db:
        return {image.name.replace("\\", "/"): int(image.image_id) for image in db.read_all_images()}


def build_basename_lookup(name_to_id: Dict[str, int]) -> Dict[str, List[int]]:
    basename_to_ids: Dict[str, List[int]] = {}
    for name, image_id in name_to_id.items():
        basename = Path(name).name
        basename_to_ids.setdefault(basename, []).append(image_id)
    return basename_to_ids


def resolve_image_id(
    candidates: Iterable[str],
    name_to_id: Dict[str, int],
    basename_to_ids: Dict[str, List[int]],
) -> Tuple[Optional[int], str]:
    for candidate in candidates:
        if candidate in name_to_id:
            return name_to_id[candidate], "exact"

    basenames = [Path(c).name for c in candidates]
    seen = set()
    for b in basenames:
        if b in seen:
            continue
        seen.add(b)
        ids = basename_to_ids.get(b, [])
        if len(ids) == 1:
            return ids[0], "basename"
        if len(ids) > 1:
            return None, "ambiguous_basename"

    return None, "missing"


def update_pose_priors(database_path: Path, updates: Dict[int, PosePrior], dry_run: bool) -> None:
    if dry_run:
        return

    with sqlite3.connect(str(database_path)) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            UPDATE images
            SET
                prior_qw = ?,
                prior_qx = ?,
                prior_qy = ?,
                prior_qz = ?,
                prior_tx = ?,
                prior_ty = ?,
                prior_tz = ?
            WHERE image_id = ?
            """,
            [
                (
                    pose.qw,
                    pose.qx,
                    pose.qy,
                    pose.qz,
                    pose.tx,
                    pose.ty,
                    pose.tz,
                    image_id,
                )
                for image_id, pose in updates.items()
            ],
        )
        conn.commit()


def main() -> None:
    args = parse_args()

    transforms_path = args.transforms_path.expanduser().resolve()
    database_path = args.database_path.expanduser().resolve()
    images_root = args.images_path.expanduser().resolve() if args.images_path else None

    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms file not found: {transforms_path}")
    if not database_path.exists():
        raise FileNotFoundError(f"database file not found: {database_path}")

    if not args.default_ext.startswith("."):
        raise ValueError("--default-ext must start with '.' (example: .png)")
    if args.index_width < 1:
        raise ValueError("--index-width must be >= 1")

    frames = load_ngp_frames(transforms_path)
    name_to_id = load_db_images_with_pycolmap(database_path)
    basename_to_ids = build_basename_lookup(name_to_id)

    updates: Dict[int, PosePrior] = {}
    matched = 0
    missing = 0
    ambiguous = 0
    no_pose = 0

    for frame in frames:
        pose = pose_from_frame(frame)
        if pose is None:
            no_pose += 1
            continue

        frame_path = frame.get("file_path")
        frame_index = frame.get("frame_index")
        candidates: List[str]
        match_label: str
        if frame_path is not None:
            candidates = build_frame_name_candidates(
                file_path_str=frame_path,
                transforms_root=transforms_path.parent,
                images_root=images_root,
                default_ext=args.default_ext,
            )
            match_label = str(frame_path)
        elif frame_index is not None:
            candidates = build_frame_index_candidates(
                frame_index=int(frame_index),
                default_ext=args.default_ext,
                index_width=args.index_width,
            )
            match_label = f"frame_index={frame_index}"
        else:
            missing += 1
            if args.strict:
                raise RuntimeError(
                    "Frame has neither file_path nor frame_index for image matching"
                )
            continue

        image_id, reason = resolve_image_id(candidates, name_to_id, basename_to_ids)
        if image_id is None:
            if reason == "ambiguous_basename":
                ambiguous += 1
            else:
                missing += 1
            if args.strict:
                raise RuntimeError(
                    f"Could not uniquely match frame '{match_label}' (reason={reason})"
                )
            continue

        updates[image_id] = pose
        matched += 1

    if not updates:
        raise RuntimeError("No pose priors to write. Check file name mapping and inputs.")

    update_pose_priors(database_path, updates, dry_run=args.dry_run)

    mode = "DRY RUN" if args.dry_run else "UPDATED"
    print(f"[{mode}] poses prepared: {len(updates)}")
    print(f"[INFO] matched frames: {matched}")
    print(f"[INFO] missing matches: {missing}")
    print(f"[INFO] ambiguous basename matches: {ambiguous}")
    print(f"[INFO] frames without pose matrix (transform_matrix/rot_mat): {no_pose}")
    print(f"[INFO] database: {database_path}")
    print(f"[INFO] transforms: {transforms_path}")


if __name__ == "__main__":
    main()
