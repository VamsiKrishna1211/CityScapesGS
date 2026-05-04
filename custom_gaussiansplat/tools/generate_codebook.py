"""
LangSplatV2 Sparse Coefficient Codebook — Top-K implementation
===========================================================
Expects a folder of .pt files, each with structure:
  {
    'features'             : Tensor (N_patches, D),
    'patch_size'           : int,
    'inference_image_shape': tuple (H, W)
  }

I/O Strategy: on first run, all .pt feature tensors are streamed into a single
flat numpy memmap (features.npy + meta.json). Subsequent runs reuse the cache.
Each __getitem__ reads exactly D*4 bytes; no full-file loads during training.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir",   type=str,   required=True, help="Folder containing .pt files")
    p.add_argument("--save_path",  type=str,   default="sparse_codebook.pt")
    p.add_argument("--L",          type=int,   default=64,  help="Global codebook size (L in paper)")
    p.add_argument("--K",          type=int,   default=4,   help="Top-K sparsity (K in paper)")
    p.add_argument("--batch_size", type=int,   default=16384, help="Number of patches per batch")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lambda_commit", type=float, default=0.25)
    p.add_argument("--no_cuda",    action="store_true")
    p.add_argument("--num_workers", type=int,  default=4,   help="DataLoader workers (4 is plenty with memmap)")
    p.add_argument("--rebuild_cache", action="store_true",  help="Force rebuild of memmap cache")
    return p.parse_args()


# ─────────────────────────────────────────────
# 1. MEMMAP CACHE + DATASET
# ─────────────────────────────────────────────

def _cache_paths(feat_dir: str):
    return (
        os.path.join(feat_dir, "_feat_cache.npy"),
        os.path.join(feat_dir, "_feat_cache_meta.json"),
    )


def _cache_is_valid(feat_dir: str) -> bool:
    """Returns True if the cache exists and covers the same set of .pt files."""
    npy_path, meta_path = _cache_paths(feat_dir)
    if not os.path.exists(npy_path) or not os.path.exists(meta_path):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    current_files = sorted(glob.glob(os.path.join(feat_dir, "*.pt")))
    # Compare filenames and modification times
    cached_files = meta.get("files", [])
    if len(current_files) != len(cached_files):
        return False
    for cur, cached in zip(current_files, cached_files):
        if os.path.basename(cur) != cached["name"]:
            return False
        if abs(os.path.getmtime(cur) - cached["mtime"]) > 1.0:
            return False
    return True


def prepare_memmap(feat_dir: str, console: Console) -> tuple[str, int, int]:
    """
    Streams all .pt files once and writes features into a flat float32 memmap.
    Returns (npy_path, total_patches, D).
    Peak RAM during this step: ~size of one .pt file.
    """
    npy_path, meta_path = _cache_paths(feat_dir)
    pt_files = sorted(glob.glob(os.path.join(feat_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in: {feat_dir}")

    console.print("\n[cyan]── Building memmap cache ────────────────────────────────────[/cyan]")

    # Pass 1: count total patches and get D (load each file once, extract shape only)
    total_patches = 0
    D = None
    file_offsets = []  # (start_row, n_patches) per file
    with Progress(TextColumn("  Scanning"), BarColumn(), MofNCompleteColumn(),
                  console=console, transient=True) as prog:
        task = prog.add_task("", total=len(pt_files))
        for path in pt_files:
            try:
                data = torch.load(path, map_location="cpu", weights_only=True)
                n, d = data["features"].shape
                if D is None:
                    D = d
                file_offsets.append((total_patches, n))
                total_patches += n
            except Exception as e:
                console.print(f"  [yellow]Skipping {os.path.basename(path)}: {e}[/yellow]")
                file_offsets.append(None)
            finally:
                prog.advance(task)

    console.print(f"  Files: [yellow]{len(pt_files)}[/yellow]  Total patches: [yellow]{total_patches:,}[/yellow]  D: [yellow]{D}[/yellow]")
    size_gb = total_patches * D * 4 / 1e9
    console.print(f"  Cache size: [cyan]{size_gb:.2f} GB[/cyan]")

    # Pass 2: allocate memmap and stream each file into it
    mm = np.memmap(npy_path, dtype=np.float32, mode="w+", shape=(total_patches, D))
    with Progress(TextColumn("  Writing "), BarColumn(), MofNCompleteColumn(),
                  console=console, transient=True) as prog:
        task = prog.add_task("", total=len(pt_files))
        for path, offset_info in zip(pt_files, file_offsets):
            if offset_info is None:
                prog.advance(task)
                continue
            start, n = offset_info
            data = torch.load(path, map_location="cpu", weights_only=True)
            feats = data["features"].float().numpy()  # (n, D)
            mm[start : start + n] = feats
            del data, feats  # free immediately before loading the next file
            prog.advance(task)

    mm.flush()
    del mm  # close the memmap writer

    # Save metadata for cache invalidation
    file_meta = [
        {"name": os.path.basename(p), "mtime": os.path.getmtime(p)}
        for p in pt_files
    ]
    with open(meta_path, "w") as f:
        json.dump({"total_patches": total_patches, "D": D, "files": file_meta}, f)

    console.print(f"  [green]✓[/green] Cache written to [cyan]{npy_path}[/cyan]\n")
    return npy_path, total_patches, D


def load_memmap_meta(feat_dir: str) -> tuple[str, int, int]:
    """Reads cached metadata. Assumes cache is valid."""
    npy_path, meta_path = _cache_paths(feat_dir)
    with open(meta_path) as f:
        meta = json.load(f)
    return npy_path, meta["total_patches"], meta["D"]


class MemmapPatchDataset(torch.utils.data.Dataset):
    """
    O(1) random patch access via numpy memmap.
    Each __getitem__ reads exactly D*4 bytes from disk (one row).
    The OS page cache handles repeated access — no data is duplicated across workers.
    """
    def __init__(self, npy_path: str, total_patches: int, D: int):
        # mode='r' → read-only, fork-safe; workers share pages via copy-on-write
        self.data = np.memmap(npy_path, dtype=np.float32, mode="r", shape=(total_patches, D))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # .copy() detaches from the mmap page so torch can own the buffer safely
        feat = torch.from_numpy(self.data[idx].copy())
        return F.normalize(feat, dim=-1)


def prepare_and_load(feat_dir: str, console: Console, rebuild: bool = False):
    """Builds (or reuses) the memmap cache and returns (npy_path, total_patches, D)."""
    if rebuild or not _cache_is_valid(feat_dir):
        return prepare_memmap(feat_dir, console)
    console.print("  [green]✓[/green] Reusing existing memmap cache")
    return load_memmap_meta(feat_dir)


def create_dataloader(npy_path: str, total_patches: int, D: int,
                      batch_size: int, num_workers: int):
    dataset = MemmapPatchDataset(npy_path, total_patches, D)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# ─────────────────────────────────────────────
# 2. SPARSE PATCH CODEBOOK (LangSplatV2 Logic)
# ─────────────────────────────────────────────

class SparseCoefficientCodebook(nn.Module):
    """
    Implements the Sparse Coefficient Field logic from LangSplatV2.
    Learns a global codebook of size L.
    Uses Top-K selection and re-normalization.
    """
    def __init__(self, L: int, K: int, D_feat: int):
        super().__init__()
        self.L = L
        self.K = K
        self.D_feat = D_feat

        # Global Codebook S: (L, D_feat)
        self.codebook = nn.Embedding(L, D_feat)
        nn.init.xavier_uniform_(self.codebook.weight)

        # Encoder: D_feat -> L soft coefficients
        self.encoder = nn.Sequential(
            nn.Linear(D_feat, 512),
            nn.ReLU(),
            nn.Linear(512, L)
        )

    def forward_soft(self, patch_feat: torch.Tensor):
        """Standard dense attention over all L codes (used for commitment)."""
        logits = self.encoder(patch_feat)
        weights = F.softmax(logits, dim=-1)
        recon = weights @ F.normalize(self.codebook.weight, dim=-1)
        return recon, weights

    def forward_sparse(self, patch_feat: torch.Tensor):
        """
        The core LangSplatV2 logic:
        1. Softmax to get L weights
        2. Keep Top-K, zero out the rest
        3. Re-normalize Top-K to sum to 1
        4. Straight-through estimator for gradients
        """
        logits = self.encoder(patch_feat)
        dense_weights = F.softmax(logits, dim=-1) # (N, L)

        # 1. Top-K Selection
        topk_weights, topk_indices = torch.topk(dense_weights, self.K, dim=-1) # (N, K)

        # 2. Re-normalization (sum to 1)
        topk_weights_norm = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 3. Reconstruct sparse weight matrix for multiplication
        sparse_weights = torch.zeros_like(dense_weights).scatter_(
            -1, topk_indices, topk_weights_norm
        ) # (N, L)

        # 4. Straight-Through Estimator (STE)
        # Forward pass is sparse/normalized, backward pass treats it as dense
        ste_weights = sparse_weights.detach() - dense_weights.detach() + dense_weights

        # 5. Reconstruction
        recon = ste_weights @ self.codebook.weight

        return recon, ste_weights, topk_indices, topk_weights_norm


# ─────────────────────────────────────────────
# 3. LOSSES
# ─────────────────────────────────────────────

def cosine_recon_loss(recon, target):
    return (1.0 - F.cosine_similarity(recon, target, dim=-1)).mean()

def commitment_loss(recon, recon_sg):
    return F.mse_loss(recon, recon_sg.detach())


# ─────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────

def train(model: SparseCoefficientCodebook, dataloader, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.01)

    console = Console()
    console.print("\n[cyan]── Training Sparse Codebook ─────────────────────────────────[/cyan]")
    console.print(f"  Epochs={args.epochs}  Batch={args.batch_size}  LR={args.lr}")
    console.print(f"  L (Codes)={args.L}  K (Sparsity)={args.K}")
    console.print(f"  Device: {device}\n")

    best_loss = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[loss]:.4f}[/cyan]  cos_sim=[green]{task.fields[cos]:.4f}[/green]  best=[yellow]{task.fields[best]:.4f}[/yellow]"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        epoch_task = progress.add_task(
            "[cyan]Training[/cyan]",
            total=args.epochs,
            loss=0.0,
            cos=0.0,
            best=best_loss
        )

        for _ in range(args.epochs):
            model.train()
            sum_loss = 0.0
            sum_cos = 0.0
            n_batches = 0

            with Progress(
                TextColumn("  {task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[cyan]{task.fields[loss]:.4f}[/cyan]  cos_sim=[green]{task.fields[cos]:.4f}[/green]"),
                console=console,
                transient=True
            ) as batch_progress:
                batch_task = batch_progress.add_task("[blue]Batches[/blue]", total=len(dataloader), loss=float('inf'), cos=0.0)

                for batch_feats in dataloader:
                    batch_feats = batch_feats.to(device)

                    # Forward pass (Sparse)
                    recon_sparse, _, _, _ = model.forward_sparse(batch_feats)
                    l_recon = cosine_recon_loss(recon_sparse, batch_feats)

                    # Commitment loss (Soft) - keeps codes from diverging
                    with torch.no_grad():
                        recon_soft, _ = model.forward_soft(batch_feats)
                    l_commit = commitment_loss(recon_sparse, recon_soft)

                    loss = l_recon + (args.lambda_commit * l_commit)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    sum_loss += loss.item()
                    with torch.no_grad():
                        cos = F.cosine_similarity(recon_sparse, batch_feats, dim=-1).mean().item()
                    sum_cos += cos
                    n_batches += 1

                    batch_progress.update(batch_task, advance=1, loss=loss.item(), cos=cos)

            scheduler.step()
            avg_loss = sum_loss / max(1, n_batches)
            avg_cos = sum_cos / max(1, n_batches)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            progress.update(
                epoch_task,
                advance=1,
                loss=avg_loss,
                cos=avg_cos,
                best=best_loss
            )

    model.load_state_dict(best_state)
    console.print(f"\n  [green]✓[/green] Restored best checkpoint  (loss={best_loss:.4f})")


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    console = Console()
    console.print("\n[cyan]── Evaluation ───────────────────────────────────────────────[/cyan]")

    cos_sparse_list = []
    total_patches = 0
    all_indices = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        eval_task = progress.add_task("[blue]Evaluating[/blue]", total=len(dataloader))

        for batch_feats in dataloader:
            batch_feats = batch_feats.to(device)
            total_patches += batch_feats.shape[0]

            recon_sparse, _, indices, _ = model.forward_sparse(batch_feats)
            cos_sparse_list.append(F.cosine_similarity(recon_sparse, batch_feats, dim=-1).cpu())
            all_indices.append(indices.cpu())

            progress.update(eval_task, advance=1)

    cos_sparse = torch.cat(cos_sparse_list).mean().item()
    flat_indices = torch.cat(all_indices).flatten()

    unique = flat_indices.unique().numel()
    counts = torch.bincount(flat_indices, minlength=model.L).float()

    console.print(f"  Total patches evaluated: [yellow]{total_patches:,}[/yellow]")
    console.print(f"  Cosine sim (Sparse K={model.K}) : [green]{cos_sparse:.4f}[/green]")
    console.print(f"  Codebook usage             : [cyan]{unique}/{model.L}[/cyan]  ({100*unique/model.L:.1f}%)")

    top5 = counts.argsort(descending=True)[:5]
    console.print("  Top-5 codewords by usage :")
    for rank, ci in enumerate(top5.tolist()):
        console.print(f"    #{rank+1}  code [yellow]{ci:4d}[/yellow]  →  [cyan]{int(counts[ci]):,}[/cyan] assignments")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")

    console = Console()
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("LangSplatV2 Top-K Sparse Codebook Generator")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")

    # Build or reuse the flat memmap cache (runs once, ~size of raw .pt data on disk)
    npy_path, total_patches, D_feat = prepare_and_load(
        args.feat_dir, console, rebuild=args.rebuild_cache
    )
    console.print(f"  Feature Dim : [yellow]{D_feat}[/yellow]   Total patches: [yellow]{total_patches:,}[/yellow]")

    model = SparseCoefficientCodebook(L=args.L, K=args.K, D_feat=D_feat).to(device)

    train_loader = create_dataloader(npy_path, total_patches, D_feat, args.batch_size, args.num_workers)
    train(model, train_loader, args, device)

    eval_loader = create_dataloader(npy_path, total_patches, D_feat, args.batch_size, args.num_workers)
    evaluate(model, eval_loader, device)

    # Save
    payload = {
        "codebook_weight" : model.codebook.weight.cpu(),
        "encoder_state"   : model.encoder.state_dict(),
        "L"               : model.L,
        "K"               : model.K,
        "D_feat"          : D_feat,
    }
    torch.save(payload, args.save_path)
    size_mb = os.path.getsize(args.save_path) / 1e6
    console.print(f"\n  [green]✓[/green] Saved to [cyan]{args.save_path}[/cyan] ([yellow]{size_mb:.2f} MB[/yellow])")


if __name__ == "__main__":
    main()
