"""
DINO Patch MLP Autoencoder — Scene-Specific Encoder
=====================================================
Trains a tiny MLP autoencoder on DINO patch features from one scene.
Overfitting is intentional: the encoder memorizes this scene's feature
distribution, giving a compact bottleneck code for every patch.

Architecture (smallest viable MLP):
  Encoder: D → hidden → bottleneck
  Decoder: bottleneck → hidden → D

I/O: reuses the same memmap cache as generate_codebook.py.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the memmap infrastructure from generate_codebook.py
from generate_codebook import (
    create_dataloader,
    prepare_and_load,
)
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
    p.add_argument("--feat_dir",    type=str, required=True, help="Folder containing .pt DINO feature files")
    p.add_argument("--save_path",   type=str, default="patch_encoder.pt")
    p.add_argument("--bottleneck",  type=int, default=32,    help="Bottleneck (code) dimension")
    p.add_argument("--hidden",      type=int, default=256,   help="Hidden layer width (encoder and decoder share this)")
    p.add_argument("--batch_size",  type=int, default=16384)
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--lr",          type=float, default=5e-3, help="Higher LR accelerates memorization")
    p.add_argument("--no_cuda",     action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--encoded_dir",  type=str, default=None,
                   help="If set, encode all .pt files and save compressed features here")
    p.add_argument("--encode_only", action="store_true",
                   help="Skip training; load encoder from --save_path and run encode_directory")
    return p.parse_args()


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class PatchAutoencoder(nn.Module):
    """
    Minimal two-layer MLP autoencoder.
    Encoder:  D  → hidden → bottleneck
    Decoder:  bottleneck → hidden → D

    Total params ≈ 2 * (D*hidden + hidden*bottleneck) + biases.
    For D=384, hidden=256, bottleneck=32: ~230 K params.
    """

    def __init__(self, D: int, hidden: int, bottleneck: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, D),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train(model: PatchAutoencoder, dataloader, args, device):
    # No weight decay — we want to memorize, not generalize
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * 1e-4
    )

    console = Console()
    console.print("\n[cyan]── Training MLP Patch Autoencoder ──────────────────────────[/cyan]")
    console.print(f"  Epochs={args.epochs}  Batch={args.batch_size}  LR={args.lr}")
    console.print(f"  Bottleneck={args.bottleneck}  Hidden={args.hidden}")
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Params: [yellow]{n_params:,}[/yellow]  Device: {device}\n")

    best_loss = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("loss=[cyan]{task.fields[loss]:.4f}[/cyan]  cos=[green]{task.fields[cos]:.4f}[/green]  best=[yellow]{task.fields[best]:.4f}[/yellow]  lr=[magenta]{task.fields[lr]:.2e}[/magenta]"),
        TimeRemainingColumn(),
        console=console,
    ) as epoch_progress:
        epoch_task = epoch_progress.add_task(
            "[cyan]Epochs[/cyan]",
            total=args.epochs,
            loss=0.0, cos=0.0, best=best_loss, lr=args.lr,
        )

        for _ in range(args.epochs):
            model.train()
            sum_loss = 0.0
            sum_cos  = 0.0
            n_batches = 0

            with Progress(
                TextColumn("  {task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("loss=[cyan]{task.fields[loss]:.4f}[/cyan]  cos=[green]{task.fields[cos]:.4f}[/green]  lr=[magenta]{task.fields[lr]:.2e}[/magenta]"),
                console=console,
                transient=True,
            ) as batch_progress:
                batch_task = batch_progress.add_task(
                    "[blue]Batches[/blue]",
                    total=len(dataloader),
                    loss=float("inf"), cos=0.0,
                    lr=optimizer.param_groups[0]["lr"],
                )

                for batch in dataloader:
                    batch = batch.to(device)  # already L2-normalized by dataset
                    recon, _ = model(batch)

                    # Cosine reconstruction loss (scale-invariant, matches DINO's feature space)
                    loss = (1.0 - F.cosine_similarity(recon, batch, dim=-1)).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        cos = F.cosine_similarity(recon, batch, dim=-1).mean().item()

                    sum_loss  += loss.item()
                    sum_cos   += cos
                    n_batches += 1

                    batch_progress.update(
                        batch_task, advance=1,
                        loss=loss.item(), cos=cos,
                        lr=optimizer.param_groups[0]["lr"],
                    )

            scheduler.step()
            avg_loss = sum_loss / max(1, n_batches)
            avg_cos  = sum_cos  / max(1, n_batches)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            epoch_progress.update(epoch_task, advance=1, loss=avg_loss, cos=avg_cos, best=best_loss, lr=optimizer.param_groups[0]["lr"])

    model.load_state_dict(best_state)
    console.print(f"\n  [green]✓[/green] Restored best checkpoint (loss={best_loss:.4f})")


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: PatchAutoencoder, dataloader, device):
    model.eval()
    console = Console()
    console.print("\n[cyan]── Evaluation ───────────────────────────────────────────────[/cyan]")

    cos_list = []
    mse_list = []
    total = 0

    for batch in dataloader:
        batch = batch.to(device)
        recon, _ = model(batch)
        cos_list.append(F.cosine_similarity(recon, batch, dim=-1).cpu())
        mse_list.append(F.mse_loss(recon, batch, reduction="none").mean(-1).cpu())
        total += batch.shape[0]

    cos = torch.cat(cos_list).mean().item()
    mse = torch.cat(mse_list).mean().item()

    console.print(f"  Total patches : [yellow]{total:,}[/yellow]")
    console.print(f"  Cosine sim    : [green]{cos:.4f}[/green]")
    console.print(f"  MSE           : [cyan]{mse:.6f}[/cyan]")


# ─────────────────────────────────────────────
# ENCODE AND SAVE
# ─────────────────────────────────────────────

import glob as _glob

@torch.no_grad()
def encode_directory(model: PatchAutoencoder, feat_dir: str, encoded_dir: str,
                     batch_size: int, device: str):
    """
    Loads each .pt file from feat_dir, encodes the features with the trained
    encoder, and saves compressed .pt files to encoded_dir in the same format:
      {'features': Tensor (N, bottleneck), 'patch_size': int, 'inference_image_shape': tuple}
    Features are L2-normalized before encoding, matching the training pipeline.
    Processing runs on `device` (GPU if available).
    """
    model.eval()
    os.makedirs(encoded_dir, exist_ok=True)
    pt_files = sorted(_glob.glob(os.path.join(feat_dir, "*.pt")))

    console = Console()
    console.print("\n[cyan]── Encoding & Saving Compressed Features ────────────────────[/cyan]")
    console.print(f"  Source : [cyan]{feat_dir}[/cyan]")
    console.print(f"  Output : [cyan]{encoded_dir}[/cyan]")
    console.print(f"  Files  : [yellow]{len(pt_files)}[/yellow]  →  bottleneck=[yellow]{model.encoder[-1].out_features}[/yellow]  device=[magenta]{device}[/magenta]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("cos=[green]{task.fields[cos]:.4f}[/green]"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        file_task = progress.add_task("[cyan]Files[/cyan]", total=len(pt_files), cos=0.0)

        for path in pt_files:
            data = torch.load(path, map_location="cpu", weights_only=True)
            raw_feats = data["features"].float()          # (N, D) — unnormalized

            # Encode in batches on GPU
            codes = []
            recon_chunks = []
            for start in range(0, len(raw_feats), batch_size):
                chunk = raw_feats[start : start + batch_size].to(device)
                chunk_norm = F.normalize(chunk, dim=-1)   # same normalization as training
                z = model.encode(chunk_norm)
                codes.append(z.cpu())
                recon_chunks.append(model.decode(z).cpu())

            encoded = torch.cat(codes, dim=0)             # (N, bottleneck)
            recon   = torch.cat(recon_chunks, dim=0)
            cos = F.cosine_similarity(recon, F.normalize(raw_feats, dim=-1), dim=-1).mean().item()

            out = {
                "features":              encoded,
                "patch_size":            data["patch_size"],
                "inference_image_shape": data["inference_image_shape"],
            }
            torch.save(out, os.path.join(encoded_dir, os.path.basename(path)))
            progress.update(file_task, advance=1, cos=cos)

    console.print(f"\n  [green]✓[/green] Compressed features saved to [cyan]{encoded_dir}[/cyan]")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")

    console = Console()
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("DINO Patch MLP Autoencoder — Scene-Specific Encoder")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")

    if args.encode_only:
        if not os.path.exists(args.save_path):
            console.print(f"[red]✗ Encoder checkpoint not found: {args.save_path}[/red]")
            return
        if args.encoded_dir is None:
            console.print("[red]✗ --encoded_dir is required with --encode_only[/red]")
            return

        ckpt = torch.load(args.save_path, map_location="cpu", weights_only=True)
        model = PatchAutoencoder(
            D=ckpt["D_feat"], hidden=ckpt["hidden"], bottleneck=ckpt["bottleneck"]
        ).to(device)
        model.encoder.load_state_dict(ckpt["encoder_state"])
        model.decoder.load_state_dict(ckpt["decoder_state"])
        console.print(f"  [green]✓[/green] Loaded encoder from [cyan]{args.save_path}[/cyan]")
        console.print(f"  D={ckpt['D_feat']}  hidden={ckpt['hidden']}  bottleneck={ckpt['bottleneck']}")

        encode_directory(model, args.feat_dir, args.encoded_dir, args.batch_size, device)
        return

    npy_path, total_patches, D_feat = prepare_and_load(
        args.feat_dir, console, rebuild=args.rebuild_cache
    )
    console.print(f"  Feature Dim: [yellow]{D_feat}[/yellow]   Total patches: [yellow]{total_patches:,}[/yellow]")

    model = PatchAutoencoder(D=D_feat, hidden=args.hidden, bottleneck=args.bottleneck).to(device)

    loader = create_dataloader(npy_path, total_patches, D_feat, args.batch_size, args.num_workers)
    train(model, loader, args, device)

    eval_loader = create_dataloader(npy_path, total_patches, D_feat, args.batch_size, args.num_workers)
    evaluate(model, eval_loader, device)

    payload = {
        "encoder_state": model.encoder.state_dict(),
        "decoder_state": model.decoder.state_dict(),
        "D_feat":        D_feat,
        "hidden":        args.hidden,
        "bottleneck":    args.bottleneck,
    }
    torch.save(payload, args.save_path)
    size_mb = os.path.getsize(args.save_path) / 1e6
    console.print(f"\n  [green]✓[/green] Saved to [cyan]{args.save_path}[/cyan] ([yellow]{size_mb:.2f} MB[/yellow])")

    if args.encoded_dir is not None:
        encode_directory(model, args.feat_dir, args.encoded_dir, args.batch_size, device)


if __name__ == "__main__":
    main()
