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

torch.set_float32_matmul_precision('high')

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
    p.add_argument("--contrastive_weight", type=float, default=0.0,
                   help="Weight for InfoNCE contrastive loss (0.0 to disable)")
    p.add_argument("--similarity_threshold", type=float, default=0.4,
                   help="Threshold for defining positive pairs from DINO feature similarity")
    p.add_argument("--temperature", type=float, default=0.07,
        help="Temperature parameter for InfoNCE loss")
    p.add_argument("--enable-mixed-precision", action="store_true",
        help="Use bfloat16 autocast for forward passes (requires Ampere+ GPU)")
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
            nn.Linear(bottleneck, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, D),
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
# CONTRASTIVE LOSS (InfoNCE)
# ─────────────────────────────────────────────

@torch.no_grad()
def get_positive_mask(features: torch.Tensor, threshold: float = 0.85) -> torch.Tensor:
    """
    Create a mask of positive pairs based on DINO feature similarity.
    Patches with cosine similarity > threshold are considered positive pairs.

    Args:
        features: L2-normalized DINO features (B, D)
        threshold: similarity threshold for positive pairs

    Returns:
        positive_mask: boolean tensor (B, B) where positive_mask[i,j] = True if i,j are positive
    """
    # Pairwise cosine similarity (features already L2-normalized)
    sim_matrix = torch.mm(features, features.t())  # (B, B)

    # Positive pairs: high similarity, excluding self-similarity
    positive_mask = sim_matrix > threshold
    positive_mask.fill_diagonal_(False)  # Don't pair with self

    return positive_mask


def infonce_loss(bottleneck_codes: torch.Tensor, positive_mask: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE / Contrastive loss on bottleneck codes.
    For each sample, pull positive pairs close and push negative pairs apart
    by turning similarity computation into a softmax classification problem.

    Args:
        bottleneck_codes: encoded codes (B, bottleneck_dim), not necessarily normalized
        positive_mask: boolean tensor (B, B) indicating positive pairs
        temperature: temperature for scaling similarities (smaller = sharper softmax)

    Returns:
        loss: scalar tensor, or zero tensor if no positive pairs exist
    """
    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=bottleneck_codes.device, dtype=bottleneck_codes.dtype)

    # Normalize codes for fair similarity computation
    normalized_codes = F.normalize(bottleneck_codes, dim=-1)
    codes_similarity = torch.mm(normalized_codes, normalized_codes.t())  # (B, B) pairwise similarities

    # Apply temperature scaling
    logits = codes_similarity / temperature

    # Log-softmax: convert similarities to log probabilities
    # For each row i, this is: log(exp(sim[i,j]) / sum_k exp(sim[i,k]))
    log_probs = F.log_softmax(logits, dim=1)  # (B, B)

    # Extract log probs of positive pairs and average (negative log-likelihood)
    mask_float = positive_mask.float()
    contrastive_loss = -(log_probs * mask_float).sum() / (mask_float.sum() + 1e-8)

    return contrastive_loss


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
    console.print(f"  Params: [yellow]{n_params:,}[/yellow]  Device: {device}")
    if args.contrastive_weight > 0.0:
        console.print(f"  Contrastive: weight={args.contrastive_weight}  threshold={args.similarity_threshold}  temp={args.temperature}")
    console.print()

    best_loss = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[info]}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        epoch_task = progress.add_task(
            "[cyan]Epochs[/cyan]",
            total=args.epochs,
            info=f"loss=0.0000  cos=0.0000  best={best_loss:.4f}  lr={args.lr:.2e}",
        )
        batch_task = progress.add_task(
            "[blue]  Batches[/blue]",
            total=len(dataloader),
            info="loss=inf  cos=0.0000",
        )

        for _ in range(args.epochs):
            model.train()
            sum_loss = 0.0
            sum_cos  = 0.0
            n_batches = 0
            progress.reset(batch_task, total=len(dataloader))

            for batch in dataloader:
                batch = batch.to(device)  # already L2-normalized by dataset

                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=args.enable_mixed_precision):
                    recon, bottleneck_codes = model(batch)

                    cos_loss = (1.0 - F.cosine_similarity(recon, batch, dim=-1)).mean()
                    mse_loss = F.mse_loss(recon, batch)
                    loss = cos_loss + 0.3 * mse_loss

                    if args.contrastive_weight > 0.0:
                        positive_pairs_mask = get_positive_mask(batch, threshold=args.similarity_threshold)
                        contrastive_loss = infonce_loss(bottleneck_codes, positive_pairs_mask, temperature=args.temperature)
                        loss = loss + args.contrastive_weight * contrastive_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    cos = 1 - cos_loss.item()

                sum_loss  += loss.item()
                sum_cos   += cos
                n_batches += 1

                progress.update(
                    batch_task, advance=1,
                    info=f"loss=[cyan]{loss.item():.4f}[/cyan]  cos=[green]{cos:.4f}[/green]  lr=[magenta]{optimizer.param_groups[0]['lr']:.2e}[/magenta]",
                )

            scheduler.step()
            avg_loss = sum_loss / max(1, n_batches)
            avg_cos  = sum_cos  / max(1, n_batches)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            progress.update(
                epoch_task, advance=1,
                info=f"loss=[cyan]{avg_loss:.4f}[/cyan]  cos=[green]{avg_cos:.4f}[/green]  best=[yellow]{best_loss:.4f}[/yellow]  lr=[magenta]{optimizer.param_groups[0]['lr']:.2e}[/magenta]",
            )

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
    model = torch.compile(model)

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
