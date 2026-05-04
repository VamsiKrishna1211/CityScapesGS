"""
DINOv2 Codebook for Scaffold-GS Anchor Features
=================================================
Tests:
  1. DinoCodebook forward pass (soft + hard VQ)
  2. Fake anchor features simulating Scaffold-GS
  3. Mini training loop with photometric + DINO + commitment loss
  4. Saves trained codebook to disk
  5. Reloads and runs inference with hard VQ

Requirements:
  pip install torch torchvision
  (DINOv2 loaded via torch.hub - needs internet on first run)
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
D_ANCHOR      = 32        # Scaffold-GS anchor feature dim
D_DINO        = 768       # DINOv2 ViT-B output dim
K             = 256       # Codebook size
N_ANCHORS     = 2048      # Fake anchors for testing
BATCH_SIZE    = 256
N_EPOCHS      = 5000
LR            = 1e-3
LAMBDA_DINO   = 1.0
LAMBDA_COMMIT = 0.25
TEMP_START    = 1.0
TEMP_END      = 0.1
SAVE_PATH     = "dino_codebook.pt"
IMAGE_SIZE    = 224       # For the DINOv2 demo pass

print(f"Device: {DEVICE}")
print(f"Config: K={K}, D_anchor={D_ANCHOR}, D_dino={D_DINO}, N_anchors={N_ANCHORS}")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. CODEBOOK MODULE
# ─────────────────────────────────────────────

class DinoCodebook(nn.Module):
    """
    Dual-purpose module that:
      - Takes anchor feat F_a (N, D_ANCHOR)
      - Returns reconstructed DINOv2 feature (N, D_DINO) via soft/hard codebook lookup
      - Codebook is K × D_DINO  → only 0.75MB for K=256, D=768
    """

    def __init__(self, K: int = 256, D_dino: int = 768, D_anchor: int = 32):
        super().__init__()
        self.K = K
        self.D_dino = D_dino

        # Shared codebook — the memory-efficient core
        self.codebook = nn.Embedding(K, D_dino)
        nn.init.xavier_uniform_(self.codebook.weight)

        # Lightweight query MLP: anchor feat → soft weights over K codewords
        self.query_mlp = nn.Sequential(
            nn.Linear(D_anchor, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, K),
        )

    # ── Soft forward (training, fully differentiable) ──────────────────
    def forward_soft(self, anchor_feat: torch.Tensor, temperature: float = 1.0):
        """
        anchor_feat : (N, D_anchor)
        returns     : dino_hat (N, D_dino), weights (N, K)
        """
        logits  = self.query_mlp(anchor_feat)               # (N, K)
        weights = F.softmax(logits / temperature, dim=-1)   # (N, K)
        dino_hat = weights @ self.codebook.weight            # (N, D_dino)
        return dino_hat, weights

    # ── Hard VQ forward (inference, memory-optimal) ────────────────────
    def forward_hard(self, anchor_feat: torch.Tensor):
        """
        Uses straight-through estimator so gradients still flow.
        anchor_feat : (N, D_anchor)
        returns     : dino_hat (N, D_dino), indices (N,)
        """
        logits  = self.query_mlp(anchor_feat)               # (N, K)
        indices = logits.argmax(dim=-1)                     # (N,)  ← int index
        # One-hot with straight-through gradient
        hard    = F.one_hot(indices, self.K).float()
        weights = hard - logits.detach() + logits           # straight-through
        dino_hat = weights @ self.codebook.weight
        return dino_hat, indices

    # ── Convenience: lookup by index only (pure inference) ─────────────
    def lookup(self, indices: torch.Tensor):
        """indices: (N,) int64 → (N, D_dino)"""
        return self.codebook(indices)

    def forward(self, anchor_feat, temperature=1.0, hard=False):
        if hard:
            return self.forward_hard(anchor_feat)
        return self.forward_soft(anchor_feat, temperature)

    def memory_stats(self):
        cb_mb   = self.K * self.D_dino * 4 / 1e6
        mlp_params = sum(p.numel() for p in self.query_mlp.parameters())
        print(f"  Codebook  : {self.K} × {self.D_dino} = {cb_mb:.2f} MB")
        print(f"  Query MLP : {mlp_params:,} params = {mlp_params*4/1e6:.3f} MB")
        print(f"  Total     : {cb_mb + mlp_params*4/1e6:.2f} MB")
        print(f"  vs naive  : 100k anchors × {self.D_dino} × 4B = "
              f"{100_000 * self.D_dino * 4 / 1e6:.1f} MB")


# ─────────────────────────────────────────────
# 2. FAKE SCAFFOLD-GS ANCHOR DATA
# ─────────────────────────────────────────────

def make_fake_scaffold_data(n_anchors, d_anchor, d_dino, device):
    """
    Simulates:
      - anchor_feats : random features as Scaffold-GS would produce
      - dino_gt      : ground-truth DINOv2 patch features (l2-normalised,
                       as ViT tokens typically are)
    In a real setup, dino_gt comes from projecting anchors → image plane
    and averaging patch tokens across training views.
    """
    torch.manual_seed(42)
    anchor_feats = torch.randn(n_anchors, d_anchor, device=device)

    # Simulate clustered DINO features (a few semantic clusters)
    n_clusters = 16
    centers = F.normalize(torch.randn(n_clusters, d_dino, device=device), dim=-1)
    assign  = torch.randint(0, n_clusters, (n_anchors,), device=device)
    noise   = 0.1 * torch.randn(n_anchors, d_dino, device=device)
    dino_gt = F.normalize(centers[assign] + noise, dim=-1)

    return anchor_feats, dino_gt


# ─────────────────────────────────────────────
# 3. LOSSES
# ─────────────────────────────────────────────

def dino_loss(dino_hat, dino_gt):
    """Cosine similarity loss — better than L2 for ViT features."""
    return (1.0 - F.cosine_similarity(dino_hat, dino_gt, dim=-1)).mean()

def commitment_loss(dino_hat, dino_hat_sg):
    """Keeps codebook entries from drifting too far."""
    return F.mse_loss(dino_hat, dino_hat_sg.detach())

def fake_photometric_loss(anchor_feats):
    """Placeholder for Scaffold-GS photometric loss."""
    return (anchor_feats ** 2).mean() * 0.0   # zero — not the focus here


# ─────────────────────────────────────────────
# 4. TEMPERATURE SCHEDULE
# ─────────────────────────────────────────────

def get_temperature(epoch, n_epochs, t_start=1.0, t_end=0.1):
    """Cosine anneal temperature from soft → hard over training."""
    progress = epoch / max(n_epochs - 1, 1)
    return t_end + 0.5 * (t_start - t_end) * (1 + np.cos(np.pi * progress))


# ─────────────────────────────────────────────
# 5. DINO FEATURE EXTRACTION (optional real image demo)
# ─────────────────────────────────────────────

def load_dinov2(model_name="dinov2_vitb14"):
    print(f"\nLoading {model_name} via torch.hub ...")
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        model = model.to(DEVICE).eval()
        print("  ✓ DINOv2 loaded")
        return model
    except Exception as e:
        print(f"  ✗ Could not load DINOv2 (no internet?): {e}")
        return None


def extract_patch_tokens(dino_model, image_tensor):
    """
    image_tensor : (1, 3, H, W) — H,W must be multiples of 14
    returns      : patch_tokens (1, N_patches, 768)
                   cls_token    (1, 768)
    """
    with torch.no_grad():
        out = dino_model.forward_features(image_tensor)
    cls_token    = out["x_norm_clstoken"]          # (1, 768)
    patch_tokens = out["x_norm_patchtokens"]        # (1, N, 768)
    return patch_tokens, cls_token


def demo_real_image(dino_model):
    """Create a dummy image and run through DINOv2 to show output shapes."""
    print("\n── DINOv2 shape demo ────────────────────────────────────────")
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    patches, cls = extract_patch_tokens(dino_model, dummy)
    n_patches = (IMAGE_SIZE // 14) ** 2
    print(f"  Input image  : {tuple(dummy.shape)}")
    print(f"  CLS token    : {tuple(cls.shape)}")
    print(f"  Patch tokens : {tuple(patches.shape)}  "
          f"← ({IMAGE_SIZE}÷14)²={n_patches} patches")
    return patches, cls


# ─────────────────────────────────────────────
# 6. MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train(model: DinoCodebook, anchor_feats: torch.Tensor, dino_gt: torch.Tensor):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

    n = anchor_feats.shape[0]
    print("\n── Training ─────────────────────────────────────────────────")
    print(f"  Anchors={n}, Epochs={N_EPOCHS}, Batch={BATCH_SIZE}, LR={LR}")
    print(f"  λ_dino={LAMBDA_DINO}, λ_commit={LAMBDA_COMMIT}")
    print(f"  Temperature: {TEMP_START} → {TEMP_END} (cosine anneal)")

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        model.train()
        temp = get_temperature(epoch, N_EPOCHS, TEMP_START, TEMP_END)

        # Random mini-batches
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, BATCH_SIZE):
            idx    = perm[start:start + BATCH_SIZE]
            fa     = anchor_feats[idx]
            dgt    = dino_gt[idx]

            # ── Soft forward ──
            dino_hat, weights = model.forward_soft(fa, temperature=temp)

            # ── Losses ──
            l_photo  = fake_photometric_loss(fa)
            l_dino   = dino_loss(dino_hat, dgt)
            # Stop-gradient version for commitment
            dino_hat_sg = model.forward_soft(fa.detach(), temperature=temp)[0]
            l_commit = commitment_loss(dino_hat, dino_hat_sg)

            loss = l_photo + LAMBDA_DINO * l_dino + LAMBDA_COMMIT * l_commit

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d}/{N_EPOCHS} | loss={avg_loss:.4f} "
                  f"| temp={temp:.3f} | best={best_loss:.4f} | {elapsed:.1f}s")

    print(f"\n  Training complete. Best loss: {best_loss:.4f}")


# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, anchor_feats, dino_gt):
    model.eval()
    print("\n── Evaluation ───────────────────────────────────────────────")

    # Soft
    dino_soft, weights_soft = model.forward_soft(anchor_feats, temperature=TEMP_END)
    cos_soft = F.cosine_similarity(dino_soft, dino_gt, dim=-1).mean().item()

    # Hard VQ
    dino_hard, indices = model.forward_hard(anchor_feats)
    cos_hard = F.cosine_similarity(dino_hard, dino_gt, dim=-1).mean().item()

    # Codebook usage (how many codewords are actually used)
    unique_codes = indices.unique().numel()

    print(f"  Cosine sim (soft) : {cos_soft:.4f}")
    print(f"  Cosine sim (hard) : {cos_hard:.4f}")
    print(f"  Codebook usage    : {unique_codes}/{K} codewords active "
          f"({100*unique_codes/K:.1f}%)")
    print(f"  Weight entropy    : "
          f"{(-weights_soft * (weights_soft + 1e-8).log()).sum(-1).mean().item():.3f} nats")

    return indices


# ─────────────────────────────────────────────
# 8. SAVE / LOAD
# ─────────────────────────────────────────────

def save_codebook(model, indices, path=SAVE_PATH):
    torch.save({
        "codebook_weight" : model.codebook.weight.cpu(),
        "query_mlp_state" : model.query_mlp.state_dict(),
        "hard_indices"    : indices.cpu(),      # ← 8-bit at inference
        "K"               : K,
        "D_dino"          : D_DINO,
        "D_anchor"        : D_ANCHOR,
    }, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n── Saved ────────────────────────────────────────────────────")
    print(f"  Path : {path}  ({size_mb:.2f} MB)")
    print(f"  Keys : codebook_weight, query_mlp_state, hard_indices")


def load_and_infer(path=SAVE_PATH, anchor_feats=None):
    print(f"\n── Load + Inference ─────────────────────────────────────────")
    ckpt  = torch.load(path, map_location=DEVICE)
    model = DinoCodebook(ckpt["K"], ckpt["D_dino"], ckpt["D_anchor"]).to(DEVICE)
    model.codebook.weight.data = ckpt["codebook_weight"].to(DEVICE)
    model.query_mlp.load_state_dict(ckpt["query_mlp_state"])
    model.eval()
    print(f"  ✓ Loaded codebook ({ckpt['K']} × {ckpt['D_dino']})")

    if anchor_feats is not None:
        with torch.no_grad():
            dino_hat, indices = model.forward_hard(anchor_feats[:8])
        print(f"  Sample indices (first 8): {indices.tolist()}")
        print(f"  Retrieved DINO feat shape: {dino_hat.shape}")

    # Hard-index memory cost
    n_anchors = ckpt["hard_indices"].shape[0]
    print(f"\n  Memory footprint at inference:")
    print(f"    Codebook          : {ckpt['K'] * ckpt['D_dino'] * 4 / 1e6:.2f} MB")
    print(f"    Indices ({n_anchors:,} pts) : "
          f"{n_anchors * 1 / 1e6:.3f} MB  (int8)")
    print(f"    vs naive storage  : "
          f"{n_anchors * ckpt['D_dino'] * 4 / 1e6:.1f} MB")

    return model


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DINOv2 Codebook — Scaffold-GS Anchor Feature Test")
    print("=" * 60)

    # ── Build model & show memory stats ──────────────────────────────
    model = DinoCodebook(K=K, D_dino=D_DINO, D_anchor=D_ANCHOR).to(DEVICE)
    print("\n── Codebook memory stats ────────────────────────────────────")
    model.memory_stats()

    # ── Fake scaffold data ───────────────────────────────────────────
    anchor_feats, dino_gt = make_fake_scaffold_data(N_ANCHORS, D_ANCHOR, D_DINO, DEVICE)
    print(f"\n── Fake Scaffold-GS data ────────────────────────────────────")
    print(f"  anchor_feats : {tuple(anchor_feats.shape)}")
    print(f"  dino_gt      : {tuple(dino_gt.shape)}  (l2-normalised, clustered)")

    # ── Quick shape sanity check ─────────────────────────────────────
    print("\n── Forward pass sanity check ────────────────────────────────")
    with torch.no_grad():
        dh_soft, w  = model.forward_soft(anchor_feats[:4], temperature=1.0)
        dh_hard, ix = model.forward_hard(anchor_feats[:4])
    print(f"  Soft  → dino_hat: {tuple(dh_soft.shape)}, weights: {tuple(w.shape)}")
    print(f"  Hard  → dino_hat: {tuple(dh_hard.shape)}, indices: {tuple(ix.shape)}")

    # ── Train ────────────────────────────────────────────────────────
    train(model, anchor_feats, dino_gt)

    # ── Evaluate ─────────────────────────────────────────────────────
    indices = evaluate(model, anchor_feats, dino_gt)

    # ── Save ─────────────────────────────────────────────────────────
    save_codebook(model, indices, SAVE_PATH)

    # ── Reload & infer ───────────────────────────────────────────────
    load_and_infer(SAVE_PATH, anchor_feats)

    # ── Optional: real DINOv2 demo ───────────────────────────────────
    print("\n── DINOv2 Hub load (optional) ───────────────────────────────")
    dino_model = load_dinov2("dinov2_vitb14")
    if dino_model is not None:
        patches, cls = demo_real_image(dino_model)
        # Show how you'd extract dino_gt for one anchor projected to patch (8,8)
        patch_u, patch_v = 8, 8
        sample_feat = patches[0, patch_v * (IMAGE_SIZE // 14) + patch_u]
        print(f"\n  Patch token at ({patch_u},{patch_v}) shape : {tuple(sample_feat.shape)}")
        sample_feat = F.normalize(sample_feat.unsqueeze(0), dim=-1)
        fake_anchor = torch.randn(1, D_ANCHOR).to(DEVICE)
        dhat, idx   = model.forward_hard(fake_anchor)
        cos         = F.cosine_similarity(dhat, sample_feat, dim=-1).item()
        print(f"  Untrained retrieval cosine sim   : {cos:.4f}  (random anchor → random code)")

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
