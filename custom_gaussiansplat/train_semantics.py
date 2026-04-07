"""Semantic and language feature fine-tuning for Gaussian Splatting models.

Two training paths are supported:

GaussianModel (standard 3DGS):
    Freezes all geometry, trains _features_semantics per-Gaussian.
    Uses SemanticRasterizer → concatenates DC + semantics → rasterize.

ScaffoldModel (Scaffold-GS, when enable_language_features=True):
    Freezes all geometry and geometric MLPs.
    Trains _anchor_lang_feat, mlp_language, language_codebook, codebook_proj.
    Uses ScaffoldSemanticRasterizer → generate_neural_gaussians → rasterize 32-dim language map.
    Supervises in PCA-compressed 32-dim space against raw 512-dim CLIP features.
    Codebook is initialized from PCA principal components before training.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from torch.utils.data import DataLoader

from dataset import ColmapDataset
from gs_types import RenderParams
from logger import GaussianSplattingLogger
from models import BaseTrainableModel, GaussianModel, ScaffoldModel, SemanticsMixin
from semantic_dataset import SemanticColmapDataset
from semantic_providers import NpySemanticProvider, SemanticTargetProvider, build_semantic_provider
from train_args import LearningRateConfig, SemanticsConfig, TrainingConfig
from train_semantics_args import SemanticStandaloneConfig, parse_semantic_args


# ---------------------------------------------------------------------------
# View / intrinsics helpers (shared by both rasterizers)
# ---------------------------------------------------------------------------


def _build_viewmat(cam: dict, device: torch.device) -> torch.Tensor:
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :3] = cam["R"]
    viewmat[:3, 3] = cam["T"]
    return viewmat.unsqueeze(0)


def _build_intrinsics(cam: dict, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


def _as_hwc(target: torch.Tensor, expected_channels: int) -> torch.Tensor:
    if target.dim() == 4 and target.shape[0] == 1:
        target = target.squeeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(-1)
    if target.dim() == 3 and target.shape[-1] != expected_channels and target.shape[0] == expected_channels:
        target = target.permute(1, 2, 0)
    if target.dim() != 3:
        raise RuntimeError(f"Unexpected semantic target shape {tuple(target.shape)}. Expected [H,W,C] or [C,H,W].")
    if target.shape[-1] != expected_channels:
        raise RuntimeError(
            f"Semantic channel mismatch: target C={target.shape[-1]} vs expected C={expected_channels}"
        )
    return target.float()


def _normalize_map(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def _interpolate_to_match(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Bilinearly resize pred [H', W', C] to match target [H, W, C] if needed."""
    if pred.shape[:2] == target.shape[:2]:
        return pred
    pred_nchw = pred.permute(2, 0, 1).unsqueeze(0)
    pred_nchw = F.interpolate(
        pred_nchw,
        size=(target.shape[0], target.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    return pred_nchw.squeeze(0).permute(1, 2, 0)


# ---------------------------------------------------------------------------
# PCA utility — used to initialize the ScaffoldModel language codebook
# ---------------------------------------------------------------------------


def compute_pca_from_clip_dir(
    clip_dir: Path,
    n_components: int,
    clip_dim: int = 512,
    sample_limit: int = 200_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute PCA over a directory of precomputed CLIP feature npy files.

    Loads ``*_s.npy`` files, flattens them to [N, clip_dim], draws up to
    ``sample_limit`` pixels, and returns the top-``n_components`` PCA directions.

    Args:
        clip_dir:     Directory containing ``{stem}_s.npy`` files of shape [H, W, clip_dim].
        n_components: Number of PCA components to extract (= codebook_size = lang_feat_dim).
        clip_dim:     Dimensionality of the raw CLIP features (default 512).
        sample_limit: Maximum number of pixel-vectors to use for PCA (for memory).

    Returns:
        pca_components: [n_components, clip_dim] — principal directions (for codebook init).
        pca_transform:  [clip_dim, n_components] — projection matrix (for target compression).
    """
    npy_files = sorted(clip_dir.glob("*_s.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No '*_s.npy' CLIP feature files found in {clip_dir}")

    # Gather a sample of pixel vectors across all images
    all_feats: List[np.ndarray] = []
    collected = 0
    for f in npy_files:
        arr = np.load(f).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])  # [H*W, clip_dim]
        if arr.shape[-1] != clip_dim:
            raise RuntimeError(
                f"Expected CLIP dim {clip_dim}, got {arr.shape[-1]} in {f.name}"
            )
        remaining = sample_limit - collected
        if arr.shape[0] > remaining:
            idx = np.random.choice(arr.shape[0], remaining, replace=False)
            arr = arr[idx]
        all_feats.append(arr)
        collected += arr.shape[0]
        if collected >= sample_limit:
            break

    features = torch.from_numpy(np.concatenate(all_feats, axis=0))  # [N, clip_dim]
    logging.getLogger("cityscape_gs.train_semantics").info(
        f"Computing PCA over {features.shape[0]} CLIP pixel vectors from {len(npy_files)} images."
    )

    # Centered covariance PCA — efficient for clip_dim (512) << N samples
    mean = features.mean(dim=0, keepdim=True)
    features_c = features - mean                                      # [N, clip_dim]
    cov = (features_c.T @ features_c) / max(features_c.shape[0] - 1, 1)  # [clip_dim, clip_dim]

    # eigh gives eigenvalues sorted ascending — take the last n_components (largest)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    top_eigenvectors = eigenvectors[:, -n_components:].flip(dims=[1])  # [clip_dim, n_components]

    pca_components = top_eigenvectors.T.contiguous()          # [n_components, clip_dim]
    pca_transform = top_eigenvectors.contiguous()              # [clip_dim, n_components]
    return pca_components, pca_transform


# ---------------------------------------------------------------------------
# GaussianModel rasterizer (original, unchanged)
# ---------------------------------------------------------------------------


class SemanticRasterizer:
    """Rasterize the per-Gaussian semantic feature field into screen space.

    Geometric attributes are detached so only _features_semantics receives gradients.
    """

    def __init__(self, model: GaussianModel) -> None:
        self.model = model

    def render(self, cam: dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (rgb_pred [H,W,3], semantic_pred [H,W,C])."""
        viewmat = _build_viewmat(cam, device=device)
        k = _build_intrinsics(cam, device=device)

        colors = torch.cat([self.model._features_dc.squeeze(1).detach(), self.model.semantics], dim=-1)

        out, _, _ = rasterization(
            means=self.model.means.detach(),
            quats=self.model.quats.detach(),
            scales=self.model.scales.detach(),
            opacities=self.model.opacities.squeeze(-1).detach(),
            colors=colors,
            viewmats=viewmat,
            Ks=k[None, ...],
            width=cam["width"],
            height=cam["height"],
            sh_degree=None,
            render_mode="RGB",
        )

        rendered = out[0].float()
        return rendered[..., :3], rendered[..., 3:]


# ---------------------------------------------------------------------------
# ScaffoldModel language rasterizer
# ---------------------------------------------------------------------------


class ScaffoldSemanticRasterizer:
    """Rasterize per-Gaussian language features from a ScaffoldModel.

    Calls generate_neural_gaussians, concatenates RGB + 32-dim language feats,
    and rasterizes jointly. Gradients flow through language features only;
    geometric outputs are detached.
    """

    def __init__(self, model: ScaffoldModel) -> None:
        self.model = model

    def render(self, cam: dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (rgb_pred [H,W,3], lang_pred [H,W,lang_feat_dim])."""
        if not self.model.enable_language_features:
            raise RuntimeError(
                "ScaffoldSemanticRasterizer requires enable_language_features=True."
            )

        viewmat = _build_viewmat(cam, device=device)
        k = _build_intrinsics(cam, device=device)

        out = self.model.generate_neural_gaussians(cam, is_training=True)
        assert out.language_features is not None, (
            "generate_neural_gaussians returned None language_features — "
            "ensure enable_language_features=True on the model."
        )

        # Detach geometric quantities so gradients only flow through language feats.
        # language_features retains gradients for mlp_language + _anchor_lang_feat.
        colors_with_lang = torch.cat(
            [out.colors.detach(), out.language_features], dim=-1
        )  # [M, 3 + lang_feat_dim]

        rendered, _, _ = rasterization(
            means=out.means.detach(),
            quats=out.quats.detach(),
            scales=out.scales.detach(),
            opacities=out.opacities.squeeze(-1).detach(),
            colors=colors_with_lang,
            viewmats=viewmat,
            Ks=k[None, ...],
            width=cam["width"],
            height=cam["height"],
            sh_degree=None,
            render_mode="RGB",
        )

        rendered = rendered[0].float()  # [H, W, 3 + lang_feat_dim]
        return rendered[..., :3], rendered[..., 3:]


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------


def _freeze_for_semantics(model: GaussianModel) -> Dict[str, bool]:
    """Freeze all GaussianModel params except _features_semantics."""
    state: Dict[str, bool] = {}
    for name, param in model.named_parameters():
        state[name] = bool(param.requires_grad)
        param.requires_grad_(name == "_features_semantics")
    return state


def _freeze_for_scaffold_language(model: ScaffoldModel) -> Dict[str, bool]:
    """Freeze all ScaffoldModel params except the language feature parameters.

    Trainable: _anchor_lang_feat, mlp_language.*, language_codebook, codebook_proj.
    Frozen:    _anchor, _offset, _anchor_feat, _scaling, _rotation, _opacity,
               mlp_opacity.*, mlp_cov.*, mlp_color.*, mlp_feature_bank.*, fourier_embedder.*,
               embedding_appearance.*.
    """
    _LANGUAGE_PARAMS = {"_anchor_lang_feat", "language_codebook", "codebook_proj"}
    _LANGUAGE_MODULES = {"mlp_language"}

    state: Dict[str, bool] = {}
    for name, param in model.named_parameters():
        state[name] = bool(param.requires_grad)
        is_language = name in _LANGUAGE_PARAMS or any(
            name.startswith(f"{m}.") for m in _LANGUAGE_MODULES
        )
        param.requires_grad_(is_language)
    return state


def _restore_requires_grad(model: BaseTrainableModel, state: Dict[str, bool]) -> None:
    for name, param in model.named_parameters():
        if name in state:
            param.requires_grad_(state[name])


# ---------------------------------------------------------------------------
# SemanticTrainer — GaussianModel path (original behaviour preserved)
# ---------------------------------------------------------------------------


class SemanticTrainer:
    """Post-training semantic fine-tuning for GaussianModel.

    Freezes all geometry, optimises _features_semantics to match 2D semantic targets.
    """

    def __init__(
        self,
        model: GaussianModel,
        dataset: SemanticColmapDataset,
        semantics_cfg: SemanticsConfig,
        lr_cfg: LearningRateConfig,
        train_cfg: TrainingConfig,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
        tb_logger: Optional[GaussianSplattingLogger] = None,
        global_step_offset: int = 0,
    ) -> None:
        if model.semantics is None:
            raise RuntimeError("Semantic fine-tuning requested but model has no semantic feature field.")

        self.model = model
        self.dataset = dataset
        self.semantics_cfg = semantics_cfg
        self.lr_cfg = lr_cfg
        self.train_cfg = train_cfg
        self.device = device
        self.logger = logger
        self.tb_logger = tb_logger
        self.global_step_offset = global_step_offset

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.train_cfg.use_low_vram)
        if self.train_cfg.use_low_vram and logger is not None:
            logger.info(
                "[yellow]⚡ Low VRAM mode enabled for semantic training:[/yellow] "
                "Using mixed precision (FP16/AMP) and gradient scaling"
            )

        self._provider: Optional[SemanticTargetProvider] = None
        self._dataloader_iter = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._rasterizer = SemanticRasterizer(model)

    def _setup_provider(self) -> None:
        self._provider = build_semantic_provider(
            provider_name=self.semantics_cfg.semantic_provider,
            semantics_dim=self.semantics_cfg.semantics_dim,
            cache_enabled=self.semantics_cfg.semantic_cache_enabled,
            semantic_model_path=self.semantics_cfg.semantic_model_path,
        )

    def _setup_dataloader(self) -> DataLoader:
        num_workers = max(0, self.train_cfg.num_workers)
        if self.train_cfg.preload and num_workers > 0:
            num_workers = 0
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.dataset.collate_fn,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        self._dataloader_iter = iter(dataloader)
        return dataloader

    def _setup_optimizer(self) -> torch.optim.Adam:
        lr = self.lr_cfg.lr_semantics if self.lr_cfg.lr_semantics is not None else self.lr_cfg.lr_sh
        self._optimizer = torch.optim.Adam([self.model._features_semantics], lr=lr)
        return self._optimizer

    def _next_batch(self) -> dict:
        try:
            return next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self._dataloader)
            return next(self._dataloader_iter)

    def _compute_loss(
        self, semantic_pred: torch.Tensor, semantic_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        semantic_pred = _normalize_map(semantic_pred)
        semantic_target = _normalize_map(semantic_target)
        semantic_loss = F.mse_loss(semantic_pred, semantic_target)
        return self.semantics_cfg.semantic_loss_weight * semantic_loss, semantic_loss

    def _train_step(self, batch: dict) -> dict:
        assert self._provider is not None
        assert self._optimizer is not None

        cam = batch["cam"]
        gt_image = batch["gt_image"]
        semantic_tensor = batch["semantic_tensor"]
        image_id = batch["image_id"]

        if not gt_image.is_cuda:
            gt_image = gt_image.to(self.device)
        if gt_image.dim() == 3:
            gt_image = gt_image.unsqueeze(0)

        with torch.cuda.amp.autocast(enabled=self.train_cfg.use_low_vram):
            rgb_pred, semantic_pred = self._rasterizer.render(cam, device=self.device)

            semantic_target = self._provider.get_target(
                image_id=image_id,
                gt_image=gt_image[0].detach(),
                dataset_semantic_tensor=semantic_tensor,
                device=self.device,
            )
            semantic_target = _as_hwc(semantic_target, expected_channels=self.semantics_cfg.semantics_dim)
            semantic_pred = _interpolate_to_match(semantic_pred, semantic_target)
            total_loss, semantic_loss = self._compute_loss(semantic_pred, semantic_target)

        if not total_loss.isfinite():
            raise RuntimeError(f"Semantic fine-tuning loss is invalid: {total_loss.item()}")

        self._optimizer.zero_grad(set_to_none=True)
        if self.train_cfg.use_low_vram:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self._optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self._optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "semantic_loss": float(semantic_loss.item()),
        }

    def _log_step(self, local_step: int, step_metrics: dict) -> None:
        global_step = self.global_step_offset + local_step
        if (
            self.tb_logger is not None
            and self.tb_logger.enabled
            and local_step % self.train_cfg.log_interval == 0
        ):
            self.tb_logger.log_losses(
                total_loss=step_metrics["total_loss"],
                l1_loss=0.0,
                ssim_loss=0.0,
                lpips_loss=0.0,
                semantic_loss=step_metrics["semantic_loss"],
                step=global_step,
            )
        if self.logger is not None and local_step % self.train_cfg.log_interval == 0:
            self.logger.info(
                "[cyan]Semantic fine-tuning[/cyan] step=%d/%d loss=%.6f",
                local_step + 1,
                self.semantics_cfg.semantic_finetune_iters,
                step_metrics["total_loss"],
            )

    def run(self) -> dict:
        self._setup_provider()
        self._dataloader = self._setup_dataloader()
        self._setup_optimizer()

        grad_state = _freeze_for_semantics(self.model)
        running_loss = 0.0
        try:
            for local_step in range(self.semantics_cfg.semantic_finetune_iters):
                batch = self._next_batch()
                step_metrics = self._train_step(batch)
                running_loss += step_metrics["total_loss"]
                self._log_step(local_step, step_metrics)
        finally:
            _restore_requires_grad(self.model, grad_state)

        avg_loss = running_loss / max(1, self.semantics_cfg.semantic_finetune_iters)
        return {"semantic_avg_loss": avg_loss, "semantic_iters": self.semantics_cfg.semantic_finetune_iters}


# ---------------------------------------------------------------------------
# ScaffoldSemanticTrainer — Scaffold-GS language feature training path
# ---------------------------------------------------------------------------


class ScaffoldSemanticTrainer:
    """Language feature fine-tuning for ScaffoldModel.

    Pipeline:
      1. Compute PCA from CLIP npy files → init codebook from PCA components.
      2. Freeze all geometry; keep _anchor_lang_feat + mlp_language + codebook trainable.
      3. Per step: render 32-dim language map → compress CLIP target to 32-dim → MSE loss.

    The supervision target (raw 512-dim CLIP features) is PCA-compressed per-step to
    lang_feat_dim (32) using the precomputed pca_transform. Training never touches 512-dim
    tensors in the backward pass — only the codebook decode at query time does.

    Args:
        model:               ScaffoldModel with enable_language_features=True.
        dataset:             SemanticColmapDataset whose semantics_path points to
                             ``*_s.npy`` files of shape [H, W, clip_dim].
        semantics_cfg:       Config; semantics_dim should equal model.clip_dim (e.g. 512).
        lr_cfg, train_cfg:   Standard training configs.
        clip_pca_sample:     How many pixel-vectors to sample for PCA computation.
    """

    def __init__(
        self,
        model: ScaffoldModel,
        dataset: SemanticColmapDataset,
        semantics_cfg: SemanticsConfig,
        lr_cfg: LearningRateConfig,
        train_cfg: TrainingConfig,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
        tb_logger: Optional[GaussianSplattingLogger] = None,
        global_step_offset: int = 0,
        clip_pca_sample: int = 200_000,
    ) -> None:
        if not model.enable_language_features:
            raise RuntimeError(
                "ScaffoldSemanticTrainer requires ScaffoldModel with enable_language_features=True."
            )
        if semantics_cfg.semantics_path is None:
            raise RuntimeError(
                "semantics_path must point to the directory of CLIP npy files."
            )

        self.model = model
        self.dataset = dataset
        self.semantics_cfg = semantics_cfg
        self.lr_cfg = lr_cfg
        self.train_cfg = train_cfg
        self.device = device
        self.logger = logger
        self.tb_logger = tb_logger
        self.global_step_offset = global_step_offset
        self.clip_pca_sample = clip_pca_sample

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.train_cfg.use_low_vram)

        # Will be set in run() before the training loop
        self._pca_transform: Optional[torch.Tensor] = None  # [clip_dim, lang_feat_dim]
        self._provider: Optional[NpySemanticProvider] = None
        self._dataloader_iter = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._rasterizer = ScaffoldSemanticRasterizer(model)

    # -- setup ---------------------------------------------------------------

    def _setup_pca_and_codebook(self) -> None:
        """Compute PCA over the CLIP features and seed the model codebook."""
        clip_dir = Path(self.semantics_cfg.semantics_path)
        if self.logger:
            self.logger.info(
                "[cyan]ScaffoldSemanticTrainer[/cyan] Computing PCA from %s ...", clip_dir
            )

        pca_components, pca_transform = compute_pca_from_clip_dir(
            clip_dir=clip_dir,
            n_components=self.model.codebook_size,
            clip_dim=self.model.clip_dim,
            sample_limit=self.clip_pca_sample,
        )
        # pca_transform: [clip_dim, lang_feat_dim] — used per-step to compress targets
        self._pca_transform = pca_transform.to(self.device)

        # Seed codebook with PCA directions — gives it semantic meaning from step 0
        self.model.init_codebook_from_pca(pca_components.to(self.device))
        if self.logger:
            self.logger.info(
                "[cyan]ScaffoldSemanticTrainer[/cyan] Codebook initialized from PCA "
                "(top %d directions of %d-dim CLIP features).",
                self.model.codebook_size,
                self.model.clip_dim,
            )

    def _setup_provider(self) -> None:
        # Provider loads raw [H, W, clip_dim] CLIP features from npy files.
        # semantics_dim = clip_dim so the channel check in the provider passes.
        # The ScaffoldSemanticTrainer compresses to lang_feat_dim internally.
        self._provider = NpySemanticProvider(
            semantics_dim=self.model.clip_dim,
            cache_enabled=self.semantics_cfg.semantic_cache_enabled,
        )

    def _setup_dataloader(self) -> DataLoader:
        num_workers = max(0, self.train_cfg.num_workers)
        if self.train_cfg.preload and num_workers > 0:
            num_workers = 0
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.dataset.collate_fn,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        self._dataloader_iter = iter(dataloader)
        return dataloader

    def _setup_optimizer(self) -> None:
        """Create optimizers for all language-feature parameters."""
        lr_lang = (
            self.lr_cfg.lr_semantics
            if self.lr_cfg.lr_semantics is not None
            else self.lr_cfg.lr_sh
        )
        lr_mlp = lr_lang * 2.0       # MLP can move faster than anchor feats
        lr_codebook = lr_lang * 0.1  # Codebook evolves slowly (bootstrapped from PCA)

        assert self.model.mlp_language is not None
        assert self.model.language_codebook is not None
        assert self.model.codebook_proj is not None

        # Group into separate optimizers for fine-grained LR control
        self._optimizer = torch.optim.Adam(
            [
                {"params": [self.model._anchor_lang_feat], "lr": lr_lang},
                {"params": list(self.model.mlp_language.parameters()), "lr": lr_mlp},
                {"params": [self.model.language_codebook, self.model.codebook_proj], "lr": lr_codebook},
            ]
        )

    # -- training loop -------------------------------------------------------

    def _next_batch(self) -> dict:
        try:
            return next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self._dataloader)
            return next(self._dataloader_iter)

    def _compress_clip_target(self, clip_target: torch.Tensor) -> torch.Tensor:
        """Project [H, W, clip_dim] CLIP features to [H, W, lang_feat_dim] via PCA.

        Uses the precomputed pca_transform from _setup_pca_and_codebook.
        Operates in FP32 to preserve precision during the linear projection.
        """
        assert self._pca_transform is not None
        pca_t = self._pca_transform.to(device=clip_target.device, dtype=torch.float32)
        return clip_target.float() @ pca_t  # [H, W, lang_feat_dim]

    def _train_step(self, batch: dict) -> dict:
        assert self._provider is not None
        assert self._optimizer is not None

        cam = batch["cam"]
        gt_image = batch["gt_image"]
        semantic_tensor = batch["semantic_tensor"]
        image_id = batch["image_id"]

        if not gt_image.is_cuda:
            gt_image = gt_image.to(self.device)
        if gt_image.dim() == 3:
            gt_image = gt_image.unsqueeze(0)

        with torch.cuda.amp.autocast(enabled=self.train_cfg.use_low_vram):
            # Render 32-dim language features for this view
            _rgb_pred, lang_pred = self._rasterizer.render(cam, device=self.device)
            # lang_pred: [H, W, lang_feat_dim]

            # Load raw CLIP target [H, W, clip_dim]
            clip_target = self._provider.get_target(
                image_id=image_id,
                gt_image=gt_image[0].detach(),
                dataset_semantic_tensor=semantic_tensor,
                device=self.device,
            )
            clip_target = _as_hwc(clip_target, expected_channels=self.model.clip_dim)

            # Compress CLIP target → 32-dim supervision signal
            lang_target = self._compress_clip_target(clip_target)

            # Spatial resolution matching
            lang_pred = _interpolate_to_match(lang_pred, lang_target)

            # Supervise in 32-dim compressed latent space.
            # We normalize both for scale-invariant MSE (consistent with GaussianModel path).
            lang_pred_n = _normalize_map(lang_pred)
            lang_target_n = _normalize_map(lang_target)
            lang_loss = F.mse_loss(lang_pred_n, lang_target_n)
            total_loss = self.semantics_cfg.semantic_loss_weight * lang_loss

        if not total_loss.isfinite():
            raise RuntimeError(f"Language feature loss is invalid at step: {total_loss.item()}")

        self._optimizer.zero_grad(set_to_none=True)
        if self.train_cfg.use_low_vram:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self._optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self._optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "semantic_loss": float(lang_loss.item()),
        }

    def _log_step(self, local_step: int, step_metrics: dict) -> None:
        global_step = self.global_step_offset + local_step
        if (
            self.tb_logger is not None
            and self.tb_logger.enabled
            and local_step % self.train_cfg.log_interval == 0
        ):
            self.tb_logger.log_losses(
                total_loss=step_metrics["total_loss"],
                l1_loss=0.0,
                ssim_loss=0.0,
                lpips_loss=0.0,
                semantic_loss=step_metrics["semantic_loss"],
                step=global_step,
            )
        if self.logger is not None and local_step % self.train_cfg.log_interval == 0:
            self.logger.info(
                "[cyan]Scaffold language training[/cyan] step=%d/%d loss=%.6f",
                local_step + 1,
                self.semantics_cfg.semantic_finetune_iters,
                step_metrics["total_loss"],
            )

    def run(self) -> dict:
        """Execute the full Scaffold language feature fine-tuning loop.

        Returns:
            Dict with ``semantic_avg_loss`` and ``semantic_iters``.
        """
        # 1. PCA init — must happen before training so codebook has semantic meaning
        self._setup_pca_and_codebook()
        self._setup_provider()
        self._dataloader = self._setup_dataloader()
        self._setup_optimizer()

        # 2. Freeze geometry, keep language params trainable
        grad_state = _freeze_for_scaffold_language(self.model)
        running_loss = 0.0

        try:
            for local_step in range(self.semantics_cfg.semantic_finetune_iters):
                batch = self._next_batch()
                step_metrics = self._train_step(batch)
                running_loss += step_metrics["total_loss"]
                self._log_step(local_step, step_metrics)
        finally:
            _restore_requires_grad(self.model, grad_state)

        avg_loss = running_loss / max(1, self.semantics_cfg.semantic_finetune_iters)
        return {
            "semantic_avg_loss": avg_loss,
            "semantic_iters": self.semantics_cfg.semantic_finetune_iters,
        }


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------


def train_semantics(
    model: BaseTrainableModel,
    dataset: SemanticColmapDataset,
    train_cfg: TrainingConfig,
    semantics_cfg: SemanticsConfig,
    lr_cfg: LearningRateConfig,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    tb_logger: Optional[GaussianSplattingLogger] = None,
    global_step_offset: int = 0,
) -> dict:
    """Run post-training semantic / language feature fine-tuning.

    Dispatches to the appropriate trainer based on model type:
      - ScaffoldModel with enable_language_features=True → ScaffoldSemanticTrainer
      - GaussianModel → SemanticTrainer (original behaviour)

    Args:
        model:    Trained model to fine-tune. For ScaffoldModel, must have
                  enable_language_features=True.
        dataset:  SemanticColmapDataset whose semantics_path points to npy files.
                  For GaussianModel: [H, W, semantics_dim] files.
                  For ScaffoldModel: [H, W, clip_dim] raw CLIP feature files (512-dim).
        ...

    Returns:
        Dict with ``semantic_avg_loss`` and ``semantic_iters``.
    """
    if isinstance(model, ScaffoldModel) and isinstance(model, SemanticsMixin):
        if not model.enable_language_features:
            raise RuntimeError(
                "train_semantics on ScaffoldModel requires enable_language_features=True. "
                "Reconstruct the model with enable_language_features=True first."
            )
        trainer: ScaffoldSemanticTrainer | SemanticTrainer = ScaffoldSemanticTrainer(
            model=model,
            dataset=dataset,
            semantics_cfg=semantics_cfg,
            lr_cfg=lr_cfg,
            train_cfg=train_cfg,
            device=device,
            logger=logger,
            tb_logger=tb_logger,
            global_step_offset=global_step_offset,
        )
    elif isinstance(model, GaussianModel):
        trainer = SemanticTrainer(
            model=model,
            dataset=dataset,
            semantics_cfg=semantics_cfg,
            lr_cfg=lr_cfg,
            train_cfg=train_cfg,
            device=device,
            logger=logger,
            tb_logger=tb_logger,
            global_step_offset=global_step_offset,
        )
    else:
        raise TypeError(
            f"train_semantics does not support model type {type(model).__name__}. "
            "Supported: GaussianModel, ScaffoldModel (with enable_language_features=True)."
        )

    return trainer.run()


# ---------------------------------------------------------------------------
# Standalone runner helpers (existing, unchanged)
# ---------------------------------------------------------------------------


def _setup_logger(output_dir: Path) -> logging.Logger:
    log = logging.getLogger("cityscape_gs.train_semantics")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    output_dir.mkdir(parents=True, exist_ok=True)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "semantic_training.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(file_handler)

    return log


def _load_model(
    cfg: SemanticStandaloneConfig,
    device: torch.device,
    logger: logging.Logger,
) -> tuple:
    logger.info("Loading model checkpoint: %s", cfg.checkpoint_path)
    model, checkpoint = GaussianModel.resume_from_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        device=str(device),
        train_semantics=True,
        semantics_dim=cfg.semantics_dim,
        strict=False,
    )
    model.to(device)
    return model, checkpoint


def _build_datasets(
    cfg: SemanticStandaloneConfig,
    device: torch.device,
) -> tuple:
    base_dataset = ColmapDataset(
        cfg.colmap_path,
        cfg.images_path,
        device=device,
        image_scale=cfg.scale,
        scene_extent_margin=1.5,
    )

    if cfg.preload:
        base_dataset.preload_all_data()

    semantic_dataset = SemanticColmapDataset(
        base_dataset=base_dataset,
        semantics_path=cfg.semantics_path if cfg.semantic_provider == "npy" else None,
        semantics_resolution=cfg.semantic_image_resolution,
    )

    return base_dataset, semantic_dataset


def _build_configs(cfg: SemanticStandaloneConfig) -> tuple:
    semantics_cfg = SemanticsConfig(
        train_semantics=True,
        semantics_path=cfg.semantics_path,
        semantics_dim=cfg.semantics_dim,
        semantic_image_resolution=cfg.semantic_image_resolution,
        semantic_loss_weight=cfg.semantic_loss_weight,
        semantic_finetune_iters=cfg.semantic_finetune_iters,
        semantic_provider=cfg.semantic_provider,
        semantic_model_path=cfg.semantic_model_path,
        semantic_cache_enabled=cfg.semantic_cache_enabled,
    )

    train_cfg = TrainingConfig(
        iterations=0,
        save_interval=0,
        log_interval=cfg.log_interval,
        num_workers=cfg.num_workers,
        preload=cfg.preload,
        enable_lpips_loss=False,
        lpips_loss_weight=0.0,
    )

    lr_cfg = LearningRateConfig(
        lr_means=0.0,
        lr_scales=0.0,
        lr_quats=0.0,
        lr_opacities=0.0,
        lr_sh=cfg.lr_sh,
        lr_semantics=cfg.lr_semantics,
    )

    return semantics_cfg, train_cfg, lr_cfg


def _save_checkpoint(
    model: GaussianModel,
    checkpoint: dict,
    metrics: dict,
    cfg: SemanticStandaloneConfig,
    tb_logger: GaussianSplattingLogger,
) -> Path:
    output_ckpt = cfg.output_dir / "semantic_model_final.pt"
    torch.save(
        {
            "iteration": int(checkpoint.get("iteration", 0)) + metrics["semantic_iters"],
            "model_state_dict": model.state_dict(),
            "base_checkpoint": str(cfg.checkpoint_path),
            "semantic_metrics": metrics,
            "tensorboard_run_name": tb_logger.run_name if cfg.tensorboard else None,
        },
        output_ckpt,
    )

    if cfg.tensorboard:
        tb_logger.log_hyperparameters({}, {"semantic/final_avg_loss": float(metrics["semantic_avg_loss"])})
        tb_logger.close()

    return output_ckpt


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------


def run_standalone_semantic_training(cfg: SemanticStandaloneConfig) -> None:
    """Run semantic fine-tuning as a standalone post-training step."""
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    log = _setup_logger(cfg.output_dir)

    model, checkpoint = _load_model(cfg, device, log)
    log.info("Building datasets...")
    _base_dataset, semantic_dataset = _build_datasets(cfg, device)
    log.info("Datasets built!")
    semantics_cfg, train_cfg, lr_cfg = _build_configs(cfg)

    log.info("Initializing tb logger...")
    tb_logger = GaussianSplattingLogger(
        log_dir=str(cfg.output_dir / "tensorboard"),
        enabled=cfg.tensorboard,
        run_name=checkpoint.get("tensorboard_run_name", None),
        purge_step=None,
    )
    log.info("Starting train_semantics...")

    metrics = train_semantics(
        model=model,
        dataset=semantic_dataset,
        train_cfg=train_cfg,
        semantics_cfg=semantics_cfg,
        lr_cfg=lr_cfg,
        device=device,
        logger=log,
        tb_logger=tb_logger,
        global_step_offset=int(checkpoint.get("iteration", 0)) + 1,
    )

    output_ckpt = _save_checkpoint(model, checkpoint, metrics, cfg, tb_logger)

    log.info("Semantic training complete. Final avg loss: %.6f", metrics["semantic_avg_loss"])
    log.info("Saved semantic checkpoint: %s", output_ckpt)


if __name__ == "__main__":
    run_standalone_semantic_training(parse_semantic_args())
