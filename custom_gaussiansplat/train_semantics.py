import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from gsplat import rasterization
from torch.utils.data import DataLoader

from dataset import ColmapDataset
from logger import GaussianSplattingLogger
from models import GaussianModel
from semantic_dataset import SemanticColmapDataset
from semantic_providers import build_semantic_provider
from train_semantics_args import SemanticStandaloneConfig, parse_semantic_args
from train_args import LearningRateConfig, SemanticsConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Small utility helpers
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


def _freeze_for_semantics(model: GaussianModel) -> Dict[str, bool]:
    state: Dict[str, bool] = {}
    for name, param in model.named_parameters():
        state[name] = bool(param.requires_grad)
        param.requires_grad_(name == "_features_semantics")
    return state


def _restore_requires_grad(model: GaussianModel, state: Dict[str, bool]) -> None:
    for name, param in model.named_parameters():
        if name in state:
            param.requires_grad_(state[name])


# ---------------------------------------------------------------------------
# SemanticRasterizer — encapsulates gsplat rasterization for semantic features
# ---------------------------------------------------------------------------


class SemanticRasterizer:
    """Rasterize the per-Gaussian semantic feature field into screen space.

    The geometric Gaussian attributes (means, quats, scales, opacities) are
    detached so that only the semantic feature tensor receives gradients.
    """

    def __init__(self, model: GaussianModel) -> None:
        self.model = model

    def render(self, cam: dict, device: torch.device) -> torch.Tensor:
        """Render semantic features for a single camera.

        Args:
            cam: Camera dict with keys R, T, fx, fy, cx, cy, width, height.
            device: Target device for intermediate tensors.

        Returns:
            rgb_pred: Rendered RGB tensor of shape ``[H, W, 3]``.
            semantic_pred: Semantic prediction tensor of shape ``[H, W, C]``.
        """
        viewmat = _build_viewmat(cam, device=device)
        k = _build_intrinsics(cam, device=device)

        colors = torch.cat([self.model._features_dc.squeeze(1).detach(), self.model.semantics], dim=-1)

        semantic_out, _, _ = rasterization(
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

        rendered = semantic_out[0].float()
        rgb_pred = rendered[..., :3]
        semantic_pred = rendered[..., 3:]

        return rgb_pred, semantic_pred


# ---------------------------------------------------------------------------
# SemanticTrainer — orchestrates the semantic fine-tuning loop
# ---------------------------------------------------------------------------


class SemanticTrainer:
    """Self-contained semantic fine-tuning trainer.

    Encapsulates provider setup, data loading, optimisation, and logging so
    that callers only need to call :meth:`run`.
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

        # Low VRAM optimizations: Initialize gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.train_cfg.use_low_vram)
        if self.train_cfg.use_low_vram and logger is not None:
            logger.info(
                "[yellow]⚡ Low VRAM mode enabled for semantic training:[/yellow] "
                "Using mixed precision (FP16/AMP) and gradient scaling"
            )

        # Lazily initialised in run()
        self._provider = None
        self._dataloader_iter = None
        self._optimizer = None
        self._rasterizer = SemanticRasterizer(model)

    # -- setup helpers -------------------------------------------------------

    def _setup_provider(self):
        """Build the semantic target provider (npy / runtime)."""
        self._provider = build_semantic_provider(
            provider_name=self.semantics_cfg.semantic_provider,
            semantics_dim=self.semantics_cfg.semantics_dim,
            cache_enabled=self.semantics_cfg.semantic_cache_enabled,
            semantic_model_path=self.semantics_cfg.semantic_model_path,
        )

    def _setup_dataloader(self) -> DataLoader:
        """Create a DataLoader with correct worker configuration."""
        num_workers = max(0, self.train_cfg.num_workers)
        if self.train_cfg.preload and num_workers > 0:
            num_workers = 0

        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.dataset.collate_fn,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        self._dataloader_iter = iter(dataloader)
        return dataloader

    def _setup_optimizer(self) -> torch.optim.Adam:
        """Create the Adam optimiser for semantic parameters only."""
        lr = self.lr_cfg.lr_semantics if self.lr_cfg.lr_semantics is not None else self.lr_cfg.lr_sh
        self._optimizer = torch.optim.Adam([self.model._features_semantics], lr=lr)
        return self._optimizer

    # -- per-step helpers ----------------------------------------------------

    def _next_batch(self) -> dict:
        """Fetch the next training batch, cycling the dataloader if exhausted."""
        try:
            return next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self._dataloader)
            return next(self._dataloader_iter)

    def _compute_loss(
        self,
        semantic_pred: torch.Tensor,
        semantic_target: torch.Tensor,
    ) -> torch.Tensor:
        """Normalise predictions & targets, then compute weighted MSE loss.

        Args:
            semantic_pred: Rendered semantic map ``[H, W, C]``.
            semantic_target: Ground-truth semantic map ``[H, W, C]``.

        Returns:
            Weighted scalar loss tensor.
        """
        semantic_pred = _normalize_map(semantic_pred)
        semantic_target = _normalize_map(semantic_target)
        semantic_loss = F.mse_loss(semantic_pred, semantic_target)
        return self.semantics_cfg.semantic_loss_weight * semantic_loss, semantic_loss

    def _train_step(self, batch: dict) -> dict:
        """Execute a single training iteration.

        Returns:
            Dict with ``total_loss`` and ``semantic_loss`` scalars.
        """
        cam = batch["cam"]
        gt_image = batch["gt_image"]
        semantic_tensor = batch["semantic_tensor"]
        image_id = batch["image_id"]

        if not gt_image.is_cuda:
            gt_image = gt_image.to(self.device)
        if gt_image.dim() == 3:
            gt_image = gt_image.unsqueeze(0)

        # Forward pass with optional mixed precision
        with torch.cuda.amp.autocast(enabled=self.train_cfg.use_low_vram):
            # Render semantic features and RGB
            rgb_pred, semantic_pred = self._rasterizer.render(cam, device=self.device)

            # Obtain ground-truth target
            semantic_target = self._provider.get_target(
                image_id=image_id,
                gt_image=gt_image[0].detach(),
                dataset_semantic_tensor=semantic_tensor,
                device=self.device,
            )
            semantic_target = _as_hwc(semantic_target, expected_channels=self.semantics_cfg.semantics_dim)

            # Spatial resolution matching
            if semantic_target.shape[:2] != semantic_pred.shape[:2]:
                pred_nchw = semantic_pred.permute(2, 0, 1).unsqueeze(0)
                pred_nchw = F.interpolate(
                    pred_nchw,
                    size=(semantic_target.shape[0], semantic_target.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                semantic_pred = pred_nchw.squeeze(0).permute(1, 2, 0)

            # Loss computation
            total_loss, semantic_loss = self._compute_loss(semantic_pred, semantic_target)

        # Validation (outside autocast for accurate checking)
        if not total_loss.isfinite():
            raise RuntimeError(f"Semantic fine-tuning loss is invalid: {total_loss.item()}")

        # Backward + update with gradient scaling
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
        """Emit TensorBoard and console logs for a single step."""
        global_step = self.global_step_offset + local_step

        if self.tb_logger is not None and self.tb_logger.enabled and (local_step % self.train_cfg.log_interval == 0):
            self.tb_logger.log_losses(
                total_loss=step_metrics["total_loss"],
                l1_loss=0.0,
                ssim_loss=0.0,
                lpips_loss=0.0,
                semantic_loss=step_metrics["semantic_loss"],
                step=global_step,
            )

        if self.logger is not None and (local_step % self.train_cfg.log_interval == 0):
            self.logger.info(
                "[cyan]Semantic fine-tuning[/cyan] step=%d/%d loss=%.6f",
                local_step + 1,
                self.semantics_cfg.semantic_finetune_iters,
                step_metrics["total_loss"],
            )

    # -- main entry-point ----------------------------------------------------

    def run(self) -> dict:
        """Execute the full semantic fine-tuning loop.

        Returns:
            Dict with ``semantic_avg_loss`` and ``semantic_iters``.
        """
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
        return {
            "semantic_avg_loss": avg_loss,
            "semantic_iters": self.semantics_cfg.semantic_finetune_iters,
        }


# ---------------------------------------------------------------------------
# Public free-function API (thin wrapper — preserves existing call-sites)
# ---------------------------------------------------------------------------


def train_semantics(
    model: GaussianModel,
    dataset: SemanticColmapDataset,
    train_cfg: TrainingConfig,
    semantics_cfg: SemanticsConfig,
    lr_cfg: LearningRateConfig,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    tb_logger: Optional[GaussianSplattingLogger] = None,
    global_step_offset: int = 0,
) -> dict:
    """Run post-training semantic-only fine-tuning.

    Only semantic Gaussian features are trainable during this stage.

    This is a convenience wrapper around :class:`SemanticTrainer` that
    preserves backward-compatible call-sites (e.g. in ``train.py``).
    """
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
    return trainer.run()


# ---------------------------------------------------------------------------
# Standalone runner helpers
# ---------------------------------------------------------------------------


def _setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("cityscape_gs.train_semantics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    output_dir.mkdir(parents=True, exist_ok=True)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "semantic_training.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    return logger


def _load_model(
    cfg: SemanticStandaloneConfig,
    device: torch.device,
    logger: logging.Logger,
) -> tuple:
    """Load the base Gaussian Splatting model and its checkpoint dict."""
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
    """Construct the base COLMAP dataset and semantic wrapper dataset."""
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
    """Bridge standalone CLI config into the training sub-configs."""
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
    """Persist the fine-tuned model to disk and finalise TensorBoard."""
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
    logger = _setup_logger(cfg.output_dir)

    model, checkpoint = _load_model(cfg, device, logger)
    logger.info("Building datasets...")
    _base_dataset, semantic_dataset = _build_datasets(cfg, device)
    logger.info("Datasets built!")
    semantics_cfg, train_cfg, lr_cfg = _build_configs(cfg)

    logger.info("Initializing tb logger...")
    tb_logger = GaussianSplattingLogger(
        log_dir=str(cfg.output_dir / "tensorboard"),
        enabled=cfg.tensorboard,
        run_name=checkpoint.get("tensorboard_run_name", None),
        purge_step=None,
    )
    logger.info("Starting train_semantics...")

    metrics = train_semantics(
        model=model,
        dataset=semantic_dataset,
        train_cfg=train_cfg,
        semantics_cfg=semantics_cfg,
        lr_cfg=lr_cfg,
        device=device,
        logger=logger,
        tb_logger=tb_logger,
        global_step_offset=int(checkpoint.get("iteration", 0)) + 1,
    )

    output_ckpt = _save_checkpoint(model, checkpoint, metrics, cfg, tb_logger)

    logger.info("Semantic training complete. Final avg loss: %.6f", metrics["semantic_avg_loss"])
    logger.info("Saved semantic checkpoint: %s", output_ckpt)


if __name__ == "__main__":
    run_standalone_semantic_training(parse_semantic_args())
