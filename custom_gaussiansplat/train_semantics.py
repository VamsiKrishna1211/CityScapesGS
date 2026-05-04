"""Semantic / language feature fine-tuning for Gaussian Splatting models.

Design
------
``SemanticTrainer`` operates through the ``SemanticsMixin`` interface and is
fully model-agnostic.  Which parameters are trained is controlled by
``SemanticsConfig.finetune_params`` (a list of group names like
``["semantics"]`` or ``["semantics", "mlp_geo"]``).

Whether a set of finetune_params requires the full photometric render path is
determined by ``model.geometry_param_group_names`` — each model declares which
of its param groups affect geometry.  The trainer never hardcodes group names.

When only ``"semantics"`` is requested the loop is identical to the previous
behaviour: geometry is detached, only the SemanticLossComputer MSE loss is
computed.

When geometry groups are added the full ``Rasterizer`` + ``LossComputer``
pipeline from ``train.py`` is activated in addition to the semantic loss,
allowing joint fine-tuning with photometric, depth, and regularization losses.

Supported provider: ``dino_encoded`` — compressed DINO patch features produced
by ``tools/train_patch_encoder.py``.  Each .pt file stores::

    {
        "features":              Tensor [N, bottleneck],
        "patch_size":            int,
        "inference_image_shape": (H, W),
    }

The bottleneck dimension must equal ``--lang-feat-dim`` / ``--semantics-dim``.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import losses
from dataset import ColmapDataset, MatrixCityDataset
from logger import GaussianSplattingLogger, configure_app_logger
from losses import SemanticLossComputer
from models import BaseTrainableModel, SemanticsMixin
from models.semantic_scaffold import SemanticScaffoldModel
from rasterizer import Rasterizer, _prepare_depth_tensor, _prepare_gt_image
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from semantic_dataset import SemanticColmapDataset
from semantic_providers import SemanticTarget, build_semantic_provider
from torch.utils.data import DataLoader
from train_args import (
    DepthConfig,
    FloaterPreventionConfig,
    LearningRateConfig,
    SHConfig,
    TensorBoardConfig,
    TrainingConfig,
)
from train_semantics_args import (
    SemanticsConfig,
    SemanticStandaloneConfig,
    parse_semantic_args,
)
from viewer_sync import TrainingViewerBundle, setup_training_viewer

# ── Unified SemanticTrainer ───────────────────────────────────────────────────


class SemanticTrainer:
    """Model-agnostic fine-tuning loop for semantic and/or geometric parameters.

    Which parameters are trained is driven by ``semantics_cfg.finetune_params``
    (group names from ``model.get_finetune_param_groups()``).  Whether those
    groups require the full render path is determined by
    ``model.geometry_param_group_names`` — the trainer never hardcodes names.

    When only ``"semantics"`` is requested: semantic MSE loss only, geometry
    detached — identical to the original SemanticTrainer behaviour.

    When geometry groups are added: activates the full Rasterizer + LossComputer
    pipeline (same losses as train.py) alongside the semantic loss.
    """

    def __init__(
        self,
        model: SemanticsMixin,
        dataset: SemanticColmapDataset,
        semantics_cfg: SemanticsConfig,
        lr_cfg: LearningRateConfig,
        lr_semantics: Optional[float],
        train_cfg: TrainingConfig,
        depth_cfg: DepthConfig,
        floater_cfg: FloaterPreventionConfig,
        sh_cfg: SHConfig,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
        tb_logger: Optional[GaussianSplattingLogger] = None,
        global_step_offset: int = 0,
        verbosity: int = 1,
        viewer_cfg: Optional[Any] = None,
        scene_extent: float = 1.0,
        tensorboard_cfg: Optional[TensorBoardConfig] = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.semantics_cfg = semantics_cfg
        self.lr_cfg = lr_cfg
        self.lr_semantics = lr_semantics
        self.train_cfg = train_cfg
        self.depth_cfg = depth_cfg
        self.floater_cfg = floater_cfg
        self.sh_cfg = sh_cfg
        self.device = device
        self.logger = logger
        self.tb_logger = tb_logger
        self.global_step_offset = global_step_offset
        self.verbosity = verbosity
        self.viewer_cfg = viewer_cfg
        self.scene_extent = scene_extent
        self.tensorboard_cfg = tensorboard_cfg

        self.scaler = torch.amp.GradScaler(enabled=self.train_cfg.use_low_vram)
        self._provider = None
        self._dataloader = None
        self._dataloader_iter = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self.progress: Optional[Progress] = None
        self.task_id: Optional[TaskID] = None

        self._sem_loss_computer = SemanticLossComputer()

        # Lazily initialised when geometry groups are requested
        self._rasterizer: Optional[Rasterizer] = None
        self._loss_computer: Optional[losses.LossComputer] = None

        # Viewer state (populated by _setup_viewer)
        self._viewer_bundle: Optional[TrainingViewerBundle] = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def viewer(self) -> Optional[Any]:
        return self._viewer_bundle.viewer if self._viewer_bundle else None

    @property
    def rerun_viewer(self) -> Optional[Any]:
        return self._viewer_bundle.rerun_viewer if self._viewer_bundle else None

    @property
    def _training_geometry(self) -> bool:
        """True when at least one requested finetune group affects geometry."""
        requested = set(self.semantics_cfg.finetune_params or [])
        return bool(requested & self.model.geometry_param_group_names)

    @property
    def _training_semantics(self) -> bool:
        return "semantics" in (self.semantics_cfg.finetune_params or ["semantics"])

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_provider(self) -> None:
        feature_dir: Optional[Path] = None
        if self.semantics_cfg.semantic_provider == "dino_encoded":
            feature_dir = self.semantics_cfg.dino_encoded_dir
        elif self.semantics_cfg.semantics_path is not None:
            feature_dir = self.semantics_cfg.semantics_path

        self._provider = build_semantic_provider(
            provider_name=self.semantics_cfg.semantic_provider,
            semantics_dim=self.model.provider_semantics_dim,
            cache_enabled=self.semantics_cfg.semantic_cache_enabled,
            semantic_model_path=self.semantics_cfg.semantic_model_path,
            feature_dir=feature_dir,
        )

    def _setup_dataloader(self) -> DataLoader:
        dl = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn,
        )
        self._dataloader_iter = iter(dl)
        return dl

    def _setup_optimizer(self) -> None:
        groups = self.model.get_finetune_param_groups()
        seen: set = set()
        params = []
        for group_name in (self.semantics_cfg.finetune_params or ["semantics"]):
            for p in groups.get(group_name, []):
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        if not params:
            raise RuntimeError(
                f"No parameters found for finetune_params={self.semantics_cfg.finetune_params}. "
                f"Available groups: {list(groups.keys())}"
            )
        lr = self.lr_semantics if self.lr_semantics is not None else self.lr_cfg.lr_sh
        self._optimizer = torch.optim.Adam(params, lr=lr)

    def _setup_full_render(self) -> None:
        self._rasterizer = Rasterizer(self.model, self.sh_cfg, packed=False, absgrad=False)
        self._loss_computer = losses.LossComputer(
            self.train_cfg, self.depth_cfg, self.floater_cfg,
            self.device, self.verbosity, self.scene_extent, self.logger,
        )

    def _setup_viewer(self) -> None:
        if self.logger is None:
            raise RuntimeError("Logger must be initialized before viewer setup")
        self._viewer_bundle = setup_training_viewer(
            model=self.model,
            device=self.device,
            viewer_cfg=self.viewer_cfg,
            verbosity=self.verbosity,
            logger=self.logger,
        )

    # ── Training step ─────────────────────────────────────────────────────────

    def _next_batch(self) -> dict:
        try:
            return next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self._dataloader)
            return next(self._dataloader_iter)

    def _train_step(self, batch: dict, local_step: int) -> Tuple[dict, int, Optional[Any], Optional[torch.Tensor]]:
        assert self._provider is not None
        assert self._optimizer is not None

        cam = batch["cam"]
        gt_image = batch["gt_image"].to(self.device)
        if gt_image.dim() == 3:
            gt_image = gt_image.unsqueeze(0)
        depth_tensor = batch.get("depth_tensor")
        semantic_tensor = batch["semantic_tensor"]
        image_id = batch["image_id"]

        render_out_for_log: Optional[Any] = None
        gt_bhwc_for_log: Optional[torch.Tensor] = None

        with torch.cuda.amp.autocast(enabled=self.train_cfg.use_low_vram):
            total_loss = torch.tensor(0.0, device=self.device)
            metrics: dict = {}

            # ── Semantic MSE loss ──────────────────────────────────────────
            if self._training_semantics:
                _rgb, sem_pred = self.model.render_semantics(
                    cam, device=self.device, detach_geometry=not self._training_geometry
                )
                raw_target = self._provider.get_target(
                    image_id=image_id,
                    gt_image=gt_image[0].detach(),
                    dataset_semantic_tensor=semantic_tensor,
                    device=self.device,
                )
                sem_target = SemanticTarget(
                    features=self.model.prepare_target(raw_target.features, self.device),
                    mode=raw_target.mode,
                    scale_factor=raw_target.scale_factor,
                )
                sem_loss, sem_metrics = self._sem_loss_computer.compute(
                    sem_pred, sem_target, weight=self.semantics_cfg.semantic_loss_weight
                )
                total_loss = total_loss + sem_loss
                metrics.update(sem_metrics)

            # ── Photometric + depth + regularization losses ────────────────
            if self._training_geometry and self._rasterizer is not None and self._loss_computer is not None:
                target_h, target_w = int(cam["height"]), int(cam["width"])
                gt_bhwc = _prepare_gt_image(gt_image[0].detach(), self.device, target_h, target_w)
                depth = _prepare_depth_tensor(depth_tensor, self.device, target_h, target_w)
                render_out = self._rasterizer.render(cam, gt_bhwc, self.device)
                loss_result = self._loss_computer.compute(
                    render_out, gt_bhwc, depth, self.model, cam, local_step
                )
                total_loss = total_loss + loss_result.total_loss
                metrics.update(loss_result.metrics)
                render_out_for_log = render_out
                gt_bhwc_for_log = gt_bhwc

        if not total_loss.isfinite():
            raise RuntimeError(f"Loss is non-finite: {total_loss.item():.6f}")

        self._optimizer.zero_grad(set_to_none=True)
        if self.train_cfg.use_low_vram:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self._optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self._optimizer.step()

        metrics["total_loss"] = float(total_loss.item())
        num_rays = cam["width"] * cam["height"]
        return metrics, num_rays, render_out_for_log, gt_bhwc_for_log

    def _log_step(
        self,
        local_step: int,
        metrics: dict,
        render_out: Optional[Any] = None,
        gt_bhwc: Optional[torch.Tensor] = None,
    ) -> None:
        global_step = self.global_step_offset + local_step
        if self.tb_logger is not None and self.tb_logger.enabled:
            if local_step % self.train_cfg.log_interval == 0:
                self.tb_logger.log_losses(
                    total_loss=metrics["total_loss"],
                    l1_loss=metrics.get("l1_loss", 0.0),
                    ssim_loss=metrics.get("ssim_loss", 0.0),
                    lpips_loss=metrics.get("lpips_loss", 0.0),
                    semantic_loss=metrics.get("semantic_loss", 0.0),
                    step=global_step,
                )
                self.tb_logger.log_model_stats(
                    num_gaussians=len(self.model.means),
                    step=global_step,
                )
                self.tb_logger.log_system_metrics(step=global_step)
                self.tb_logger.flush()

            tb_image_interval = (
                self.tensorboard_cfg.tb_image_interval if self.tensorboard_cfg is not None else 500
            )
            if render_out is not None and gt_bhwc is not None and local_step % tb_image_interval == 0:
                self.tb_logger.log_images(
                    rendered=render_out.render[0].detach(),
                    ground_truth=gt_bhwc[0].detach(),
                    alpha=(
                        render_out.alpha[0].detach()
                        if render_out.alpha.dim() >= 3
                        else render_out.alpha.detach()
                    ),
                    rendered_depth_map=render_out.depth_map[0].detach(),
                    step=global_step,
                )

            tb_histogram_interval = (
                self.tensorboard_cfg.tb_histogram_interval if self.tensorboard_cfg is not None else 1000
            )
            if local_step > 0 and local_step % tb_histogram_interval == 0:
                self.tb_logger.log_gaussian_histograms(self.model, step=global_step)

        if self.logger is not None and local_step % self.train_cfg.log_interval == 0:
            self.logger.info(
                "step=%d/%d  loss=%.6f",
                local_step + 1,
                self.semantics_cfg.semantic_finetune_iters,
                metrics["total_loss"],
            )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the fine-tuning loop.

        Returns dict with ``semantic_avg_loss`` and ``semantic_iters``.
        """
        self.model.setup_semantic_training(self.semantics_cfg, self.device)
        self._setup_provider()
        self._dataloader = self._setup_dataloader()
        self._setup_optimizer()
        if self._training_geometry:
            self._setup_full_render()
        self._setup_viewer()

        if self.verbosity >= 1:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                expand=False,
            )
            self.progress.start()
            self.task_id = self.progress.add_task(
                "[cyan]Fine-tuning...",
                total=self.semantics_cfg.semantic_finetune_iters,
            )

        groups = self.model.get_finetune_param_groups()
        all_trainable = [
            p for g in (self.semantics_cfg.finetune_params or ["semantics"])
            for p in groups.get(g, [])
        ]
        grad_state = self.model.freeze_params_except(all_trainable)

        running_loss = 0.0
        try:
            for local_step in range(self.semantics_cfg.semantic_finetune_iters):
                step_start_time = time.perf_counter()
                batch = self._next_batch()
                metrics, num_rays, render_out, gt_bhwc = self._train_step(batch, local_step)
                running_loss += metrics["total_loss"]
                self._log_step(local_step, metrics, render_out, gt_bhwc)

                if self.progress is not None and self.task_id is not None:
                    self.progress.update(
                        self.task_id, advance=1,
                        description=f"[cyan]Fine-tuning... loss={metrics['total_loss']:.6f}",
                    )

                if self.viewer is not None:
                    step_time = time.perf_counter() - step_start_time
                    self.viewer.render_tab_state.num_train_rays_per_sec = (
                        num_rays / step_time if step_time > 0 else 0
                    )
                    self.viewer.update(local_step, num_rays)
                if self.rerun_viewer is not None:
                    self.rerun_viewer.update(local_step)
        finally:
            self.model.restore_grad_state(grad_state)
            if self.progress is not None:
                self.progress.stop()

        avg_loss = running_loss / max(1, self.semantics_cfg.semantic_finetune_iters)
        return {
            "semantic_avg_loss": avg_loss,
            "semantic_iters": self.semantics_cfg.semantic_finetune_iters,
        }


# ── Public dispatch API (used by train.py's joint training path) ──────────────


def train_semantics(
    model: BaseTrainableModel,
    dataset: SemanticColmapDataset,
    train_cfg: TrainingConfig,
    semantics_cfg: SemanticsConfig,
    lr_cfg: LearningRateConfig,
    lr_semantics: Optional[float],
    depth_cfg: DepthConfig,
    floater_cfg: FloaterPreventionConfig,
    sh_cfg: SHConfig,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    tb_logger: Optional[GaussianSplattingLogger] = None,
    global_step_offset: int = 0,
    verbosity: int = 1,
    viewer_cfg: Optional[Any] = None,
    scene_extent: float = 1.0,
    tensorboard_cfg: Optional[TensorBoardConfig] = None,
) -> dict:
    """Run post-training semantic / language feature fine-tuning.

    The model must implement SemanticsMixin.
    """
    if not isinstance(model, SemanticsMixin):
        raise TypeError(
            f"train_semantics requires a SemanticsMixin model, got {type(model).__name__}."
        )
    trainer = SemanticTrainer(
        model=model,
        dataset=dataset,
        semantics_cfg=semantics_cfg,
        lr_cfg=lr_cfg,
        lr_semantics=lr_semantics,
        train_cfg=train_cfg,
        depth_cfg=depth_cfg,
        floater_cfg=floater_cfg,
        sh_cfg=sh_cfg,
        device=device,
        logger=logger,
        tb_logger=tb_logger,
        global_step_offset=global_step_offset,
        verbosity=verbosity,
        viewer_cfg=viewer_cfg,
        scene_extent=scene_extent,
        tensorboard_cfg=tensorboard_cfg,
    )
    return trainer.run()


# ── Standalone runner (CLI helpers + entry point) ─────────────────────────────


def _setup_logger(output_dir: Path, verbosity: int = 1) -> logging.Logger:
    return configure_app_logger(
        verbosity=verbosity,
        output_dir=output_dir,
        log_filename="semantic_training.log",
        logger_name="cityscape_gs.train_semantics",
    )


def _load_model(
    cfg: SemanticStandaloneConfig,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[SemanticScaffoldModel, dict]:
    """Load and return (SemanticScaffoldModel, checkpoint_dict)."""
    logger.info("Loading checkpoint: %s", cfg.checkpoint_path)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", {})

    auto_type = "scaffold" if "_anchor" in state_dict else "gaussian"
    if cfg.model_type == "auto":
        model_type = auto_type
        logger.info("Auto-detected model type: %s", model_type)
    else:
        model_type = cfg.model_type
        if model_type != auto_type:
            logger.warning(
                "Requested --model-type=%s but checkpoint looks like %s. Proceeding.",
                model_type, auto_type,
            )

    if model_type != "scaffold":
        raise NotImplementedError(
            "Only scaffold models are supported in this semantic training script."
        )

    is_semantic_ckpt = "lang_feat_dim" in checkpoint

    if is_semantic_ckpt:
        logger.info("Detected existing semantic checkpoint — resuming.")
        model = SemanticScaffoldModel.from_checkpoint(
            checkpoint_path=cfg.checkpoint_path, device=device,
        )
    else:
        logger.info(
            "Geometry-only checkpoint — adding lang features (lang_feat_dim=%d, semantics_dim=%d).",
            cfg.lang_feat_dim, cfg.semantics.semantics_dim,
        )
        model = SemanticScaffoldModel.from_pretrained_base(
            checkpoint_path=cfg.checkpoint_path, device=device,
            lang_feat_dim=cfg.lang_feat_dim, semantics_dim=cfg.semantics.semantics_dim,
            **asdict(cfg.model.scaffold),
        )

    model.to(device)
    return model, checkpoint


def _build_datasets(
    cfg: SemanticStandaloneConfig, device: torch.device
) -> SemanticColmapDataset:
    r = cfg.required
    if r.dataset_type == "matrixcity":
        base_dataset = MatrixCityDataset(
            matrixcity_paths=r.matrixcity_paths,
            matrixcity_depth_paths=r.matrixcity_depth_paths or None,
            matrixcity_max_init_points=r.matrixcity_max_init_points,
            device=device,
            image_scale=r.scale,
        )
    else:
        base_dataset = ColmapDataset(
            r.colmap_path,
            r.images_path,
            device=device,
            image_scale=r.scale,
            scene_extent_margin=1.5,
        )

    if cfg.training.preload:
        base_dataset.preload_all_data()

    return SemanticColmapDataset(
        base_dataset=base_dataset,
        semantics_path=cfg.semantics.semantics_path,
        semantics_resolution=cfg.semantics.semantic_image_resolution,
    )


def _build_train_cfg(cfg: SemanticStandaloneConfig) -> TrainingConfig:
    """Strip geometry-only fields from TrainingConfig for the semantic loop."""
    return dataclasses.replace(
        cfg.training,
        iterations=0,
        save_interval=0,
        enable_lpips_loss=False,
        lpips_loss_weight=0.0,
    )


def _save_checkpoint(
    model: SemanticScaffoldModel,
    base_checkpoint: dict,
    metrics: dict,
    cfg: SemanticStandaloneConfig,
    tb_logger: GaussianSplattingLogger,
) -> Path:
    output_ckpt = cfg.output_dir / "semantic_model_final.pt"
    torch.save(
        {
            "iteration": int(base_checkpoint.get("iteration", 0)) + metrics["semantic_iters"],
            "model_state_dict": model.state_dict(),
            "language_mlp": model.mlp_language.state_dict(),
            "lang_feat_dim": model.lang_feat_dim,
            "base_checkpoint": str(cfg.checkpoint_path),
            "semantic_metrics": metrics,
            "tensorboard_run_name": (
                tb_logger.run_name if cfg.tensorboard.tensorboard else None
            ),
        },
        output_ckpt,
    )
    return output_ckpt


def main() -> None:
    cfg = parse_semantic_args()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(cfg.output_dir, verbosity=cfg.runtime.verbosity)

    tb_logger = GaussianSplattingLogger(
        log_dir=str(cfg.output_dir / "tensorboard"),
        run_name=f"semantic_{cfg.output_dir.name}",
        enabled=cfg.tensorboard.tensorboard,
    )
    if cfg.tensorboard.tensorboard:
        logger.info(
            "[green]📊 TensorBoard:[/green] Logging to run [cyan]%s[/cyan]",
            tb_logger.run_name,
        )
        logger.info("[dim]Run: tensorboard --logdir=%s[/dim]", cfg.output_dir / "tensorboard")
        logger.info("[dim]Run directory: %s[/dim]", tb_logger.log_dir)

    model, base_checkpoint = _load_model(cfg, device, logger)
    semantic_dataset = _build_datasets(cfg, device)
    train_cfg = _build_train_cfg(cfg)

    logger.info(
        "Starting fine-tuning: %d iters, provider=%s, dim=%d, groups=%s",
        cfg.semantics.semantic_finetune_iters,
        cfg.semantics.semantic_provider,
        cfg.lang_feat_dim,
        cfg.semantics.finetune_params,
    )

    metrics = train_semantics(
        model=model,
        dataset=semantic_dataset,
        train_cfg=train_cfg,
        semantics_cfg=cfg.semantics,
        lr_cfg=cfg.learning_rates,
        lr_semantics=cfg.lr_semantics,
        depth_cfg=cfg.depth,
        floater_cfg=cfg.floater_prevention,
        sh_cfg=cfg.sh,
        device=device,
        logger=logger,
        tb_logger=tb_logger,
        verbosity=cfg.runtime.verbosity,
        viewer_cfg=cfg.viewer,
        tensorboard_cfg=cfg.tensorboard,
    )

    output_ckpt = _save_checkpoint(model, base_checkpoint, metrics, cfg, tb_logger)
    logger.info("Semantic checkpoint saved to %s", output_ckpt)
    logger.info("Average loss: %.6f", metrics["semantic_avg_loss"])


if __name__ == "__main__":
    main()
