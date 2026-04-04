"""Factory for creating trainable model instances based on configuration."""

from pathlib import Path
from typing import Optional

import torch
from models import BaseTrainableModel, GaussianModel, ScaffoldModel


class ModelFactory:
    """Encapsulates model instantiation logic — the single place for type branching."""

    @staticmethod
    def create(
        model_type: str,
        init_points,
        init_colors: Optional[torch.Tensor] = None,
        sh_degree: int = 3,
        console=None,
        **scaffold_kwargs,
    ) -> BaseTrainableModel | ScaffoldModel | GaussianModel:
        """Create a model instance based on model_type string.

        Args:
            model_type: "gaussian" or "scaffold"
            init_points: Initial point cloud
            init_colors: Initial colors (GaussianModel only)
            sh_degree: Spherical harmonic degree
            console: Logger console
            **scaffold_kwargs: Additional args for ScaffoldModel (feat_dim, n_offsets, etc.)

        Returns:
            Instantiated BaseTrainableModel subclass.
        """
        if model_type == "scaffold":
            return ScaffoldModel(
                init_points=init_points,
                sh_degree=sh_degree,
                console=console,
                **scaffold_kwargs,
            )
        elif model_type == "gaussian":
            return GaussianModel(
                init_points=init_points,
                init_colors=init_colors,
                sh_degree=sh_degree,
                console=console,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported: 'gaussian', 'scaffold'")

    @staticmethod
    def resume(
        model_type: str,
        checkpoint_path: Path,
        device: torch.device,
        sh_degree: int = 3,
        **scaffold_kwargs,
    ) -> tuple[BaseTrainableModel | ScaffoldModel | GaussianModel, dict]:
        """Resume a model from checkpoint.

        Args:
            model_type: "gaussian" or "scaffold"
            checkpoint_path: Path to checkpoint file
            device: Device to load on
            sh_degree: Spherical harmonic degree
            **scaffold_kwargs: Additional args for ScaffoldModel initialization

        Returns:
            (model, checkpoint_dict)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if model_type == "scaffold":
            model = ScaffoldModel(init_points=None, sh_degree=sh_degree, **scaffold_kwargs)
            model.to(device)
            state_dict = checkpoint["model_state_dict"]

            # Reconstruct appearance embedding if it was saved in the checkpoint.
            # set_appearance is called during training after the dataset is known;
            # for a bare resume we derive the camera count directly from the weight shape.
            if "embedding_appearance.embedding.weight" in state_dict:
                num_cameras = state_dict["embedding_appearance.embedding.weight"].shape[0]
                model.set_appearance(num_cameras)
                if model.embedding_appearance is not None:
                    model.embedding_appearance.to(device)

            # ScaffoldModel initialises all geometric parameters and buffers as
            # torch.empty(0).  load_state_dict uses copy_() which enforces matching
            # shapes, so we must directly replace storage for every entry in the
            # state dict before delegating to load_state_dict.
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in state_dict:
                        param.data = state_dict[name].to(device)
                for name, buf in model.named_buffers():
                    if name in state_dict:
                        buf.data = state_dict[name].to(device)

            model.load_state_dict(state_dict, strict=False)

            # Sync lod_offsets now that anchor data is populated.
            if model._anchor.numel() > 0:
                model.lod_offsets = [model._anchor.shape[0]]
        elif model_type == "gaussian":
            model, _ = GaussianModel.resume_from_checkpoint(
                checkpoint_path=checkpoint_path,
                device=str(device),
                sh_degree=sh_degree,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model, checkpoint
