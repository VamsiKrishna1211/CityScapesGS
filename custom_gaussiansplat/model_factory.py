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
            model.load_state_dict(checkpoint["model_state_dict"])
        elif model_type == "gaussian":
            model, _ = GaussianModel.resume_from_checkpoint(
                checkpoint_path=checkpoint_path,
                device=str(device),
                sh_degree=sh_degree,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model, checkpoint
