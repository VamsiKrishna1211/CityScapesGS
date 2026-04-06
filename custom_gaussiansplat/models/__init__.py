"""Model implementations for trainable Gaussian Splatting.

Provides:
- BaseTrainableModel: Abstract base class with unified contract
- NeuralRenderingMixin: Capability mixin for dynamic neural Gaussian generation
- GaussianModel: Standard 3DGS implementation
- ScaffoldModel: Scaffold-GS with neural MLPs for Gaussian generation
"""

from .base import BaseTrainableModel, NeuralRenderingMixin, SemanticsMixin
from .gaussian import GaussianModel
from .scaffold import ScaffoldModel

__all__ = ["BaseTrainableModel", "NeuralRenderingMixin", "SemanticsMixin", "GaussianModel", "ScaffoldModel"]
