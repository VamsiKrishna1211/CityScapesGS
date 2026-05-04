"""Model implementations for trainable Gaussian Splatting.

Provides:
- BaseTrainableModel: Abstract base class with unified contract
- NeuralRenderingMixin: Capability mixin for dynamic neural Gaussian generation
- SemanticsMixin: Capability mixin for semantic / language feature learning
- GaussianModel: Standard 3DGS implementation (visual only)
- ScaffoldModel: Scaffold-GS with neural MLPs for Gaussian generation (visual only)
- SemanticGaussianModel: GaussianModel + per-Gaussian semantic features
- SemanticScaffoldModel: ScaffoldModel + per-anchor language features + CLIP codebook

Visual training (train.py):   uses GaussianModel / ScaffoldModel via ModelFactory.
Semantic training (train_semantics.py): uses SemanticGaussianModel / SemanticScaffoldModel.
"""

from .base import BaseTrainableModel, NeuralRenderingMixin, SemanticsMixin
from .gaussian import GaussianModel
from .scaffold import ScaffoldModel
from .semantic_gaussian import SemanticGaussianModel
from .semantic_scaffold import SemanticScaffoldModel

__all__ = [
    "BaseTrainableModel",
    "NeuralRenderingMixin",
    "SemanticsMixin",
    "GaussianModel",
    "ScaffoldModel",
    "SemanticGaussianModel",
    "SemanticScaffoldModel",
]
