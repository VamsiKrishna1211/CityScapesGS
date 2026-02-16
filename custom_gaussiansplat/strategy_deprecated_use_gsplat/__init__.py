"""
Strategy module for Gaussian Splatting densification and pruning.

This module provides a modular approach to managing Gaussian densification,
following the architecture from the gsplat library. It separates the
densification logic from the model, allowing for different strategies
(e.g., default 3DGS, MCMC) to be easily swapped.
"""

from .base import Strategy, ViewportInfo
from .default import DefaultStrategy

__all__ = ["Strategy", "ViewportInfo", "DefaultStrategy"]
