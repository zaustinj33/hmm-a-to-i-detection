# File: hmm_variant_detector/__init__.py
"""
HMM Variant Detector Package

A Hidden Markov Model implementation for detecting A->G variants in NGS data.
"""

from .hmm import VariantHMM
from .utils import generate_synthetic_data, evaluate_predictions

__version__ = "0.1.0"
__all__ = ["VariantHMM", "generate_synthetic_data", "evaluate_predictions"]