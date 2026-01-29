"""
Causal Inference Module
=======================

Implements rigorous causal inference methods to separate
correlation from causation and attribute policy impacts.
"""

from .engine import CausalInferenceEngine
from .methods import (
    DifferenceInDifferences,
    SyntheticControl,
    RegressionDiscontinuity,
    InstrumentalVariables,
    PropensityScoreMatching,
)
from .robustness import RobustnessChecker

__all__ = [
    "CausalInferenceEngine",
    "DifferenceInDifferences",
    "SyntheticControl",
    "RegressionDiscontinuity",
    "InstrumentalVariables",
    "PropensityScoreMatching",
    "RobustnessChecker",
]
