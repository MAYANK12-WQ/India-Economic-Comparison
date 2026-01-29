"""
Uncertainty Quantification Module
=================================

Every estimate comes with confidence intervals.
Quantifies and visualizes uncertainty in economic analysis.
"""

from .quantifier import UncertaintyQuantifier
from .methods import (
    BootstrapEstimator,
    BayesianEstimator,
    MonteCarloSimulator,
    SensitivityAnalyzer,
)

__all__ = [
    "UncertaintyQuantifier",
    "BootstrapEstimator",
    "BayesianEstimator",
    "MonteCarloSimulator",
    "SensitivityAnalyzer",
]
