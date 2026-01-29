"""
India Economic Comparison System
================================

A comprehensive, transparent, and methodologically rigorous system for comparing
India's economic performance across the 2004-2014 and 2014-2024 periods.

Core Modules:
- data_pipeline: Data fetching, cleaning, and validation
- data_skepticism: Cross-validation and reliability scoring
- causal_inference: Policy impact attribution
- counterfactual: What-if scenario simulation
- uncertainty: Confidence interval quantification
- debate_assistant: Real-time argument response
- ethical_framework: Bias detection and balanced presentation
- visualization: Interactive charts and dashboards
- api: RESTful and GraphQL endpoints
"""

__version__ = "1.0.0"
__author__ = "Economic Analysis Team"

from .data_pipeline import DataPipeline
from .data_skepticism import DataSkepticismEngine
from .causal_inference import CausalInferenceEngine
from .counterfactual import CounterfactualSimulator
from .uncertainty import UncertaintyQuantifier
from .debate_assistant import DebateAssistant
from .ethical_framework import EthicalFramework
from .predictive_forensics import PredictiveForensics

__all__ = [
    "DataPipeline",
    "DataSkepticismEngine",
    "CausalInferenceEngine",
    "CounterfactualSimulator",
    "UncertaintyQuantifier",
    "DebateAssistant",
    "EthicalFramework",
    "PredictiveForensics",
]
