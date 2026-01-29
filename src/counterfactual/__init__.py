"""
Counterfactual Simulation Module
================================

Answers "what if" questions by simulating alternative scenarios.
"""

from .simulator import CounterfactualSimulator
from .scenarios import (
    ScenarioBuilder,
    GlobalConditionScenario,
    PolicyScenario,
    DemographicScenario,
)

__all__ = [
    "CounterfactualSimulator",
    "ScenarioBuilder",
    "GlobalConditionScenario",
    "PolicyScenario",
    "DemographicScenario",
]
