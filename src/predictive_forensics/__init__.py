"""
Predictive Forensics Module
===========================

Tests political claims against actual outcomes.
Validates promises vs delivery.
"""

from .forensics import PredictiveForensics
from .claim_tracker import ClaimTracker
from .outcome_validator import OutcomeValidator

__all__ = [
    "PredictiveForensics",
    "ClaimTracker",
    "OutcomeValidator",
]
