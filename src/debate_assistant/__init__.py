"""
Debate Assistant Module
=======================

Real-time response generator for economic debates.
Provides instant counter-arguments with data backing.
"""

from .assistant import DebateAssistant
from .argument_library import ArgumentLibrary
from .response_generator import ResponseGenerator

__all__ = [
    "DebateAssistant",
    "ArgumentLibrary",
    "ResponseGenerator",
]
