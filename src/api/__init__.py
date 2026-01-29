"""
API Module
==========

RESTful and GraphQL APIs for accessing comparison data.
"""

from .rest_api import create_app
from .routes import router

__all__ = [
    "create_app",
    "router",
]
