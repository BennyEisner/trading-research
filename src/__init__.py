#!/usr/bin/env python3

"""
Financial Returns API - Source Package
ML-powered trading strategy research and ensemble management
"""

__version__ = "1.0.0"
__author__ = "Financial Returns Research"

# Package-level exports for common imports
from .strategies.base import BaseStrategy
from .strategies.ensemble import EnsembleManager

__all__ = [
    "BaseStrategy",
    "EnsembleManager",
]