"""
Configuration management for Lasso Feature Selector.

This module provides centralized configuration management with support for
default settings, parameter validation, and configuration file loading.
"""

from .settings import LassoConfig
from .defaults import DEFAULT_CONFIGS, HYPERPARAMETER_RANGES

__all__ = [
    "LassoConfig",
    "DEFAULT_CONFIGS", 
    "HYPERPARAMETER_RANGES",
]
