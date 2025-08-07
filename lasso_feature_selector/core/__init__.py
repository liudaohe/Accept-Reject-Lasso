"""
Core module for Lasso Feature Selector.

This module contains the fundamental interfaces, base classes, and type definitions
that form the foundation of the feature selection framework.
"""

from .interfaces import GlobalLassoProvider, SubsetLassoProvider
from .base import BaseLassoMethod, BaseRescueMethod
from .types import FeatureSet, Metadata, SelectionResult

__all__ = [
    "GlobalLassoProvider",
    "SubsetLassoProvider", 
    "BaseLassoMethod",
    "BaseRescueMethod",
    "FeatureSet",
    "Metadata", 
    "SelectionResult",
]
