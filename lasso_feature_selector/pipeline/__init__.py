"""
Pipeline module for Lasso Feature Selector.

This module contains the main orchestrator that coordinates all
feature selection and rescue operations.
"""

from .orchestrator import LassoFeatureSelector

__all__ = [
    "LassoFeatureSelector",
]
