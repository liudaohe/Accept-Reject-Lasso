"""
Feature rescue module for Lasso Feature Selector.

This module contains components for identifying and rescuing features
that may have been missed by the primary selection process due to
high correlations or other complex relationships.
"""

from .problem_groups import ProblemGroupIdentifier
from .clustering import ClusteringAnalyzer
from .co_occurrence import CoOccurrenceRescue

__all__ = [
    "ProblemGroupIdentifier",
    "ClusteringAnalyzer", 
    "CoOccurrenceRescue",
]
