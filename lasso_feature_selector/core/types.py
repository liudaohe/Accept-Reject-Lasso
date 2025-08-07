"""
Type definitions for the Lasso Feature Selector package.

This module defines common types used throughout the package to ensure
type safety and improve code documentation.
"""

from typing import Set, Dict, Any, List, Tuple, Union, Optional, Callable
import pandas as pd
import numpy as np

# Basic types
FeatureSet = Set[str]
Metadata = Dict[str, Any]
FeatureList = List[str]
ProblemGroups = List[List[str]]

# Selection result types
SelectionResult = Tuple[FeatureSet, Metadata]
RescueResult = FeatureSet

# Data types
DataFrame = pd.DataFrame
Series = pd.Series
Array = np.ndarray

# Configuration types
ConfigDict = Dict[str, Any]
HyperParams = Dict[str, Union[int, float, str, bool]]

# Function signatures
GlobalLassoProvider = Callable[[DataFrame, Series], SelectionResult]
SubsetLassoProvider = Callable[[DataFrame, Series, Metadata], FeatureSet]

# Evaluation types
EvaluationMetrics = Dict[str, Union[int, float]]
TrueFeatureParams = Dict[str, Any]

# Clustering types
ClusterLabels = np.ndarray
SilhouetteScores = np.ndarray

# Rescue types
CoOccurrenceCounter = Dict[frozenset, int]
BasisGroups = List[List[str]]

__all__ = [
    # Basic types
    "FeatureSet",
    "Metadata", 
    "FeatureList",
    "ProblemGroups",
    
    # Result types
    "SelectionResult",
    "RescueResult",
    
    # Data types
    "DataFrame",
    "Series", 
    "Array",
    
    # Configuration types
    "ConfigDict",
    "HyperParams",
    
    # Function signatures
    "GlobalLassoProvider",
    "SubsetLassoProvider",
    
    # Evaluation types
    "EvaluationMetrics",
    "TrueFeatureParams",
    
    # Clustering types
    "ClusterLabels",
    "SilhouetteScores",
    
    # Rescue types
    "CoOccurrenceCounter",
    "BasisGroups",
]
