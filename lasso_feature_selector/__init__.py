"""
Lasso Feature Selector: A comprehensive feature selection library with rescue mechanisms.

This package provides various Lasso-based feature selection methods with advanced
rescue mechanisms for handling high-dimensional data with complex correlation structures.

Main Components:
- Multiple Lasso variants (Adaptive, Random, Stability Selection, etc.)
- Feature rescue mechanisms based on problem group analysis
- Unified interface for easy method comparison
- Comprehensive evaluation and visualization tools

Author: [Your Name]
Version: 1.0.0
"""

from .core.interfaces import GlobalLassoProvider, SubsetLassoProvider
from .core.base import BaseLassoMethod, BaseRescueMethod
from .pipeline.orchestrator import LassoFeatureSelector
from .config.settings import LassoConfig

# Import all method implementations
from .methods.adaptive_lasso import AdaptiveLassoMethod
from .methods.random_lasso import RandomLassoMethod
from .methods.stability_selection import StabilitySelectionMethod
from .methods.lasso_cv import LassoCVMethod
from .methods.elastic_net import ElasticNetMethod

# Import rescue components
from .rescue.problem_groups import ProblemGroupIdentifier
from .rescue.clustering import ClusteringAnalyzer
from .rescue.co_occurrence import CoOccurrenceRescue

# Import utilities
from .utils.evaluation import evaluate_selection
from .utils.visualization import plot_selection_results

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    # Core interfaces
    "GlobalLassoProvider",
    "SubsetLassoProvider", 
    "BaseLassoMethod",
    "BaseRescueMethod",
    
    # Main pipeline
    "LassoFeatureSelector",
    "LassoConfig",
    
    # Lasso methods
    "AdaptiveLassoMethod",
    "RandomLassoMethod", 
    "StabilitySelectionMethod",
    "LassoCVMethod",
    "ElasticNetMethod",
    
    # Rescue components
    "ProblemGroupIdentifier",
    "ClusteringAnalyzer",
    "CoOccurrenceRescue",
    
    # Utilities
    "evaluate_selection",
    "plot_selection_results",
]
