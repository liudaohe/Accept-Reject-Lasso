"""
Base classes for the Lasso Feature Selector package.

This module provides concrete base implementations that can be extended
by specific feature selection and rescue methods.
"""

import warnings
from abc import ABC
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from .interfaces import LassoMethodInterface, RescueMethodInterface
from .types import DataFrame, Series, FeatureSet, Metadata, SelectionResult

class BaseLassoMethod(LassoMethodInterface):
    """
    Base implementation for Lasso-based feature selection methods.
    
    This class provides common functionality and default implementations
    that can be shared across different Lasso variants.
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1, **kwargs):
        """
        Initialize base Lasso method.
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
            **kwargs: Additional hyperparameters
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.hyperparameters = kwargs
        self._setup_warnings()
    
    def _setup_warnings(self):
        """Configure warning filters for cleaner output."""
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return current hyperparameter configuration."""
        base_params = {
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
        base_params.update(self.hyperparameters)
        return base_params
    
    def set_hyperparameters(self, **kwargs) -> None:
        """Update hyperparameter configuration."""
        if 'random_state' in kwargs:
            self.random_state = kwargs.pop('random_state')
        if 'n_jobs' in kwargs:
            self.n_jobs = kwargs.pop('n_jobs')
        self.hyperparameters.update(kwargs)
    
    def _validate_input(self, X: DataFrame, y: Series) -> None:
        """Validate input data format and consistency."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        if X.empty:
            raise ValueError("X cannot be empty")
        if y.empty:
            raise ValueError("y cannot be empty")
    
    def _prepare_metadata(self, **kwargs) -> Metadata:
        """Prepare metadata dictionary with common information."""
        metadata = {
            'random_state': self.random_state,
            'method_name': self.name,
        }
        metadata.update(kwargs)
        return metadata

class BaseRescueMethod(RescueMethodInterface):
    """
    Base implementation for feature rescue methods.
    
    This class provides common functionality for identifying and rescuing
    features that may have been missed by primary selection methods.
    """
    
    def __init__(self, 
                 correlation_threshold: float = 0.8,
                 min_group_size: int = 2,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize base rescue method.
        
        Args:
            correlation_threshold: Threshold for identifying high correlations
            min_group_size: Minimum size for problem groups
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        self.correlation_threshold = correlation_threshold
        self.min_group_size = min_group_size
        self.random_state = random_state
        self.parameters = kwargs
    
    def _validate_input(self, X: DataFrame, y: Optional[Series] = None) -> None:
        """Validate input data for rescue operations."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if y is not None and not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
    
    def _build_correlation_graph(self, X: DataFrame) -> Dict[str, set]:
        """
        Build adjacency graph based on feature correlations.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary mapping each feature to its highly correlated neighbors
        """
        corr_matrix = X.corr().abs()
        adj = {node: set() for node in X.columns}
        
        # Find high correlation pairs
        high_corr_pairs = corr_matrix[corr_matrix > self.correlation_threshold].stack().reset_index()
        high_corr_pairs = high_corr_pairs[high_corr_pairs['level_0'] != high_corr_pairs['level_1']]
        
        # Build adjacency list
        for _, row in high_corr_pairs.iterrows():
            adj[row['level_0']].add(row['level_1'])
            adj[row['level_1']].add(row['level_0'])
        
        return adj
    
    def _find_connected_components(self, adj: Dict[str, set]) -> list:
        """
        Find connected components in the correlation graph using BFS.
        
        Args:
            adj: Adjacency list representation of correlation graph
            
        Returns:
            List of connected components (problem groups)
        """
        problem_groups = []
        visited = set()
        
        for node in adj.keys():
            if node not in visited:
                # BFS to find connected component
                current_group = set()
                queue = [node]
                head = 0
                visited.add(node)
                
                while head < len(queue):
                    current_node = queue[head]
                    head += 1
                    current_group.add(current_node)
                    
                    for neighbor in adj.get(current_node, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                # Only keep groups that meet minimum size requirement
                if len(current_group) >= self.min_group_size:
                    problem_groups.append(sorted(list(current_group)))
        
        return problem_groups
