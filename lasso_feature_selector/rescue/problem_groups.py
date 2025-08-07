"""
Problem group identification for feature rescue.

This module identifies groups of highly correlated features that may
cause issues in feature selection due to multicollinearity.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set
from ..core.base import BaseRescueMethod
from ..core.types import DataFrame, ProblemGroups
from ..config.defaults import RESCUE_DEFAULTS

class ProblemGroupIdentifier(BaseRescueMethod):
    """
    Identifies groups of highly correlated features (problem groups).
    
    Problem groups are sets of features with high pairwise correlations
    that may interfere with stable feature selection.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize problem group identifier.
        
        Args:
            **kwargs: Configuration overrides
        """
        config = {**RESCUE_DEFAULTS, **kwargs}
        super().__init__(**config)
    
    @property
    def name(self) -> str:
        """Return the method name."""
        return "Problem Group Identifier"
    
    def identify_problem_groups(self, X: DataFrame, **kwargs) -> ProblemGroups:
        """
        Identify groups of highly correlated features.
        
        Args:
            X: Feature matrix
            **kwargs: Additional parameters
            
        Returns:
            List of problem groups (each group is a list of feature names)
        """
        self._validate_input(X)
        
        correlation_threshold = kwargs.get('correlation_threshold', self.correlation_threshold)
        
        if self.parameters.get('verbose', True):
            print(f"\\n🔍 算法洞察 (步骤2): 正在识别高度相关的问题组 (阈值: {correlation_threshold})...")
        
        # Build correlation graph
        adj = self._build_correlation_graph(X)
        
        # Find connected components (problem groups)
        problem_groups = self._find_connected_components(adj)
        
        if self.parameters.get('verbose', True):
            print(f"  ✅ 共识别出 {len(problem_groups)} 个问题组。")
        
        return problem_groups
    
    def rescue_features(self, X: DataFrame, y: pd.Series, base_features: Set[str], 
                       problem_groups: ProblemGroups, **kwargs) -> Set[str]:
        """
        This method doesn't directly rescue features but provides problem groups
        for other rescue methods to use.
        
        Args:
            X: Feature matrix
            y: Target vector
            base_features: Initially selected features
            problem_groups: Identified problem groups
            **kwargs: Additional parameters
            
        Returns:
            Empty set (this method only identifies, doesn't rescue)
        """
        return set()
    
    def get_group_statistics(self, X: DataFrame, problem_groups: ProblemGroups) -> Dict:
        """
        Calculate statistics for identified problem groups.
        
        Args:
            X: Feature matrix
            problem_groups: List of problem groups
            
        Returns:
            Dictionary with group statistics
        """
        stats = {
            'num_groups': len(problem_groups),
            'group_sizes': [len(group) for group in problem_groups],
            'total_features_in_groups': sum(len(group) for group in problem_groups),
            'largest_group_size': max(len(group) for group in problem_groups) if problem_groups else 0,
            'average_group_size': sum(len(group) for group in problem_groups) / len(problem_groups) if problem_groups else 0
        }
        
        # Calculate average within-group correlation for each group
        group_correlations = []
        for group in problem_groups:
            if len(group) > 1:
                group_corr_matrix = X[group].corr().abs()
                # Get upper triangle (excluding diagonal)
                upper_triangle = group_corr_matrix.where(
                    np.triu(np.ones(group_corr_matrix.shape), k=1).astype(bool)
                )
                avg_corr = upper_triangle.stack().mean()
                group_correlations.append(avg_corr)
        
        if group_correlations:
            stats['average_within_group_correlation'] = sum(group_correlations) / len(group_correlations)
            stats['max_within_group_correlation'] = max(group_correlations)
            stats['min_within_group_correlation'] = min(group_correlations)
        else:
            stats['average_within_group_correlation'] = 0
            stats['max_within_group_correlation'] = 0
            stats['min_within_group_correlation'] = 0
        
        return stats
    
    def visualize_groups(self, X: DataFrame, problem_groups: ProblemGroups, max_groups: int = 10):
        """
        Create a visualization of problem groups (correlation heatmaps).
        
        Args:
            X: Feature matrix
            problem_groups: List of problem groups
            max_groups: Maximum number of groups to visualize
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Sort groups by size (largest first)
            sorted_groups = sorted(problem_groups, key=len, reverse=True)
            groups_to_plot = sorted_groups[:max_groups]
            
            n_groups = len(groups_to_plot)
            if n_groups == 0:
                print("No problem groups to visualize.")
                return
            
            # Create subplots
            fig, axes = plt.subplots(1, min(n_groups, 4), figsize=(4*min(n_groups, 4), 4))
            if n_groups == 1:
                axes = [axes]
            
            for i, group in enumerate(groups_to_plot[:4]):  # Limit to 4 plots
                if len(group) > 1:
                    corr_matrix = X[group].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                              square=True, ax=axes[i] if n_groups > 1 else axes[0])
                    axes[i].set_title(f'Group {i+1} (size: {len(group)})')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].tick_params(axis='y', rotation=0)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib and seaborn are required for visualization.")
            print("Install with: pip install matplotlib seaborn")
    
    def export_groups(self, problem_groups: ProblemGroups, filepath: str):
        """
        Export problem groups to a file.
        
        Args:
            problem_groups: List of problem groups
            filepath: Output file path
        """
        import json
        
        # Convert to serializable format
        groups_data = {
            'num_groups': len(problem_groups),
            'groups': [{'id': i, 'features': group, 'size': len(group)} 
                      for i, group in enumerate(problem_groups)]
        }
        
        with open(filepath, 'w') as f:
            json.dump(groups_data, f, indent=2)
        
        print(f"Problem groups exported to {filepath}")
