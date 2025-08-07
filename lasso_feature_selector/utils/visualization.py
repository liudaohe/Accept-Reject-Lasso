"""
Visualization utilities for feature selection results.

This module provides plotting and visualization tools for analyzing
feature selection performance and results.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Set
import warnings

from ..core.types import FeatureSet

def plot_selection_results(method_results: Dict[str, FeatureSet], 
                          evaluation_results: Optional[pd.DataFrame] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Create comprehensive plots for feature selection results.
    
    Args:
        method_results: Dictionary mapping method names to selected features
        evaluation_results: DataFrame with evaluation metrics
        save_path: Optional path to save the plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib and seaborn are required for visualization.")
        print("Install with: pip install matplotlib seaborn")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Number of selected features by method
    ax1 = plt.subplot(2, 3, 1)
    methods = list(method_results.keys())
    feature_counts = [len(method_results[method]) for method in methods]
    
    bars = ax1.bar(methods, feature_counts)
    ax1.set_title('Number of Selected Features by Method')
    ax1.set_ylabel('Number of Features')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, feature_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    # Plot 2: Performance metrics (if evaluation results provided)
    if evaluation_results is not None and 'F1-Score' in evaluation_results.columns:
        ax2 = plt.subplot(2, 3, 2)
        metrics = ['Precision', 'Recall', 'F1-Score']
        x = np.arange(len(evaluation_results))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            if metric in evaluation_results.columns:
                ax2.bar(x + i*width, evaluation_results[metric], width, label=metric)
        
        ax2.set_title('Performance Metrics by Method')
        ax2.set_ylabel('Score')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(evaluation_results['method'], rotation=45)
        ax2.legend()
        ax2.set_ylim(0, 1)
    
    # Plot 3: Feature overlap heatmap
    ax3 = plt.subplot(2, 3, 3)
    n_methods = len(methods)
    overlap_matrix = np.zeros((n_methods, n_methods))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                set1, set2 = method_results[method1], method_results[method2]
                union = len(set1 | set2)
                jaccard = len(set1 & set2) / union if union > 0 else 0
                overlap_matrix[i, j] = jaccard
    
    sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=methods, yticklabels=methods, ax=ax3)
    ax3.set_title('Method Similarity (Jaccard Index)')
    
    # Plot 4: Feature frequency across methods
    ax4 = plt.subplot(2, 3, 4)
    all_features = set.union(*method_results.values()) if method_results else set()
    feature_counts = {}
    for feature in all_features:
        count = sum(1 for features in method_results.values() if feature in features)
        feature_counts[feature] = count
    
    # Show top 20 most frequently selected features
    top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    if top_features:
        features, counts = zip(*top_features)
        ax4.barh(range(len(features)), counts)
        ax4.set_yticks(range(len(features)))
        ax4.set_yticklabels(features)
        ax4.set_xlabel('Selection Frequency')
        ax4.set_title('Top 20 Most Selected Features')
        ax4.invert_yaxis()
    
    # Plot 5: Venn diagram for up to 3 methods
    if len(methods) <= 3:
        ax5 = plt.subplot(2, 3, 5)
        try:
            from matplotlib_venn import venn2, venn3
            
            if len(methods) == 2:
                venn2([method_results[methods[0]], method_results[methods[1]]], 
                      set_labels=methods, ax=ax5)
            elif len(methods) == 3:
                venn3([method_results[methods[0]], method_results[methods[1]], 
                       method_results[methods[2]]], set_labels=methods, ax=ax5)
            
            ax5.set_title('Feature Selection Overlap')
        except ImportError:
            ax5.text(0.5, 0.5, 'matplotlib-venn required\nfor Venn diagrams', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Feature Selection Overlap (Venn)')
    
    # Plot 6: Performance vs. Selection Size scatter
    if evaluation_results is not None and 'F1-Score' in evaluation_results.columns:
        ax6 = plt.subplot(2, 3, 6)
        sizes = [len(method_results[method]) for method in evaluation_results['method']]
        f1_scores = evaluation_results['F1-Score']
        
        ax6.scatter(sizes, f1_scores, s=100, alpha=0.7)
        for i, method in enumerate(evaluation_results['method']):
            ax6.annotate(method, (sizes[i], f1_scores[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax6.set_xlabel('Number of Selected Features')
        ax6.set_ylabel('F1-Score')
        ax6.set_title('Performance vs. Selection Size')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    
    plt.show()

class SelectionVisualizer:
    """
    Advanced visualization class for feature selection analysis.
    """
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required visualization libraries are available."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.plt = plt
            self.sns = sns
            self._has_plotting = True
        except ImportError:
            self._has_plotting = False
            warnings.warn("Matplotlib and seaborn are required for visualization.")
    
    def plot_method_comparison(self, evaluation_df: pd.DataFrame, 
                             save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive comparison plot of different methods.
        
        Args:
            evaluation_df: DataFrame with evaluation results
            save_path: Optional path to save the plot
        """
        if not self._has_plotting:
            return
        
        fig, axes = self.plt.subplots(2, 2, figsize=(15, 10))
        
        # Metrics comparison
        metrics = ['Precision', 'Recall', 'F1-Score']
        available_metrics = [m for m in metrics if m in evaluation_df.columns]
        
        if available_metrics:
            ax = axes[0, 0]
            evaluation_df.set_index('method')[available_metrics].plot(kind='bar', ax=ax)
            ax.set_title('Performance Metrics Comparison')
            ax.set_ylabel('Score')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
        
        # Feature count comparison
        if '总选择数' in evaluation_df.columns:
            ax = axes[0, 1]
            ax.bar(evaluation_df['method'], evaluation_df['总选择数'])
            ax.set_title('Number of Selected Features')
            ax.set_ylabel('Feature Count')
            ax.tick_params(axis='x', rotation=45)
        
        # TP/FP/FN breakdown
        if all(col in evaluation_df.columns for col in ['TP', 'FP', 'FN']):
            ax = axes[1, 0]
            x = np.arange(len(evaluation_df))
            width = 0.25
            
            ax.bar(x - width, evaluation_df['TP'], width, label='True Positive', color='green')
            ax.bar(x, evaluation_df['FP'], width, label='False Positive', color='red')
            ax.bar(x + width, evaluation_df['FN'], width, label='False Negative', color='orange')
            
            ax.set_title('Classification Breakdown')
            ax.set_ylabel('Count')
            ax.set_xticks(x)
            ax.set_xticklabels(evaluation_df['method'], rotation=45)
            ax.legend()
        
        # F1-Score ranking
        if 'F1-Score' in evaluation_df.columns:
            ax = axes[1, 1]
            sorted_df = evaluation_df.sort_values('F1-Score', ascending=True)
            colors = self.plt.cm.RdYlGn(sorted_df['F1-Score'])
            ax.barh(range(len(sorted_df)), sorted_df['F1-Score'], color=colors)
            ax.set_yticks(range(len(sorted_df)))
            ax.set_yticklabels(sorted_df['method'])
            ax.set_xlabel('F1-Score')
            ax.set_title('Method Ranking by F1-Score')
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.plt.show()
    
    def plot_stability_analysis(self, stability_results: Dict[str, Any],
                              save_path: Optional[str] = None) -> None:
        """
        Visualize stability analysis results.
        
        Args:
            stability_results: Results from stability analysis
            save_path: Optional path to save the plot
        """
        if not self._has_plotting or not stability_results:
            return
        
        methods = list(stability_results.keys())
        
        fig, axes = self.plt.subplots(1, 2, figsize=(15, 6))
        
        # Stability scores
        ax = axes[0]
        similarities = [stability_results[method]['mean_jaccard_similarity'] for method in methods]
        errors = [stability_results[method]['std_jaccard_similarity'] for method in methods]
        
        bars = ax.bar(methods, similarities, yerr=errors, capsize=5)
        ax.set_title('Method Stability (Jaccard Similarity)')
        ax.set_ylabel('Mean Jaccard Similarity')
        ax.tick_params(axis='x', rotation=45)
        
        # Stable feature counts
        ax = axes[1]
        stable_counts = [stability_results[method]['stable_feature_count'] for method in methods]
        ax.bar(methods, stable_counts)
        ax.set_title('Stable Features Count')
        ax.set_ylabel('Number of Stable Features')
        ax.tick_params(axis='x', rotation=45)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.plt.show()
