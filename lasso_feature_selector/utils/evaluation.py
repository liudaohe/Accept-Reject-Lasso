"""
Evaluation utilities for feature selection methods.

This module provides comprehensive evaluation metrics and tools
for assessing feature selection performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Set, Optional, List
from ..core.types import FeatureSet, EvaluationMetrics, TrueFeatureParams

def evaluate_selection(selected_features: FeatureSet, 
                      true_features_params: TrueFeatureParams) -> EvaluationMetrics:
    """
    Evaluate feature selection performance using precision, recall, and F1-score.
    
    This function implements the evaluation logic from the original code,
    handling different types of true features (determined important, 
    false redundancy, true redundancy groups).
    
    Args:
        selected_features: Set of selected feature names
        true_features_params: Dictionary with true feature configuration
        
    Returns:
        Dictionary with evaluation metrics
    """
    p = true_features_params
    tp, fp, fn = 0, 0, 0
    
    # Define true feature sets
    true_di = {f'DI_{j+1}' for j in range(p['n_determined_important'])}
    true_fr = {f'FR_g{i+1}_v{j+1}' for i in range(p['n_false_redundancy_groups']) 
               for j in range(p['n_vars_per_group'])}
    
    # True redundancy groups
    all_tr_groups = [{f'TR_g{i+1}_v{j+1}' for j in range(p['n_vars_per_group'])} 
                     for i in range(p['n_true_redundancy_groups'])]
    
    # Important non-redundant features
    true_important_non_tr = true_di.union(true_fr)
    
    # All known features
    all_known_features = true_important_non_tr.union(*all_tr_groups)
    
    # Unimportant features
    all_u_features = set(f'U_{i+1}' for i in range(p['p'] - len(all_known_features)))
    
    # Calculate TP and FN for non-redundant important features
    tp += len(selected_features.intersection(true_important_non_tr))
    fn += len(true_important_non_tr - selected_features)
    
    # Calculate FP for unimportant features
    fp += len(selected_features.intersection(all_u_features))
    
    # Handle true redundancy groups (only one representative needed per group)
    for tr_group in all_tr_groups:
        num_selected = len(selected_features.intersection(tr_group))
        if num_selected == 0:
            fn += 1  # Should have selected at least one from this group
        else:
            tp += 1  # Correctly selected from this group
            fp += (num_selected - 1)  # Extra selections are false positives
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "总选择数": len(selected_features),
        "TP": tp,
        "FP": fp, 
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    }

class FeatureEvaluator:
    """
    Comprehensive feature selection evaluator.
    
    This class provides various evaluation methods and comparative
    analysis tools for feature selection results.
    """
    
    def __init__(self, true_features_params: Optional[TrueFeatureParams] = None):
        """
        Initialize evaluator.
        
        Args:
            true_features_params: True feature configuration for synthetic data
        """
        self.true_features_params = true_features_params
    
    def evaluate_single_method(self, selected_features: FeatureSet, 
                             method_name: str = "Unknown") -> Dict[str, Any]:
        """
        Evaluate a single feature selection method.
        
        Args:
            selected_features: Selected features
            method_name: Name of the method
            
        Returns:
            Evaluation results dictionary
        """
        if self.true_features_params is None:
            return {
                "method": method_name,
                "selected_count": len(selected_features),
                "selected_features": list(selected_features),
                "error": "No true features provided for evaluation"
            }
        
        metrics = evaluate_selection(selected_features, self.true_features_params)
        
        return {
            "method": method_name,
            "selected_features": list(selected_features),
            **metrics
        }
    
    def compare_methods(self, method_results: Dict[str, FeatureSet]) -> pd.DataFrame:
        """
        Compare multiple feature selection methods.
        
        Args:
            method_results: Dictionary mapping method names to selected features
            
        Returns:
            DataFrame with comparative results
        """
        if self.true_features_params is None:
            raise ValueError("True features parameters required for comparison")
        
        results = []
        for method_name, selected_features in method_results.items():
            result = self.evaluate_single_method(selected_features, method_name)
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # Sort by F1-Score descending
        if 'F1-Score' in df.columns:
            df = df.sort_values('F1-Score', ascending=False)
        
        return df
    
    def analyze_feature_overlap(self, method_results: Dict[str, FeatureSet]) -> Dict[str, Any]:
        """
        Analyze overlap between different methods' selections.
        
        Args:
            method_results: Dictionary mapping method names to selected features
            
        Returns:
            Dictionary with overlap analysis
        """
        methods = list(method_results.keys())
        n_methods = len(methods)
        
        # Pairwise overlaps
        pairwise_overlaps = {}
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                method1, method2 = methods[i], methods[j]
                set1, set2 = method_results[method1], method_results[method2]
                
                overlap = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = overlap / union if union > 0 else 0
                
                pairwise_overlaps[f"{method1}_vs_{method2}"] = {
                    "overlap_count": overlap,
                    "jaccard_similarity": jaccard,
                    "method1_unique": len(set1 - set2),
                    "method2_unique": len(set2 - set1)
                }
        
        # Features selected by all methods
        all_features = set.intersection(*method_results.values()) if method_results else set()
        
        # Features selected by any method
        any_features = set.union(*method_results.values()) if method_results else set()
        
        # Feature frequency
        feature_counts = {}
        for features in method_results.values():
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        return {
            "pairwise_overlaps": pairwise_overlaps,
            "consensus_features": list(all_features),
            "consensus_count": len(all_features),
            "total_unique_features": len(any_features),
            "feature_selection_frequency": feature_counts,
            "most_selected_features": sorted(feature_counts.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]
        }
    
    def stability_analysis(self, multiple_runs: List[Dict[str, FeatureSet]]) -> Dict[str, Any]:
        """
        Analyze stability of feature selection across multiple runs.
        
        Args:
            multiple_runs: List of dictionaries, each containing method results for one run
            
        Returns:
            Dictionary with stability analysis
        """
        if not multiple_runs:
            return {"error": "No runs provided"}
        
        methods = list(multiple_runs[0].keys())
        stability_results = {}
        
        for method in methods:
            method_selections = [run[method] for run in multiple_runs if method in run]
            
            if not method_selections:
                continue
            
            # Calculate pairwise Jaccard similarities
            n_runs = len(method_selections)
            similarities = []
            
            for i in range(n_runs):
                for j in range(i + 1, n_runs):
                    set1, set2 = method_selections[i], method_selections[j]
                    union = len(set1 | set2)
                    jaccard = len(set1 & set2) / union if union > 0 else 0
                    similarities.append(jaccard)
            
            # Features selected in all runs
            stable_features = set.intersection(*method_selections) if method_selections else set()
            
            # Average selection frequency
            all_features = set.union(*method_selections) if method_selections else set()
            feature_frequencies = {}
            for feature in all_features:
                count = sum(1 for selection in method_selections if feature in selection)
                feature_frequencies[feature] = count / n_runs
            
            stability_results[method] = {
                "mean_jaccard_similarity": np.mean(similarities) if similarities else 0,
                "std_jaccard_similarity": np.std(similarities) if similarities else 0,
                "stable_features": list(stable_features),
                "stable_feature_count": len(stable_features),
                "feature_frequencies": feature_frequencies,
                "most_stable_features": sorted(feature_frequencies.items(), 
                                             key=lambda x: x[1], reverse=True)[:10]
            }
        
        return stability_results
