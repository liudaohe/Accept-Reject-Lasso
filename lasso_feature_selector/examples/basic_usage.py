"""
Basic usage examples for Lasso Feature Selector.

This script demonstrates the fundamental functionality of the library
with simple, easy-to-follow examples.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from lasso_feature_selector import LassoFeatureSelector, LassoConfig
from lasso_feature_selector.utils import evaluate_selection

def create_sample_data(n_samples=1000, n_features=100, n_informative=20, noise=0.1, random_state=42):
    """Create sample regression data for demonstration."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )
    
    # Convert to DataFrame with meaningful column names
    feature_names = [f'feature_{i:03d}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series

def example_basic_selection():
    """Example 1: Basic feature selection with a single method."""
    print("=" * 60)
    print("Example 1: Basic Feature Selection")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=500, n_features=50, n_informative=10)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize selector
    selector = LassoFeatureSelector()
    
    # Run feature selection
    selected_features = selector.select_features(
        X, y, 
        method='adaptive_lasso',
        enable_rescue=True
    )
    
    print(f"Selected {len(selected_features)} features:")
    print(f"Features: {sorted(list(selected_features))[:10]}...")  # Show first 10
    
    return selected_features

def example_method_comparison():
    """Example 2: Compare multiple feature selection methods."""
    print("\\n" + "=" * 60)
    print("Example 2: Method Comparison")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=800, n_features=80, n_informative=15)
    
    # Initialize selector
    selector = LassoFeatureSelector()
    
    # Compare multiple methods
    methods = ['adaptive_lasso', 'random_lasso', 'stability_selection', 'lasso_cv']
    
    print(f"Comparing {len(methods)} methods...")
    comparison_results = selector.compare_methods(X, y, methods, enable_rescue=True)
    
    print("\\nComparison Results:")
    print(comparison_results[['method', 'selected_count']])
    
    return comparison_results

def example_custom_configuration():
    """Example 3: Using custom configuration."""
    print("\\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = LassoConfig()
    
    # Modify some parameters
    config.set('adaptive_lasso', 'global_cv_folds', 10)
    config.set('random_lasso', 'global_B', 300)
    config.set('rescue', 'correlation_threshold', 0.9)
    
    print("Custom configuration:")
    print(f"- Adaptive Lasso CV folds: {config.get('adaptive_lasso', 'global_cv_folds')}")
    print(f"- Random Lasso bootstrap iterations: {config.get('random_lasso', 'global_B')}")
    print(f"- Rescue correlation threshold: {config.get('rescue', 'correlation_threshold')}")
    
    # Create sample data
    X, y = create_sample_data(n_samples=600, n_features=60, n_informative=12)
    
    # Initialize selector with custom config
    selector = LassoFeatureSelector(config=config)
    
    # Run selection
    selected_features = selector.select_features(X, y, method='adaptive_lasso')
    
    print(f"\\nSelected {len(selected_features)} features with custom configuration")
    
    return selected_features

def example_detailed_results():
    """Example 4: Getting detailed results and metadata."""
    print("\\n" + "=" * 60)
    print("Example 4: Detailed Results")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=400, n_features=40, n_informative=8)
    
    # Initialize selector
    selector = LassoFeatureSelector()
    
    # Get detailed results
    results = selector.select_features(
        X, y, 
        method='stability_selection',
        enable_rescue=True,
        return_metadata=True
    )
    
    print("Detailed Results:")
    print(f"- Method: {results['method']}")
    print(f"- Base features: {len(results['base_features'])}")
    print(f"- Rescued features: {len(results['rescued_features'])}")
    print(f"- Final features: {len(results['final_features'])}")
    print(f"- Rescue enabled: {results['rescue_enabled']}")
    
    # Show some metadata
    metadata = results['metadata']
    print(f"\\nMetadata keys: {list(metadata.keys())}")
    
    return results

def example_evaluation():
    """Example 5: Evaluation with synthetic true features."""
    print("\\n" + "=" * 60)
    print("Example 5: Performance Evaluation")
    print("=" * 60)
    
    # Create sample data with known structure
    X, y = create_sample_data(n_samples=500, n_features=50, n_informative=10)
    
    # Define true feature parameters (for synthetic evaluation)
    true_features_params = {
        'n_determined_important': 10,
        'n_false_redundancy_groups': 2,
        'n_vars_per_group': 3,
        'n_true_redundancy_groups': 2,
        'p': 50
    }
    
    # Initialize selector
    selector = LassoFeatureSelector()
    
    # Run selection
    selected_features = selector.select_features(X, y, method='lasso_cv')
    
    # Evaluate performance
    metrics = evaluate_selection(selected_features, true_features_params)
    
    print("Performance Metrics:")
    print(f"- Selected features: {metrics['总选择数']}")
    print(f"- True Positives: {metrics['TP']}")
    print(f"- False Positives: {metrics['FP']}")
    print(f"- False Negatives: {metrics['FN']}")
    print(f"- Precision: {metrics['Precision']:.3f}")
    print(f"- Recall: {metrics['Recall']:.3f}")
    print(f"- F1-Score: {metrics['F1-Score']:.3f}")
    
    return metrics

def example_save_load_results():
    """Example 6: Saving and loading results."""
    print("\\n" + "=" * 60)
    print("Example 6: Save and Load Results")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=300, n_features=30, n_informative=6)
    
    # Initialize selector
    selector = LassoFeatureSelector()
    
    # Run selection
    selected_features = selector.select_features(
        X, y, 
        method='elastic_net',
        return_metadata=True
    )
    
    # Save results
    try:
        selector.save_results('selection_results.json', format='json')
        print("Results saved to 'selection_results.json'")
        
        # Load and display
        import json
        with open('selection_results.json', 'r') as f:
            loaded_results = json.load(f)
        
        print(f"Loaded results - Method: {loaded_results['method']}")
        print(f"Final features count: {len(loaded_results['final_features'])}")
        
    except Exception as e:
        print(f"Error saving/loading results: {e}")
    
    return selected_features

def main():
    """Run all examples."""
    print("Lasso Feature Selector - Basic Usage Examples")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run examples
        example_basic_selection()
        example_method_comparison()
        example_custom_configuration()
        example_detailed_results()
        example_evaluation()
        example_save_load_results()
        
        print("\\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\\nError running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
