#!/usr/bin/env python3
"""
ARL Feature Selection Algorithm Pipeline

Command-line interface for ARL (Adaptive Rescue Lasso) feature selection.

Usage:
    python arl_pipeline.py --input data.csv --target target_column --output results.json
"""

import argparse
import pandas as pd
import json
import sys
from pathlib import Path

try:
    from lasso_feature_selector import LassoFeatureSelector
except ImportError:
    print("Error: lasso_feature_selector not found. Please install dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='ARL Feature Selection Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file with features')
    parser.add_argument('--target', '-t', required=True,
                       help='Target CSV file or column name')
    parser.add_argument('--output', '-o', default='arl_results.json',
                       help='Output JSON file for results')
    
    # Method selection
    parser.add_argument('--method', default='adaptive_lasso',
                       choices=['adaptive_lasso', 'random_lasso', 'stability_selection', 
                               'lasso_cv', 'elastic_net'],
                       help='Base Lasso method')
    
    # ARL core parameters  
    parser.add_argument('--correlation_threshold', type=float, default=0.8,
                       help='Correlation threshold for problem groups (README default: 0.8)')
    parser.add_argument('--silhouette_threshold', type=float, default=0.5,
                       help='Silhouette threshold for clustering quality (README default: 0.5)')
    parser.add_argument('--n_final_clusters', type=int, default=50,
                       help='Number of data subsets for ARL (README default: 50)')
    parser.add_argument('--co_occurrence_threshold', type=int, default=1,
                       help='Co-occurrence threshold for feature rescue (README default: 1)')
    parser.add_argument('--min_subset_size', type=int, default=20,
                       help='Minimum subset size (README default: 20)')



    # Performance parameters 
    parser.add_argument('--n_jobs', type=int, default=64,
                       help='Number of parallel jobs (README default: 64)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed (README default: 42)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Load data
        if args.verbose:
            print(f"Loading data from {args.input}...")
        
        X = pd.read_csv(args.input)
        
        if Path(args.target).exists():
            y = pd.read_csv(args.target).squeeze()
        else:
            y = X[args.target]
            X = X.drop(columns=[args.target])
        
        if args.verbose:
            print(f"Data shape: {X.shape}, Target shape: {y.shape}")

        optimized_n_clusters = args.n_final_clusters
        optimized_co_threshold = args.co_occurrence_threshold

        # Run ARL algorithm
        if args.verbose:
            print(f"\n🚀 运行ARL算法 (基础方法: {args.method})...")

        selector = LassoFeatureSelector(enable_logging=args.verbose)
        
        results = selector.select_features(
            X, y,
            method=args.method,
            enable_rescue=True,
            return_metadata=True,
            correlation_threshold=args.correlation_threshold,
            silhouette_threshold=args.silhouette_threshold,
            n_final_clusters=optimized_n_clusters,
            co_occurrence_threshold=optimized_co_threshold,
            random_state=args.random_state,
            n_jobs=args.n_jobs
        )
        
        # Prepare output
        output_data = {
            'algorithm': 'ARL',
            'base_method': args.method,
            'selected_features': list(results['final_features']),
            'n_selected': len(results['final_features']),
            'n_base_features': len(results['base_features']),
            'n_rescued_features': len(results.get('rescued_features', set())),
            'parameters': {
                'correlation_threshold': args.correlation_threshold,
                'silhouette_threshold': args.silhouette_threshold,
                'n_final_clusters': optimized_n_clusters,
                'co_occurrence_threshold': optimized_co_threshold,
                'random_state': args.random_state
            },
            'hyperparameter_optimization': {
                'enabled': args.optimize_hyperparams,
                'original_n_clusters': args.n_final_clusters,
                'original_co_threshold': args.co_occurrence_threshold,
                'optimized_n_clusters': optimized_n_clusters,
                'optimized_co_threshold': optimized_co_threshold
            }
        }


        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if args.verbose:
            print(f"Results saved to {args.output}")
            print(f"ARL selected {output_data['n_selected']} features:")
            print(f"  - Base features: {output_data['n_base_features']}")
            print(f"  - Rescued features: {output_data['n_rescued_features']}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
