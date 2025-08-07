"""
Default configurations for all Lasso methods and rescue procedures.

This module centralizes all default hyperparameter settings to ensure
consistency and make it easy to modify global defaults.
"""

import numpy as np

# Common settings across all methods
COMMON_DEFAULTS = {
    'random_state': 42,
    'n_jobs': 64,  # 修改默认值为64，避免使用-1占用所有核心
    'verbose': True,
}

# Cross-validation settings
CV_DEFAULTS = {
    'global_cv_folds': 5,
    'subset_cv_folds': 3,
}

# Convergence settings
CONVERGENCE_DEFAULTS = {
    'global_tolerance': 0.005,
    'subset_tolerance': 0.01,
    'max_iter': 2000,
}

# Method-specific default configurations
ADAPTIVE_LASSO_DEFAULTS = {
    **COMMON_DEFAULTS,
    **CV_DEFAULTS,
    **CONVERGENCE_DEFAULTS,
    'ridge_alphas': list(np.logspace(-5, 5, 100)),
    'weight_regularization': 1e-10,
}

RANDOM_LASSO_DEFAULTS = {
    **COMMON_DEFAULTS,
    **CONVERGENCE_DEFAULTS,
    'global_B': 200,
    'subset_B': 50,
    'q1_fraction': 0.1,
    'q2_fraction': 0.1,
    'alpha_cv_folds': 5,
    'alpha_tolerance': 0.005,
}

STABILITY_SELECTION_DEFAULTS = {
    **COMMON_DEFAULTS,
    **CONVERGENCE_DEFAULTS,
    'global_B': 200,
    'subset_B': 50,
    'selection_threshold': 0.75,
    'subsample_fraction': 0.5,
    'ridge_alpha': 1.0,  # 保留作为fallback
    'ridge_alphas': list(np.logspace(-5, 5, 100)),  # RidgeCV候选alpha值
    'ridge_cv_folds': 3,  # Ridge交叉验证折数
    'weight_regularization': 1e-10,
    'alpha_cv_folds': 5,
    'alpha_tolerance': 0.005,
}

LASSO_CV_DEFAULTS = {
    **COMMON_DEFAULTS,
    **CV_DEFAULTS,
    **CONVERGENCE_DEFAULTS,
}

ELASTIC_NET_DEFAULTS = {
    **COMMON_DEFAULTS,
    **CV_DEFAULTS,
    **CONVERGENCE_DEFAULTS,
    # 移除l1_ratios配置，使用sklearn默认值以匹配notebook
}

# Rescue method defaults
RESCUE_DEFAULTS = {
    **COMMON_DEFAULTS,
    'correlation_threshold': 0.8,
    'min_group_size': 2,
    'silhouette_threshold': 0.5,  # 修正：与notebook一致
    'n_final_clusters': 50,
    'min_subset_size': 20,        # 修正：与notebook一致
    'co_occurrence_threshold': 1,
    'use_unified_apriori': True,  # 启用统一Apriori算法
    'use_pruning_optimization': True,  # 保留原有剪枝优化作为后备
}

# Combine all defaults
DEFAULT_CONFIGS = {
    'adaptive_lasso': ADAPTIVE_LASSO_DEFAULTS,
    'random_lasso': RANDOM_LASSO_DEFAULTS,
    'stability_selection': STABILITY_SELECTION_DEFAULTS,
    'lasso_cv': LASSO_CV_DEFAULTS,
    'elastic_net': ELASTIC_NET_DEFAULTS,
    'rescue': RESCUE_DEFAULTS,
    'common': COMMON_DEFAULTS,
}

# Hyperparameter validation ranges
HYPERPARAMETER_RANGES = {
    # Common parameters
    'random_state': {'type': int, 'min': 0, 'max': 2**31 - 1},
    'n_jobs': {'type': int, 'min': -1, 'max': None},
    'verbose': {'type': bool},
    
    # CV parameters
    'global_cv_folds': {'type': int, 'min': 2, 'max': 20},
    'subset_cv_folds': {'type': int, 'min': 2, 'max': 10},
    
    # Convergence parameters
    'global_tolerance': {'type': float, 'min': 1e-6, 'max': 1e-1},
    'subset_tolerance': {'type': float, 'min': 1e-6, 'max': 1e-1},
    'max_iter': {'type': int, 'min': 100, 'max': 10000},
    
    # Adaptive Lasso
    'weight_regularization': {'type': float, 'min': 1e-12, 'max': 1e-6},
    
    # Random Lasso
    'global_B': {'type': int, 'min': 10, 'max': 1000},
    'subset_B': {'type': int, 'min': 5, 'max': 200},
    'q1_fraction': {'type': float, 'min': 0.01, 'max': 0.5},
    'q2_fraction': {'type': float, 'min': 0.01, 'max': 0.5},
    
    # Stability Selection
    'selection_threshold': {'type': float, 'min': 0.5, 'max': 0.99},
    'subsample_fraction': {'type': float, 'min': 0.1, 'max': 0.9},
    'ridge_alpha': {'type': float, 'min': 0.01, 'max': 100.0},
    
    # Rescue parameters
    'correlation_threshold': {'type': float, 'min': 0.5, 'max': 0.99},
    'min_group_size': {'type': int, 'min': 2, 'max': 20},
    'silhouette_threshold': {'type': float, 'min': 0.0, 'max': 1.0},
    'n_final_clusters': {'type': int, 'min': 2, 'max': 200},
    'min_subset_size': {'type': int, 'min': 5, 'max': 100},
    'co_occurrence_threshold': {'type': int, 'min': 0, 'max': 10},
}
