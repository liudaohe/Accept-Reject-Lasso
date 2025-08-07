"""
Stability Selection feature selection method.

This module implements Stability Selection which combines subsampling
with adaptive Lasso to improve feature selection stability.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from typing import Set, Dict, Any

from ..core.base import BaseLassoMethod
from ..core.types import DataFrame, Series, FeatureSet, Metadata, SelectionResult
from ..config.defaults import STABILITY_SELECTION_DEFAULTS

class StabilitySelectionMethod(BaseLassoMethod):
    """
    Stability Selection feature selection method.
    
    Stability Selection uses subsampling combined with adaptive Lasso
    to identify features that are consistently selected across different
    data subsets, improving selection stability.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Stability Selection method.
        
        Args:
            **kwargs: Hyperparameter overrides
        """
        # Merge with defaults
        config = {**STABILITY_SELECTION_DEFAULTS, **kwargs}
        super().__init__(**config)
        
        # Extract method-specific parameters
        self.global_B = config['global_B']
        self.subset_B = config['subset_B']
        self.selection_threshold = config['selection_threshold']
        self.subsample_fraction = config['subsample_fraction']
        self.ridge_alpha = config['ridge_alpha']  # 保留作为fallback
        self.ridge_alphas = config.get('ridge_alphas', list(np.logspace(-5, 5, 100)))  # CV候选值
        self.ridge_cv_folds = config.get('ridge_cv_folds', 3)  # Ridge CV折数
        self.weight_regularization = config['weight_regularization']
        self.alpha_cv_folds = config['alpha_cv_folds']
        self.alpha_tolerance = config['alpha_tolerance']
        self.subset_tolerance = config['subset_tolerance']
    
    @property
    def name(self) -> str:
        """Return method name."""
        return "Stability Selection"
    
    @property
    def description(self) -> str:
        """Return method description."""
        return "Stability Selection with adaptive Lasso using RidgeCV for automatic alpha selection and subsampling for robust feature selection"
    
    def _run_stability_selection_core(self, X: DataFrame, y: Series, B: int,
                                    threshold: float, global_alpha: float, verbose: bool = True) -> FeatureSet:
        """Core Stability Selection algorithm implementation."""
        if verbose:
            print(f"    [StabilitySelection] 正在运行 (B={B}, threshold={threshold}, alpha={global_alpha:.4f})...")
            print(f"    [StabilitySelection] 使用RidgeCV自动选择alpha (候选范围: {len(self.ridge_alphas)}个值, 10^-5到10^5)")

        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # 用于统计RidgeCV选择的alpha值
        ridge_alphas_selected = []
        
        # Robustness checks
        if n_samples < 2:
            return set()
        
        feature_names = X.columns
        selection_counts = np.zeros(n_features)
        subsample_size = n_samples // 2  # 与notebook完全一致：固定50%
        
        if subsample_size == 0:
            return set()
        
        for i in range(B):
            # Subsample without replacement
            sub_indices = np.random.choice(n_samples, subsample_size, replace=False)
            X_sub, y_sub = X.iloc[sub_indices], y.iloc[sub_indices]
            
            try:
                # Step 1: Ridge regression with CV for adaptive weights
                # 使用RidgeCV自动选择最佳alpha，提高稳定性
                ridge_model = RidgeCV(
                    alphas=self.ridge_alphas,
                    cv=self.ridge_cv_folds,
                    scoring='r2'  # 使用R²作为评估指标
                ).fit(X_sub, y_sub)

                # 记录选择的alpha值用于统计
                ridge_alphas_selected.append(ridge_model.alpha_)

                weights = 1.0 / (np.abs(ridge_model.coef_) + self.weight_regularization)
                
                # Step 2: Apply weights and run Lasso
                X_sub_adaptive = X_sub / weights
                lasso_model = Lasso(
                    alpha=global_alpha,
                    random_state=self.random_state,
                    tol=self.subset_tolerance
                ).fit(X_sub_adaptive, y_sub)
                
                # Step 3: Record selected features
                selection_counts[lasso_model.coef_ != 0] += 1
                
            except Exception:
                # Skip if fitting fails on this subset
                continue
        
        # Calculate selection probabilities and apply threshold
        selection_probs = selection_counts / B
        final_selected_mask = selection_probs >= threshold

        # 输出RidgeCV统计信息
        if verbose and len(ridge_alphas_selected) > 0:
            mean_ridge_alpha = np.mean(ridge_alphas_selected)
            std_ridge_alpha = np.std(ridge_alphas_selected)
            print(f"    [StabilitySelection] RidgeCV统计: 平均alpha={mean_ridge_alpha:.4f}±{std_ridge_alpha:.4f}")
            print(f"    [StabilitySelection] alpha范围: [{np.min(ridge_alphas_selected):.4f}, {np.max(ridge_alphas_selected):.4f}]")

        return set(feature_names[final_selected_mask])
    
    def global_provider(self):
        """Return global feature selection provider."""
        def _global_stability_selection(X: DataFrame, y: Series) -> SelectionResult:
            """Global Stability Selection implementation."""
            self._validate_input(X, y)
            
            if self.hyperparameters.get('verbose', True):
                print("📦 [StabilitySelection Global] 正在启动...")
            
            # Step 1: Calculate global alpha using LassoCV
            if self.hyperparameters.get('verbose', True):
                print("    正在使用LassoCV计算全局alpha...")
            
            lasso_cv_model = LassoCV(
                cv=self.alpha_cv_folds,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tol=self.alpha_tolerance
            ).fit(X, y)
            global_alpha = lasso_cv_model.alpha_
            
            if self.hyperparameters.get('verbose', True):
                print(f"    计算完成。全局 Alpha: {global_alpha:.4f}")
            
            # Step 2: Run Stability Selection
            selected_features = self._run_stability_selection_core(
                X=X,
                y=y,
                B=self.global_B,
                threshold=self.selection_threshold,
                global_alpha=global_alpha,
                verbose=self.hyperparameters.get('verbose', True)
            )
            
            # Prepare metadata
            metadata = self._prepare_metadata(
                global_alpha=global_alpha,
                selection_threshold=self.selection_threshold,
                subsample_fraction=self.subsample_fraction,
                B=self.global_B
            )
            
            if self.hyperparameters.get('verbose', True):
                print(f"🏁 [StabilitySelection Global] 运行完成。选出 {len(selected_features)} 个特征。")
            
            return selected_features, metadata
        
        return _global_stability_selection
    
    def subset_provider(self):
        """Return subset feature selection provider."""
        def _subset_stability_selection(X_subset: DataFrame, y_subset: Series, metadata: Metadata) -> FeatureSet:
            """Subset Stability Selection implementation."""
            self._validate_input(X_subset, y_subset)
            
            # Extract global alpha from metadata
            global_alpha = metadata.get('global_alpha')
            if global_alpha is None:
                import warnings
                warnings.warn("⚠️ 警告: 未在元数据中找到 'global_alpha'，将使用默认值 0.01。")
                global_alpha = 0.01
            
            # Run Stability Selection on subset
            selected_features = self._run_stability_selection_core(
                X=X_subset,
                y=y_subset,
                B=self.subset_B,
                threshold=self.selection_threshold,
                global_alpha=global_alpha,
                verbose=False
            )
            
            return selected_features
        
        return _subset_stability_selection
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return current hyperparameter configuration."""
        base_params = super().get_hyperparameters()
        stability_params = {
            'global_B': self.global_B,
            'subset_B': self.subset_B,
            'selection_threshold': self.selection_threshold,
            'subsample_fraction': self.subsample_fraction,
            'ridge_alpha': self.ridge_alpha,
            'ridge_alphas': self.ridge_alphas,
            'ridge_cv_folds': self.ridge_cv_folds,
            'weight_regularization': self.weight_regularization,
            'alpha_cv_folds': self.alpha_cv_folds,
            'alpha_tolerance': self.alpha_tolerance,
            'subset_tolerance': self.subset_tolerance,
        }
        base_params.update(stability_params)
        return base_params

    def set_hyperparameters(self, **kwargs) -> None:
        """Update hyperparameter configuration."""
        # Handle method-specific parameters
        if 'global_B' in kwargs:
            self.global_B = kwargs.pop('global_B')
        if 'subset_B' in kwargs:
            self.subset_B = kwargs.pop('subset_B')
        if 'selection_threshold' in kwargs:
            self.selection_threshold = kwargs.pop('selection_threshold')
        if 'subsample_fraction' in kwargs:
            self.subsample_fraction = kwargs.pop('subsample_fraction')
        if 'ridge_alpha' in kwargs:
            self.ridge_alpha = kwargs.pop('ridge_alpha')
        if 'ridge_alphas' in kwargs:
            self.ridge_alphas = kwargs.pop('ridge_alphas')
        if 'ridge_cv_folds' in kwargs:
            self.ridge_cv_folds = kwargs.pop('ridge_cv_folds')
        if 'weight_regularization' in kwargs:
            self.weight_regularization = kwargs.pop('weight_regularization')
        if 'alpha_cv_folds' in kwargs:
            self.alpha_cv_folds = kwargs.pop('alpha_cv_folds')
        if 'alpha_tolerance' in kwargs:
            self.alpha_tolerance = kwargs.pop('alpha_tolerance')
        if 'subset_tolerance' in kwargs:
            self.subset_tolerance = kwargs.pop('subset_tolerance')

        # Handle base parameters
        super().set_hyperparameters(**kwargs)
