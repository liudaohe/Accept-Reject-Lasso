"""
Random Lasso feature selection method.

This module implements the Random Lasso method which uses a two-stage
bootstrap procedure with random feature sampling to improve stability.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from joblib import Parallel, delayed
from typing import Set, Dict, Any, Tuple
from tqdm.auto import tqdm

from ..core.base import BaseLassoMethod
from ..core.types import DataFrame, Series, FeatureSet, Metadata, SelectionResult
from ..config.defaults import RANDOM_LASSO_DEFAULTS

class RandomLassoMethod(BaseLassoMethod):
    """
    Random Lasso feature selection method.
    
    Random Lasso uses a two-stage bootstrap procedure:
    1. Random feature sampling to estimate feature importance
    2. Importance-weighted sampling for final selection
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Random Lasso method.
        
        Args:
            **kwargs: Hyperparameter overrides
        """
        # Merge with defaults
        config = {**RANDOM_LASSO_DEFAULTS, **kwargs}
        super().__init__(**config)
        
        # Extract method-specific parameters
        self.global_B = config['global_B']
        self.subset_B = config['subset_B']
        self.q1_fraction = config['q1_fraction']
        self.q2_fraction = config['q2_fraction']
        self.alpha_cv_folds = config['alpha_cv_folds']
        self.alpha_tolerance = config['alpha_tolerance']
        self.max_iter = config['max_iter']
    
    @property
    def name(self) -> str:
        """Return method name."""
        return "Random Lasso"
    
    @property
    def description(self) -> str:
        """Return method description."""
        return "Random Lasso with two-stage bootstrap feature sampling"
    
    def _run_step1_iteration(self, X: DataFrame, y: Series, q1: int, alpha: float, iteration_seed: int) -> np.ndarray:
        """Run single iteration of step 1 (random feature sampling)."""
        n_samples, n_features = X.shape
        rng = np.random.default_rng(iteration_seed)
        
        # Bootstrap sampling
        boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X.iloc[boot_indices], y.iloc[boot_indices]
        
        # Random feature sampling
        feature_indices = rng.choice(n_features, size=q1, replace=False)
        X_boot_subset = X_boot.iloc[:, feature_indices]
        
        # Fit Lasso
        model = Lasso(
            alpha=alpha,
            random_state=iteration_seed,
            max_iter=self.max_iter
        )
        model.fit(X_boot_subset, y_boot)
        
        # Map coefficients back to full feature space
        coefs = np.zeros(n_features)
        coefs[feature_indices] = model.coef_
        
        return coefs
    
    def _run_step2_iteration(self, X: DataFrame, y: Series, q2: int, alpha: float, 
                           selection_probs: np.ndarray, iteration_seed: int) -> np.ndarray:
        """Run single iteration of step 2 (importance-weighted sampling)."""
        n_samples, n_features = X.shape
        rng = np.random.default_rng(iteration_seed)
        
        # Bootstrap sampling
        boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X.iloc[boot_indices], y.iloc[boot_indices]
        
        # Importance-weighted feature sampling
        feature_indices = rng.choice(n_features, size=q2, replace=False, p=selection_probs)
        X_boot_subset = X_boot.iloc[:, feature_indices]
        
        # Fit Lasso
        model = Lasso(
            alpha=alpha,
            random_state=iteration_seed,
            max_iter=self.max_iter
        )
        model.fit(X_boot_subset, y_boot)
        
        # Map coefficients back to full feature space
        coefs = np.zeros(n_features)
        coefs[feature_indices] = model.coef_
        
        return coefs
    
    def _run_random_lasso_core(self, X: DataFrame, y: Series, B: int, q1: int, q2: int,
                              alpha: float, selection_threshold_n: int, verbose: bool = True) -> Tuple[FeatureSet, pd.Series]:
        """Core Random Lasso algorithm implementation."""
        n_samples, n_features = X.shape
        
        if verbose:
            print(f"📦 [RandomLasso Step 1] 正在并行计算特征重要性 (B={B}, q1={q1})...")
        
        # Step 1: Random feature sampling for importance estimation
        rng_main = np.random.default_rng(self.random_state)
        step1_seeds = rng_main.integers(0, np.iinfo(np.int32).max, size=B)
        
        step1_coefs_list = Parallel(n_jobs=self.n_jobs, verbose=(10 if verbose else 0))(
            delayed(self._run_step1_iteration)(X, y, q1, alpha, seed) for seed in step1_seeds
        )
        
        # Calculate feature importance
        avg_coefs_step1 = np.mean(step1_coefs_list, axis=0)
        importance_measures = np.abs(avg_coefs_step1)
        
        if verbose:
            print("   步骤 1 完成。")
            print(f"📦 [RandomLasso Step 2] 正在根据重要性并行选择变量 (B={B}, q2={q2})...")
        
        # Step 2: Importance-weighted sampling
        sum_importance = np.sum(importance_measures)
        if sum_importance == 0:
            selection_probs = np.full(n_features, 1/n_features)
        else:
            selection_probs = importance_measures / sum_importance
        
        step2_seeds = rng_main.integers(0, np.iinfo(np.int32).max, size=B)
        step2_coefs_list = Parallel(n_jobs=self.n_jobs, verbose=(10 if verbose else 0))(
            delayed(self._run_step2_iteration)(X, y, q2, alpha, selection_probs, seed) for seed in step2_seeds
        )
        
        # Final coefficient averaging
        final_coefficients = np.mean(step2_coefs_list, axis=0)
        final_coefs_series = pd.Series(final_coefficients, index=X.columns)
        
        if verbose:
            print("   步骤 2 完成。")
        
        # Apply selection threshold
        threshold_val = 1.0 / selection_threshold_n if selection_threshold_n > 0 else 0
        selected_features_mask = np.abs(final_coefficients) > threshold_val
        selected_features = set(X.columns[selected_features_mask])
        
        return selected_features, final_coefs_series
    
    def global_provider(self):
        """Return global feature selection provider."""
        def _global_random_lasso(X: DataFrame, y: Series) -> SelectionResult:
            """Global Random Lasso implementation."""
            self._validate_input(X, y)
            
            if self.hyperparameters.get('verbose', True):
                print("🚀 [RandomLasso Provider] 正在启动...")
            
            n_samples, n_features = X.shape
            
            # Calculate global alpha using LassoCV
            if self.hyperparameters.get('verbose', True):
                print("   正在使用LassoCV计算全局alpha...")
            
            lasso_cv_model = LassoCV(
                cv=self.alpha_cv_folds,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tol=self.alpha_tolerance
            ).fit(X, y)
            global_alpha = lasso_cv_model.alpha_
            
            if self.hyperparameters.get('verbose', True):
                print(f"   计算完成。全局 Alpha: {global_alpha:.4f}")
            
            # Calculate q1 and q2
            q1 = max(1, int(n_features * self.q1_fraction))
            q2 = max(1, int(n_features * self.q2_fraction))
            
            # Run Random Lasso
            selected_features, final_coefficients = self._run_random_lasso_core(
                X=X, y=y, B=self.global_B, q1=q1, q2=q2,
                alpha=global_alpha,
                selection_threshold_n=n_samples,
                verbose=self.hyperparameters.get('verbose', True)
            )
            
            # Prepare metadata
            metadata = self._prepare_metadata(
                final_coefficients=final_coefficients,
                global_alpha=global_alpha,
                B=self.global_B,
                q1=q1,
                q2=q2
            )
            
            if self.hyperparameters.get('verbose', True):
                print(f"🏁 [RandomLasso Provider] 运行完成。选出 {len(selected_features)} 个特征。")
            
            return selected_features, metadata
        
        return _global_random_lasso

    def subset_provider(self):
        """Return subset feature selection provider."""
        def _subset_random_lasso(X_subset: DataFrame, y_subset: Series, metadata: Metadata) -> FeatureSet:
            """Subset Random Lasso implementation."""
            self._validate_input(X_subset, y_subset)

            # Extract parameters from metadata
            alpha = metadata.get('global_alpha')
            if alpha is None:
                import warnings
                warnings.warn("未在元数据中找到 'global_alpha'，将使用默认值 0.01。")
                alpha = 0.01

            q1 = metadata.get('q1')
            if q1 is None:
                import warnings
                warnings.warn("未在元数据中找到 'q1'，将使用特征总数的10%。")
                q1 = max(1, int(X_subset.shape[1] * 0.1))

            q2 = metadata.get('q2')
            if q2 is None:
                import warnings
                warnings.warn("未在元数据中找到 'q2'，将使用特征总数的10%。")
                q2 = max(1, int(X_subset.shape[1] * 0.1))

            # Run Random Lasso on subset
            selected_features, _ = self._run_random_lasso_core(
                X=X_subset, y=y_subset, B=self.subset_B, q1=q1, q2=q2,
                alpha=alpha,
                selection_threshold_n=X_subset.shape[0],
                verbose=False
            )

            return selected_features

        return _subset_random_lasso

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return current hyperparameter configuration."""
        base_params = super().get_hyperparameters()
        random_params = {
            'global_B': self.global_B,
            'subset_B': self.subset_B,
            'q1_fraction': self.q1_fraction,
            'q2_fraction': self.q2_fraction,
            'alpha_cv_folds': self.alpha_cv_folds,
            'alpha_tolerance': self.alpha_tolerance,
            'max_iter': self.max_iter,
        }
        base_params.update(random_params)
        return base_params

    def set_hyperparameters(self, **kwargs) -> None:
        """Update hyperparameter configuration."""
        # Handle method-specific parameters
        if 'global_B' in kwargs:
            self.global_B = kwargs.pop('global_B')
        if 'subset_B' in kwargs:
            self.subset_B = kwargs.pop('subset_B')
        if 'q1_fraction' in kwargs:
            self.q1_fraction = kwargs.pop('q1_fraction')
        if 'q2_fraction' in kwargs:
            self.q2_fraction = kwargs.pop('q2_fraction')
        if 'alpha_cv_folds' in kwargs:
            self.alpha_cv_folds = kwargs.pop('alpha_cv_folds')
        if 'alpha_tolerance' in kwargs:
            self.alpha_tolerance = kwargs.pop('alpha_tolerance')
        if 'max_iter' in kwargs:
            self.max_iter = kwargs.pop('max_iter')

        # Handle base parameters
        super().set_hyperparameters(**kwargs)
