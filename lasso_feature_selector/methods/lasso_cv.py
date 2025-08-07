"""
Standard LassoCV feature selection method.

This module implements the standard LassoCV method which automatically
selects the optimal regularization parameter through cross-validation.
"""

import pandas as pd
from sklearn.linear_model import LassoCV
from typing import Set, Dict, Any

from ..core.base import BaseLassoMethod
from ..core.types import DataFrame, Series, FeatureSet, Metadata, SelectionResult
from ..config.defaults import LASSO_CV_DEFAULTS

class LassoCVMethod(BaseLassoMethod):
    """
    Standard LassoCV feature selection method.
    
    This method uses scikit-learn's LassoCV to automatically select
    the optimal regularization parameter through cross-validation.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize LassoCV method.
        
        Args:
            **kwargs: Hyperparameter overrides
        """
        # Merge with defaults
        config = {**LASSO_CV_DEFAULTS, **kwargs}
        super().__init__(**config)
        
        # Extract method-specific parameters
        self.global_cv_folds = config['global_cv_folds']
        self.subset_cv_folds = config['subset_cv_folds']
        self.global_tolerance = config['global_tolerance']
        self.subset_tolerance = config['subset_tolerance']
    
    @property
    def name(self) -> str:
        """Return method name."""
        return "LassoCV"
    
    @property
    def description(self) -> str:
        """Return method description."""
        return "Standard LassoCV with automatic alpha selection via cross-validation"
    
    def global_provider(self):
        """Return global feature selection provider."""
        def _global_lasso_cv(X: DataFrame, y: Series) -> SelectionResult:
            """Global LassoCV implementation."""
            self._validate_input(X, y)
            
            if self.hyperparameters.get('verbose', True):
                print("📦 [LassoCV Global] 正在运行...")
            
            # Initialize and fit LassoCV model
            model = LassoCV(
                cv=self.global_cv_folds,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tol=self.global_tolerance
            ).fit(X, y)
            
            # Extract selected features
            selected_features = set(X.columns[model.coef_ != 0])
            
            # Prepare metadata
            metadata = self._prepare_metadata(
                global_alpha=model.alpha_,
                coefficients=model.coef_,
                cv_scores=getattr(model, 'mse_path_', None)
            )
            
            if self.hyperparameters.get('verbose', True):
                print(f"🏁 [LassoCV Global] 运行完成。找到最佳 alpha: {model.alpha_:.6f}，选出 {len(selected_features)} 个特征。")
            
            return selected_features, metadata
        
        return _global_lasso_cv
    
    def subset_provider(self):
        """Return subset feature selection provider."""
        def _subset_lasso_cv(X_subset: DataFrame, y_subset: Series, metadata: Metadata) -> FeatureSet:
            """Subset LassoCV implementation."""
            self._validate_input(X_subset, y_subset)
            
            # Run LassoCV on subset with independent alpha optimization
            model = LassoCV(
                cv=self.subset_cv_folds,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tol=self.subset_tolerance
            ).fit(X_subset, y_subset)
            
            # Extract selected features
            selected_features = set(X_subset.columns[model.coef_ != 0])
            
            return selected_features
        
        return _subset_lasso_cv
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return current hyperparameter configuration."""
        base_params = super().get_hyperparameters()
        lasso_cv_params = {
            'global_cv_folds': self.global_cv_folds,
            'subset_cv_folds': self.subset_cv_folds,
            'global_tolerance': self.global_tolerance,
            'subset_tolerance': self.subset_tolerance,
        }
        base_params.update(lasso_cv_params)
        return base_params
    
    def set_hyperparameters(self, **kwargs) -> None:
        """Update hyperparameter configuration."""
        # Handle method-specific parameters
        if 'global_cv_folds' in kwargs:
            self.global_cv_folds = kwargs.pop('global_cv_folds')
        if 'subset_cv_folds' in kwargs:
            self.subset_cv_folds = kwargs.pop('subset_cv_folds')
        if 'global_tolerance' in kwargs:
            self.global_tolerance = kwargs.pop('global_tolerance')
        if 'subset_tolerance' in kwargs:
            self.subset_tolerance = kwargs.pop('subset_tolerance')
        
        # Handle base parameters
        super().set_hyperparameters(**kwargs)
