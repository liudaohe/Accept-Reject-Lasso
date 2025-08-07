"""
Adaptive Lasso feature selection method.

This module implements the Adaptive Lasso method which uses Ridge regression
coefficients to create adaptive weights for the Lasso penalty.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from typing import Set, Dict, Any, Tuple

from ..core.base import BaseLassoMethod
from ..core.types import DataFrame, Series, FeatureSet, Metadata, SelectionResult
from ..config.defaults import ADAPTIVE_LASSO_DEFAULTS

class AdaptiveLassoMethod(BaseLassoMethod):
    """
    Adaptive Lasso feature selection method.
    
    The Adaptive Lasso uses Ridge regression coefficients to create adaptive
    weights that penalize different features differently, allowing for better
    feature selection in the presence of correlated predictors.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Adaptive Lasso method.
        
        Args:
            **kwargs: Hyperparameter overrides
        """
        # Merge with defaults
        config = {**ADAPTIVE_LASSO_DEFAULTS, **kwargs}
        super().__init__(**config)
        
        # Extract method-specific parameters
        self.ridge_alphas = config['ridge_alphas']
        self.weight_regularization = config['weight_regularization']
        self.global_cv_folds = config['global_cv_folds']
        self.subset_cv_folds = config['subset_cv_folds']
        self.global_tolerance = config['global_tolerance']
        self.subset_tolerance = config['subset_tolerance']
    
    @property
    def name(self) -> str:
        """Return method name."""
        return "Adaptive Lasso"
    
    @property
    def description(self) -> str:
        """Return method description."""
        return "Adaptive Lasso with Ridge-based weights for improved feature selection"
    
    def global_provider(self):
        """Return global feature selection provider."""
        def _global_adaptive_lasso(X: DataFrame, y: Series) -> SelectionResult:
            """Global Adaptive Lasso implementation."""
            self._validate_input(X, y)
            
            # Step 1: Ridge regression for weights
            ridge = RidgeCV(
                alphas=self.ridge_alphas,
                cv=self.global_cv_folds
            ).fit(X, y)
            
            # Step 2: Calculate adaptive weights
            weights = 1.0 / (np.abs(ridge.coef_) + self.weight_regularization)
            
            # Step 3: Apply weights and run Lasso
            X_adaptive = X / weights
            model = LassoCV(
                cv=self.global_cv_folds,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tol=self.global_tolerance
            ).fit(X_adaptive, y)
            
            # Step 4: Get final coefficients and selected features
            final_coefs = model.coef_ / weights
            selected_features = set(X.columns[final_coefs != 0])
            
            # Prepare metadata
            metadata = self._prepare_metadata(
                global_alpha=model.alpha_,
                ridge_alpha=ridge.alpha_,
                weights=weights,
                final_coefficients=final_coefs
            )
            
            return selected_features, metadata
        
        return _global_adaptive_lasso
    
    def subset_provider(self):
        """Return subset feature selection provider."""
        def _subset_adaptive_lasso(X_subset: DataFrame, y_subset: Series, metadata: Metadata) -> FeatureSet:
            """Subset Adaptive Lasso implementation."""
            self._validate_input(X_subset, y_subset)
            
            # Step 1: Ridge regression for weights (recalculated for subset)
            ridge = RidgeCV(
                alphas=self.ridge_alphas,
                cv=self.subset_cv_folds
            ).fit(X_subset, y_subset)
            
            # Step 2: Calculate adaptive weights
            weights = 1.0 / (np.abs(ridge.coef_) + self.weight_regularization)
            
            # Step 3: Apply weights and run Lasso
            X_adaptive_subset = X_subset / weights
            model = LassoCV(
                cv=self.subset_cv_folds,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tol=self.subset_tolerance
            ).fit(X_adaptive_subset, y_subset)
            
            # Step 4: Get final coefficients and selected features
            final_coefs = model.coef_ / weights
            selected_features = set(X_subset.columns[final_coefs != 0])
            
            return selected_features
        
        return _subset_adaptive_lasso
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return current hyperparameter configuration."""
        base_params = super().get_hyperparameters()
        adaptive_params = {
            'ridge_alphas': self.ridge_alphas,
            'weight_regularization': self.weight_regularization,
            'global_cv_folds': self.global_cv_folds,
            'subset_cv_folds': self.subset_cv_folds,
            'global_tolerance': self.global_tolerance,
            'subset_tolerance': self.subset_tolerance,
        }
        base_params.update(adaptive_params)
        return base_params
    
    def set_hyperparameters(self, **kwargs) -> None:
        """Update hyperparameter configuration."""
        # Handle method-specific parameters
        if 'ridge_alphas' in kwargs:
            self.ridge_alphas = kwargs.pop('ridge_alphas')
        if 'weight_regularization' in kwargs:
            self.weight_regularization = kwargs.pop('weight_regularization')
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
