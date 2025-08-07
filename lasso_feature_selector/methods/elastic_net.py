"""
Elastic Net feature selection method.

This module implements the Elastic Net method which combines L1 and L2
regularization for feature selection with automatic hyperparameter tuning.
"""

import pandas as pd
from sklearn.linear_model import ElasticNetCV
from typing import Set, Dict, Any

from ..core.base import BaseLassoMethod
from ..core.types import DataFrame, Series, FeatureSet, Metadata, SelectionResult
from ..config.defaults import ELASTIC_NET_DEFAULTS

class ElasticNetMethod(BaseLassoMethod):
    """
    Elastic Net feature selection method.
    
    Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization,
    automatically selecting both alpha and l1_ratio through cross-validation.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Elastic Net method.
        
        Args:
            **kwargs: Hyperparameter overrides
        """
        # Merge with defaults
        config = {**ELASTIC_NET_DEFAULTS, **kwargs}
        super().__init__(**config)
        
        # Extract method-specific parameters
        # 不再使用自定义l1_ratios，使用sklearn默认值以匹配notebook
        self.global_cv_folds = config['global_cv_folds']
        self.subset_cv_folds = config['subset_cv_folds']
        self.global_tolerance = config['global_tolerance']
        self.subset_tolerance = config['subset_tolerance']
    
    @property
    def name(self) -> str:
        """Return method name."""
        return "Elastic Net"
    
    @property
    def description(self) -> str:
        """Return method description."""
        return "Elastic Net with automatic alpha and l1_ratio selection via cross-validation"
    
    def global_provider(self):
        """Return global feature selection provider."""
        def _global_elastic_net(X: DataFrame, y: Series) -> SelectionResult:
            """Global Elastic Net implementation."""
            self._validate_input(X, y)
            
            if self.hyperparameters.get('verbose', True):
                print("📦 [ElasticNetCV Global] 正在运行...")
            
            # Initialize and fit ElasticNetCV model
            # 注意：为了与notebook完全一致，不指定l1_ratios，使用sklearn默认值
            model = ElasticNetCV(
                cv=self.global_cv_folds,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tol=self.global_tolerance,
                max_iter=1000  # 明确使用sklearn默认值，而不是package的2000
                # 不指定l1_ratios参数，使用sklearn默认值以匹配notebook
            ).fit(X, y)
            
            # Extract selected features
            selected_features = set(X.columns[model.coef_ != 0])
            
            # Prepare metadata
            metadata = self._prepare_metadata(
                global_alpha=model.alpha_,
                global_l1_ratio=model.l1_ratio_,
                coefficients=model.coef_,
                cv_scores=getattr(model, 'mse_path_', None)
            )
            
            if self.hyperparameters.get('verbose', True):
                print(f"🏁 [ElasticNetCV Global] 运行完成。找到最佳 alpha: {model.alpha_:.6f}，"
                      f"最佳 l1_ratio: {model.l1_ratio_:.3f}，选出 {len(selected_features)} 个特征。")
            
            return selected_features, metadata
        
        return _global_elastic_net
    
    def subset_provider(self):
        """Return subset feature selection provider."""
        def _subset_elastic_net(X_subset: DataFrame, y_subset: Series, metadata: Metadata) -> FeatureSet:
            """Subset Elastic Net implementation."""
            self._validate_input(X_subset, y_subset)
            
            # Run ElasticNetCV on subset with independent parameter optimization
            # 注意：为了与notebook完全一致，不指定l1_ratios，使用sklearn默认值
            model = ElasticNetCV(
                cv=self.subset_cv_folds,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tol=self.subset_tolerance,
                max_iter=1000  # 明确使用sklearn默认值，而不是package的2000
                # 不指定l1_ratios参数，使用sklearn默认值以匹配notebook
            ).fit(X_subset, y_subset)
            
            # Extract selected features
            selected_features = set(X_subset.columns[model.coef_ != 0])
            
            return selected_features
        
        return _subset_elastic_net
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return current hyperparameter configuration."""
        base_params = super().get_hyperparameters()
        elastic_net_params = {
            # 不再包含l1_ratios，使用sklearn默认值
            'global_cv_folds': self.global_cv_folds,
            'subset_cv_folds': self.subset_cv_folds,
            'global_tolerance': self.global_tolerance,
            'subset_tolerance': self.subset_tolerance,
        }
        base_params.update(elastic_net_params)
        return base_params
    
    def set_hyperparameters(self, **kwargs) -> None:
        """Update hyperparameter configuration."""
        # Handle method-specific parameters
        # 移除l1_ratios处理，使用sklearn默认值
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
