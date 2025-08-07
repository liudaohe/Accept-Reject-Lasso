"""
Main orchestrator for Lasso Feature Selector.

This module provides the primary interface for running feature selection
with optional rescue mechanisms.
"""

import pandas as pd
from typing import Dict, Any, Optional, List, Union
from sklearn.preprocessing import StandardScaler

from ..core.interfaces import LassoMethodInterface
from ..core.types import DataFrame, Series, FeatureSet, Metadata
from ..config.settings import LassoConfig
from ..methods import (
    AdaptiveLassoMethod, RandomLassoMethod, StabilitySelectionMethod,
    LassoCVMethod, ElasticNetMethod
)
from ..rescue import ProblemGroupIdentifier, ClusteringAnalyzer, CoOccurrenceRescue
from ..utils.logging import LassoLogger
from ..utils.evaluation import FeatureEvaluator

class LassoFeatureSelector:
    """
    Main orchestrator for Lasso-based feature selection with rescue mechanisms.
    
    This class provides a unified interface for running various Lasso methods
    with optional feature rescue based on problem group analysis.
    """
    
    # Available method implementations
    AVAILABLE_METHODS = {
        'adaptive_lasso': AdaptiveLassoMethod,
        'random_lasso': RandomLassoMethod,
        'stability_selection': StabilitySelectionMethod,
        'lasso_cv': LassoCVMethod,
        'elastic_net': ElasticNetMethod
    }
    
    def __init__(self, 
                 config: Optional[LassoConfig] = None,
                 enable_logging: bool = True,
                 log_file: Optional[str] = None):
        """
        Initialize the feature selector.
        
        Args:
            config: Configuration object (uses defaults if None)
            enable_logging: Whether to enable logging
            log_file: Optional log file path
        """
        self.config = config if config is not None else LassoConfig()
        
        # Initialize logging
        if enable_logging:
            self.logger = LassoLogger(log_file=log_file)
        else:
            self.logger = None
        
        # Initialize components
        self.problem_identifier = ProblemGroupIdentifier(**self.config.get('rescue'))
        self.clustering_analyzer = ClusteringAnalyzer(**self.config.get('rescue'))
        self.rescue_method = CoOccurrenceRescue(**self.config.get('rescue'))
        
        # Initialize evaluator
        self.evaluator = FeatureEvaluator()
        
        # Storage for results
        self.last_results = {}
    
    def _log(self, level: str, message: str, **kwargs) -> None:
        """Helper method for logging."""
        if self.logger:
            getattr(self.logger, level)(message, **kwargs)
    
    def _validate_method(self, method: Union[str, LassoMethodInterface]) -> LassoMethodInterface:
        """
        Validate and return method instance.
        
        Args:
            method: Method name or instance
            
        Returns:
            Method instance
        """
        if isinstance(method, str):
            if method not in self.AVAILABLE_METHODS:
                raise ValueError(f"Unknown method: {method}. Available: {list(self.AVAILABLE_METHODS.keys())}")
            
            method_class = self.AVAILABLE_METHODS[method]
            method_config = self.config.get(method)
            return method_class(**method_config)
        
        elif isinstance(method, LassoMethodInterface):
            return method
        
        else:
            raise TypeError("Method must be string name or LassoMethodInterface instance")
    
    def select_features(self,
                       X: DataFrame,
                       y: Series,
                       method: Union[str, LassoMethodInterface],
                       enable_rescue: bool = True,
                       scale_features: bool = True,
                       return_metadata: bool = False,
                       # 常用超参数直接传入
                       correlation_threshold: Optional[float] = None,
                       n_final_clusters: Optional[int] = None,
                       min_subset_size: Optional[int] = None,
                       co_occurrence_threshold: Optional[int] = None,
                       silhouette_threshold: Optional[float] = None,
                       global_cv_folds: Optional[int] = None,
                       global_tolerance: Optional[float] = None,
                       random_state: Optional[int] = None,
                       **kwargs) -> Union[FeatureSet, Dict[str, Any]]:
        """
        Perform feature selection with optional rescue.

        Args:
            X: Feature matrix
            y: Target vector
            method: Method name or instance
            enable_rescue: Whether to enable feature rescue
            scale_features: Whether to scale features before selection
            return_metadata: Whether to return detailed metadata

            # 常用超参数（可选，会覆盖config中的设置）
            correlation_threshold: 特征相关性阈值 (0.6-0.95)
            n_final_clusters: 最终聚类数量 (20-200)
            min_subset_size: 子集最小样本数 (10-100)
            co_occurrence_threshold: 共现拯救阈值 (1-5)
            silhouette_threshold: 轮廓系数阈值 (0.3-0.8)
            global_cv_folds: 全局交叉验证折数 (3-10)
            global_tolerance: 全局收敛容忍度 (0.001-0.01)
            random_state: 随机种子
            **kwargs: 其他方法特定参数

        Returns:
            Selected features or detailed results dictionary
        """
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")

        # 处理直接传入的超参数（临时覆盖config设置）
        original_config = {}
        if correlation_threshold is not None:
            original_config[('rescue', 'correlation_threshold')] = self.config.get('rescue', 'correlation_threshold')
            self.config.set('rescue', 'correlation_threshold', correlation_threshold)
        if n_final_clusters is not None:
            original_config[('rescue', 'n_final_clusters')] = self.config.get('rescue', 'n_final_clusters')
            self.config.set('rescue', 'n_final_clusters', n_final_clusters)
        if min_subset_size is not None:
            original_config[('rescue', 'min_subset_size')] = self.config.get('rescue', 'min_subset_size')
            self.config.set('rescue', 'min_subset_size', min_subset_size)
        if co_occurrence_threshold is not None:
            original_config[('rescue', 'co_occurrence_threshold')] = self.config.get('rescue', 'co_occurrence_threshold')
            self.config.set('rescue', 'co_occurrence_threshold', co_occurrence_threshold)
        if silhouette_threshold is not None:
            original_config[('rescue', 'silhouette_threshold')] = self.config.get('rescue', 'silhouette_threshold')
            self.config.set('rescue', 'silhouette_threshold', silhouette_threshold)
        if global_cv_folds is not None:
            original_config[('adaptive_lasso', 'global_cv_folds')] = self.config.get('adaptive_lasso', 'global_cv_folds')
            self.config.set('adaptive_lasso', 'global_cv_folds', global_cv_folds)
        if global_tolerance is not None:
            original_config[('adaptive_lasso', 'global_tolerance')] = self.config.get('adaptive_lasso', 'global_tolerance')
            self.config.set('adaptive_lasso', 'global_tolerance', global_tolerance)
        if random_state is not None:
            original_config[('adaptive_lasso', 'random_state')] = self.config.get('adaptive_lasso', 'random_state')
            self.config.set('adaptive_lasso', 'random_state', random_state)
            original_config[('rescue', 'random_state')] = self.config.get('rescue', 'random_state')
            self.config.set('rescue', 'random_state', random_state)

        # 处理其他kwargs参数
        for key, value in kwargs.items():
            if hasattr(self.config, 'set'):
                # 尝试设置参数，如果失败则忽略
                try:
                    # 先尝试adaptive_lasso
                    if key in ['ridge_alphas', 'weight_regularization', 'subset_cv_folds', 'subset_tolerance', 'n_jobs', 'max_iter']:
                        original_config[('adaptive_lasso', key)] = self.config.get('adaptive_lasso', key)
                        self.config.set('adaptive_lasso', key, value)
                    # 再尝试rescue
                    elif key in ['min_group_size']:
                        original_config[('rescue', key)] = self.config.get('rescue', key)
                        self.config.set('rescue', key, value)
                    # 最后尝试random_lasso
                    elif key in ['B', 'q1', 'q2', 'alpha', 'selection_threshold_n']:
                        original_config[('random_lasso', key)] = self.config.get('random_lasso', key)
                        self.config.set('random_lasso', key, value)
                except:
                    self._log('warning', f"Unknown parameter: {key}")

        try:
            # Initialize method
            method_instance = self._validate_method(method)

            self._log('info', f"Starting feature selection with {method_instance.name}")
            if self.logger:
                self.logger.log_data_info(X.shape, y.shape, list(X.columns))

            # Scale features if requested
            X_processed = X.copy()
            scaler = None
            if scale_features:
                scaler = StandardScaler()
                X_processed = pd.DataFrame(
                    scaler.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
                self._log('debug', "Features scaled using StandardScaler")

            # Run global feature selection
            self._log('info', f"Running global {method_instance.name} selection")
            global_provider = method_instance.global_provider()
            base_features, metadata = global_provider(X_processed, y)

            self._log('info', f"Global selection completed: {len(base_features)} features selected")

            # Store base results
            results = {
                'method': method_instance.name,
                'base_features': base_features,
                'metadata': metadata,
                'scaler': scaler,
                'rescue_enabled': enable_rescue
            }

            # Feature rescue if enabled
            if enable_rescue:
                self._log('info', "Starting feature rescue process")
                rescued_features = self._run_rescue_pipeline(
                    X_processed, y, base_features, metadata, method_instance
                )

                final_features = base_features.union(rescued_features)
                results.update({
                    'rescued_features': rescued_features,
                    'final_features': final_features
                })

                if self.logger:
                    self.logger.log_rescue_result(
                        len(base_features), len(rescued_features), len(final_features)
                    )
            else:
                results['final_features'] = base_features

            # Store results for later access
            self.last_results = results

            if return_metadata:
                return results
            else:
                return results['final_features']

        finally:
            # 恢复原始配置
            for (section, key), original_value in original_config.items():
                self.config.set(section, key, original_value)
    
    def _run_rescue_pipeline(self,
                           X: DataFrame,
                           y: Series,
                           base_features: FeatureSet,
                           metadata: Metadata,
                           method_instance: LassoMethodInterface) -> FeatureSet:
        """
        Run the complete feature rescue pipeline.
        
        Args:
            X: Processed feature matrix
            y: Target vector
            base_features: Initially selected features
            metadata: Metadata from global selection
            method_instance: Method instance for subset selection
            
        Returns:
            Set of rescued features
        """
        # Step 1: Identify problem groups
        self._log('info', "Identifying problem groups")
        problem_groups = self.problem_identifier.identify_problem_groups(X)
        
        if self.logger:
            self.logger.log_problem_groups(
                problem_groups, 
                self.problem_identifier.correlation_threshold
            )
        
        if not problem_groups:
            self._log('warning', "No problem groups identified, skipping rescue")
            return set()
        
        # Step 2: Evaluate clustering basis
        self._log('info', "Evaluating clustering basis groups")
        basis_evaluation = self.clustering_analyzer.evaluate_problem_groups_as_basis(
            X, problem_groups
        )
        
        if basis_evaluation.empty:
            self._log('warning', "No suitable clustering basis found, using all problem groups")
            clustering_basis_groups = problem_groups
        else:
            clustering_basis_groups = [
                row['feature_list'] for _, row in basis_evaluation.iterrows()
            ]
            self._log('info', f"Found {len(clustering_basis_groups)} suitable basis groups")
        
        # Step 3: Run rescue
        self._log('info', "Executing co-occurrence rescue")
        rescued_features = self.rescue_method.rescue_features(
            X=X,
            y=y,
            base_features=base_features,
            problem_groups=problem_groups,
            clustering_basis_groups=clustering_basis_groups,
            subset_lasso_provider=method_instance.subset_provider(),
            metadata=metadata
        )
        
        return rescued_features
    
    def compare_methods(self,
                       X: DataFrame,
                       y: Series,
                       methods: List[Union[str, LassoMethodInterface]],
                       enable_rescue: bool = True,
                       scale_features: bool = True,
                       true_features_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Compare multiple feature selection methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            methods: List of methods to compare
            enable_rescue: Whether to enable rescue for all methods
            scale_features: Whether to scale features
            true_features_params: True feature parameters for evaluation
            
        Returns:
            DataFrame with comparison results
        """
        self._log('info', f"Comparing {len(methods)} methods")
        
        # Set up evaluator if true features provided
        if true_features_params:
            self.evaluator = FeatureEvaluator(true_features_params)
        
        results = {}
        detailed_results = {}
        
        for method in methods:
            method_instance = self._validate_method(method)
            method_name = method_instance.name
            
            self._log('info', f"Running {method_name}")
            
            try:
                result = self.select_features(
                    X, y, method_instance,
                    enable_rescue=enable_rescue,
                    scale_features=scale_features,
                    return_metadata=True
                )
                
                results[method_name] = result['final_features']
                detailed_results[method_name] = result
                
            except Exception as e:
                self._log('error', f"Error running {method_name}: {str(e)}")
                results[method_name] = set()
        
        # Store detailed results
        self.last_results = {
            'comparison_results': detailed_results,
            'method_selections': results
        }
        
        # Evaluate if true features provided
        if true_features_params:
            comparison_df = self.evaluator.compare_methods(results)
            return comparison_df
        else:
            # Return basic comparison
            basic_results = []
            for method_name, features in results.items():
                basic_results.append({
                    'method': method_name,
                    'selected_count': len(features),
                    'selected_features': list(features)
                })
            return pd.DataFrame(basic_results)
    
    def get_method_info(self, method: Union[str, LassoMethodInterface]) -> Dict[str, Any]:
        """
        Get information about a specific method.
        
        Args:
            method: Method name or instance
            
        Returns:
            Dictionary with method information
        """
        method_instance = self._validate_method(method)
        
        return {
            'name': method_instance.name,
            'description': method_instance.description,
            'hyperparameters': method_instance.get_hyperparameters(),
            'config_key': next((k for k, v in self.AVAILABLE_METHODS.items() 
                              if v == type(method_instance)), 'unknown')
        }
    
    def list_available_methods(self) -> List[str]:
        """Get list of available method names."""
        return list(self.AVAILABLE_METHODS.keys())
    
    def get_last_results(self) -> Dict[str, Any]:
        """Get results from the last operation."""
        return self.last_results.copy()
    
    def save_results(self, filepath: str, format: str = 'json') -> None:
        """
        Save last results to file.
        
        Args:
            filepath: Output file path
            format: Output format ('json' or 'pickle')
        """
        if not self.last_results:
            raise ValueError("No results to save. Run feature selection first.")
        
        if format == 'json':
            import json
            # Convert sets to lists for JSON serialization
            serializable_results = self._make_json_serializable(self.last_results)
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        elif format == 'pickle':
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.last_results, f)
        else:
            raise ValueError("Format must be 'json' or 'pickle'")
        
        self._log('info', f"Results saved to {filepath}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        else:
            return obj
