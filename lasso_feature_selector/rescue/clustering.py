"""
Clustering analysis for feature rescue.

This module provides clustering-based analysis to identify optimal
groupings for feature rescue procedures.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional

from ..core.types import DataFrame, ProblemGroups, ClusterLabels
from ..config.defaults import RESCUE_DEFAULTS

class ClusteringAnalyzer:
    """
    Analyzes clustering potential of problem groups for feature rescue.
    
    This class evaluates whether problem groups can serve as effective
    clustering bases by testing their structural coherence.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize clustering analyzer.
        
        Args:
            **kwargs: Configuration parameters
        """
        config = {**RESCUE_DEFAULTS, **kwargs}
        self.silhouette_threshold = config['silhouette_threshold']
        self.random_state = config['random_state']
        self.verbose = config.get('verbose', True)
        self.n_final_clusters = config['n_final_clusters']
        self.min_subset_size = config['min_subset_size']
    
    def evaluate_problem_groups_as_basis(self, 
                                       data: DataFrame,
                                       problem_groups: ProblemGroups,
                                       silhouette_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Evaluate predefined problem groups as clustering basis.
        
        This method directly evaluates the structural quality of problem groups
        without searching for new groupings.
        
        Args:
            data: Scaled feature data
            problem_groups: Pre-identified problem groups
            silhouette_threshold: Threshold for filtering results
            
        Returns:
            DataFrame with high-quality groups that exceed the threshold
        """
        if silhouette_threshold is None:
            silhouette_threshold = self.silhouette_threshold
        
        if self.verbose:
            print("🕵️ 开始执行基于问题组的评估策略...")

        X = data.copy()

        # Use problem groups as candidate communities
        candidate_groups = problem_groups
        if self.verbose:
            print(f"\\n--- 步骤 1: 加载了 {len(candidate_groups)} 个预定义的问题组作为候选社区。 ---")

        # Evaluate each problem group with k = group_size
        if self.verbose:
            print("\\n--- 步骤 2: 根据 k=组大小 的规则评估各问题组 ---")
        
        results = []
        for group in tqdm(candidate_groups, desc="正在分析问题组", disable=not self.verbose):
            group_size = len(group)
            
            # Minimum size requirement for meaningful clustering
            if group_size < 3:
                continue
            
            # Set k = group size (core logic)
            k = group_size
            sub_X = X[group]
            
            # Ensure sufficient samples for clustering
            if sub_X.shape[0] <= k:
                continue
            
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
                labels = kmeans.fit_predict(sub_X)
                
                # Check if clustering produced multiple clusters
                if len(np.unique(labels)) < 2:
                    continue
                
                score = silhouette_score(sub_X, labels)
                results.append({
                    'group_size_k': k,
                    'silhouette_score': score,
                    'features': ", ".join(group),
                    'feature_list': group
                })
                
            except (ValueError, Exception):
                # Skip groups that cause clustering errors
                continue
        
        if not results:
            if self.verbose:
                print("⚠️ 步骤 2 警告: 未能成功计算任何问题组的轮廓系数。")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results).sort_values(by='silhouette_score', ascending=False).reset_index(drop=True)
        
        if self.verbose:
            print("✅ 步骤 2 完成!")
        
        # Filter by threshold
        if self.verbose:
            print(f"\\n--- 步骤 3: 筛选轮廓系数 > {silhouette_threshold} 的结果 ---")

        final_groups = results_df[results_df['silhouette_score'] > silhouette_threshold]

        if final_groups.empty:
            if self.verbose:
                print(f"❌ 未找到任何轮廓系数高于 {silhouette_threshold} 的高结构性问题组。")
        else:
            if self.verbose:
                print(f"🏆 成功找到 {len(final_groups)} 个高结构性问题组作为聚类依据！")
        
        return final_groups
    
    def partition_data_by_basis_groups(self, X: DataFrame, basis_groups: List[List[str]], 
                                     n_clusters: Optional[int] = None) -> ClusterLabels:
        """
        Partition data based on clustering basis groups.
        
        Args:
            X: Feature matrix
            basis_groups: High-quality groups to use as clustering basis
            n_clusters: Number of final clusters
            
        Returns:
            Cluster labels for each sample
        """
        if n_clusters is None:
            n_clusters = self.n_final_clusters
        
        if self.verbose:
            print(f"\\n🔍 算法洞察 (步骤3): 正在根据 {len(basis_groups)} 个依据组进行最终聚类 (k={n_clusters})...")

        # Extract all features from basis groups
        basis_features = sorted(list(set(feat for group in basis_groups for feat in group)))

        if not basis_features:
            if self.verbose:
                print("  ❌ 错误：依据组中不包含任何特征，无法聚类。")
            return np.array([])

        # Perform final clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init='auto')
        final_clusters = kmeans.fit_predict(X[basis_features])

        if self.verbose:
            print(f"  ✅ 数据已成功划分为 {n_clusters} 个子集。")
        
        return final_clusters
    
    def analyze_cluster_quality(self, X: DataFrame, labels: ClusterLabels) -> Dict[str, Any]:
        """
        Analyze the quality of clustering results.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary with clustering quality metrics
        """
        if len(np.unique(labels)) < 2:
            return {
                'silhouette_score': 0,
                'n_clusters': len(np.unique(labels)),
                'cluster_sizes': [len(labels)],
                'error': 'Only one cluster found'
            }
        
        try:
            silhouette_avg = silhouette_score(X, labels)
        except ValueError:
            silhouette_avg = 0
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        return {
            'silhouette_score': silhouette_avg,
            'n_clusters': len(unique_labels),
            'cluster_sizes': counts.tolist(),
            'min_cluster_size': int(np.min(counts)),
            'max_cluster_size': int(np.max(counts)),
            'avg_cluster_size': float(np.mean(counts)),
            'cluster_size_std': float(np.std(counts))
        }
    
    def find_optimal_clusters(self, X: DataFrame, features: List[str], 
                            k_range: range = range(2, 11)) -> Dict[str, Any]:
        """
        Find optimal number of clusters for a given feature set.
        
        Args:
            X: Feature matrix
            features: List of features to use for clustering
            k_range: Range of k values to test
            
        Returns:
            Dictionary with optimal clustering results
        """
        if not features:
            return {'error': 'No features provided'}
        
        sub_X = X[features]
        results = []
        
        for k in k_range:
            if k >= sub_X.shape[0]:
                continue
            
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
                labels = kmeans.fit_predict(sub_X)
                
                if len(np.unique(labels)) < 2:
                    continue
                
                score = silhouette_score(sub_X, labels)
                results.append({
                    'k': k,
                    'silhouette_score': score,
                    'inertia': kmeans.inertia_
                })
            except (ValueError, Exception):
                continue
        
        if not results:
            return {'error': 'No valid clustering found'}
        
        # Find best k based on silhouette score
        best_result = max(results, key=lambda x: x['silhouette_score'])
        
        return {
            'optimal_k': best_result['k'],
            'best_silhouette_score': best_result['silhouette_score'],
            'all_results': results
        }
