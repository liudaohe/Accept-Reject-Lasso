"""
Co-occurrence based feature rescue.

This module implements the co-occurrence rescue algorithm that identifies
features to rescue based on their joint selection patterns across subsets.
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from typing import Set, List, Dict, Any
from tqdm.auto import tqdm

from ..core.base import BaseRescueMethod
from ..core.types import DataFrame, Series, FeatureSet, Metadata, ProblemGroups, CoOccurrenceCounter
from ..config.defaults import RESCUE_DEFAULTS

class CoOccurrenceRescue(BaseRescueMethod):
    """
    Co-occurrence based feature rescue method.
    
    This method rescues features based on their co-occurrence patterns
    in subset selections, identifying features that are consistently
    selected together across different data subsets.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize co-occurrence rescue method.
        
        Args:
            **kwargs: Configuration overrides
        """
        config = {**RESCUE_DEFAULTS, **kwargs}
        super().__init__(**config)
        
        self.co_occurrence_threshold = config['co_occurrence_threshold']
        self.n_final_clusters = config['n_final_clusters']
        self.min_subset_size = config['min_subset_size']
    
    @property
    def name(self) -> str:
        """Return the method name."""
        return "Co-occurrence Rescue"
    
    def identify_problem_groups(self, X: DataFrame, **kwargs) -> ProblemGroups:
        """
        Use the base class implementation for problem group identification.
        
        Args:
            X: Feature matrix
            **kwargs: Additional parameters
            
        Returns:
            List of problem groups
        """
        return self._find_connected_components(self._build_correlation_graph(X))
    
    def rescue_features_by_co_occurrence(self,
                                       problem_groups: ProblemGroups,
                                       subset_selections: List[FeatureSet],
                                       threshold: int = None) -> FeatureSet:
        """
        Execute co-occurrence rescue algorithm with unified Apriori-based pruning.

        Args:
            problem_groups: List of problem groups
            subset_selections: Feature selections from different subsets
            threshold: Co-occurrence threshold (default: use instance threshold)

        Returns:
            Set of rescued features
        """
        if threshold is None:
            threshold = self.co_occurrence_threshold

        if self.parameters.get('verbose', True):
            print(f"\\n🔍 算法洞察 (步骤5): 正在根据子集分析结果拯救特征 (共现阈值: > {threshold})...")

        rescued_features = set()

        # Use unified Apriori-based algorithm if enabled
        if self.parameters.get('use_unified_apriori', True):
            rescued_features = self._unified_apriori_rescue(
                problem_groups, subset_selections, threshold
            )
        else:
            # Fallback to original implementation
            rescued_features = self._original_co_occurrence_rescue(
                problem_groups, subset_selections, threshold
            )

        if self.parameters.get('verbose', True):
            print(f"  ✅ 成功拯救了 {len(rescued_features)} 个新特征。")

        return rescued_features
    
    def rescue_features(self, 
                       X: DataFrame,
                       y: Series,
                       base_features: FeatureSet,
                       problem_groups: ProblemGroups,
                       **kwargs) -> FeatureSet:
        """
        Main rescue method that orchestrates the entire rescue process.
        
        Args:
            X: Feature matrix
            y: Target vector
            base_features: Initially selected features
            problem_groups: Identified problem groups
            **kwargs: Additional parameters including subset_lasso_provider and metadata
            
        Returns:
            Set of rescued features
        """
        # Extract required parameters
        subset_lasso_provider = kwargs.get('subset_lasso_provider')
        metadata = kwargs.get('metadata', {})
        clustering_basis_groups = kwargs.get('clustering_basis_groups', [])
        
        if subset_lasso_provider is None:
            raise ValueError("subset_lasso_provider is required for feature rescue")
        
        if not problem_groups:
            return set()
        
        # Use clustering basis groups if provided, otherwise use all problem groups
        if not clustering_basis_groups:
            if self.parameters.get('verbose', True):
                print("\\n  ⚠️ 警告: 外部未提供聚类依据组，使用所有问题组作为后备方案。")
            clustering_basis_groups = problem_groups
        
        # Import clustering analyzer for data partitioning
        from .clustering import ClusteringAnalyzer
        clustering_analyzer = ClusteringAnalyzer(
            n_final_clusters=self.n_final_clusters,
            random_state=self.random_state,
            verbose=self.parameters.get('verbose', True)
        )
        
        # Partition data based on clustering basis groups
        final_clusters = clustering_analyzer.partition_data_by_basis_groups(
            X, clustering_basis_groups, self.n_final_clusters
        )
        
        if final_clusters.size == 0:
            return set()
        
        # Run Lasso on subsets
        if self.parameters.get('verbose', True):
            print(f"\\n🔍 算法洞察 (步骤4): 正在对 {self.n_final_clusters} 个子集运行Lasso分析...")
        
        # Prepare features for subset analysis
        features_for_subset_analysis = sorted(
            list(set(feat for group in problem_groups for feat in group) | base_features)
        )
        
        subset_selections = []
        for i in tqdm(range(self.n_final_clusters), desc="  分析数据子集", 
                     disable=not self.parameters.get('verbose', True)):
            subset_indices = np.where(final_clusters == i)[0]
            
            if len(subset_indices) < self.min_subset_size:
                continue
            
            X_subset = X.iloc[subset_indices][features_for_subset_analysis]
            y_subset = y.iloc[subset_indices]
            
            # Run subset Lasso provider
            selected_set = subset_lasso_provider(X_subset, y_subset, metadata)
            
            if selected_set:
                subset_selections.append(selected_set)
        
        if self.parameters.get('verbose', True):
            print(f"  ✅ 在 {len(subset_selections)} 个子集上完成了Lasso分析。")
        
        # Execute co-occurrence rescue
        rescued_features = self.rescue_features_by_co_occurrence(
            problem_groups, subset_selections, self.co_occurrence_threshold
        )
        
        return rescued_features

    def _unified_apriori_rescue(self,
                               problem_groups: ProblemGroups,
                               subset_selections: List[FeatureSet],
                               threshold: int) -> FeatureSet:
        """
        统一的Apriori算法实现，从1-项集开始逐层剪枝到k-项集。

        核心思想：
        1. 从1-项集开始，统计每个特征的出现频率
        2. 只有频繁的1-项集才能组成候选2-项集
        3. 只有频繁的2-项集才能组成候选3-项集
        4. 以此类推，大幅减少需要检查的组合数量

        Apriori原理：如果一个k-项集不频繁，那么包含它的所有(k+1)-项集都不频繁

        Args:
            problem_groups: 问题组列表
            subset_selections: 子集选择结果
            threshold: 共现阈值

        Returns:
            拯救的特征集合
        """
        if self.parameters.get('verbose', True):
            print(f"  🚀 使用统一Apriori算法进行频繁项集挖掘...")

        rescued_features = set()
        total_combinations_checked = 0
        total_combinations_pruned = 0

        for group_idx, group in enumerate(problem_groups):
            if len(group) < 2:
                continue

            # 为当前组构建事务数据库（每个子集选择是一个事务）
            transactions = []
            for selection in subset_selections:
                intersection = set(group).intersection(selection)
                if intersection:
                    transactions.append(intersection)

            if not transactions:
                continue

            # 执行Apriori算法，从1-项集开始逐层剪枝
            frequent_itemsets, stats = self._apriori_frequent_mining(
                transactions, threshold, sorted(list(group))
            )

            # 统计信息
            checked, pruned = stats
            total_combinations_checked += checked
            total_combinations_pruned += pruned

            # 收集所有频繁项集中的特征（只考虑2-项集及以上）
            for size, itemsets in frequent_itemsets.items():
                if size >= 2:
                    for itemset in itemsets:
                        rescued_features.update(itemset)

        if self.parameters.get('verbose', True):
            print(f"  📊 Apriori统计: 检查了 {total_combinations_checked} 个组合，剪枝了 {total_combinations_pruned} 个组合")
            if total_combinations_checked + total_combinations_pruned > 0:
                pruning_ratio = total_combinations_pruned / (total_combinations_checked + total_combinations_pruned)
                print(f"  📊 剪枝效率: {pruning_ratio:.2%}")

        return rescued_features

    def _apriori_frequent_mining(self, transactions: List[Set[str]],
                                min_support: int,
                                all_items: List[str]) -> tuple:
        """
        改进的Apriori频繁项集挖掘算法

        从1-项集开始，逐层向上挖掘频繁项集，充分利用Apriori原理进行剪枝：
        如果一个k-项集不频繁，那么包含它的所有(k+1)-项集都不频繁

        Args:
            transactions: 事务列表，每个事务是一个特征集合
            min_support: 最小支持度（绝对频次）
            all_items: 所有可能的项目列表

        Returns:
            (frequent_itemsets_dict, (combinations_checked, combinations_pruned))
        """
        frequent_itemsets = {}
        combinations_checked = 0
        combinations_pruned = 0

        # 第1层：统计1-项集频率（这是所有后续剪枝的基础）
        item_counts = Counter()
        for transaction in transactions:
            for item in transaction:
                if item in all_items:
                    item_counts[item] += 1

        # 找出频繁1-项集
        frequent_1_itemsets = []
        for item, count in item_counts.items():
            combinations_checked += 1
            if count > min_support:
                frequent_1_itemsets.append(frozenset([item]))
            else:
                combinations_pruned += 1  # 不频繁的1-项集被剪枝

        if not frequent_1_itemsets:
            return {}, (combinations_checked, combinations_pruned)

        frequent_itemsets[1] = frequent_1_itemsets

        # 从2-项集开始迭代，每一层都基于上一层的频繁项集
        k = 2
        current_frequent = frequent_1_itemsets

        while current_frequent and k <= len(all_items):
            # 生成候选k-项集（只从频繁(k-1)-项集生成）
            candidates = self._generate_candidates_apriori(current_frequent, k)

            if not candidates:
                break

            # 计算候选项集的支持度
            candidate_counts = Counter()
            for transaction in transactions:
                for candidate in candidates:
                    if candidate.issubset(transaction):
                        candidate_counts[candidate] += 1

            # 找出频繁k-项集
            frequent_k_itemsets = []
            for candidate in candidates:
                combinations_checked += 1
                count = candidate_counts.get(candidate, 0)
                if count > min_support:
                    frequent_k_itemsets.append(candidate)
                else:
                    combinations_pruned += 1  # 不频繁的k-项集被剪枝

            if frequent_k_itemsets:
                frequent_itemsets[k] = frequent_k_itemsets
                current_frequent = frequent_k_itemsets
            else:
                current_frequent = []

            k += 1

        return frequent_itemsets, (combinations_checked, combinations_pruned)

    def _generate_candidates_apriori(self, frequent_k_minus_1: List[frozenset], k: int) -> List[frozenset]:
        """
        基于Apriori原理从(k-1)-频繁项集生成k-候选项集

        核心思想：
        1. 连接步骤：两个(k-1)-项集如果前(k-2)个元素相同，则可以连接生成k-项集
        2. 剪枝步骤：如果候选k-项集的任何(k-1)-子集不频繁，则剪枝该候选

        Args:
            frequent_k_minus_1: (k-1)-频繁项集列表
            k: 当前项集大小

        Returns:
            k-候选项集列表
        """
        candidates = []
        n = len(frequent_k_minus_1)

        # 连接步骤：从频繁(k-1)-项集生成候选k-项集
        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = sorted(list(frequent_k_minus_1[i]))
                itemset2 = sorted(list(frequent_k_minus_1[j]))

                # 只有当两个(k-1)-项集的前(k-2)个元素相同时才能连接
                if k == 2:
                    # 对于2-项集，任意两个1-项集都可以连接
                    candidate = frozenset(itemset1 + itemset2)
                    candidates.append(candidate)
                else:
                    # 对于k>=3的项集，需要前(k-2)个元素相同
                    if itemset1[:-1] == itemset2[:-1]:
                        candidate = frozenset(itemset1 + [itemset2[-1]])

                        # 剪枝步骤：检查候选项集的所有(k-1)-子集是否都是频繁的
                        if not self._has_infrequent_subset_apriori(candidate, frequent_k_minus_1, k-1):
                            candidates.append(candidate)
                        # 如果有不频繁的子集，则该候选被剪枝（不添加到candidates中）

        return candidates

    def _has_infrequent_subset_apriori(self, candidate: frozenset,
                                      frequent_k_minus_1: List[frozenset],
                                      k_minus_1: int) -> bool:
        """
        基于Apriori原理检查候选项集是否包含非频繁的(k-1)-子集

        Apriori原理：如果一个项集不频繁，那么包含它的所有超集都不频繁
        反过来说：如果一个项集要频繁，那么它的所有子集都必须频繁

        Args:
            candidate: 候选k-项集
            frequent_k_minus_1: 频繁(k-1)-项集列表
            k_minus_1: 子集大小

        Returns:
            如果包含非频繁子集则返回True（需要剪枝）
        """
        candidate_list = list(candidate)
        frequent_set = set(frequent_k_minus_1)

        # 检查候选项集的所有(k-1)-子集是否都在频繁(k-1)-项集中
        for subset in combinations(candidate_list, k_minus_1):
            if frozenset(subset) not in frequent_set:
                # 发现非频繁子集，根据Apriori原理，该候选必须剪枝
                return True

        # 所有子集都频繁，该候选可以保留
        return False

    def _original_co_occurrence_rescue(self,
                                     problem_groups: ProblemGroups,
                                     subset_selections: List[FeatureSet],
                                     threshold: int) -> FeatureSet:
        """
        原始的共现拯救算法实现（作为后备方案）
        """
        # 特征预筛选
        feature_counts = Counter()
        for selection in subset_selections:
            for feature in selection:
                feature_counts[feature] += 1

        min_appearances = max(2, threshold + 1)
        frequent_features = {feat for feat, count in feature_counts.items() if count >= min_appearances}

        if self.parameters.get('verbose', True):
            total_unique_features = len(feature_counts)
            print(f"  📊 优化统计: {total_unique_features} 个唯一特征中，{len(frequent_features)} 个出现≥{min_appearances}次")

        rescued_features = set()
        co_occurrence_counter = Counter()

        # 分析每个问题组内的共现模式
        for group in problem_groups:
            group_frequent = set(group).intersection(frequent_features)

            if len(group_frequent) < 2:
                continue

            # 检查与每个子集选择的交集
            for selection in subset_selections:
                intersection = group_frequent.intersection(selection)

                if len(intersection) < 2:
                    continue

                # 生成所有可能的大小≥2的子集
                for k in range(2, len(intersection) + 1):
                    for subset in combinations(sorted(list(intersection)), k):
                        co_occurrence_counter[frozenset(subset)] += 1

        # 拯救超过阈值的特征
        for feature_set, count in co_occurrence_counter.items():
            if count > threshold:
                rescued_features.update(feature_set)

        return rescued_features

    def _co_occurrence_with_pruning(self,
                                   problem_groups: ProblemGroups,
                                   subset_selections: List[FeatureSet],
                                   frequent_features: FeatureSet,
                                   threshold: int) -> FeatureSet:
        """
        Co-occurrence analysis with hierarchical pruning optimization.

        This method uses a bottom-up approach:
        1. First check all 2-feature combinations
        2. Only check larger combinations that contain at least one valid 2-feature pair

        Args:
            problem_groups: List of problem groups
            subset_selections: Feature selections from different subsets
            frequent_features: Pre-filtered frequent features
            threshold: Co-occurrence threshold

        Returns:
            Set of rescued features
        """
        rescued_features = set()

        if self.parameters.get('verbose', True):
            print(f"  🚀 使用分层剪枝优化进行共现分析...")

        total_combinations_checked = 0
        total_combinations_pruned = 0

        for group_idx, group in enumerate(problem_groups):
            # Only consider frequent features within this group
            group_frequent = set(group).intersection(frequent_features)

            if len(group_frequent) < 2:
                continue

            group_frequent_list = sorted(list(group_frequent))

            # Step 1: Check all 2-feature combinations and identify valid pairs
            valid_pairs = set()
            pair_counts = {}

            for pair in combinations(group_frequent_list, 2):
                count = 0
                for selection in subset_selections:
                    if set(pair).issubset(selection):
                        count += 1

                total_combinations_checked += 1
                pair_counts[frozenset(pair)] = count

                if count > threshold:
                    rescued_features.update(pair)
                    valid_pairs.add(frozenset(pair))

            # Step 2: For larger combinations, only check those containing valid pairs
            for k in range(3, len(group_frequent_list) + 1):
                for subset in combinations(group_frequent_list, k):
                    subset_set = set(subset)

                    # Pruning: Check if this subset contains at least one valid 2-feature pair
                    contains_valid_pair = False
                    for pair in combinations(subset, 2):
                        if frozenset(pair) in valid_pairs:
                            contains_valid_pair = True
                            break

                    if not contains_valid_pair:
                        total_combinations_pruned += 1
                        continue  # Prune this combination

                    # Count co-occurrences for this combination
                    count = 0
                    for selection in subset_selections:
                        if subset_set.issubset(selection):
                            count += 1

                    total_combinations_checked += 1

                    if count > threshold:
                        rescued_features.update(subset)

        if self.parameters.get('verbose', True):
            print(f"  📊 剪枝统计: 检查了 {total_combinations_checked} 个组合，剪枝了 {total_combinations_pruned} 个组合")
            if total_combinations_checked + total_combinations_pruned > 0:
                pruning_ratio = total_combinations_pruned / (total_combinations_checked + total_combinations_pruned)
                print(f"  📊 剪枝比例: {pruning_ratio:.2%}")

        return rescued_features

    def analyze_co_occurrence_patterns(self,
                                     problem_groups: ProblemGroups,
                                     subset_selections: List[FeatureSet]) -> Dict[str, Any]:
        """
        Analyze co-occurrence patterns without rescuing features.
        
        Args:
            problem_groups: List of problem groups
            subset_selections: Feature selections from different subsets
            
        Returns:
            Dictionary with co-occurrence analysis results
        """
        co_occurrence_counter = Counter()
        group_analysis = []
        
        for group_idx, group in enumerate(problem_groups):
            group_set = set(group)
            group_co_occurrences = []
            
            for selection in subset_selections:
                intersection = group_set.intersection(selection)
                
                if len(intersection) >= 2:
                    for k in range(2, len(intersection) + 1):
                        for subset in combinations(sorted(list(intersection)), k):
                            co_occurrence_counter[frozenset(subset)] += 1
                            group_co_occurrences.append(subset)
            
            group_analysis.append({
                'group_id': group_idx,
                'group_size': len(group),
                'features': group,
                'co_occurrences_found': len(group_co_occurrences),
                'unique_co_occurrence_patterns': len(set(group_co_occurrences))
            })
        
        # Analyze threshold effects
        threshold_analysis = {}
        for threshold in range(0, 6):
            rescued_count = sum(1 for count in co_occurrence_counter.values() if count > threshold)
            rescued_features = set()
            for feature_set, count in co_occurrence_counter.items():
                if count > threshold:
                    rescued_features.update(feature_set)
            
            threshold_analysis[threshold] = {
                'rescued_patterns': rescued_count,
                'rescued_features': len(rescued_features),
                'rescued_feature_list': list(rescued_features)
            }
        
        return {
            'total_co_occurrence_patterns': len(co_occurrence_counter),
            'group_analysis': group_analysis,
            'threshold_analysis': threshold_analysis,
            'co_occurrence_counter': dict(co_occurrence_counter),
            'most_frequent_patterns': co_occurrence_counter.most_common(10)
        }
    
    def get_rescue_statistics(self, 
                            base_features: FeatureSet,
                            rescued_features: FeatureSet,
                            problem_groups: ProblemGroups) -> Dict[str, Any]:
        """
        Calculate statistics about the rescue process.
        
        Args:
            base_features: Originally selected features
            rescued_features: Features rescued by the process
            problem_groups: Problem groups used in rescue
            
        Returns:
            Dictionary with rescue statistics
        """
        total_features_in_groups = set(feat for group in problem_groups for feat in group)
        
        return {
            'base_feature_count': len(base_features),
            'rescued_feature_count': len(rescued_features),
            'total_final_features': len(base_features | rescued_features),
            'rescue_rate': len(rescued_features) / len(total_features_in_groups) if total_features_in_groups else 0,
            'overlap_with_base': len(base_features & rescued_features),
            'new_features_rescued': len(rescued_features - base_features),
            'problem_groups_count': len(problem_groups),
            'features_in_problem_groups': len(total_features_in_groups)
        }
