# ARL 特征选择算法

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/yourusername/arl-feature-selector)

ARL (Adaptive Rescue Lasso) 算法：一种针对高维数据复杂相关结构的创新特征选择方法。

[English](README.md) | [文档](docs/) | [示例](examples/)

## 🚀 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 一行命令运行

```bash
# 基础用法
python arl_pipeline.py --input data.csv --target target_column --output results.json

# 自定义参数
python arl_pipeline.py --input features.csv --target y \
    --correlation_threshold 0.9 \
    --n_final_clusters 30 \
    --method adaptive_lasso

# 高性能模式
python arl_pipeline.py --input large_data.csv --target target \
    --n_jobs 32 \
    --n_final_clusters 20 \
    --verbose
```

## 📋 目录

- [ARL 算法概述](#arl-算法概述)
- [参数配置](#参数配置)
- [自定义方法集成](#自定义方法集成)
- [复杂度分析](#复杂度分析)
- [API 参考](#api-参考)

## 🧠 ARL 算法概述

ARL (Adaptive Rescue Lasso) 是一种创新的特征选择算法，专门设计用于处理高维数据中的复杂相关结构问题。

### 核心创新

ARL 算法通过以下 5 个阶段解决传统 Lasso 在高相关特征上的局限性：

1. **问题组识别**：自动检测高度相关的特征簇
2. **聚类分析**：使用轮廓系数评估特征组的结构质量
3. **数据分区**：基于高质量特征组创建数据子集
4. **子集分析**：在每个数据子集上独立运行 Lasso 选择
5. **共现模式分析**：识别在多个子集中一致出现的特征模式

### 算法优势

- **处理多重共线性**：有效解决高相关特征导致的选择不稳定问题
- **保持预测性能**：在提高特征选择稳定性的同时保持模型预测能力
- **自适应机制**：根据数据特征自动调整选择策略
- **高效优化**：通过频繁特征预筛选大幅降低共现分析复杂度
- **可扩展性**：支持大规模高维数据处理

## ⚙️ 参数配置

### ARL 核心参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `correlation_threshold` | 0.8 | 0.5-0.99 | 问题组相关性阈值 |
| `silhouette_threshold` | 0.5 | 0.0-1.0 | 聚类质量阈值 |
| `n_final_clusters` | 50 | 2-200 | 数据子集数量 |
| `co_occurrence_threshold` | 1 | 0-10 | 共现模式阈值 |
| `min_group_size` | 2 | 2-20 | 最小问题组大小 |
| `min_subset_size` | 20 | 5-100 | 最小子集样本大小 |

### Adaptive Lasso 超参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `global_cv_folds` | 5 | 2-20 | 全局选择交叉验证折数 |
| `subset_cv_folds` | 3 | 2-10 | 子集选择交叉验证折数 |
| `global_tolerance` | 0.005 | 1e-6 到 1e-1 | 全局收敛容忍度 |
| `subset_tolerance` | 0.01 | 1e-6 到 1e-1 | 子集收敛容忍度 |
| `weight_regularization` | 1e-10 | 1e-12 到 1e-6 | 岭权重正则化 |
| `ridge_alphas` | logspace(-5,5,100) | 数组 | 岭回归 alpha 候选值 |

### Random Lasso 超参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `global_B` | 200 | 10-1000 | 全局随机采样次数 |
| `subset_B` | 50 | 5-200 | 子集随机采样次数 |
| `q1_fraction` | 0.1 | 0.01-0.5 | 特征采样比例（阶段1） |
| `q2_fraction` | 0.1 | 0.01-0.5 | 特征采样比例（阶段2） |
| `alpha_cv_folds` | 5 | 2-10 | alpha 选择的 CV 折数 |
| `alpha_tolerance` | 0.005 | 1e-6 到 1e-1 | Alpha 优化容忍度 |

### Stability Selection 超参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `global_B` | 200 | 10-1000 | 全局子采样次数 |
| `subset_B` | 50 | 5-200 | 子集子采样次数 |
| `selection_threshold` | 0.75 | 0.5-0.99 | 特征选择阈值 |
| `subsample_fraction` | 0.5 | 0.1-0.9 | 子采样比例 |
| `ridge_alpha` | 1.0 | 0.01-100 | 岭正则化参数 |

### LassoCV 超参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `cv_folds` | 5 | 2-20 | 交叉验证折数 |
| `alphas` | None | 数组 | Alpha 候选值（None 为自动） |
| `max_iter` | 2000 | 100-10000 | 最大迭代次数 |
| `tolerance` | 0.005 | 1e-6 到 1e-1 | 收敛容忍度 |

### Elastic Net 超参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `l1_ratio` | 0.5 | 0.0-1.0 | L1 正则化比例 |
| `cv_folds` | 5 | 2-20 | 交叉验证折数 |
| `alphas` | None | 数组 | Alpha 候选值 |
| `max_iter` | 2000 | 100-10000 | 最大迭代次数 |

### 通用参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `random_state` | 42 | 0-2³¹ | 随机种子 |
| `n_jobs` | -1 | -1 到 ∞ | 并行作业数量 |
| `verbose` | True | bool | 详细输出 |

## 🔧 自定义方法集成

### 接口要求

要集成自定义 Lasso 方法到 ARL 框架，需要实现两个核心接口：

1. **GlobalLassoProvider**: 全局特征选择
2. **SubsetLassoProvider**: 子集特征选择

### 实现示例

```python
from lasso_feature_selector.core.base import BaseLassoMethod

class CustomLassoMethod(BaseLassoMethod):
    @property
    def name(self) -> str:
        return "Custom Lasso"

    def global_provider(self):
        def _global_lasso(X, y):
            # 实现全局选择逻辑
            # 返回: (selected_features_set, metadata_dict)
            pass
        return _global_lasso

    def subset_provider(self):
        def _subset_lasso(X_subset, y_subset, metadata):
            # 实现子集选择逻辑
            # 返回: selected_features_set
            pass
        return _subset_lasso
```

### 元数据传递

元数据用于在全局和子集选择之间传递关键参数：

- `global_alpha`: 全局选择的正则化参数
- `coefficients`: 模型系数
- `method_name`: 方法标识符

## 📊 复杂度分析

### ARL 算法时间复杂度

基于算法伪代码和代码实现分析，其中 k 为 Lasso 最大迭代次数（默认2000）：

| 阶段 | 复杂度 | 说明 |
|------|--------|------|
| **1. 全局 Lasso** | O(n·p·k) | 标准 Lasso 复杂度，k为最大迭代次数 |
| **2. 问题组识别** | O(p²) | 相关矩阵计算 + BFS连通分量查找 |
| **3. 聚类分析** | O(G·n·g·c) | G个组，每组g个特征，c次K-means迭代 |
| **4. 数据分区** | O(n·b·c) | n个样本，b个依据特征，c次K-means迭代 |
| **5. 子集 Lasso** | O(m·(n/m)·f·k) | m个子集，f为问题组特征+初选特征数 |
| **6. 共现分析** | O(G·m·2^{g'}) | **已优化**：g'为频繁特征数，g' ≪ g |

**总复杂度**: O(n·p·k + n·f·k + p² + G·n·g·c + G·m·2^{g'})

**关键参数**:
- k = 2000 (Lasso最大迭代次数)
- m = 50 (数据子集数量，n_final_clusters)
- G ≪ p (问题组数量远小于特征数)
- g ≪ p (每组特征数远小于总特征数)
- f ≪ p (问题组特征+初选特征数，远小于总特征数)
- g' ≪ g (频繁特征数，通常为g的20-40%)
- c ≈ 10 (K-means迭代次数)

**关键优化**:
1. **子集Lasso优化**: 仅在问题组特征∪初选特征上运行，特征数从p降到f
2. **共现分析优化**: 预筛选出现≥(threshold+1)次的特征，搜索空间从2^g降到2^{g'}

**实际开销**:
- 聚类分析：由于G·g ≪ p，可忽略不计
- 子集Lasso：由于f ≪ p，复杂度显著低于全局Lasso
- 共现分析：通过频繁特征预筛选，从指数级降到实际可处理水平
- 总体相比标准Lasso增加约50-100%的计算量（而非指数级增长）

## � 使用示例

### Python API

```python
from lasso_feature_selector import LassoFeatureSelector
import pandas as pd

# 加载数据
X = pd.read_csv('features.csv')
y = pd.read_csv('target.csv').squeeze()

# 运行 ARL 算法
selector = LassoFeatureSelector()
selected_features = selector.select_features(
    X, y,
    method='adaptive_lasso',
    correlation_threshold=0.8,
    n_final_clusters=50
)

print(f"选择了 {len(selected_features)} 个特征")
```

## 📖 API 参考

### 主要接口

```python
# 基础用法
selector = LassoFeatureSelector()
selected_features = selector.select_features(X, y, method='adaptive_lasso')

# 自定义参数
selected_features = selector.select_features(
    X, y,
    method='adaptive_lasso',
    correlation_threshold=0.9,
    n_final_clusters=30,
    co_occurrence_threshold=2
)

# 方法比较
comparison = selector.compare_methods(X, y, ['adaptive_lasso', 'random_lasso'])
```

## 🤝 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细指南。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

##  研究与引用

如果您在研究中使用 ARL 算法，请引用：

```bibtex
@software{arl_feature_selector,
  title={ARL: Adaptive Rescue Lasso for High-Dimensional Feature Selection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/arl-feature-selector},
  version={1.0.0}
}
```

## 📞 支持

- 📧 邮箱：your.email@example.com
- 🐛 问题：[GitHub Issues](https://github.com/yourusername/arl-feature-selector/issues)
- 📖 文档：[完整文档](https://arl-feature-selector.readthedocs.io/)

---

**ARL 算法 - 为高维特征选择而生 ❤️**
