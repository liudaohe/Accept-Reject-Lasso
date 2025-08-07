# Accept-Reject Lasso 特征选择算法

[![paper](https://img.shields.io/badge/arXiv-2508.04646-b31b1b.svg)](https://arxiv.org/abs/2508.04646)


## 快速开始

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

## 目录

- [ARL 算法概述](#arl-算法概述)
- [参数配置](#参数配置)
- [自定义方法集成](#自定义方法集成)
- [API 参考](#api-参考)

## ARL 算法概述

ARL (Accept-Reject Lasso) 是一种创新的特征选择算法，用以解决Lasso在高相关性特征之间的选择不确定性

### 算法优势

通过合适的超参数
能够解决Lasso方法在一组相关特征组随意选择其中一部分作为代表从而导致可能遗漏特征的问题
且不引入过多的多重共线性问题

## ⚙️ 参数配置

### ARL 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `correlation_threshold` | 0.8 | 问题组相关性阈值 |
| `silhouette_threshold` | 0.5 | 聚类质量阈值 |
| `n_final_clusters` | 50 | 数据子集数量 |
| `co_occurrence_threshold` | 1 | 0-10 | 共现模式阈值 |
| `min_subset_size` | 20 | 最小子集样本大小 |

### 基准Lasso 默认超参数设置可见文章


### 通用参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|
| `random_state` | 42 | 随机种子 |
| `n_jobs` | 64 | 并行作业数量 |

## 自定义方法集成

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



## 使用示例

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

## API 参考

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


## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

##  研究与引用

如果您在研究中使用 ARL 算法，请引用：

```bibtex
@misc{Accept-Reject-Lasso,
      title={Accept-Reject Lasso}, 
      author={Yanxin Liu and Yunqi Zhang},
      year={2025},
      eprint={2508.04646},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
```


