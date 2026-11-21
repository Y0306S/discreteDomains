import os
import pandas as pd

import numpy as np
import pandas as pd

METRIC_NAME_MAP = {
    'acc': 'accuracy',
    'accuracy': 'accuracy',
    'auc': 'area under the ROC curve',
    'f1': 'F1 score',
    'mse': 'mean squared error',
    'rmse': 'root mean squared error',
    'mae': 'mean absolute error',
    'r2': 'R-squared',
}
def to_human_metric(metric: str) -> str:
    m = (metric or '').lower()
    return METRIC_NAME_MAP.get(m, m or 'score')


COMMON_LABEL_CANDIDATES = [
    'target','label','y','class','binaryClass','defects','rej','Purchase'
]
def resolve_label_series(df: pd.DataFrame, label_column):
    # 传入明确的 index / -1 / 列名
    if label_column is not None:
        if isinstance(label_column, int):
            return df.iloc[:, label_column], df.columns[label_column]
        else:
            return df[label_column], label_column
    # 未提供：用候选名找
    for name in COMMON_LABEL_CANDIDATES:
        if name in df.columns:
            return df[name], name
    # 仍没有：用最后一列
    return df.iloc[:, -1], df.columns[-1]

# ---- 3) 特征类型统计（含“整数低基数”的 one-hot/类别型检测） ----
def feature_type_counts(df: pd.DataFrame, label_name: str) -> dict:
    X = df.drop(columns=[label_name], errors='ignore')
    n_features = int(X.shape[1])
    if n_features == 0:
        return {'n_features': 0, 'n_numeric': 0, 'n_categorical': 0}

    # object/boolean => 直接按类别特征计数
    obj_bool_cols = list(X.select_dtypes(include=['object','bool']).columns)

    # 数值列中，检测“低基数整数”当作类别（包含 0/1 one-hot）
    num_df = X.select_dtypes(include=['number'])
    cat_like_numeric = []
    for c in num_df.columns:
        ser = num_df[c]
        # 只考虑整数或近似整数
        if pd.api.types.is_integer_dtype(ser) or np.allclose(ser.dropna() % 1, 0):
            uniq = ser.dropna().unique()
            # 小基数阈值：<=10 或 占样本的 <=2%
            if len(uniq) <= 10 or len(uniq) <= max(1, int(0.02 * len(ser))):
                cat_like_numeric.append(c)

    cat_cols = set(obj_bool_cols) | set(cat_like_numeric)
    n_categorical = int(len(cat_cols))
    n_numeric = int(n_features - n_categorical)
    return {
        'n_features': n_features,
        'n_numeric': n_numeric,
        'n_categorical': n_categorical,
    }

# ---- 4) 缺失值画像 ----
def missing_profile(df: pd.DataFrame, label_name: str) -> dict:
    X = df.drop(columns=[label_name], errors='ignore')
    cells = int(X.shape[0] * max(1, X.shape[1]))
    miss_total = int(X.isna().sum().sum())
    overall = float(miss_total / cells) if cells > 0 else 0.0
    per_col = (X.isna().sum() / max(1, X.shape[0])).sort_values(ascending=False)
    top_cols = per_col[per_col > 0].head(5)
    return {
        'missing_rate': round(overall, 6),
        'missing_top': [{'col': c, 'rate': round(float(r), 6)} for c, r in top_cols.items()]
    }

# ---- 5) 分类：类别分布（最多展示前 8 类，其余合并 others） ----
def class_distribution(y: pd.Series, topk=8) -> dict:
    vc = y.value_counts(dropna=False)
    total = int(vc.sum())
    items = [(str(k), int(v)) for k, v in vc.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    head, others = items[:topk], items[topk:]
    dist = {k: round(v/total, 6) for k, v in head}
    if others:
        dist['others'] = round(sum(v for _, v in others)/total, 6)
    return {'n_classes': int(len(vc)), 'distribution': dist}

# ---- 6) 回归：目标统计 ----
def regression_target_stats(y: pd.Series) -> dict:
    arr = pd.to_numeric(y, errors='coerce').values
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {}
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'median': float(np.median(arr)),
        'max': float(np.max(arr)),
    }

# ---- 7) 总画像构建 ----
def build_dataset_profile(df: pd.DataFrame, label_column, task_type_str: str, metric_name: str) -> dict:
    """
    task_type_str: 'CLS' or 'RGS'
    metric_name: original name which will be mapped to human-readable form
    """

    assert task_type_str in ['cls', 'rgs']
    # y : target 1d array , label_name : target column name
    y, label_name = resolve_label_series(df, label_column)
    ft = feature_type_counts(df, label_name)
    miss = missing_profile(df, label_name)

    prof = {
        'n_samples': int(df.shape[0]),
        **ft,
        **miss,
        'task_type': 'classification' if task_type_str == 'CLS' else 'regression',
        'metric': to_human_metric(metric_name),
    }

    if task_type_str == 'cls':
        prof['class_distribution'] = class_distribution(y, topk=8)
    else:
        prof['target_stats'] = regression_target_stats(y)
    return prof





def get_dataset_info(task_type, dataset):
    assert task_type in ['cls', 'rgs']
    datasets_dir = './datasets_bundle'

    if task_type == 'rgs':
        dataset_path = os.path.join(datasets_dir, 'rgs_datasets_downsampled', dataset + '_downsampled.csv')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(datasets_dir, 'rgs_datasets', dataset + '.csv')
    else:
        dataset_path = os.path.join(datasets_dir, 'cls_datasets_downsampled', dataset + '_downsampled.csv')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(datasets_dir, 'cls_datasets', dataset + '.csv')

    label_column = -1
    header = 'infer'

    sep = ','

    return dataset_path, label_column, header, sep
