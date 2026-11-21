import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_RUNS_DIR = ROOT / "experiment_runs"

import numpy as np
import pandas as pd
import sklearn
import openml
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------------------------
# Version-safe OneHotEncoder kwargs.
# sklearn < 1.2 uses `sparse=False`; sklearn >= 1.2 renamed it `sparse_output`.
# ---------------------------------------------------------------------------
_sklearn_version = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
_OHE_SPARSE_KWARG = (
    {'sparse_output': False} if _sklearn_version >= (1, 2) else {'sparse': False}
)


class DataProfiler:
    def __init__(self, subsample=2000):
        self.subsample = subsample

    def _normalize_task_type(self, task_type_raw, task_type_id=None):
        """Map OpenML task fields to a canonical task type label."""
        text = str(task_type_raw).lower() if task_type_raw is not None else ""
        if "classification" in text:
            return "classification"
        if "regression" in text:
            return "regression"
        if task_type_id == 1:
            return "classification"
        if task_type_id == 2:
            return "regression"
        return "unknown"

    def _get_task_type_from_openml(self, task_id):
        """Best-effort OpenML lookup for task type."""
        if task_id is None:
            return None, "inferred"
        try:
            task = openml.tasks.get_task(int(task_id))
            task_type = self._normalize_task_type(
                getattr(task, "task_type", None),
                getattr(task, "task_type_id", None)
            )
            if task_type != "unknown":
                return task_type, "openml"
        except Exception:
            pass
        return None, "inferred"

    def _get_task_type(self, y):
        """Infer classification vs regression from the target vector."""
        if (
            y.dtype == 'object'
            or str(y.dtype).startswith('cat')
            or (y.nunique() < 20 and y.dtype == 'int')
        ):
            return 'classification'
        return 'regression'

    def _get_preprocessor(self, X):
        num_cols = X.select_dtypes(include=['number']).columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns

        num_pipe = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scl', StandardScaler())
        ])

        cat_pipe = Pipeline([
            ('imp', SimpleImputer(strategy='constant', fill_value='missing')),
            ('enc', OneHotEncoder(handle_unknown='ignore', **_OHE_SPARSE_KWARG))
        ])

        return ColumnTransformer([
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ])

    def _get_feature_breakdown(self, X):
        """Count feature types for prompt-level dataset description."""
        numeric = int(len(X.select_dtypes(include=['number']).columns))
        categorical = int(len(X.select_dtypes(include=['object', 'category']).columns))
        boolean = int(len(X.select_dtypes(include=['bool']).columns))
        known = numeric + categorical + boolean
        other = max(int(X.shape[1]) - known, 0)
        return {
            "numeric": numeric,
            "categorical_or_string": categorical,
            "boolean": boolean,
            "other": other
        }

    def _get_class_distribution(self, y):
        """Return exact class distribution percentages by class label."""
        if len(y) == 0:
            return {}
        counts = y.astype(str).value_counts(dropna=False).sort_index()
        total = float(counts.sum())
        return {label: round((count / total) * 100.0, 4) for label, count in counts.items()}

    def profile(self, X, y, user_desc="", task_id=None):
        # Subsample for speed
        if len(X) > self.subsample:
            X_small = X.sample(n=self.subsample, random_state=42)
            y_small = y[X_small.index]
        else:
            X_small, y_small = X, y

        task_type, task_type_source = self._get_task_type_from_openml(task_id)
        if task_type is None:
            task_type = self._get_task_type(y_small)
        prep      = self._get_preprocessor(X_small)

        if task_type == 'classification':
            probes = {
                "Linear": LogisticRegression(max_iter=1000, solver='lbfgs'),
                "KNN":    KNeighborsClassifier(n_neighbors=3),
                "Tree":   DecisionTreeClassifier(max_depth=2)
            }
            metric = 'balanced_accuracy'
        else:
            probes = {
                "Linear": Ridge(),
                "KNN":    KNeighborsRegressor(n_neighbors=3),
                "Tree":   DecisionTreeRegressor(max_depth=2)
            }
            metric = 'r2'

        class_distribution = self._get_class_distribution(y) if task_type == 'classification' else {}
        feature_breakdown = self._get_feature_breakdown(X)

        scores = {}
        for name, model in probes.items():
            try:
                pipe = Pipeline([('prep', prep), ('clf', model)])
                s    = np.mean(cross_val_score(pipe, X_small, y_small, cv=3, scoring=metric))
                scores[name] = round(float(s), 4)
            except Exception:
                scores[name] = 0.0

        return {
            "meta": {
                "description": user_desc,
                "rows":        int(X.shape[0]),
                "features":    int(X.shape[1]),
                "sparsity":    round((X.isna().sum().sum() / X.size), 3) if X.size > 0 else 0,
                "task_type":   task_type,
                "task_type_source": task_type_source,
                "metric_name": metric,
                "metric_direction": "higher_is_better" if metric in ("balanced_accuracy", "r2") else "lower_is_better",
                "feature_breakdown": feature_breakdown,
                "num_classes": int(y.nunique()) if task_type == 'classification' else 1,
                "class_distribution": class_distribution
            },
            "behavior": scores
        }
