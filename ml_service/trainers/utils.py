from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def detect_types(df: pd.DataFrame, cols: list[str]):
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in cols if c not in numeric]
    return numeric, categorical

def prepare_xy(df: pd.DataFrame, features: list[str], target: str, task: str,
               standardize: bool, for_transformer: bool):
    df = df.copy()
    if target not in df.columns:
        raise ValueError("Target not in columns")
    X_raw = df[features]
    y = df[target]

    # classification: ensure categorical y
    if task == "classification":
        if pd.api.types.is_numeric_dtype(y):
            # keep numeric labels
            pass
        else:
            y = y.astype("category").cat.codes

    num_cols, cat_cols = detect_types(df, features)

    meta = {
        "features": features,
        "target": target,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "task": task,
        "standardize": standardize,
        "for_transformer": for_transformer,
    }

    if for_transformer:
        # transformer handles its own embedding/normalization later
        X = X_raw
        return X, y, meta

    # for sklearn/xgb/nn (tabular MLP) use ColumnTransformer
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler() if standardize else "passthrough", num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

    pre = ColumnTransformer(transformers)
    pipe = Pipeline([("pre", pre)])
    X = pipe.fit_transform(X_raw)

    meta["preprocess"] = pipe
    return X, y.values, meta

def split_data(X, y, val_size=0.1, test_size=0.2, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state)
    return {"X_train": X_train, "X_val": X_val, "X_test": X_test, "y_train": y_train, "y_val": y_val, "y_test": y_test}
