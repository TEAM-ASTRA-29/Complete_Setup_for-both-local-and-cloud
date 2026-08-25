"""
backend/preprocessor.py
========================
Full preprocessing pipeline — works on raw OR pre-processed CSVs.
Exposes a DataProfiler and a PreprocessingPipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# ─────────────────────────────────────────────────────────────
# DATA PROFILER
# ─────────────────────────────────────────────────────────────
class DataProfiler:
    """Analyse a dataframe and return a structured profile."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def profile(self) -> dict:
        df = self.df
        n_rows, n_cols = df.shape
        numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols       = df.select_dtypes(include=["object","category"]).columns.tolist()
        bool_cols      = df.select_dtypes(include=["bool"]).columns.tolist()

        null_counts    = df.isnull().sum()
        null_pct       = (null_counts / n_rows * 100).round(2)
        dup_rows       = int(df.duplicated().sum())

        col_details = []
        for col in df.columns:
            dtype   = str(df[col].dtype)
            n_null  = int(null_counts[col])
            n_uniq  = int(df[col].nunique())
            is_num  = col in numeric_cols

            detail = {
                "column": col, "dtype": dtype,
                "nulls": n_null, "null_pct": round(null_pct[col], 2),
                "unique": n_uniq,
                "is_numeric": is_num,
                "is_categorical": col in cat_cols,
            }
            if is_num:
                detail.update({
                    "mean":   round(float(df[col].mean()), 4),
                    "std":    round(float(df[col].std()),  4),
                    "min":    round(float(df[col].min()),  4),
                    "max":    round(float(df[col].max()),  4),
                    "median": round(float(df[col].median()),4),
                    "skew":   round(float(df[col].skew()), 4),
                })
            col_details.append(detail)

        # Detect if data looks already preprocessed
        all_numeric      = len(cat_cols) == 0 and len(bool_cols) == 0
        no_nulls         = null_counts.sum() == 0
        values_0_to_1    = all(df[c].between(0,1).all() for c in numeric_cols[:5]) if numeric_cols else False
        looks_preprocessed = all_numeric and no_nulls

        return {
            "n_rows": n_rows, "n_cols": n_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": cat_cols,
            "bool_cols": bool_cols,
            "null_total": int(null_counts.sum()),
            "duplicate_rows": dup_rows,
            "col_details": col_details,
            "looks_preprocessed": looks_preprocessed,
            "all_numeric": all_numeric,
            "values_0_to_1": values_0_to_1,
        }


# ─────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────
class PreprocessingPipeline:
    """
    Flexible pipeline that handles both raw and pre-processed data.
    Steps are individually togglable.
    """

    def __init__(self, config: dict):
        """
        config keys:
          target_col      : str
          drop_cols       : list[str]
          handle_nulls    : 'mean' | 'median' | 'mode' | 'drop' | 'none'
          handle_cats     : 'label' | 'onehot' | 'drop' | 'none'
          handle_dups     : bool
          scale_method    : 'standard' | 'minmax' | 'none'
          drop_low_var    : bool
          already_clean   : bool  (skip most steps)
        """
        self.cfg = config
        self.encoders  = {}
        self.scaler    = None
        self.log       = []

    def fit_transform(self, df: pd.DataFrame):
        cfg    = self.cfg
        target = cfg["target_col"]
        df     = df.copy()

        # Drop user-specified columns
        drop = [c for c in cfg.get("drop_cols", []) if c in df.columns and c != target]
        if drop:
            df.drop(columns=drop, inplace=True)
            self.log.append(f"Dropped columns: {drop}")

        # Separate target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found.")

        y  = df[target].copy()
        X  = df.drop(columns=[target])

        # If data already clean, skip heavy steps
        already_clean = cfg.get("already_clean", False)

        # ── Duplicate rows ──
        if not already_clean and cfg.get("handle_dups", False):
            before = len(X)
            mask   = ~X.duplicated()
            X, y   = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
            self.log.append(f"Removed {before - len(X)} duplicate rows")

        # ── Categorical encoding ──
        cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
        handle_cats = cfg.get("handle_cats", "label")

        if cat_cols and handle_cats != "none":
            if already_clean:
                X.drop(columns=cat_cols, inplace=True, errors="ignore")
                self.log.append(f"Skipped encoding (data marked as preprocessed): dropped {cat_cols}")
            elif handle_cats == "label":
                for col in cat_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.encoders[col] = list(le.classes_)
                self.log.append(f"Label-encoded: {cat_cols}")
            elif handle_cats == "onehot":
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
                self.log.append(f"One-hot encoded: {cat_cols}")
            elif handle_cats == "drop":
                X.drop(columns=cat_cols, inplace=True, errors="ignore")
                self.log.append(f"Dropped categorical cols: {cat_cols}")

        # ── Null handling ──
        handle_nulls = cfg.get("handle_nulls", "mean")
        if not already_clean and handle_nulls != "none":
            null_cols = X.columns[X.isnull().any()].tolist()
            if null_cols:
                if handle_nulls == "drop":
                    before = len(X)
                    mask   = X.notna().all(axis=1)
                    X, y   = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
                    self.log.append(f"Dropped {before-len(X)} rows with nulls")
                elif handle_nulls == "mean":
                    for c in null_cols:
                        if X[c].dtype in [np.float64, np.int64, float, int]:
                            X[c] = X[c].fillna(X[c].mean())
                    self.log.append(f"Mean-filled nulls in: {null_cols}")
                elif handle_nulls == "median":
                    for c in null_cols:
                        if X[c].dtype in [np.float64, np.int64, float, int]:
                            X[c] = X[c].fillna(X[c].median())
                    self.log.append(f"Median-filled nulls in: {null_cols}")
                elif handle_nulls == "mode":
                    for c in null_cols:
                        X[c] = X[c].fillna(X[c].mode()[0])
                    self.log.append(f"Mode-filled nulls in: {null_cols}")

        # Drop target nulls
        null_y = y.isnull()
        if null_y.any():
            X = X[~null_y].reset_index(drop=True)
            y = y[~null_y].reset_index(drop=True)
            self.log.append(f"Dropped {null_y.sum()} rows with null target")

        # ── Final numeric fill (safety net) ──
        for c in X.select_dtypes(include=[np.number]).columns:
            if X[c].isnull().any():
                X[c] = X[c].fillna(X[c].mean())

        # ── Low variance drop ──
        if not already_clean and cfg.get("drop_low_var", False):
            var    = X.var()
            low_v  = var[var < 1e-6].index.tolist()
            if low_v:
                X.drop(columns=low_v, inplace=True)
                self.log.append(f"Dropped low-variance cols: {low_v}")

        # ── Feature scaling ──
        scale_method = cfg.get("scale_method", "none")
        if scale_method == "standard":
            self.scaler = StandardScaler()
            X_scaled    = pd.DataFrame(self.scaler.fit_transform(X),
                                        columns=X.columns)
            self.log.append("Applied StandardScaler")
            X = X_scaled
        elif scale_method == "minmax":
            self.scaler = MinMaxScaler()
            X_scaled    = pd.DataFrame(self.scaler.fit_transform(X),
                                        columns=X.columns)
            self.log.append("Applied MinMaxScaler")
            X = X_scaled

        self.feature_names = list(X.columns)
        return X, y

    def get_log(self) -> list:
        return self.log
