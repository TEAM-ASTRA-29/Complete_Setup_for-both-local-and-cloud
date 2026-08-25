"""
backend/models.py
=================
Model registry, builder, trainer, and smart recommender.
"""

import time, os, joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                              accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, roc_auc_score)
# Regression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
# Classification
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# REGISTRIES
# ─────────────────────────────────────────────────────────────
REGRESSION_MODELS = {
    "Random Forest":       "rf",
    "XGBoost":             "xgb",
    "Gradient Boosting":   "gbm",
    "Extra Trees":         "et",
    "AdaBoost":            "ada",
    "Decision Tree":       "dt",
    "Ridge Regression":    "ridge",
    "Lasso Regression":    "lasso",
    "ElasticNet":          "enet",
    "Linear Regression":   "lr",
    "SVR (SVM)":           "svr",
    "K-Nearest Neighbors": "knn",
    "MLP Neural Network":  "mlp",
}

CLASSIFICATION_MODELS = {
    "Random Forest":       "rf",
    "XGBoost":             "xgb",
    "Gradient Boosting":   "gbm",
    "Extra Trees":         "et",
    "AdaBoost":            "ada",
    "Decision Tree":       "dt",
    "Logistic Regression": "logreg",
    "SVC (SVM)":           "svc",
    "K-Nearest Neighbors": "knn",
    "MLP Neural Network":  "mlp",
}

# Descriptions shown to user
MODEL_INFO = {
    "rf":     {"full":"Random Forest",       "type":"Ensemble (Bagging)",
               "strengths":"Robust, handles high dims, low overfitting",
               "weaknesses":"Slow on very large data, memory heavy"},
    "xgb":    {"full":"XGBoost",             "type":"Ensemble (Boosting)",
               "strengths":"Best accuracy on tabular data, fast",
               "weaknesses":"Many hyperparameters to tune"},
    "gbm":    {"full":"Gradient Boosting",   "type":"Ensemble (Boosting)",
               "strengths":"High accuracy, handles mixed types",
               "weaknesses":"Slower training than XGBoost"},
    "et":     {"full":"Extra Trees",         "type":"Ensemble (Bagging)",
               "strengths":"Very fast, low variance",
               "weaknesses":"Slightly worse than RF on some datasets"},
    "ada":    {"full":"AdaBoost",            "type":"Ensemble (Boosting)",
               "strengths":"Simple, interpretable boosting",
               "weaknesses":"Sensitive to outliers and noise"},
    "dt":     {"full":"Decision Tree",       "type":"Tree",
               "strengths":"Interpretable, fast",
               "weaknesses":"High variance, prone to overfitting"},
    "ridge":  {"full":"Ridge Regression",    "type":"Linear (L2 reg)",
               "strengths":"Fast, handles multicollinearity",
               "weaknesses":"Assumes linear relationships"},
    "lasso":  {"full":"Lasso Regression",    "type":"Linear (L1 reg)",
               "strengths":"Feature selection via sparsity",
               "weaknesses":"Drops correlated features"},
    "enet":   {"full":"ElasticNet",          "type":"Linear (L1+L2)",
               "strengths":"Balances Ridge and Lasso",
               "weaknesses":"Requires careful alpha/l1_ratio tuning"},
    "lr":     {"full":"Linear Regression",   "type":"Linear",
               "strengths":"Extremely fast, interpretable",
               "weaknesses":"Only linear relationships"},
    "svr":    {"full":"SVR (SVM)",           "type":"Kernel SVM",
               "strengths":"Good on small datasets, kernel trick",
               "weaknesses":"Slow on large data, needs scaling"},
    "svc":    {"full":"SVC (SVM)",           "type":"Kernel SVM",
               "strengths":"Good on small datasets, kernel trick",
               "weaknesses":"Slow on large data, needs scaling"},
    "knn":    {"full":"K-Nearest Neighbors", "type":"Instance-based",
               "strengths":"Simple, no training time",
               "weaknesses":"Slow at inference, needs scaling"},
    "mlp":    {"full":"MLP Neural Network",  "type":"Neural Network",
               "strengths":"Learns complex patterns",
               "weaknesses":"Black box, needs tuning"},
    "logreg": {"full":"Logistic Regression", "type":"Linear",
               "strengths":"Fast, probabilistic output",
               "weaknesses":"Assumes linear boundary"},
}

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "saved_sessions")
os.makedirs(PERSIST_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SMART RECOMMENDER
# ─────────────────────────────────────────────────────────────
def recommend_models(df: pd.DataFrame, target: str, task: str,
                     user_rules: dict = None) -> dict:
    """
    Analyse dataset characteristics and recommend top models with reasons.
    user_rules: dict with keys like 'prefer_speed', 'prefer_accuracy',
                'prefer_interpretable', 'max_training_time_s', 'small_model'
    """
    n_rows, n_cols = df.shape
    n_features = n_cols - 1
    has_nulls  = df.isnull().sum().sum() > 0
    num_cols   = df.select_dtypes(include=[np.number]).shape[1]
    cat_ratio  = 1 - (num_cols / max(n_cols, 1))

    scores = {}   # model_key -> score (higher = more recommended)
    reasons = {}  # model_key -> list of reason strings

    registry = REGRESSION_MODELS if task == "Regression" else CLASSIFICATION_MODELS

    for name, key in registry.items():
        s = 50  # base score
        r = []

        # ── Size-based rules ──
        if n_rows > 50_000:
            if key in ("svr","svc","knn"):
                s -= 30; r.append("⚠️ Slow on large datasets (>50k rows)")
            if key in ("rf","xgb","gbm","et"):
                s += 15; r.append("✅ Scales well to large datasets")
        elif n_rows < 1_000:
            if key in ("svr","svc","knn","logreg","ridge","lasso","lr"):
                s += 15; r.append("✅ Works well on small datasets")
            if key in ("mlp","xgb","gbm"):
                s -= 10; r.append("⚠️ May overfit on very small datasets")

        # ── Feature count ──
        if n_features > 50:
            if key in ("lasso","enet"):
                s += 20; r.append("✅ Lasso/ElasticNet performs feature selection automatically")
            if key in ("knn","svc","svr"):
                s -= 15; r.append("⚠️ Struggles with very high-dimensional data")
            if key in ("rf","et","xgb"):
                s += 10; r.append("✅ Handles high feature count well")

        # ── Missing data ──
        if has_nulls:
            if key in ("xgb","gbm","rf","et"):
                s += 10; r.append("✅ Tolerates missing values")
            if key in ("knn","svc","svr","mlp"):
                s -= 10; r.append("⚠️ Sensitive to missing data — ensure imputation")

        # ── Categorical features ──
        if cat_ratio > 0.3:
            if key in ("rf","et","xgb","gbm","dt","ada"):
                s += 10; r.append("✅ Handles categorical features well")

        # ── User rules ──
        if user_rules:
            if user_rules.get("prefer_speed"):
                if key in ("lr","ridge","lasso","dt","knn","logreg"):
                    s += 25; r.append("⚡ Fast training (user preference)")
                if key in ("mlp","svr","svc","gbm"):
                    s -= 15; r.append("🐢 Relatively slow (user prefers speed)")

            if user_rules.get("prefer_accuracy"):
                if key in ("xgb","gbm","rf","et","mlp"):
                    s += 25; r.append("🎯 High accuracy models (user preference)")

            if user_rules.get("prefer_interpretable"):
                if key in ("lr","ridge","lasso","logreg","dt"):
                    s += 25; r.append("🔍 Interpretable model (user preference)")
                if key in ("mlp","xgb","gbm","rf"):
                    s -= 15; r.append("🔒 Black-box model (low interpretability)")

            if user_rules.get("small_model"):
                if key in ("lr","ridge","lasso","logreg","dt","knn"):
                    s += 20; r.append("💾 Small model size (user preference)")
                if key in ("rf","et","mlp"):
                    s -= 20; r.append("📦 Large model size")

            max_time = user_rules.get("max_training_time_s")
            if max_time:
                if max_time < 5:
                    if key in ("mlp","svr","svc","gbm"):
                        s -= 30; r.append(f"⏱️ Likely exceeds {max_time}s time limit")
                    if key in ("lr","ridge","lasso","logreg","dt","knn"):
                        s += 20; r.append(f"⚡ Usually trains in <{max_time}s")

        if not r:
            r.append("✓ Suitable for this dataset")

        scores[name] = max(0, s)
        reasons[name] = r

    # Sort by score descending
    sorted_models = sorted(scores.items(), key=lambda x: -x[1])
    return {
        "ranked": sorted_models,
        "reasons": reasons,
        "dataset_profile": {
            "n_rows": n_rows, "n_features": n_features,
            "has_nulls": has_nulls, "cat_ratio": round(cat_ratio, 2),
        }
    }

# ─────────────────────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────────────────────
def build_model(key: str, params: dict, task: str):
    p = params
    if task == "Regression":
        MAP = {
            "rf":    lambda: RandomForestRegressor(**p, n_jobs=-1),
            "xgb":   lambda: XGBRegressor(**p, verbosity=0, n_jobs=-1),
            "gbm":   lambda: GradientBoostingRegressor(**p),
            "et":    lambda: ExtraTreesRegressor(**p, n_jobs=-1),
            "ada":   lambda: AdaBoostRegressor(**p),
            "dt":    lambda: DecisionTreeRegressor(**p),
            "ridge": lambda: Ridge(**p),
            "lasso": lambda: Lasso(**p),
            "enet":  lambda: ElasticNet(**p),
            "lr":    lambda: LinearRegression(),
            "svr":   lambda: SVR(**p),
            "knn":   lambda: KNeighborsRegressor(**p),
            "mlp":   lambda: MLPRegressor(**p, max_iter=500),
        }
    else:
        MAP = {
            "rf":     lambda: RandomForestClassifier(**p, n_jobs=-1),
            "xgb":    lambda: XGBClassifier(**p, verbosity=0,
                                            eval_metric="logloss", n_jobs=-1),
            "gbm":    lambda: GradientBoostingClassifier(**p),
            "et":     lambda: ExtraTreesClassifier(**p, n_jobs=-1),
            "ada":    lambda: AdaBoostClassifier(**p),
            "dt":     lambda: DecisionTreeClassifier(**p),
            "logreg": lambda: LogisticRegression(**p, max_iter=500),
            "svc":    lambda: SVC(**p, probability=True),
            "knn":    lambda: KNeighborsClassifier(**p),
            "mlp":    lambda: MLPClassifier(**p, max_iter=500),
        }
    return MAP[key]()

# ─────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────
def train_one(name: str, mkey: str, params: dict,
              X_tr, X_te, y_tr, y_te,
              task: str, scale: bool, persist_dir: str) -> dict:

    X_tr, X_te = X_tr.copy(), X_te.copy()
    if scale:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    model = build_model(mkey, params, task)

    proc = psutil.Process()
    m0   = proc.memory_info().rss / (1024 ** 2)
    t0   = time.time()
    model.fit(X_tr, y_tr)
    rt   = time.time() - t0
    mem  = max(0, proc.memory_info().rss / (1024 ** 2) - m0)

    tmp_path = os.path.join(persist_dir, f"_tmp_{mkey}.pkl")
    joblib.dump(model, tmp_path)
    size = os.path.getsize(tmp_path) / (1024 ** 2)

    fi = None
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_.tolist()
    elif hasattr(model, "coef_"):
        c = np.abs(model.coef_)
        if c.ndim > 1: c = c.mean(axis=0)
        total = c.sum() + 1e-9
        fi = (c / total).tolist()

    result = {
        "name": name, "key": mkey,
        "params": {k: str(v) for k, v in params.items()},
        "runtime": round(rt, 4),
        "memory_mb": round(mem, 2),
        "model_size_mb": round(size, 4),
        "feature_importances": fi,
    }

    if task == "Regression":
        y_pred = model.predict(X_te)
        result.update({
            "y_pred": y_pred.tolist(),
            "r2":   round(float(r2_score(y_te, y_pred)), 6),
            "rmse": round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 6),
            "mae":  round(float(mean_absolute_error(y_te, y_pred)), 6),
            "mse":  round(float(mean_squared_error(y_te, y_pred)), 6),
        })
    else:
        y_pred = model.predict(X_te)
        classes = list(np.unique(y_te))
        avg = "binary" if len(classes) == 2 else "weighted"
        y_prob = None
        auc = None
        try:
            y_prob = model.predict_proba(X_te)
            if len(classes) == 2:
                auc = round(float(roc_auc_score(y_te, y_prob[:,1])), 6)
                y_prob = y_prob[:,1].tolist()
            else:
                auc = round(float(roc_auc_score(y_te, y_prob,
                                                  multi_class="ovr", average="weighted")), 6)
                y_prob = y_prob.tolist()
        except Exception:
            pass
        result.update({
            "y_pred":    y_pred.tolist(),
            "y_prob":    y_prob,
            "accuracy":  round(float(accuracy_score(y_te, y_pred)), 6),
            "f1":        round(float(f1_score(y_te, y_pred, average=avg, zero_division=0)), 6),
            "precision": round(float(precision_score(y_te, y_pred, average=avg, zero_division=0)), 6),
            "recall":    round(float(recall_score(y_te, y_pred, average=avg, zero_division=0)), 6),
            "auc":       auc,
            "conf_matrix": confusion_matrix(y_te, y_pred).tolist(),
        })
    return result


# ─────────────────────────────────────────────────────────────
# CLOUD vs LOCAL SIMULATION
# ─────────────────────────────────────────────────────────────
def simulate_cloud_vs_local(result: dict, task: str,
                             cloud_spec: str = "t3.medium") -> dict:
    """
    Simulate how the model would perform on cloud vs localhost.
    Uses rough multipliers based on cloud instance type and model type.
    """
    CLOUD_SPECS = {
        "t3.micro":   {"cpu_mult": 0.5,  "mem_mult": 0.4,  "vcpu": 2,  "ram_gb": 1},
        "t3.medium":  {"cpu_mult": 1.0,  "mem_mult": 1.0,  "vcpu": 2,  "ram_gb": 4},
        "t3.large":   {"cpu_mult": 2.0,  "mem_mult": 2.0,  "vcpu": 2,  "ram_gb": 8},
        "c5.xlarge":  {"cpu_mult": 4.0,  "mem_mult": 3.0,  "vcpu": 4,  "ram_gb": 8},
        "c5.4xlarge": {"cpu_mult": 8.0,  "mem_mult": 6.0,  "vcpu": 16, "ram_gb": 32},
        "m5.4xlarge": {"cpu_mult": 6.0,  "mem_mult": 8.0,  "vcpu": 16, "ram_gb": 64},
        "p3.2xlarge": {"cpu_mult": 12.0, "mem_mult": 10.0, "vcpu": 8,  "ram_gb": 61},
    }
    spec  = CLOUD_SPECS.get(cloud_spec, CLOUD_SPECS["t3.medium"])
    local = psutil.virtual_memory()
    local_ram_gb  = local.total / (1024**3)
    local_vcpu    = psutil.cpu_count(logical=True)

    local_rt = result["runtime"]
    # Cloud speedup depends on CPU multiplier relative to local
    local_cpu_mult = max(1.0, local_vcpu / 2.0)
    cloud_rt = local_rt * (local_cpu_mult / spec["cpu_mult"])
    cloud_rt = max(0.001, cloud_rt)

    local_mem  = result["memory_mb"]
    cloud_mem  = local_mem   # memory used by model is same

    # Estimate cost per run (AWS on-demand pricing approximation)
    COST_PER_HOUR = {
        "t3.micro": 0.0104, "t3.medium": 0.0416, "t3.large": 0.0832,
        "c5.xlarge": 0.17, "c5.4xlarge": 0.68, "m5.4xlarge": 0.768,
        "p3.2xlarge": 3.06,
    }
    cost_per_hr = COST_PER_HOUR.get(cloud_spec, 0.0416)
    cloud_cost_cents = (cloud_rt / 3600) * cost_per_hr * 100  # cents

    return {
        "local": {
            "runtime_s":  local_rt,
            "memory_mb":  local_mem,
            "vcpu":       local_vcpu,
            "ram_gb":     round(local_ram_gb, 1),
            "cost_cents": 0.0,
        },
        "cloud": {
            "instance":   cloud_spec,
            "runtime_s":  round(cloud_rt, 4),
            "memory_mb":  cloud_mem,
            "vcpu":       spec["vcpu"],
            "ram_gb":     spec["ram_gb"],
            "cost_cents": round(cloud_cost_cents, 6),
            "speedup":    round(local_rt / max(cloud_rt, 0.001), 2),
        }
    }
