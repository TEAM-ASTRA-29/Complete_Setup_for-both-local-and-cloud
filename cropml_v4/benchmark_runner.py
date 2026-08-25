"""
benchmark_runner.py
====================
Headless (no Streamlit UI) training + real system-metrics logger.

Purpose: run the EXACT SAME command on your local machine and inside the
Docker container on EC2, so the two result JSON files are directly
comparable — no more guessed/simulated numbers.

Usage:
    python benchmark_runner.py \
        --data data/karnataka_crop.csv \
        --target yield \
        --task Regression \
        --models rf,xgb,gbm,dt \
        --env local \
        --label "my-laptop"

    # On EC2 (inside the container):
    python benchmark_runner.py \
        --data data/karnataka_crop.csv \
        --target yield \
        --task Regression \
        --models rf,xgb,gbm,dt \
        --env cloud \
        --label "t3.medium"

Output:
    benchmark_results/<env>_<label>_<timestamp>.json

Each model's entry contains REAL measured values:
    - runtime_s          : wall-clock model.fit() time
    - cpu_percent_avg    : average CPU utilisation sampled during fit
    - cpu_percent_max    : peak CPU utilisation sampled during fit
    - memory_mb          : RSS memory delta during fit
    - model_size_mb      : size of the pickled model on disk
    - throughput_rows_s  : training rows processed per second
    - r2/rmse/mae/mse  OR  accuracy/f1/precision/recall/auc
Plus environment metadata (hostname, OS, vCPU count, total RAM, label).
"""

import argparse
import json
import os
import platform
import socket
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.preprocessor import PreprocessingPipeline
from backend.models import (
    REGRESSION_MODELS, CLASSIFICATION_MODELS, build_model,
)
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# CPU SAMPLER — polls CPU% in a background thread during model.fit()
# psutil.cpu_percent() needs repeated calls over time to be meaningful;
# a single call in train_one() (as the Streamlit app does) is not real
# monitoring. This is what makes the number "real" instead of guessed.
# ─────────────────────────────────────────────────────────────
class CPUSampler:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        # first call always returns 0.0 / garbage, prime it
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(interval=self.interval))

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            return 0.0, 0.0
        return round(float(np.mean(self.samples)), 2), round(float(np.max(self.samples)), 2)


def env_metadata(env_label: str, instance_label: str) -> dict:
    return {
        "environment": env_label,               # "local" or "cloud"
        "instance_label": instance_label,        # e.g. "my-laptop" or "t3.medium"
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "vcpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "timestamp": datetime.now().isoformat(),
    }


def train_and_measure(name, mkey, X_tr, X_te, y_tr, y_te, task, scale):
    X_tr, X_te = X_tr.copy(), X_te.copy()
    if scale:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    model = build_model(mkey, {}, task)

    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024 ** 2)

    sampler = CPUSampler(interval=0.1)
    sampler.start()
    t0 = time.time()
    model.fit(X_tr, y_tr)
    runtime_s = time.time() - t0
    cpu_avg, cpu_max = sampler.stop()

    mem_after = proc.memory_info().rss / (1024 ** 2)
    mem_delta = max(0.0, mem_after - mem_before)

    tmp_path = os.path.join(RESULTS_DIR, f"_tmp_{mkey}.pkl")
    import joblib
    joblib.dump(model, tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
    os.remove(tmp_path)

    n_train = len(X_tr)
    throughput = round(n_train / runtime_s, 2) if runtime_s > 0 else None

    result = {
        "name": name,
        "key": mkey,
        "runtime_s": round(runtime_s, 4),
        "cpu_percent_avg": cpu_avg,
        "cpu_percent_max": cpu_max,
        "memory_mb": round(mem_delta, 2),
        "model_size_mb": round(size_mb, 4),
        "throughput_rows_s": throughput,
        "n_train_rows": n_train,
        "n_test_rows": len(X_te),
    }

    y_pred = model.predict(X_te)
    if task == "Regression":
        result.update({
            "r2": round(float(r2_score(y_te, y_pred)), 6),
            "rmse": round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 6),
            "mae": round(float(mean_absolute_error(y_te, y_pred)), 6),
            "mse": round(float(mean_squared_error(y_te, y_pred)), 6),
        })
    else:
        classes = list(np.unique(y_te))
        avg = "binary" if len(classes) == 2 else "weighted"
        result.update({
            "accuracy": round(float(accuracy_score(y_te, y_pred)), 6),
            "f1": round(float(f1_score(y_te, y_pred, average=avg, zero_division=0)), 6),
            "precision": round(float(precision_score(y_te, y_pred, average=avg, zero_division=0)), 6),
            "recall": round(float(recall_score(y_te, y_pred, average=avg, zero_division=0)), 6),
        })
    return result


def main():
    ap = argparse.ArgumentParser(description="Headless CropML benchmark runner")
    ap.add_argument("--data", required=True, help="Path to CSV dataset")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--task", required=True, choices=["Regression", "Classification"])
    ap.add_argument("--models", required=True,
                     help="Comma-separated model keys, e.g. rf,xgb,gbm,dt "
                          "(see backend/models.py REGRESSION_MODELS / CLASSIFICATION_MODELS)")
    ap.add_argument("--env", required=True, choices=["local", "cloud"])
    ap.add_argument("--label", required=True,
                     help="Free-text label, e.g. 'my-laptop' or 't3.medium'")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", action="store_true",
                     help="Apply StandardScaler before training (recommended for svr/svc/knn/mlp)")
    ap.add_argument("--repeats", type=int, default=1,
                     help="Repeat each model N times and keep all runs (for statistical stability)")
    args = ap.parse_args()

    print(f"Loading dataset: {args.data}")
    df = pd.read_csv(args.data)
    print(f"  {df.shape[0]:,} rows x {df.shape[1]} columns")

    pipe_cfg = {
        "target_col": args.target,
        "drop_cols": [],
        "handle_cats": "label",
        "handle_nulls": "mean",
        "handle_dups": True,
        "drop_low_var": False,
        "scale_method": "none",
        "already_clean": False,
    }
    pipeline = PreprocessingPipeline(pipe_cfg)
    X, y = pipeline.fit_transform(df)
    print(f"  Preprocessed -> {X.shape[1]} features")
    for line in pipeline.get_log():
        print(f"    - {line}")

    if args.task == "Classification":
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    registry = REGRESSION_MODELS if args.task == "Regression" else CLASSIFICATION_MODELS
    key_to_name = {v: k for k, v in registry.items()}
    model_keys = [m.strip() for m in args.models.split(",")]

    invalid = [m for m in model_keys if m not in key_to_name]
    if invalid:
        print(f"ERROR: unknown model keys {invalid}. Valid keys: {list(key_to_name.keys())}")
        sys.exit(1)

    meta = env_metadata(args.env, args.label)
    meta.update({
        "dataset": os.path.basename(args.data),
        "task": args.task,
        "target": args.target,
        "n_rows": df.shape[0],
        "n_features": X.shape[1],
        "test_size": args.test_size,
        "seed": args.seed,
        "repeats": args.repeats,
    })

    all_results = []
    for mkey in model_keys:
        name = key_to_name[mkey]
        for rep in range(args.repeats):
            print(f"Training {name} (run {rep+1}/{args.repeats}) on {args.env}:{args.label} ...")
            r = train_and_measure(name, mkey, X_tr, X_te, y_tr, y_te, args.task, args.scale)
            r["run_index"] = rep
            all_results.append(r)
            metric = r.get("r2", r.get("accuracy"))
            print(f"  runtime={r['runtime_s']}s  cpu_avg={r['cpu_percent_avg']}%  "
                  f"mem={r['memory_mb']}MB  metric={metric}")

    output = {"meta": meta, "results": all_results}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = args.label.replace(" ", "-").replace("/", "-")
    out_path = os.path.join(RESULTS_DIR, f"{args.env}_{safe_label}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
