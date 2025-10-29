#!/usr/bin/env python3
"""
AutoML for the taxi dataset using FLAML (fast, time-budgeted).

What it does
------------
1) Loads your CSV -> runs your Preprocessing + FeatureEngineering -> creates next-hour target (leakage-safe)
2) Chronological split (train_fraction from config)
3) Runs FLAML with a time budget you choose (default 180 sec)
4) Reports:
   - Total time budget + actual runtime
   - Best model name and its MAE/RMSE on the validation split
   - Leaderboard of all estimators FLAML tried (MAE/RMSE on the same validation split)
5) Saves the best model to models/automl/best_model.pkl

Run
---
conda activate ml-taxi-pipeline
pip install flaml
# optional (recommended): pip install lightgbm xgboost catboost

python tools/automl_flaml.py --config config/config.yaml --metric mae --time_budget 300
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml
from flaml import AutoML
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump


# ---- make your repo modules importable (same pattern as your entrypoints) ----
ROOT   = Path(__file__).resolve().parents[1]  # project root (.. from tools/)
ML_SRC = ROOT / "ml" / "src"
for p in (ROOT, ML_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Ensure relative paths in config.yaml resolve correctly
os.chdir(ROOT)

# Import your pipeline steps
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.feature_engineering import FeatureEngineeringPipeline


# ---------------- utils ----------------
def read_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def build_supervised_frame(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load raw CSV -> preprocessing -> feature engineering -> next-hour target.
    Drop the raw source_col to avoid leakage (matches your training pipeline).
    """
    csv_path = cfg["data_manager"]["csv_path"]
    # best-effort parse of datetime
    try:
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
    except Exception:
        df = pd.read_csv(csv_path)

    # 1) preprocessing
    df = PreprocessingPipeline(cfg).run(df)

    # 2) feature engineering
    df = FeatureEngineeringPipeline(cfg).run(df)

    # 3) make target (shift -horizon)
    tcfg = cfg["training"]["target"]
    target_name = tcfg["target_name"]
    source_col  = tcfg["source_col"]
    horizon     = int(tcfg["horizon"])

    df[target_name] = df[source_col].shift(-horizon)
    df = df.dropna(subset=[target_name]).reset_index(drop=True)

    # 4) drop raw source col (to mirror your training feature set)
    if source_col in df.columns:
        df = df.drop(columns=[source_col])

    return df


def chrono_split(df: pd.DataFrame, target_name: str, frac: float):
    """
    Chronological split (no shuffle), returns (X_train, y_train, X_valid, y_valid).

    Works whether 'datetime' exists (not dropped in preprocessing) or not.
    If present, 'datetime' is excluded from features; otherwise we just ignore it.
    """
    n = len(df)
    split = int(n * frac)
    train = df.iloc[:split].copy()
    valid = df.iloc[split:].copy()

    drop_cols = [target_name]
    if "datetime" in train.columns:
        drop_cols.append("datetime")

    X_train = train.drop(columns=drop_cols, errors="ignore")
    y_train = train[target_name]
    X_valid = valid.drop(columns=drop_cols, errors="ignore")
    y_valid = valid[target_name]
    return X_train, y_train, X_valid, y_valid


def safe_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def try_refit_and_eval(est_name: str, config: Dict[str, Any], X_train, y_train, X_valid, y_valid):
    """
    Refit the best config for an estimator on train and compute MAE/RMSE on valid.
    FLAML stores best_config_per_estimator; we instantiate a new AutoML for each
    estimator to refit quickly using that best config.
    """
    automl = AutoML()
    settings = {
        "task": "regression",
        "metric": "mae",
        "estimator_list": [est_name],
        "log_file_name": None,
        "verbose": 0,
        "time_budget": 5,
        "eval_method": "holdout",
        "sample": False,
        "train_time_limit": 5,
    }
    try:
        t0 = time.time()
        automl.fit(
            X_train=X_train, y_train=y_train,
            X_val=X_valid, y_val=y_valid,
            **settings,
            config_constraints={est_name: config},   # pin to best config
            refit_full=True
        )
        y_hat = automl.predict(X_valid)
        mae = float(mean_absolute_error(y_valid, y_hat))
        rmse = safe_rmse(y_valid, y_hat)
        secs = time.time() - t0
        return mae, rmse, secs
    except Exception:
        return np.inf, np.inf, 0.0


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    ap.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mape", "r2"],
                    help="Optimization metric for FLAML")
    ap.add_argument("--time_budget", type=int, default=180, help="Total time budget (seconds)")
    ap.add_argument("--train_fraction", type=float, default=None,
                    help="Override train_fraction in config for the chrono split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = read_config(Path(args.config))
    df  = build_supervised_frame(cfg)

    target_name   = cfg["training"]["target"]["target_name"]
    train_fraction = float(args.train_fraction or cfg["training"].get("train_fraction", 0.8))

    X_train, y_train, X_valid, y_valid = chrono_split(df, target_name, train_fraction)

    print("\n================ AutoML (FLAML) ================")
    print(f"Rows total: {len(df)} | train: {len(X_train)} | valid: {len(X_valid)}")
    print(f"Metric: {args.metric.upper()} | Time budget: {args.time_budget} sec | Seed: {args.seed}")
    print("================================================\n")

    automl = AutoML()
    start = time.time()
    automl.fit(
        X_train=X_train, y_train=y_train,
        task="regression",
        metric=args.metric,
        time_budget=args.time_budget,   # ← upper bound on wall-clock time
        eval_method="holdout",
        X_val=X_valid, y_val=y_valid,   # fixed chronological holdout
        log_file_name="flaml_automl.log",
        seed=args.seed,
        # You can optionally narrow the search space:
        # estimator_list=["lgbm", "xgboost", "rf", "extra_tree", "lrl1", "catboost"],
    )
    elapsed = time.time() - start

    # Best model results
    y_hat = automl.predict(X_valid)
    best_mae = float(mean_absolute_error(y_valid, y_hat))
    best_rmse = safe_rmse(y_valid, y_hat)

    print("Best estimator :", automl.best_estimator)
    print("Best config    :", automl.best_config)
    print(f"Best {args.metric.upper()} :", round(automl.best_loss, 4))
    print(f"MAE (valid)    :", round(best_mae, 4))
    print(f"RMSE (valid)   :", round(best_rmse, 4))
    print(f"Time budget    : {args.time_budget} sec")
    print(f"Actual runtime : {round(elapsed, 1)} sec\n")

    # Leaderboard of all estimators tried (refit best-per-estimator and score on the same valid split)
    rows: List[Dict[str, Any]] = []
    bcfg_per_est = getattr(automl, "best_config_per_estimator", {}) or {}
    for est_name, cfg_est in bcfg_per_est.items():
        mae, rmse, secs = try_refit_and_eval(est_name, cfg_est, X_train, y_train, X_valid, y_valid)
        rows.append({"estimator": est_name, "MAE": mae, "RMSE": rmse, "refit_secs": secs})

    if rows:
        lb = pd.DataFrame(rows).sort_values(["MAE", "RMSE"])
        print("============== Leaderboard (validation metrics) ==============")
        try:
            print(lb.to_string(index=False, formatters={
                "MAE":  lambda v: f"{v:.4f}" if np.isfinite(v) else "nan",
                "RMSE": lambda v: f"{v:.4f}" if np.isfinite(v) else "nan",
                "refit_secs": lambda v: f"{v:.2f}",
            }))
        except Exception:
            print(lb)
    else:
        print("No estimator leaderboard available (FLAML didn’t try multiple estimators).")

    # Save the overall best model
    out_dir = Path("models/automl")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(automl.model, out_dir / "best_model.pkl")
    print(f"\nSaved best model to: {out_dir/'best_model.pkl'}")
    print("Point pipeline_runner.model_path to this file if you want to serve it.\n")


if __name__ == "__main__":
    main()


