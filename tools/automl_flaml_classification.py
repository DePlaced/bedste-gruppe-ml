#!/usr/bin/env python3
"""
AutoML for classification using FLAML.

What it does
------------
1) Loads your CSV -> runs your Preprocessing + FeatureEngineering
2) Chronological split (mushroom_fraction from config)
3) Runs FLAML with a time budget you choose (default 180 sec)
4) Reports:
   - Total time budget + actual runtime
   - Best model name and its F1/Accuracy on the validation split
   - Leaderboard of all estimators FLAML tried (F1/Accuracy on the same validation split)
5) Saves the best model to models/automl/best_model.pkl

Run
---
conda activate ml-mushroom-pipeline
pip install flaml
# optional (recommended): pip install lightgbm xgboost catboost

python tools/automl_flaml_classification.py --config config/config.yaml --metric f1 --time_budget 300
"""

import argparse
from datetime import datetime
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml
from flaml import AutoML
from sklearn.metrics import f1_score, accuracy_score
from joblib import dump

# ---- make your repo modules importable ----
ROOT = Path(__file__).resolve().parents[1]
ML_SRC = ROOT / "ml" / "src"
for p in (ROOT, ML_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

os.chdir(ROOT)
from pipelines.preprocessing import PreprocessingPipeline


# ---------------- utils ----------------
def read_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def build_supervised_frame(cfg: Dict[str, Any]) -> pd.DataFrame:
    csv_path = cfg["data_manager"]["csv_path"]
    df = pd.read_csv(csv_path)
    df = PreprocessingPipeline(cfg).run(df)
    return df


def chrono_split(df: pd.DataFrame, target_name: str, frac: float):
    n = len(df)
    split = int(n * frac)
    train = df.iloc[:split].copy()
    valid = df.iloc[split:].copy()

    drop_cols = [target_name]

    X_train = train.drop(columns=drop_cols, errors="ignore")
    y_train = train[target_name]
    X_valid = valid.drop(columns=drop_cols, errors="ignore")
    y_valid = valid[target_name]
    return X_train, y_train, X_valid, y_valid


def try_refit_and_eval(est_name: str, config: Dict[str, Any], X_train, y_train, X_valid, y_valid):
    """Refit each estimator on train and evaluate F1/Accuracy on validation."""
    automl = AutoML()
    settings = {
        "task": "classification",
        "metric": "f1",
        "estimator_list": [est_name],
        "time_budget": 5,
        "eval_method": "holdout",
        "log_file_name": None,
        "verbose": 0,
    }

    try:
        t0 = time.time()
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_valid,
            y_val=y_valid,
            **settings,
            

        )
        y_hat = automl.predict(X_valid)
        f1 = f1_score(y_valid, y_hat, average="weighted")
        acc = accuracy_score(y_valid, y_hat)
        secs = time.time() - t0
        return f1, acc, secs
    except Exception as e:
        print(f"Error refitting {est_name}: {e}")
        return 0.0, 0.0, 0.0


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    ap.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["accuracy", "f1", "macro_f1", "micro_f1", "roc_auc", "log_loss"],
        help="Optimization metric for FLAML",
    )
    ap.add_argument("--time_budget", type=int, default=180, help="Total time budget (seconds)")
    ap.add_argument("--train_fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = read_config(Path(args.config))
    df = build_supervised_frame(cfg)

    target_name = cfg["training"]["target"]["target_name"]
    train_fraction = float(args.train_fraction or cfg["training"].get("train_fraction", 0.8))

    X_train, y_train, X_valid, y_valid = chrono_split(df, target_name, train_fraction)

    print("\n================ AutoML (FLAML - Classification) ================")
    print(f"Rows total: {len(df)} | train: {len(X_train)} | valid: {len(X_valid)}")
    print(f"Metric: {args.metric.upper()} | Time budget: {args.time_budget} sec | Seed: {args.seed}")
    print("===============================================================\n")

    automl = AutoML()
    start = time.time()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="classification",
        metric=args.metric,
        time_budget=args.time_budget,
        eval_method="holdout",
        X_val=X_valid,
        y_val=y_valid,
        log_file_name="flaml_automl.log",
        seed=args.seed,
        estimator_list=["lgbm", "xgboost", "rf", "extra_tree", "lrl1"]
    )
    elapsed = time.time() - start

    y_hat = automl.predict(X_valid)
    best_f1 = f1_score(y_valid, y_hat, average='weighted')
    best_acc = accuracy_score(y_valid, y_hat)

    #Ensures path is valid
    out_dir = Path("models/automl")
    out_dir.mkdir(parents=True, exist_ok=True)

    #Creates run log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    txt_file = out_dir / f"run_{timestamp}.txt"

    # Open the file in write mode
    try:
        with open(txt_file, "w") as f:
            # Helper function to print to both terminal and file
            def log(*args, **kwargs):
                print(*args, **kwargs)  # print to terminal
                print(*args, **kwargs, file=f)  # write to file

            log("Best estimator :", automl.best_estimator)
            log("Best config    :", automl.best_config)

            log("best metric")
            if args.metric in ["f1", "accuracy", "macro_f1", "micro_f1"]:
                log(f"Best {args.metric.upper()} (FLAML):", round(1 - automl.best_loss, 4))
            else:
                log(f"Best {args.metric.upper()} (FLAML):", round(automl.best_loss, 4))

            txt_out = ""
            log(f"F1 score (valid) :", round(best_f1, 4))
            log(f"Accuracy (valid) :", round(best_acc, 4))
            log(f"Time budget      : {args.time_budget} sec")
            log(f"Actual runtime   : {round(elapsed, 1)} sec\n")

            # Leaderboard
            rows: List[Dict[str, Any]] = []
            bcfg_per_est = getattr(automl, "best_config_per_estimator", {}) or {}
            for est_name, cfg_est in bcfg_per_est.items():
                f1, acc, secs = try_refit_and_eval(est_name, cfg_est, X_train, y_train, X_valid, y_valid)
                rows.append({"estimator": est_name, "F1": f1, "Accuracy": acc, "refit_secs": secs})
            log(bcfg_per_est)

            if rows:
                lb = pd.DataFrame(rows).sort_values(["F1", "Accuracy"], ascending=False)
                log("============== Leaderboard (validation metrics) ==============")
                log(lb.to_string(index=False, formatters={
                    "F1": lambda v: f"{v:.4f}",
                    "Accuracy": lambda v: f"{v:.4f}",
                    "refit_secs": lambda v: f"{v:.2f}",
                }))
            else:
                log("No estimator leaderboard available.")


            dump(automl.model, out_dir / "best_model.pkl")
            log(f"\nSaved best model to: {out_dir/'best_model.pkl'}")
            log("Point pipeline_runner.model_path to this file if you want to serve it.\n")
    except Exception as e:
        print(f"Error refitting {est_name}: {e}")



if __name__ == "__main__":
    main()
