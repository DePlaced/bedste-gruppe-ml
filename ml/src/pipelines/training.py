# src/pipelines/training.py
import os
import json
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.dummy import DummyClassifier


class TrainingPipeline:
    """
    TRAINING:
      1) Use all engineered features except target
      2) Random train/test split (classification)
      3) Train ExtraTreesClassifier
      4) Evaluate against DummyClassifier baseline
      5) Save metrics
      6) Return trained ExtraTrees model
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def run(self, df: pd.DataFrame):
        # --------- TARGET SETUP ----------
        tcfg = self.cfg["training"]["target"]
        target_col = tcfg["source_col"]

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found after preprocessing.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if X.shape[1] == 0:
            raise ValueError("No features selected. Check preprocessing/drop_columns.")

        # --------- TRAIN / TEST SPLIT ----------
        frac = float(self.cfg["training"]["train_fraction"])
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=(1.0 - frac),
            shuffle=True,
            stratify=y
        )

        # --------- MAIN MODEL: ExtraTreesClassifier ----------
        tr_cfg = self.cfg["training"]
        et_params = tr_cfg["extra_trees_params"]
        model = ExtraTreesClassifier(**et_params)

        # Train ExtraTrees
        model.fit(X_train, y_train)

        # --------- PREDICTIONS ----------
        y_pred = model.predict(X_test)

        # --------- BASELINE: DummyClassifier ----------
        dcfg = tr_cfg.get(
            "dummy",
            {"enabled": True, "strategy": "most_frequent", "random_state": 42}
        )
        dummy_metrics = None
        if dcfg.get("enabled", True):
            dummy = DummyClassifier(
                strategy=dcfg.get("strategy", "most_frequent"),
                random_state=dcfg.get("random_state", 42)
            )
            # Dummy ignores X but we pass it for API consistency
            dummy.fit(X_train, y_train)
            y_pred_dummy = dummy.predict(X_test)

            acc_d = float(accuracy_score(y_test, y_pred_dummy))
            precision_d, recall_d, f1_d, _ = precision_recall_fscore_support(
                y_test,
                y_pred_dummy,
                average="weighted",
                zero_division=0
            )
            dummy_metrics = {
                "accuracy": round(acc_d, 4),
                "precision_weighted": round(float(precision_d), 4),
                "recall_weighted": round(float(recall_d), 4),
                "f1_weighted": round(float(f1_d), 4),
                "strategy": dcfg.get("strategy", "most_frequent"),
            }

        # --------- MODEL METRICS ----------
        acc = float(accuracy_score(y_test, y_pred))

        # Use weighted average so it works even if classes are imbalanced
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        model_metrics = {
            "accuracy": round(acc, 4),
            "precision_weighted": round(float(precision), 4),
            "recall_weighted": round(float(recall), 4),
            "f1_weighted": round(float(f1), 4),
        }

        metrics = {
            "model": model_metrics,
            "baseline_dummy": dummy_metrics,
        }

        # --------- SAVE METRICS ----------
        metrics_path = self.cfg["reports"]["metrics_path"]
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return model
