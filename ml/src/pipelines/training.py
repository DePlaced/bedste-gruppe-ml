# src/pipelines/training.py
import os
import json
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.dummy import DummyClassifier

class TrainingPipeline:
    """
    TRAINING:
      1) Use all engineered features except target
      2) Random train/test split (classification)
      3) Train GBC
      4) Save metrics
      5) Return model
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def run(self, df: pd.DataFrame):
        tcfg = self.cfg["training"]["target"]
        target_col = tcfg["source_col"]

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found after preprocessing.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if X.shape[1] == 0:
            raise ValueError("No features selected. Check preprocessing/drop_columns.")

        frac = float(self.cfg["training"]["train_fraction"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1.0 - frac), shuffle=True, stratify=y
        )

        base_params = self.cfg["training"]["gbc_params"]
        model = GradientBoostingClassifier(**base_params)

        gs_cfg = self.cfg["training"].get("grid_search", {"enabled": False})
        if gs_cfg.get("enabled", False):
            search = GridSearchCV(
                model,
                param_grid=gs_cfg["param_grid"],
                scoring="accuracy",
                cv=5,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_

        model.fit(X_train, y_train)

        # ------- METRICS: accuracy + precision/recall/F1 -------
        y_pred = model.predict(X_test)

        # ------- BASELINE: DummyClassifier -------
        dcfg = self.cfg["training"].get("dummy", {"enabled": True, "strategy": "most_frequent", "random_state": 42})
        dummy_metrics = None
        if dcfg.get("enabled", True):
            dummy = DummyClassifier(
                strategy=dcfg.get("strategy", "most_frequent"),
                random_state=dcfg.get("random_state", 42)
            )
            # Fit on the same split; Dummy ignores X but we pass it for API consistency
            dummy.fit(X_train, y_train)
            y_pred_dummy = dummy.predict(X_test)

            acc_d = float(accuracy_score(y_test, y_pred_dummy))
            precision_d, recall_d, f1_d, _ = precision_recall_fscore_support(
                y_test, y_pred_dummy, average="weighted", zero_division=0
            )
            dummy_metrics = {
                "accuracy": round(acc_d, 4),
                "precision_weighted": round(float(precision_d), 4),
                "recall_weighted": round(float(recall_d), 4),
                "f1_weighted": round(float(f1_d), 4),
                "strategy": dcfg.get("strategy", "most_frequent")
            }

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

        metrics = {"model": model_metrics}
        if dummy_metrics is not None:
            metrics["baseline_dummy"] = dummy_metrics

        metrics_path = self.cfg["reports"]["metrics_path"]
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return model
