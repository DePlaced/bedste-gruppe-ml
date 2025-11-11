# src/pipelines/training.py
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class TrainingPipeline:
    """
    TRAINING:
      1) Use all engineered features except target
      2) Random train/test split (classification)
      3) Train GBC (optionally grid-search)
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

        acc = float(accuracy_score(y_test, y_pred))

        # Use weighted average so it works even if classes are imbalanced
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        metrics = {
            "accuracy": round(acc, 4),
            "precision_weighted": round(float(precision), 4),
            "recall_weighted": round(float(recall), 4),
            "f1_weighted": round(float(f1), 4),
        }

        metrics_path = self.cfg["reports"]["metrics_path"]
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return model
