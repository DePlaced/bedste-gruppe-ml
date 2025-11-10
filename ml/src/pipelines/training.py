# src/pipelines/training.py
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Classification metrics: Accuracy + weighted F1.
    Works with string labels (e.g. 'EDIBLE' / 'POISONOUS').
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))
    return {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}


class TrainingPipeline:
    """
    TRAINING (classification):
      1) Build target column (optional shift -horizon, but for mushrooms horizon=0 is typical)
      2) Use all engineered features except target + raw source_col
      3) Random shuffled train/test split
      4) Train GradientBoostingClassifier (optionally grid-search)
      5) Save metrics
      6) Return model
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def _make_target(self, df: pd.DataFrame) -> pd.DataFrame:
        tcfg = self.cfg["training"]["target"]
        df = df.copy()

        source_col = tcfg["source_col"]      # e.g. "poisonous"
        target_name = tcfg["target_name"]    # e.g. "target"
        horizon = tcfg.get("horizon", 0)

        # For mushrooms (non-time-series), horizon should ideally be 0.
        if horizon and horizon > 0:
            df[target_name] = df[source_col].shift(-horizon)
            df = df.dropna(subset=[target_name]).reset_index(drop=True)
        else:
            df[target_name] = df[source_col]

        return df

    def _split(self, X: pd.DataFrame, y: pd.Series, frac: float):
        """
        Standard shuffled train/test split for classification.
        Stratify to preserve class balance.
        """
        return train_test_split(
            X,
            y,
            test_size=(1.0 - frac),
            shuffle=True,
            stratify=y,
            random_state=42,
        )

    def run(self, df: pd.DataFrame):
        # 1) Build target column
        df = self._make_target(df)
        tcfg = self.cfg["training"]["target"]
        target_name = tcfg["target_name"]
        source_col = tcfg["source_col"]

        # 2) Build feature matrix and target vector
        feature_cols = [c for c in df.columns if c not in [target_name, source_col]]
        X = df[feature_cols]
        y = df[target_name]

        if X.shape[1] == 0:
            raise ValueError("No features selected. Check preprocessing/feature_engineering and drop lists.")

        frac = float(self.cfg["training"]["train_fraction"])
        X_train, X_test, y_train, y_test = self._split(X, y, frac)

        # 3) Base classifier
        base_params = self.cfg["training"]["gbr_params"]
        model = GradientBoostingClassifier(**base_params)

        # 4) Optional GridSearchCV
        gs_cfg = self.cfg["training"].get("grid_search", {"enabled": False})
        if gs_cfg.get("enabled", False):
            search = GridSearchCV(
                model,
                param_grid=gs_cfg["param_grid"],
                scoring="accuracy",   # 👈 classification scoring
                cv=5,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
        else:
            model.fit(X_train, y_train)

        # 5) Evaluate
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        # 6) Save metrics
        metrics_path = self.cfg["reports"]["metrics_path"]
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return model
