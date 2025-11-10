# src/pipelines/training.py
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


def _rmse(y_true, y_pred) -> float:
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        from sklearn.metrics import mean_squared_error
        return float(mean_squared_error(y_true, y_pred, squared=False))


def compute_metrics(y_true, y_pred, mape_min: float = 1.0) -> Dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse(y_true, y_pred)

    mask = y_true >= mape_min
    mape = np.nan if not mask.any() else 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": (None if np.isnan(mape) else round(float(mape), 2))}


class TrainingPipeline:
    """
    TRAINING:
      1) Build next-hour target (shift -horizon)
      2) Use all engineered features except target + raw source_col
      3) Chronological split
      4) Train GBR (optionally grid-search)
      5) Save metrics
      6) Return model
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def _make_target(self, df: pd.DataFrame) -> pd.DataFrame:
        tcfg = self.cfg["training"]["target"]
        df = df.copy()
        df[tcfg["target_name"]] = df[tcfg["source_col"]].shift(-tcfg["horizon"])
        df = df.dropna(subset=[tcfg["target_name"]]).reset_index(drop=True)
        return df

    def _split(self, X: pd.DataFrame, y: pd.Series, frac: float):
        return train_test_split(X, y, test_size=(1.0 - frac), shuffle=False)

    def run(self, df: pd.DataFrame):
        df = self._make_target(df)
        tcfg = self.cfg["training"]["target"]
        target_name = tcfg["target_name"]
        source_col = tcfg["source_col"]

        feature_cols = [c for c in df.columns if c not in [target_name, source_col]]
        X = df[feature_cols]
        y = df[target_name]

        if X.shape[1] == 0:
            raise ValueError("No features selected. Check preprocessing/feature_engineering and drop lists.")

        frac = float(self.cfg["training"]["train_fraction"])
        X_train, X_test, y_train, y_test = self._split(X, y, frac)

        base_params = self.cfg["training"]["gbr_params"]
        model = GradientBoostingRegressor(**base_params)

        gs_cfg = self.cfg["training"].get("grid_search", {"enabled": False})
        if gs_cfg.get("enabled", False):
            tscv = TimeSeriesSplit(n_splits=5)
            search = GridSearchCV(
                model,
                param_grid=gs_cfg["param_grid"],
                scoring="neg_mean_absolute_error",
                cv=tscv,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        metrics_path = self.cfg["reports"]["metrics_path"]
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return model
