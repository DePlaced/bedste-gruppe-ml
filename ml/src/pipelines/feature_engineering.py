import joblib
import pandas as pd
from typing import Any, Dict


class FeatureEngineeringPipeline:
    """
    TRAINING:
      - One-hot encode all categorical features (except target)
      - Return: [one-hot features..., target]

    INFERENCE:
      - One-hot encode all categorical features
      - Align to model.feature_names_in_:
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        t_cfg = config.get("training", {}).get("target", {})
        self.target_col = t_cfg.get("source_col")

    # ---------- helpers ----------
    @staticmethod
    def _separate_target(self, df: pd.DataFrame):
        """Split df into (X, y) based on target_col."""
        if self.target_col is not None and self.target_col in df.columns:
            y = df[self.target_col].copy()
            X = df.drop(columns=[self.target_col]).copy()
        else:
            y = None
            X = df.copy()
        return X, y

    @staticmethod
    def _one_hot_encode(X: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(X, drop_first=False, dtype=int)

    # ---------- TRAINING ----------
    def training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Training-time feature engineering:
          - one-hot encode all non-target columns
          - add target back
        """
        X, y = self._separate_target(df)
        X_enc = self._one_hot_encode(X)

        if y is not None:
            X_enc[self.target_col] = y.values

        return X_enc

    # ---------- INFERENCE ----------
    def inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inference-time feature engineering:
          1) one-hot encode all non-target columns
          2) load the model and read feature_names_in_
          3) reindex to those columns, filling missing with 0
        """
        if df.empty:
            raise ValueError("No rows provided to FeatureEngineeringPipeline.inference().")

        # 2) one-hot encode current row(s)
        df_encoded = self._one_hot_encode(df)

        # 3) load model to get expected feature names
        model_path = self.cfg["pipeline_runner"]["model_path"]
        model = joblib.load(model_path)

        expected = getattr(model, "feature_names_in_", None)
        df_filled = df_encoded.reindex(columns=expected, fill_value=0)

        return df_filled
