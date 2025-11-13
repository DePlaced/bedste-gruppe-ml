# src/pipelines/preprocessing.py
import pandas as pd
from typing import Dict, List, Any


class PreprocessingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["preprocessing"]
        t_cfg = config.get("training", {}).get("target", {})
        self.target_col = t_cfg.get("source_col")

    # ---------- helpers ----------
    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        keep = [c for c in df.columns if c not in columns]
        return df[keep].copy()

    def one_hot_encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode all columns except the target.
        Target is kept unchanged so training can still find it.
        """
        df = df.copy()

        # Separate target if present
        y = None
        if self.target_col is not None and self.target_col in df.columns:
            y = df[self.target_col]
            df = df.drop(columns=[self.target_col])

        # One-hot encode all remaining columns
        x_encoded = pd.get_dummies(df, drop_first=False, dtype=int)

        # Add target back if we had it
        if y is not None:
            x_encoded[self.target_col] = y.values

        return x_encoded

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    # ---------- TRAINING ----------
    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop duplicate rows
        df = self._drop_duplicates(df)
        df = df.reset_index(drop=True)

        # Drop configured columns
        if self.cfg.get("drop_columns"):
            df = self.drop_columns(df, self.cfg["drop_columns"])

        # One-hot encode
        df = self.one_hot_encode_features(df)

        return df

    # ---------- INFERENCE ----------
    def inference(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop configured columns - just because it's annoying to change the body of the raw json coming from the api endpoint
        if self.cfg.get("drop_columns"):
            df = self.drop_columns(df, self.cfg["drop_columns"])

        # One-hot encode
        df = self.one_hot_encode_features(df)

        return df