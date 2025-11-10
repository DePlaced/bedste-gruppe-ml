import pandas as pd
from typing import Dict, List, Any


class PreprocessingPipeline:
    """
    Preprocessing for mushroom dataset:
      1) Drop duplicates
      2) Drop columns listed in config["preprocessing"]["drop_columns"]
      3) One-hot encode all feature columns (keep target raw)
         - target column name comes from config["training"]["target"]["source_col"]
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.preprocessing_cfg = self.cfg["preprocessing"]
        self.target_col = self.cfg["training"]["target"]["source_col"]

    # ---------- helpers ----------
    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Drop unwanted columns if they exist."""
        keep = [c for c in df.columns if c not in columns]
        return df[keep].copy()

    @staticmethod
    def one_hot_encode_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        One-hot encode all columns except the target column.
        Target is kept unchanged so training can still find it.
        """
        # If target is missing, just encode everything
        if target_col not in df.columns:
            return pd.get_dummies(df, drop_first=False, dtype=int)

        # Split features and target
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols]
        y = df[target_col]

        # One-hot encode features
        X_encoded = pd.get_dummies(X, drop_first=False, dtype=int)

        # Add target back
        X_encoded[target_col] = y

        return X_encoded

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        return df.drop_duplicates()

    # ---------- orchestrated run ----------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Drop duplicates
        df = self._drop_duplicates(df)

        # Step 2: Drop unwanted columns
        drop_cols = self.preprocessing_cfg.get("drop_columns", [])
        if drop_cols:
            df = self.drop_columns(df, drop_cols)

        # Step 3: One-hot encode all features, keeping target raw
        df = self.one_hot_encode_features(df, self.target_col)

        # Step 4: Reset index for cleanliness
        return df.reset_index(drop=True)
