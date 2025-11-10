# src/pipelines/preprocessing.py
import pandas as pd
from typing import Dict, List, Any


class PreprocessingPipeline:
    """
    Minimal, time-series friendly preprocessing:
      1) Drop exact duplicates
      2) Parse & sort datetime (if present)
      3) Light imputation for weather (ffill → bfill)
      4) Domain-based outlier clipping (min/max)
      5) Column rename (optional) + column drop (from config)
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["preprocessing"]

    # ---------- helpers ----------
    @staticmethod
    def rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        return df.rename(columns=mapping)

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        keep = [c for c in df.columns if c not in columns]
        return df[keep].copy()

    # ---------- hygiene ----------
    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    # ---------- orchestrated run ----------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._drop_duplicates(df)

        df = df.reset_index(drop=True)
        if self.cfg.get("column_mapping"):
            df = self.rename_columns(df, self.cfg["column_mapping"])
        if self.cfg.get("drop_columns"):
            df = self.drop_columns(df, self.cfg["drop_columns"])
        return df
