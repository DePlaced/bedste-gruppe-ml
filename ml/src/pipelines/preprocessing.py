# src/pipelines/preprocessing.py
import pandas as pd
from typing import Dict, List, Any


class PreprocessingPipeline:
    """
    Minimal, time-series friendly preprocessing:
      1) Drop exact duplicates
      2) Drop columns, that have been chosen for removable in the data analysis step
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["preprocessing"]

    # ---------- helpers ----------
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
        if self.cfg.get("drop_columns"):
            df = self.drop_columns(df, self.cfg["drop_columns"])
        return df
