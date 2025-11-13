
import pandas as pd
from typing import Dict, List, Any

class PreprocessingPipeline:
    """
     TRAINING: drop duplicates, reset index, drop configured columns
     INFERENCE: drop configured columns
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["preprocessing"]
        t_cfg = config.get("training", {}).get("target", {})
        self.target_col = t_cfg.get("source_col")

    # ---------- helpers ----------
    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        keep = [c for c in df.columns if c not in columns]
        return df[keep].copy()

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    # ---------- TRAINING ----------
    def training(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._drop_duplicates(df)
        df = df.reset_index(drop=True)

        # Drop configured columns
        if self.cfg.get("drop_columns"):
            df = self.drop_columns(df, self.cfg["drop_columns"])

        return df

    # ---------- INFERENCE ----------
    def inference(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.get("drop_columns"):
            df = self.drop_columns(df, self.cfg["drop_columns"])

        return df
