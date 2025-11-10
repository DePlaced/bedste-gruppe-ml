# src/pipelines/feature_engineering.py
import pandas as pd
from typing import Dict, Any, List, Tuple


class FeatureEngineeringPipeline:
    """
    Minimal feature engineering for next-hour taxi demand:
      - Rolling means of past demand (uses shift(1) so only history)
      - Weather deltas = current minus previous hour (t - t-1)
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["feature_engineering"]

    def _rolling_pickups(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        wins = [w for w in self.cfg.get("rolling_windows", [3, 6]) if isinstance(w, int) and w > 0]
        created: List[str] = []

        if "taxi_pickups" in df.columns:
            for w in wins:
                col = f"pickups_roll{w}"
                df[col] = df["taxi_pickups"].shift(1).rolling(w).mean()
                created.append(col)

        return df, created

    def _weather_deltas(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        created: List[str] = []
        for col in self.cfg.get("weather_deltas", []):
            if col in df.columns:
                out = f"d_{col}"
                df[out] = df[col] - df[col].shift(1)
                created.append(out)
        return df, created

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df, created_roll = self._rolling_pickups(df)
        df, created_delta = self._weather_deltas(df)

        na_sensitive = created_roll + created_delta
        if na_sensitive:
            df = df.dropna(subset=na_sensitive)
        return df.reset_index(drop=True)

