# ===============================================================
# DataManager — TRAINING + INFERENCE (CSV only)
# Path: ml/src/common/data_manager.py
# ===============================================================
import os
import pandas as pd
from typing import Dict, Any, Optional


class DataManager:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        dm = self.cfg["data_manager"]
        # === TRAINING ===
        self.raw_csv = dm["csv_path"]
        # === INFERENCE ===
        self.prod_db_csv = dm["prod_database_path"]
        self.rt_csv = dm["real_time_data_prod_path"]
        self.pred_csv = dm["predictions_path"]

    # ------------------- CSV I/O -------------------
    @staticmethod
    def _read_csv(path: str) -> pd.DataFrame:
        """
        Read a CSV with best-effort datetime parsing for 'datetime'.
        Returns empty DataFrame if file doesn't exist.
        """
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            return pd.read_csv(path, parse_dates=["datetime"])
        except Exception:
            return pd.read_csv(path)

    @staticmethod
    def _write_csv(df: pd.DataFrame, path: str) -> None:
        """Write a DataFrame to CSV, creating parent dirs if needed."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    # ================= TRAINING =================
    def load_raw_csv(self) -> pd.DataFrame:
        """Load the training CSV."""
        return self._read_csv(self.raw_csv)

    # ================= INFERENCE ================
    def initialize_prod_database(self) -> None:
        """
        Initialize rolling production DB from TRAINING CSV,
        and clear any previous predictions.
        """
        df = self._read_csv(self.raw_csv)
        self._write_csv(df, self.prod_db_csv)
        if os.path.exists(self.pred_csv):
            os.remove(self.pred_csv)

    def load_prod_data(self) -> pd.DataFrame:
        """Load the rolling production DB CSV."""
        return self._read_csv(self.prod_db_csv)

    def save_prod_data(self, df: pd.DataFrame) -> None:
        """Persist the rolling production DB CSV."""
        self._write_csv(df, self.prod_db_csv)

    def load_real_time(self) -> pd.DataFrame:
        """Load the real-time (stream) CSV."""
        return self._read_csv(self.rt_csv)

    def load_prediction_data(self) -> pd.DataFrame:
        """Load the predictions CSV."""
        return self._read_csv(self.pred_csv)

    def save_predictions(self, df_pred: pd.DataFrame, current_timestamp) -> None:
        """
        Append prediction rows to predictions CSV.
        Overwrite on the very first inference timestamp to start fresh.
        """
        first_ts = pd.to_datetime(self.cfg["pipeline_runner"]["first_timestamp"])
        existing = self.load_prediction_data()
        if existing.empty or pd.to_datetime(current_timestamp) == first_ts:
            out = df_pred.copy()
        else:
            out = pd.concat([existing, df_pred], ignore_index=True)

        # Ensure sorted & unique by datetime
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
            out = out.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")

        self._write_csv(out, self.pred_csv)

    # ---------- small helpers ----------
    @staticmethod
    def append_data(current: Optional[pd.DataFrame], new_rows: pd.DataFrame) -> pd.DataFrame:
        """
        Append new_rows into current, sort by datetime, drop duplicates.
        """
        if new_rows is None or new_rows.empty:
            return (current.copy() if current is not None else pd.DataFrame())

        if current is None or current.empty:
            out = new_rows.copy()
        else:
            out = pd.concat([current, new_rows], ignore_index=True)

        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
            out = out.drop_duplicates(subset=["datetime"]).sort_values("datetime")

        return out.reset_index(drop=True)

    @staticmethod
    def get_timestamp_data(df: pd.DataFrame, timestamp) -> pd.DataFrame:
        """Return the single row matching `timestamp` in the 'datetime' column."""
        if df is None or df.empty or "datetime" not in df.columns:
            return pd.DataFrame()
        ts = pd.to_datetime(timestamp)
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        return df.loc[dt == ts].copy()

    @staticmethod
    def get_n_last_points(df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Return the last n rows of df (safe with empty)."""
        if df is None or df.empty:
            return df
        return df.tail(n).reset_index(drop=True)

    # ---------- enrich predictions with actuals ----------
    def update_predictions_with_actuals(self, actual_col: str = "taxi_pickups") -> None:
        """
        Enrich predictions.csv with actuals and errors whenever they become available.
        Adds/updates: ['actual', 'error', 'abs_error'].
        Safe against re-running many times and duplicate labels.
        Ensures integer-looking values write as integers in CSV by casting to Pandas Int64.
        """
        preds = self.load_prediction_data()
        prod  = self.load_prod_data()

        # Basic guards
        if preds.empty or prod.empty:
            return
        if "datetime" not in preds.columns or "datetime" not in prod.columns:
            return
        if actual_col not in prod.columns:
            return

        preds = preds.copy()
        prod  = prod.copy()

        # Parse datetimes
        preds["datetime"] = pd.to_datetime(preds["datetime"], errors="coerce")
        prod["datetime"]  = pd.to_datetime(prod["datetime"], errors="coerce")

        # Drop any previous enrichment columns to avoid duplicate labels after merge
        preds = preds.drop(columns=["actual", "error", "abs_error"], errors="ignore")

        # Merge latest actuals from production DB
        merged = preds.merge(
            prod[["datetime", actual_col]],
            on="datetime",
            how="left",
            validate="one_to_one"
        ).rename(columns={actual_col: "actual"})

        # To compute errors, make sure numeric
        merged["prediction"] = pd.to_numeric(merged["prediction"], errors="coerce")
        merged["actual"]     = pd.to_numeric(merged["actual"], errors="coerce")

        # Compute errors (NaN if actual not available yet)
        merged["error"]     = merged["prediction"] - merged["actual"]
        merged["abs_error"] = merged["error"].abs()

        # De-duplicate on datetime just in case
        merged = merged.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")

        # ---- KEY: cast to nullable integers so CSV shows 10 (not 10.0), but keeps missing as <NA> ----
        for col in ["prediction", "actual", "error", "abs_error"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").round()
                merged[col] = merged[col].astype("Int64")  # nullable int: prints as plain int; supports <NA>

        self._write_csv(merged, self.pred_csv)
