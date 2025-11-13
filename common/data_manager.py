# ===============================================================
# DataManager — TRAINING + optional logging (CSV only)
# Path: ml/src/common/data_manager.py
# ===============================================================
import os
import pandas as pd
from typing import Dict, Any


class DataManager:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        dm = self.cfg["data_manager"]
        # === TRAINING ===
        self.raw_csv = dm["raw_data_path"]
        # === PROD DB ===
        self.prod_db_csv = dm["prod_database_path"]
        # === PREDICTIONS ===
        self.pred_csv = dm["predictions_path"]

    # ------------------- CSV I/O -------------------
    @staticmethod
    def _read_csv(path: str) -> pd.DataFrame:
        """Read a CSV; return empty DataFrame if file doesn't exist."""
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)

    @staticmethod
    def _write_csv(df: pd.DataFrame, path: str) -> None:
        """Write a DataFrame to CSV, creating parent dirs if needed."""
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        df.to_csv(path, index=False)


    # ================= TRAINING =================
    def load_raw_csv(self) -> pd.DataFrame:
        return self._read_csv(self.raw_csv)

    # ================= PROD DB =================
    def load_prod_data(self) -> pd.DataFrame:
        """Load the production DB CSV."""
        return self._read_csv(self.prod_db_csv)

    def save_prod_data(self, df: pd.DataFrame) -> None:
        """Persist the production DB CSV."""
        self._write_csv(df, self.prod_db_csv)

    def append_row_to_prod(self, row_dict: Dict[str, Any]) -> int:
        prod = self.load_prod_data()
        new_row = pd.DataFrame([row_dict])

        if prod.empty:
            # First row in prod DB
            new_row = new_row.reset_index(drop=True)
            new_row.insert(0, "id", 0)
            self.save_prod_data(new_row)
            return 0

        next_id = int(prod["id"].max()) + 1
        new_row.insert(0, "id", next_id)

        # Align columns: ensure new_row has same columns as prod
        for col in prod.columns:
            if col not in new_row.columns:
                new_row[col] = pd.NA

        # Ensure prod has any new columns from new_row
        for col in new_row.columns:
            if col not in prod.columns:
                prod[col] = pd.NA

        prod = pd.concat([prod, new_row[prod.columns]], ignore_index=True)
        self.save_prod_data(prod)

        return next_id

    # ================= PREDICTIONS ==============
    def load_prediction_data(self) -> pd.DataFrame:
        """Load the predictions CSV."""
        return self._read_csv(self.pred_csv)

    def save_predictions(self, prediction: pd.DataFrame) -> None:
        path = self.pred_csv
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not os.path.exists(path):
            prediction.to_csv(path, index=False, header=True, mode="w")
        else:
            prediction.to_csv(path, index=False, header=False, mode="a")

    def update_predictions_with_actuals(self, actual_col: str) -> None:
        preds = self.load_prediction_data()
        prod  = self.load_prod_data()

        if preds.empty or prod.empty:
            return

        if "id" not in preds.columns or "id" not in prod.columns:
            return

        if actual_col not in prod.columns:
            return

        preds = preds.copy()
        prod  = prod.copy()

        # Drop previous enrichment columns to avoid duplicates
        preds = preds.drop(columns=["actual", "correct"], errors="ignore")

        merged = preds.merge(
            prod[["id", actual_col]],
            on="id",
            how="left",
            validate="one_to_one"
        ).rename(columns={actual_col: "actual"})

        # correct = prediction == actual (when both exist)
        merged["correct"] = merged["prediction"] == merged["actual"]

        # Sort and drop potential duplicate ids
        merged = merged.sort_values("id").drop_duplicates(subset=["id"], keep="last")

        # Save back to predictions.csv
        self._write_csv(merged, self.pred_csv)
