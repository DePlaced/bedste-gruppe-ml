# ===============================================================
# PipelineRunner — TRAINING + INFERENCE orchestration
# Path: ml/src/pipelines/pipeline_runner.py
# ===============================================================
from typing import Dict, Any
import pandas as pd

from common.data_manager import DataManager
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.training import TrainingPipeline
from pipelines.postprocessing import PostprocessingPipeline
from pipelines.inference import InferencePipeline


class PipelineRunner:
    """
    Orchestrates the full life-cycle:
      • TRAINING:   raw → preprocess → FE → train → save model
      • INFERENCE:  (stream/UI row) append → last-N → preprocess → FE → predict t+1
                     → save prediction (shifted time) → persist DB → backfill actuals
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        self.cfg = config
        self.dm = data_manager

        # Shared steps
        self.prep = PreprocessingPipeline(config)
        self.post = PostprocessingPipeline(config)

        # === TRAINING ===
        self.train = TrainingPipeline(config)

        # === INFERENCE ===
        self.inf = InferencePipeline(config)
        self.prod_df = self.dm.load_prod_data()   # rolling production DB CSV
        self.rt_df = self.dm.load_real_time()     # real-time CSV (UI stream)

    # ============================== TRAINING ==============================
    def run_training(self) -> None:
        df = self.dm.load_raw_csv()
        if df is None or df.empty:
            raise ValueError(
                "[training] Loaded dataframe is empty. "
                "Check data_manager.csv_path and the file integrity."
            )

        df = self.prep.run(df)
        if df.empty:
            raise ValueError("[training] Dataframe empty after preprocessing.")

        model = self.train.run(df)
        self.post.run_train(model)

    # ============================== INFERENCE (batch loop) ==============================
    def run_inference(self, current_timestamp: pd.Timestamp) -> None:
        """
        Single streaming step driven by a timestamp that must exist in the
        real_time_data_prod.csv (the "incoming" feed).
        """
        # 1) pick current incoming row
        rt_row = self.dm.get_timestamp_data(self.rt_df, current_timestamp)
        if rt_row is None or rt_row.empty:
            raise ValueError(
                f"[inference] No row found in real_time_data_prod at {current_timestamp}. "
                "Ensure the streaming CSV contains that timestamp."
            )

        # 2) update in-memory production DB
        self.prod_df = self.dm.append_data(self.prod_df, rt_row)

        # 3) build batch for features
        n = int(self.cfg["pipeline_runner"]["batch_size"])
        batch = self.dm.get_n_last_points(self.prod_df, n)
        batch = self.prep.run(batch)
        batch = self.fe.run(batch)
        if batch is None or batch.empty:
            raise ValueError(
                "[inference] All rows contain NaNs after preprocessing/feature engineering. "
                "Increase batch_size in config or ensure streaming rows include required inputs."
            )

        # 4) drop raw source column to avoid unseen-at-fit features
        source_col = self.cfg["training"]["target"]["source_col"]  # usually 'poisonous'
        if source_col in batch.columns:
            batch = batch.drop(columns=[source_col])

        # 5) predict next hour (use last row)
        y_hat = self.inf.run(batch)

        # 6) format & save prediction; persist DB
        df_pred = self.post.run_inference(y_pred=y_hat, current_timestamp=current_timestamp)
        self.dm.save_predictions(df_pred, current_timestamp)
        self.dm.save_prod_data(self.prod_df)

        # 7) backfill actuals for any predictions whose ground truth just arrived
        self.dm.update_predictions_with_actuals(actual_col=source_col)

    # ============================== INFERENCE (UI helper) ==============================
    def predict_from_ui_row(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append ONE incoming UI row (must include 'datetime'), build features from last-N rows,
        predict t+1, save, backfill actuals, and return a JSON-friendly payload.
        """
        if "datetime" not in row_dict:
            raise ValueError("Incoming UI row must include 'datetime'.")

        # 1) Convert dict → DataFrame (single-row), normalize datetime and append to rolling DB
        rt_row = pd.DataFrame([row_dict])
        rt_row["datetime"] = pd.to_datetime(rt_row["datetime"], errors="coerce")
        if rt_row["datetime"].isna().any():
            raise ValueError("Invalid 'datetime' in UI row; could not parse to timestamp.")
        self.prod_df = self.dm.append_data(self.prod_df, rt_row)

        # 2) Build last-N batch and create features
        n = int(self.cfg["pipeline_runner"]["batch_size"])
        batch = self.dm.get_n_last_points(self.prod_df, n)
        batch = self.prep.run(batch)
        batch = self.fe.run(batch)
        if batch is None or batch.empty:
            raise ValueError(
                "[inference] No usable rows after preprocessing/feature engineering. "
                "Check batch_size and that the input row contains the required columns."
            )

        # 3) Drop the raw source column used to build the training target
        source_col = self.cfg["training"]["target"]["source_col"]
        if source_col in batch.columns:
            batch = batch.drop(columns=[source_col])

        # 4) Predict next step (t+1 relative to the UI row's timestamp)
        current_ts = pd.to_datetime(row_dict["datetime"])
        y_hat = self.inf.run(batch)
        df_pred = self.post.run_inference(y_pred=y_hat, current_timestamp=current_ts)

        # 5) Persist prediction + DB; then backfill actuals that are now available
        self.dm.save_predictions(df_pred, current_ts)
        self.dm.save_prod_data(self.prod_df)
        self.dm.update_predictions_with_actuals(actual_col=source_col)

        # 6) Produce a compact response for the API/UI — **return native Python types**
        pred_val = df_pred["prediction"].iloc[0]
        return {
            "input_timestamp": str(current_ts),
            "prediction_timestamp": str(df_pred["datetime"].iloc[0]),
            "prediction": int(pred_val) if float(pred_val).is_integer() else float(pred_val),
            "used_rows": int(n),
        }
