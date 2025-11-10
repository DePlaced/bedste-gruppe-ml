# ===============================================================
# postprocessing.py
# ---------------------------------------------------------------
# TRAINING: save the model
# INFERENCE: format the prediction row (optionally as integer)
# ===============================================================

from typing import Dict, Any
import os
import math
import joblib
import pandas as pd


class PostprocessingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    # ----------------------------- TRAINING -----------------------------
    def run_train(self, model) -> None:
        """Persist the trained model to the configured path."""
        model_path = self.cfg["pipeline_runner"]["model_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

    # ----------------------------- INFERENCE ----------------------------
    def run_inference(self, y_pred: float, current_timestamp) -> pd.DataFrame:
        """
        Create a one-row DataFrame with:
          - datetime: current_timestamp + time_increment
          - prediction: formatted number (float or int per config)

        Config (optional):
          inference:
            output_integer: true|false
            integer_strategy: "round"|"floor"|"ceil"
            min_value: 0
        """
        ts_now = pd.to_datetime(current_timestamp)
        increment = pd.Timedelta(self.cfg["pipeline_runner"]["time_increment"])
        ts_pred = ts_now + increment

        inf_cfg = self.cfg.get("inference", {})
        as_int = bool(inf_cfg.get("output_integer", False))
        strategy = str(inf_cfg.get("integer_strategy", "round")).lower()
        min_value = float(inf_cfg.get("min_value", 0))

        val = float(y_pred)
        val = max(min_value, val)  # clamp negatives

        if as_int:
            if strategy == "floor":
                val = math.floor(val)
            elif strategy == "ceil":
                val = math.ceil(val)
            else:
                val = int(round(val))

        return pd.DataFrame({"datetime": [ts_pred], "prediction": [val]})



