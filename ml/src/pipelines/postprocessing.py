# ===============================================================
# postprocessing.py
# ---------------------------------------------------------------
# TRAINING: save the model
# INFERENCE: format the prediction row
# ===============================================================

from typing import Dict, Any
import os
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
    def run_inference(self, y_pred, current_timestamp) -> pd.DataFrame:
        ts_now = pd.to_datetime(current_timestamp)
        increment = pd.Timedelta(self.cfg["pipeline_runner"]["time_increment"])
        ts_pred = ts_now + increment

        inf_cfg = self.cfg.get("inference", {})

        # ---- New style: classification config ----
        if "output_type" in inf_cfg:
            output_type = str(inf_cfg.get("output_type", "label")).lower()

            if output_type == "label":
                # y_pred is expected to be a class label, e.g. "EDIBLE" / "POISONOUS"
                val = y_pred

            elif output_type == "probability":
                # y_pred is expected to be a probability (0–1)
                val = float(y_pred)
                # clamp to [0, 1] just to be safe
                val = max(0.0, min(1.0, val))

            else:
                # Fallback: treat as numeric (regression-style)
                val = float(y_pred)

            return pd.DataFrame({"datetime": [ts_pred], "prediction": [val]})