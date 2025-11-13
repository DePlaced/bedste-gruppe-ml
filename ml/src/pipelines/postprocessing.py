# ===============================================================
# postprocessing.py
# TRAINING: save the model
# INFERENE: get the label and confidence percentage from the result
# ===============================================================
from typing import Dict, Any
import os
import joblib
import pandas as pd

from common.data_manager import DataManager


class PostprocessingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    # ----------------------------- TRAINING -----------------------------
    def train(self, model) -> None:
        model_path = self.cfg["pipeline_runner"]["model_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

    # ----------------------------- INFERENCE -----------------------------
    def inference(self, df_inf: Dict[str, Any]) -> pd.DataFrame:
        y_hat = df_inf["label"]
        confidence = df_inf.get("confidence")

        return pd.DataFrame([{
            "prediction": y_hat,
            "confidence": confidence,
        }])

