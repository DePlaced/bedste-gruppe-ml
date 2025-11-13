# ml/src/pipelines/inference.py

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any


class InferencePipeline:
    """
    Loads the trained model and returns prediction info from a prepared batch.

    Assumes:
      - df_prep is already preprocessed & one-hot encoded
      - We may be missing some dummy columns that existed at training time

    Steps:
      1) Align columns to the model's training schema (feature_names_in_):
         - add missing one-hot columns filled with 0
         - drop any extra columns
      2) Predict on the last row in the batch
      3) Return label + confidence
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def run(self, df_prep: pd.DataFrame) -> Dict[str, Any]:
        # 1) Load the trained model
        model_path = self.cfg["pipeline_runner"]["model_path"]
        model = joblib.load(model_path)

        # 2) Align columns to what the model expects
        expected = getattr(model, "feature_names_in_", None)

        if expected is not None:
            expected = list(expected)
            # Reindex:
            #  - all expected columns present
            #  - missing ones filled with 0
            #  - any extra cols dropped
            x_aligned = df_prep.reindex(columns=expected, fill_value=0)
        else:
            # Fallback: assume df_prep already matches
            x_aligned = df_prep.copy()

        if x_aligned.empty:
            raise ValueError("No rows available for inference after alignment.")

        # 3) Use the last row for prediction (you can change this to all rows if you like)
        last_row = x_aligned.tail(1)

        # 4) Predict label
        y_pred = model.predict(last_row)[0]

        # 5) Predict confidence (if supported)
        confidence: float | None = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(last_row)[0]  # shape: (n_classes,)
            confidence = float(np.max(proba))

        return {
            "label": y_pred,
            "confidence": confidence,
        }
