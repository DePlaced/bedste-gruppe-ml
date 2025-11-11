# ===============================================================
# InferencePipeline — INFERENCE ONLY (pipeline module)
# Path: ml/src/pipelines/inference.py
# ===============================================================
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any


class InferencePipeline:
    """
    Loads the trained model and returns prediction info from a prepared batch.

    Steps:
      1) Align columns to the model's training schema (feature_names_in_):
         - add missing one-hot columns filled with 0
         - drop any extra columns
      2) ffill/bfill remaining NaNs within the batch
      3) Drop any still-NaN rows
      4) Predict on the last valid row
      5) Return label + confidence + per-class probabilities
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def run(self, x: pd.DataFrame) -> Dict[str, Any]:
        model_path = self.cfg["pipeline_runner"]["model_path"]
        model = joblib.load(model_path)

        expected = getattr(model, "feature_names_in_", None)

        if expected is None:
            x_aligned = x.copy()
        else:
            expected = list(expected)
            x_aligned = x.copy()

            # 1) Add missing columns (0 for one-hot dummies)
            for col in expected:
                if col not in x_aligned.columns:
                    x_aligned[col] = 0

            # 2) Drop any extra columns
            x_aligned = x_aligned.loc[:, expected]

        # 3) Fill missing values within the batch
        x_aligned = x_aligned.ffill().bfill()

        # 4) Drop rows that still contain NaNs
        x_valid = x_aligned.dropna(axis=0, how="any")
        if x_valid.empty:
            raise ValueError(
                "All rows contain NaNs after preprocessing/feature alignment."
            )

        # 5) Predict on the last valid row
        last_row = x_valid.tail(1)

        # Predicted label
        y_pred = model.predict(last_row)[0]

        confidence: float | None = None
        class_probs: Dict[str, float] = {}

        # If classifier supports probabilities, expose them
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(last_row)[0]       # shape: (n_classes,)
            classes = getattr(model, "classes_", None)     # class labels

            if classes is not None:
                class_probs = {
                    str(cls): float(p) for cls, p in zip(classes, proba)
                }
                confidence = float(np.max(proba))

        return {
            "label": y_pred,
            "confidence": confidence,
            "class_probs": class_probs,
        }