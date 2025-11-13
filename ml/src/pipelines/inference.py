# ml/src/pipelines/inference.py

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any


class InferencePipeline:
    """
    Loads the trained model and returns prediction info from a prepared input.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def run(self, df_feats: pd.DataFrame) -> Dict[str, Any]:
        if df_feats.empty:
            raise ValueError("No rows available for inference.")

        # 1) Load the trained model
        model_path = self.cfg["pipeline_runner"]["model_path"]
        model = joblib.load(model_path)

        # 2) Use the last row for prediction
        last_row = df_feats.tail(1)

        # 3) Predict label
        y_pred = model.predict(last_row)[0]

        # 4) Predict confidence (if supported)
        confidence: float | None = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(last_row)[0]  # shape: (n_classes,)
            confidence = float(np.max(proba))

        return {
            "label": y_pred,
            "confidence": confidence,
        }
