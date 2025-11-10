# ===============================================================
# InferencePipeline — INFERENCE ONLY (pipeline module)
# ===============================================================
import joblib
import pandas as pd
from typing import Dict, Any


class InferencePipeline:
    """
    Loads the trained model and returns the LAST prediction from a prepared batch.

    Robustness steps:
      1) Align columns to the model's training schema (feature_names_in_)
      2) ffill/bfill remaining NaNs using only the batch history
      3) Drop any still-NaN rows
      4) Predict on the last valid row
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def run(self, x: pd.DataFrame) -> float:
        model_path = self.cfg["pipeline_runner"]["model_path"]
        model = joblib.load(model_path)

        expected = getattr(model, "feature_names_in_", None)
        if expected is None:
            x_aligned = x.copy()
        else:
            expected = list(expected)
            missing = [c for c in expected if c not in x.columns]
            if missing:
                raise ValueError(
                    "Inference features are missing columns the model was trained on.\n"
                    f"Missing: {missing}\n"
                    f"Available: {list(x.columns)}"
                )
            x_aligned = x.loc[:, expected].copy()

        x_aligned = x_aligned.ffill().bfill()
        x_valid = x_aligned.dropna(axis=0, how="any")
        if x_valid.empty:
            raise ValueError(
                "All rows contain NaNs after preprocessing/feature engineering. "
                "Increase batch_size in config or ensure streaming rows contain required inputs."
            )

        last_row = x_valid.tail(1)
        y_pred = model.predict(last_row)
        return float(y_pred[0])

