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
      1) Align columns to the model's training schema (feature_names_in_):
         - add missing one-hot columns filled with 0
         - drop any extra columns
      2) ffill/bfill remaining NaNs using only the batch history
      3) Drop any still-NaN rows
      4) Predict on the last valid row
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def run(self, x: pd.DataFrame):
        model_path = self.cfg["pipeline_runner"]["model_path"]
        model = joblib.load(model_path)

        expected = getattr(model, "feature_names_in_", None)

        if expected is None:
            # Model doesn't expose feature_names_in_ (unlikely for sklearn);
            # just use whatever is in x.
            x_aligned = x.copy()
        else:
            expected = list(expected)
            x_aligned = x.copy()

            # 1) Add any missing columns (set to 0 for one-hot dummies)
            for col in expected:
                if col not in x_aligned.columns:
                    x_aligned[col] = 0

            # 2) Drop any extra columns not used at training time
            x_aligned = x_aligned.loc[:, expected]

        # 3) Fill missing values within the batch (if any)
        x_aligned = x_aligned.ffill().bfill()

        # 4) Drop rows that still contain NaNs
        x_valid = x_aligned.dropna(axis=0, how="any")
        if x_valid.empty:
            raise ValueError(
                "All rows contain NaNs after preprocessing/feature alignment. "
                "Increase batch_size in config or ensure streaming rows contain required inputs."
            )

        # 5) Predict on the last valid row
        last_row = x_valid.tail(1)
        y_pred = model.predict(last_row)

        # For classification, this is typically a class label (e.g. 'EDIBLE' / 'POISONOUS')
        return y_pred[0]
