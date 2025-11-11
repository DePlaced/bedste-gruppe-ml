# ===============================================================
# postprocessing.py
# TRAINING: save the model
# ===============================================================
from typing import Dict, Any
import os
import joblib


class PostprocessingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    # ----------------------------- TRAINING -----------------------------
    def run_train(self, model) -> None:
        """Persist the trained model to the configured path."""
        model_path = self.cfg["pipeline_runner"]["model_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
