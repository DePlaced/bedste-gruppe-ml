# ===============================================================
# PipelineRunner — TRAINING + API inference orchestration
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
    Orchestrates the life-cycle:
      • TRAINING: raw → preprocess → train → save model
      • API INFERENCE: JSON row → DataFrame → preprocess → predict → label
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        self.cfg = config
        self.dm = data_manager

        self.prep = PreprocessingPipeline(config)
        self.post = PostprocessingPipeline(config)
        self.train = TrainingPipeline(config)
        self.inf = InferencePipeline(config)

    # ============================== TRAINING ==============================
    def run_training(self) -> None:
        df = self.dm.load_raw_csv()
        if df is None or df.empty:
            raise ValueError(
                "[training] Loaded dataframe is empty. "
            )

        df = self.prep.run(df)

        if df.empty:
            raise ValueError("[training] Dataframe empty after preprocessing.")

        model = self.train.run(df)
        self.post.run_train(model)

    def predict_and_log(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        API-style prediction:

          1) Append incoming row to production DB with a new 'id'
          2) Preprocess that single row
          3) Run classifier to get label + confidence + class_probs
          4) Log (id, prediction, confidence) into predictions.csv
          5) Merge any available actuals from prod DB into predictions.csv
          6) Return {id, prediction, confidence, class_probs}
        """
        # 1) append to prod DB, get new id
        row_id = self.dm.append_row_to_prod(row_dict)

        # 2) build a single-row DataFrame with the new id included
        df = pd.DataFrame([row_dict]).copy()
        df.insert(0, "id", row_id)

        # 3) preprocess
        df_prep = self.prep.run(df)

        # 4) drop target if present
        source_col = self.cfg["training"]["target"]["source_col"]  # 'poisonous'
        if source_col in df_prep.columns:
            df_prep = df_prep.drop(columns=[source_col])

        # 5) predict (now returns label + confidence + class_probs)
        inf_result = self.inf.run(df_prep)
        y_hat = inf_result["label"]
        confidence = inf_result.get("confidence")
        class_probs = inf_result.get("class_probs", {})

        # 6) log prediction to predictions.csv
        df_pred = pd.DataFrame([{
            "id": row_id,
            "prediction": y_hat,
            "confidence": confidence,
        }])
        self.dm.save_predictions(df_pred)

        # 7) sync actuals from prod DB into predictions.csv (based on id)
        self.dm.update_predictions_with_actuals(actual_col=source_col)

        # 8) return payload for API
        return {
            "id": row_id,
            "prediction": y_hat,
            "confidence": confidence,
            "class_probs": class_probs,
        }
