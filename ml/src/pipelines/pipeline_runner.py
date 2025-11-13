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
from pipelines.feature_engineering import FeatureEngineeringPipeline


class PipelineRunner:
    """
        TRAINING: raw → preprocess → feature engineering → training → postprocess → save model
        INFERENCE: raw JSON → preprocess → feature engineering → inference → postprocess → save input and prediction
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        self.cfg = config
        self.dm = data_manager

        self.prep = PreprocessingPipeline(config)
        self.post = PostprocessingPipeline(config)
        self.fe = FeatureEngineeringPipeline(config)
        self.train = TrainingPipeline(config)
        self.inf = InferencePipeline(config)

    # ============================== TRAINING ==============================
    def run_training(self) -> None:
        df = self.dm.load_raw_csv()

        # 1) preprocess
        df = self.prep.training(df)

        # 2) feature engineering - does nothing
        df = self.fe.training(df)

        # 3) training
        model = self.train.run(df)

        # 4) postprocess
        self.post.training(model)

    # ============================== INFERENCE ==============================
    def run_prediction(self, row_dict: Dict[str, Any]) -> None:
        df = pd.DataFrame([row_dict])

        # 1) preprocess
        df_prep = self.prep.inference(df)

        # 2) feature engineering - does nothing
        df = self.fe.inference(df)

        # 3) inference
        prediction = self.inf.run(df_prep)

        # 4) postprocess
        df_post = self.post.inference(prediction)

        # 5) save prediction and input
        row_id = self.dm.append_row_to_prod(row_dict)
        df_post.insert(0, "id", row_id)
        self.dm.save_predictions(df_post)
