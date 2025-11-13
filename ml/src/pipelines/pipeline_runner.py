# ml/src/pipelines/pipeline_runner.py

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
    TRAINING:
      raw → preprocess → feature engineering → training → postprocess → save model

    INFERENCE:
      raw JSON → preprocess → feature engineering → inference → postprocess → save input and prediction
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        self.cfg = config
        self.dm = data_manager

        self.prep = PreprocessingPipeline(config)
        self.post = PostprocessingPipeline(config)
        self.feat = FeatureEngineeringPipeline(config)
        self.train = TrainingPipeline(config)
        self.inf = InferencePipeline(config)

    # ============================== TRAINING ==============================
    def run_training(self) -> None:
        df = self.dm.load_raw_csv()

        # 1) preprocess
        df_prep = self.prep.training(df)

        # 2) feature engineering (one-hot + store feature cols)
        df_feat = self.feat.training(df_prep)

        # 3) training (df now has one-hot features + target)
        model = self.train.run(df_feat)

        # 4) postprocess
        self.post.training(model)

    # ============================== INFERENCE ==============================
    def run_prediction(self, row_dict: Dict[str, Any]) -> None:
        df_raw = pd.DataFrame([row_dict])

        # 1) preprocess
        df_prep = self.prep.inference(df_raw)

        # 2) feature engineering (one-hot + align + fill missing with 0)
        df_feat = self.feat.inference(df_prep)

        # 3) inference
        prediction = self.inf.run(df_feat)

        # 4) postprocess
        df_post = self.post.inference(prediction)

        # 5) save prediction and input
        row_id = self.dm.append_row_to_prod(row_dict)
        df_post.insert(0, "id", row_id)
        self.dm.save_predictions(df_post)
