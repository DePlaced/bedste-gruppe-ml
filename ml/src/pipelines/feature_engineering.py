# ============================================
# Feature Engineering Pipeline
# ============================================

import pandas as pd
from typing import Any, Dict


class FeatureEngineeringPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def training(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def inference(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
