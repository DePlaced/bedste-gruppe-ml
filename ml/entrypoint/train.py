# ml/entrypoint/train.py
import os
import sys
from pathlib import Path
import yaml

ROOT   = Path(__file__).resolve().parents[2]  # project root
ML_SRC = ROOT / "ml" / "src"

for p in (ROOT, ML_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

os.chdir(ROOT)

from common.data_manager import DataManager
from pipelines.pipeline_runner import PipelineRunner


def read_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    cfg = read_config(ROOT / "config" / "config.yaml")
    dm = DataManager(cfg)
    runner = PipelineRunner(cfg, dm)

    runner.run_training()
    print("Training complete. Model saved to:", cfg["pipeline_runner"]["model_path"])
