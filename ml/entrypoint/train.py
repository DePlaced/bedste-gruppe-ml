# ml/entrypoint/train.py
import os
import sys
from pathlib import Path
import yaml

# --- resolve important paths ---
ROOT   = Path(__file__).resolve().parents[2]  # project root
ML_SRC = ROOT / "ml" / "src"                  # where pipelines live

# Put ROOT (for 'common') and ML_SRC (for 'pipelines') on sys.path
for p in (ROOT, ML_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Ensure relative paths in config.yaml resolve correctly
os.chdir(ROOT)

# Now imports will work
from common.data_manager import DataManager
from pipelines.pipeline_runner import PipelineRunner
# --------------------------------

def read_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = read_config(ROOT / "config" / "config.yaml")
    dm  = DataManager(cfg)
    runner = PipelineRunner(cfg, dm)
    runner.run_training()
    print("Training finished. Model saved to:", cfg["pipeline_runner"]["model_path"])


