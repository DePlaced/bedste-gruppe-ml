"""
Flask inference API for the mushroom classification pipeline

Endpoints
---------
GET  /health
GET  /predictions?n=10
POST /predict
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yaml

# --------- Resolve project paths ---------
ROOT   = Path(__file__).resolve().parents[2]  # project root
ML_SRC = ROOT / "ml" / "src"

for p in (ROOT, ML_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

os.chdir(ROOT)

from common.data_manager import DataManager
from pipelines.pipeline_runner import PipelineRunner


# ---------------- Helpers ----------------
def read_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_native(obj):
    """
    Recursively convert pandas/NumPy scalars into Python-native
    types so Flask's jsonify can serialize them.
    """
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat(sep=" ")
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


# ---------------- App & bootstrap ----------------
app = Flask(__name__)

CFG = read_config(ROOT / "config" / "config.yaml")
DM  = DataManager(CFG)
RUNNER = PipelineRunner(CFG, DM)

# ---------------- Routes ----------------
@app.get("/")
def health():
    return jsonify({"status": "ok"}), 200

@app.get("/predictions")
def predictions():
    n = int(request.args.get("n", 10))

    df_pred = DM.load_prediction_data()
    if df_pred.empty:
        return jsonify({"rows": 0, "predictions": []}), 200

    df_prod = DM.load_prod_data()

    if "id" not in df_pred.columns or "id" not in df_prod.columns:
        return jsonify({"error": "Both prediction and prod data must have an 'id' column."}), 500

    df_merged = df_pred.merge(
        df_prod,
        on="id",
        how="left",
        suffixes=("_pred", "_prod")
    )

    # 4) Get last n rows from the merged frame
    tail = df_merged.tail(n)

    payload = {
        "rows": int(len(df_pred)),
        "predictions": tail.to_dict(orient="records")
    }

    return jsonify(_to_native(payload)), 200

@app.post("/predictions")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        if not payload:
            return jsonify({"status": "bad", "reason": "Empty or invalid JSON payload"}), 400

        RUNNER.run_prediction(payload)

        return jsonify({"status": "success"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)