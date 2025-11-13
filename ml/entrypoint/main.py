#!/usr/bin/env python3
"""
Flask inference API for the mushroom classification pipeline (no streaming).

Endpoints
---------
GET  /health
GET  /predictions/tail?n=10   (optional, if you log predictions)
POST /predict      -> accept a full JSON row, predict
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yaml
import traceback

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
    df = DM.load_prediction_data()
    if df.empty:
        return jsonify({"rows": 0, "tail": []}), 200

    tail = df.tail(n)
    payload = {
        "rows": int(len(df)),
        "predictions": tail.to_dict(orient="records"),
    }
    return jsonify(_to_native(payload)), 200


@app.post("/predictions")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        if not payload:
            return jsonify({"status": "bad", "reason": "Empty or invalid JSON payload"}), 400

        RUNNER.run_prediction(payload)

        return jsonify(_to_native({"status": "success"})), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
