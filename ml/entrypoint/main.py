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
    """
    Show the last N predictions from predictions.csv
    """
    try:
        n = int(request.args.get("n", 10))
    except Exception:
        n = 10

    pred_path = ROOT / CFG["data_manager"]["predictions_path"]
    df = _load_csv(pred_path)
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
    """
    Accept a full JSON row (same feature columns as training, but NOT 'poisonous'),
    append it to the production DB, run preprocessing + classification,
    log the prediction in predictions.csv, and return {id, prediction}.
    """
    try:
        payload = request.get_json(silent=True) or {}
        if not payload:
            return jsonify({"status": "error", "message": "Empty or invalid JSON payload"}), 400

        # Ignore any datetime the client might send
        payload.pop("datetime", None)

        result = RUNNER.predict_and_log(payload)
        return jsonify(_to_native({**result})), 200

    except Exception as e:
        return jsonify({"message": str(e)}), 500

# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
