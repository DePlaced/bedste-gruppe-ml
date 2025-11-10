#!/usr/bin/env python3
"""
Flask inference API for the mushroom classification pipeline (CSV-only), placed under ml/entrypoint/.

Endpoints
---------
GET  /health
GET  /db/status
GET  /predictions/tail?n=10
POST /predict/next         -> take the next unprocessed datetime from real_time_data_prod.csv, append to DB, predict
POST /predict/from-row     -> accept a full JSON row (with 'datetime'), append to DB, predict
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

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
    try:
        return pd.read_csv(path, parse_dates=["datetime"])
    except Exception:
        return pd.read_csv(path)


def pick_next_row(df_rt: pd.DataFrame, df_db: pd.DataFrame, inc: pd.Timedelta) -> Dict[str, Any]:
    """
    Choose the 'next' timestamp from the real-time stream for classification:

      - If DB has rows:
          next_ts = max(DB.datetime) + inc
          pick the row in real_time_data_prod.csv with datetime == next_ts
      - Else (DB empty):
          use the first row in the real-time CSV (smallest datetime)

    Returns that row as a plain dict (for PipelineRunner.predict_from_ui_row).
    """
    if df_rt.empty or ("datetime" not in df_rt.columns):
        raise ValueError("real_time_data_prod.csv is empty or lacks 'datetime' column.")

    # DB has data: move forward in time
    if df_db is not None and not df_db.empty and ("datetime" in df_db.columns):
        db_max = pd.to_datetime(df_db["datetime"], errors="coerce").max()
        target_ts = db_max + inc
        row = df_rt.loc[pd.to_datetime(df_rt["datetime"], errors="coerce") == target_ts]
        if row.empty:
            raise ValueError(
                f"No next row found at {target_ts}. "
                f"Ensure real_time_data_prod.csv contains that timestamp."
            )
        return row.iloc[0].to_dict()

    # DB empty → start with the earliest row of the stream
    row = df_rt.sort_values("datetime").iloc[0]
    return row.to_dict()


def _to_native(obj):
    """
    Recursively convert pandas/NumPy scalars and timestamps into Python-native
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


@app.get("/db/status")
def db_status():
    """
    Show basic info about the production DB (database_prod.csv):
      - number of rows
      - min & max datetime
    """
    db_path = ROOT / CFG["data_manager"]["prod_database_path"]
    df_db = _load_csv(db_path)
    if df_db.empty or "datetime" not in df_db.columns:
        return jsonify({"rows": 0, "min_datetime": None, "max_datetime": None, "path": str(db_path)}), 200

    dts = pd.to_datetime(df_db["datetime"], errors="coerce")
    payload = {
        "rows": int(len(df_db)),
        "min_datetime": (None if dts.isna().all() else str(dts.min())),
        "max_datetime": (None if dts.isna().all() else str(dts.max())),
        "path": str(db_path),
    }
    return jsonify(_to_native(payload)), 200


@app.get("/predictions/tail")
def predictions_tail():
    """
    Show the last N predictions from predictions.csv.
    This is classification, so 'prediction' will typically be 'EDIBLE' / 'POISONOUS'
    and may also contain 'actual' / 'correct' if update_predictions_with_actuals was run.
    """
    try:
        n = int(request.args.get("n", 10))
    except Exception:
        n = 10

    pred_path = ROOT / CFG["data_manager"]["predictions_path"]
    df = _load_csv(pred_path)
    if df.empty:
        return jsonify({"rows": 0, "tail": [], "path": str(pred_path)}), 200

    if "datetime" in df.columns:
        df = df.sort_values("datetime")

    tail = df.tail(n)
    payload = {
        "rows": int(len(df)),
        "tail": tail.to_dict(orient="records"),
        "path": str(pred_path),
    }
    return jsonify(_to_native(payload)), 200


@app.post("/predict/next")
def predict_next():
    """
    Use the 'next' unseen datetime from real_time_data_prod.csv to simulate a streaming UI event.

    Flow:
      1) Determine next datetime based on production DB max(datetime) + time_increment.
      2) Pick that row from real_time_data_prod.csv (which does NOT have 'poisonous').
      3) PipelineRunner.predict_from_ui_row(...) will:
           - append to production DB,
           - preprocess,
           - classify (EDIBLE/POISONOUS),
           - save prediction to predictions.csv,
           - backfill actuals from DB when available.
      4) Return a compact JSON payload.
    """
    try:
        rt_path = ROOT / CFG["data_manager"]["real_time_data_prod_path"]
        db_path = ROOT / CFG["data_manager"]["prod_database_path"]

        df_rt = _load_csv(rt_path)
        df_db = _load_csv(db_path)
        inc   = pd.Timedelta(CFG["pipeline_runner"]["time_increment"])

        row_dict = pick_next_row(df_rt, df_db, inc)
        result = RUNNER.predict_from_ui_row(row_dict)

        # Ensure JSON-native types
        return jsonify(_to_native({"status": "success", **result})), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/predict/from-row")
def predict_from_row():
    """
    Accept a full JSON row (must include 'datetime' and the same feature columns as training,
    but NOT 'poisonous'), append it to the DB, run preprocessing + classification, save,
    and return the prediction.
    """
    try:
        payload = request.get_json(silent=True) or {}
        if "datetime" not in payload:
            return jsonify({"status": "error", "message": "JSON must include 'datetime'"}), 400

        result = RUNNER.predict_from_ui_row(payload)
        return jsonify(_to_native({"status": "success", **result})), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------- Main ----------------
if __name__ == "__main__":
    # Do NOT re-seed DB here (to avoid wiping history).
    app.run(host="0.0.0.0", port=5001, debug=False)
