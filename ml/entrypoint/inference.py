# ===============================================================
# INFERENCE ENTRYPOINT (batch evaluation over prod DB)
# ===============================================================
import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import yaml

# --- resolve important paths ---
ROOT   = Path(__file__).resolve().parents[2]   # project root
ML_SRC = ROOT / "ml" / "src"                   # where pipelines/ live

# Put ROOT (for 'common') and ML_SRC (for 'pipelines') on sys.path
for p in (ROOT, ML_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Ensure relative paths in config.yaml resolve correctly
os.chdir(ROOT)

from common.data_manager import DataManager
from pipelines.pipeline_runner import PipelineRunner


def read_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def show_results(dm: DataManager, cfg: dict, tail_rows: int = 10) -> None:
    """
    Load predictions & production DB, merge on row_id, and show a concise summary.

    For classification:
      - actual_col = training.target.source_col (e.g. 'poisonous')
      - prediction column contains labels (e.g. 'EDIBLE' / 'POISONOUS')
    """
    actual_col = cfg["training"]["target"]["source_col"]  # e.g. 'poisonous'

    preds  = dm.load_prediction_data()
    actual = dm.load_prod_data()

    if preds is None or preds.empty:
        print("No predictions found. Did you run inference without errors?")
        return

    if "row_id" not in preds.columns:
        print("Missing 'row_id' column in predictions.csv.")
        return

    if actual is None or actual.empty:
        print("Production DB is empty; cannot compute accuracy.")
        return

    if actual_col not in actual.columns:
        print(f"Column '{actual_col}' not found in production DB. Available columns: {list(actual.columns)}")
        return

    # Reset index on actual so we have a stable row_id to join on
    actual_reset = actual.reset_index().rename(columns={"index": "row_id"})

    merged = pd.merge(
        preds[["row_id", "prediction"]],
        actual_reset[["row_id", actual_col]],
        on="row_id",
        how="inner",
        validate="one_to_one"
    ).sort_values("row_id")

    if merged.empty:
        print("No matching row_id between predictions and actuals.")
        return

    # ---- Classification metrics (string labels) ----
    correct = (merged["prediction"] == merged[actual_col])
    accuracy = float(correct.mean())

    print("\n========= Inference Summary (Classification) =========")
    print(f"Rows predicted: {len(preds)}")
    print(f"Rows matched with actuals: {len(merged)}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nLabel distribution (actual vs predicted):")
    print("Actual:\n", merged[actual_col].value_counts())
    print("\nPredicted:\n", merged["prediction"].value_counts())

    print("\nLast predictions vs actuals (by row_id):")
    print(merged.tail(tail_rows).to_string(index=False))

    # Optional: very simple confusion table
    try:
        ct = pd.crosstab(merged[actual_col], merged["prediction"],
                         rownames=["actual"], colnames=["predicted"])
        print("\nConfusion table:\n", ct)
    except Exception as e:
        print(f"(Could not build confusion table: {e})")


if __name__ == "__main__":
    cfg = read_config(ROOT / "config" / "config.yaml")

    parser = argparse.ArgumentParser(description="Batch inference/evaluation runner")
    parser.add_argument("--seed", action="store_true",
                        help="Initialize production DB from training CSV and clear predictions (one-time)")
    args = parser.parse_args()

    dm = DataManager(cfg)

    # --- Seed ONLY if explicitly requested ---
    if args.seed:
        dm.initialize_prod_database()
        print("Seeded production DB from training CSV and cleared predictions.")

    runner = PipelineRunner(cfg, dm)

    # --- Load production DB ---
    prod_df = dm.load_prod_data()
    if prod_df is None or prod_df.empty:
        print("Production DB is empty. Nothing to infer on. "
              "Run with --seed to initialize from training CSV.")
        sys.exit(0)

    # --- Run predictions for ALL rows in production DB ---
    tcfg = cfg["training"]["target"]
    target_col = tcfg["source_col"]

    # We'll use the row index as row_id
    prod_reset = prod_df.reset_index().rename(columns={"index": "row_id"})

    preds_rows = []
    for _, row in prod_reset.iterrows():
        row_id = int(row["row_id"])

        # Build feature dict (drop target if present)
        row_dict = row.drop(labels=["row_id"]).to_dict()
        row_dict.pop(target_col, None)

        result = runner.predict_from_features(row_dict)  # {"prediction": label}
        preds_rows.append({"row_id": row_id, "prediction": result["prediction"]})

    pred_df = pd.DataFrame(preds_rows)

    # Overwrite predictions.csv with fresh predictions
    pred_path = cfg["data_manager"]["predictions_path"]
    os.makedirs(Path(pred_path).parent, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)
    print(f"\nWrote {len(pred_df)} predictions to: {pred_path}")

    # Show summary
    show_results(dm, cfg, tail_rows=10)
