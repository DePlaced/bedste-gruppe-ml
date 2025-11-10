# ===============================================================
# INFERENCE ENTRYPOINT (CSV streaming) — advances by DB state
# ===============================================================
import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import yaml
import numpy as np

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


def show_results(dm: DataManager, actual_col: str = "taxi_pickups", tail_rows: int = 10, plot: bool = False):
    """Load predictions & production DB, merge on datetime, and show a concise summary."""
    preds  = dm.load_prediction_data()
    actual = dm.load_prod_data()

    if preds is None or preds.empty:
        print("No predictions found. Did the timestamps in config match your streaming CSV?")
        return

    if "datetime" not in preds.columns or "datetime" not in actual.columns:
        print("Missing 'datetime' column in predictions or production database.")
        return

    if actual_col not in actual.columns:
        print(f"Column '{actual_col}' not found in production DB. Available columns: {list(actual.columns)}")
        return

    merged = pd.merge(
        preds[["datetime", "prediction"]],
        actual[["datetime", actual_col]],
        on="datetime",
        how="inner",
        validate="one_to_one"
    ).sort_values("datetime")

    if merged.empty:
        print("No matching timestamps between predictions and actuals.")
        return

    err = merged["prediction"] - merged[actual_col]
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))

    print("\n========= Inference Summary =========")
    print(f"Rows predicted: {len(preds)}")
    print(f"Rows matched with actuals: {len(merged)}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    print("\nLast predictions vs actuals:")
    print(merged.tail(tail_rows).to_string(index=False))

    if plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            plt.figure(figsize=(12, 5))
            plt.plot(merged["datetime"], merged[actual_col], label=f"Actual ({actual_col})", marker="o", linewidth=2)
            plt.plot(merged["datetime"], merged["prediction"], label="Predicted", marker="s", linewidth=2)
            plt.title(f"Predicted vs Actual (RMSE: {rmse:.2f})")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)

            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    cfg = read_config(ROOT / "config" / "config.yaml")

    parser = argparse.ArgumentParser(description="Inference runner")
    parser.add_argument("--seed", action="store_true", help="Initialize rolling DB and clear predictions (one-time)")
    parser.add_argument("--steps", type=int, default=None, help="Override num_inference_steps from config")
    args = parser.parse_args()

    dm = DataManager(cfg)

    # --- Seed ONLY if explicitly requested ---
    if args.seed:
        dm.initialize_prod_database()
        print("Seeded rolling DB from training CSV and cleared predictions.")
        # If you called with --seed only, you might want to exit here; keep going if you also set steps>0.

    runner = PipelineRunner(cfg, dm)

    # Steps: CLI override wins; else config
    steps = int(cfg["pipeline_runner"]["num_inference_steps"]) if args.steps is None else int(args.steps)
    inc   = pd.Timedelta(cfg["pipeline_runner"]["time_increment"])

    # --- Determine starting timestamp dynamically from DB state ---
    prod_df = dm.load_prod_data()
    if prod_df is not None and not prod_df.empty and "datetime" in prod_df.columns:
        ts = pd.to_datetime(prod_df["datetime"], errors="coerce").max() + inc
    else:
        # Fallback to configured first_timestamp if DB is empty
        ts = pd.to_datetime(cfg["pipeline_runner"]["first_timestamp"])

    for i in range(steps):
        print(f"[{i+1}/{steps}] Inference @ {ts}")
        runner.run_inference(ts)
        ts += inc  # move to the next hour for the next iteration

    print("\nInference complete. Predictions saved to:", cfg["data_manager"]["predictions_path"])

    # Show summary (will print "No matching..." until the second run when actuals exist)
    show_results(dm, actual_col="taxi_pickups", tail_rows=10, plot=False)
