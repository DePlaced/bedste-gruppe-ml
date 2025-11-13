import os
import sys

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = "models/extra_trees.pkl"
TEST_CSV = "data/debug/test_set.csv"
OUT_CSV = "reports/perm_importance.csv"
OUT_PNG = "reports/perm_importance.png"

# --------- Resolve project paths ---------
ROOT   = Path(__file__).resolve().parents[1]  # project root
ML_SRC = ROOT / "ml" / "src"

for p in (ROOT, ML_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

os.chdir(ROOT)

def main():
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test set not found at {TEST_CSV}. Did you save it in TrainingPipeline?")

    df = pd.read_csv(TEST_CSV)
    y_test = df["__target__"]
    X_test = df.drop(columns=["__target__"])

    model = joblib.load(MODEL_PATH)

    print("Computing permutation importance...")
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importances = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    importances.to_csv(OUT_CSV, index=False)
    print(f"Saved permutation importance to {OUT_CSV}")

    # Bar plot (top 20)
    top = importances.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"][::-1], top["importance_mean"][::-1])
    plt.xlabel("Mean decrease in score (permutation importance)")
    plt.title("Top 20 Features – Permutation Importance")
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG)
    print(f"Saved permutation importance plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
