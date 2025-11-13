import os
import json
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, brier_score_loss
from sklearn.dummy import DummyClassifier
from typing import Dict, Any

from sklearn.preprocessing import LabelBinarizer


class TrainingPipeline:
    """
    TRAINING:
      1) Use all engineered features except target
      2) Random train/test split (classification)
      3) Train ExtraTreesClassifier
      4) Evaluate against DummyClassifier baseline
      5) Save metrics
      6) Return trained ExtraTrees model
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def run(self, df: pd.DataFrame):
        # --------- TARGET SETUP ----------
        tcfg = self.cfg["training"]["target"]
        target_col = tcfg["source_col"]

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found after preprocessing/feature engineering.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if X.shape[1] == 0:
            raise ValueError("No features selected. Check preprocessing/feature engineering.")

        # --------- TRAIN / TEST SPLIT ----------
        frac = float(self.cfg["training"]["train_fraction"])
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=(1.0 - frac),
            shuffle=True,
            stratify=y
        )

        # --------- SAVE TEST SPLIT FOR ANALYSIS TOOLS ----------
        os.makedirs("data/debug", exist_ok=True)
        df_test_dbg = X_test.copy()
        df_test_dbg["__target__"] = y_test.values
        df_test_dbg.to_csv("data/debug/test_set.csv", index=False)

        # --------- MAIN MODEL: ExtraTreesClassifier ----------
        tr_cfg = self.cfg["training"]

        if "extra_trees_params" in tr_cfg:
            et_params = tr_cfg["extra_trees_params"]
        else:
            raise KeyError(
                f"'extra_trees_params' not found under training in config. "
                f"Got keys: {list(tr_cfg.keys())}"
            )

        model = ExtraTreesClassifier(**et_params)
        calib_cfg = tr_cfg.get("calibration")

        method = calib_cfg.get("method")
        cv = calib_cfg.get("cv")

        model = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv=cv,
        )

        model.fit(X_train, y_train)

        # --------- PREDICTIONS ----------
        y_pred = model.predict(X_test)

        # --------- BASELINE: DummyClassifier ----------
        dcfg = tr_cfg.get(
            "dummy",
            {"enabled": True, "strategy": "most_frequent", "random_state": 42}
        )
        dummy_metrics = None
        if dcfg.get("enabled", True):
            dummy = DummyClassifier(
                strategy=dcfg.get("strategy", "most_frequent"),
                random_state=dcfg.get("random_state", 42)
            )
            # Dummy ignores X but we pass it for API consistency
            dummy.fit(X_train, y_train)
            y_pred_dummy = dummy.predict(X_test)

            acc_d = float(accuracy_score(y_test, y_pred_dummy))
            precision_d, recall_d, f1_d, _ = precision_recall_fscore_support(
                y_test,
                y_pred_dummy,
                average="weighted",
                zero_division=0
            )
            dummy_metrics = {
                "accuracy": round(acc_d, 4),
                "precision_weighted": round(float(precision_d), 4),
                "recall_weighted": round(float(recall_d), 4),
                "f1_weighted": round(float(f1_d), 4),
                "strategy": dcfg.get("strategy"),
            }

        # --------- MODEL METRICS ----------
        acc = float(accuracy_score(y_test, y_pred))

        # Use weighted average so it works even if classes are imbalanced
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        # Calculate brier score, to check how well calibrated the model is
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test).ravel()
        proba_pos = model.predict_proba(X_test)[:, list(model.classes_).index("POISONOUS")]
        brier = brier_score_loss(y_test_bin, proba_pos)

        model_metrics = {
            "accuracy": round(acc, 4),
            "precision_weighted": round(float(precision), 4),
            "recall_weighted": round(float(recall), 4),
            "f1_weighted": round(float(f1), 4),
            "brier_score": round(brier, 6),
        }

        metrics = {
            "model": model_metrics,
            "baseline_dummy": dummy_metrics,
        }

        # --------- SAVE METRICS ----------
        metrics_path = self.cfg["reports"]["metrics_path"]
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return model