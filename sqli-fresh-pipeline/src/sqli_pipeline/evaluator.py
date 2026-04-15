from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import (
    ARTIFACT_DIR,
    BEST_MODEL_NAME,
    LABEL_COLUMN,
    MODEL_METADATA_NAME,
    QUERY_COLUMN,
    REPORT_DIR,
)
from .data import load_dataset
from .trainer import compute_metrics


def load_model_bundle(artifact_dir: str | Path = ARTIFACT_DIR):
    artifact_dir = Path(artifact_dir)
    model = joblib.load(artifact_dir / BEST_MODEL_NAME)
    metadata = json.loads((artifact_dir / MODEL_METADATA_NAME).read_text(encoding="utf-8"))
    return model, metadata


def evaluate_saved_model(
    data_path: str | Path | None = None,
    artifact_dir: str | Path = ARTIFACT_DIR,
    report_dir: str | Path = REPORT_DIR,
) -> Dict:
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if data_path is None:
        data_path = artifact_dir / "holdout_test.csv"

    df = load_dataset(data_path)
    model, metadata = load_model_bundle(artifact_dir)
    threshold = float(metadata["threshold"])

    y_true = df[LABEL_COLUMN].to_numpy()
    y_score = model.predict_proba(df[QUERY_COLUMN].tolist())[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_score)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report = {
        "model": metadata["best_model"],
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "evaluated_rows": int(len(df)),
        "data_path": str(Path(data_path)),
    }

    predictions = pd.DataFrame(
        {
            QUERY_COLUMN: df[QUERY_COLUMN],
            LABEL_COLUMN: y_true,
            "malicious_probability": np.round(y_score, 6),
            "prediction": y_pred,
        }
    )
    predictions.to_csv(report_dir / "holdout_predictions.csv", index=False)
    (report_dir / "best_model_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    cm_frame = pd.DataFrame(cm, index=["actual_benign", "actual_attack"], columns=["pred_benign", "pred_attack"])
    cm_frame.to_csv(report_dir / "confusion_matrix.csv")
    return report

