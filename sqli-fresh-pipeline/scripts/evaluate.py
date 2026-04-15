"""Entry point script for evaluating the saved best SQLi detection model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqli_pipeline.config import ARTIFACT_DIR, REPORT_DIR  # noqa: E402
from sqli_pipeline.evaluator import evaluate_saved_model  # noqa: E402


def main() -> None:
    report = evaluate_saved_model(
        data_path=None,       # defaults to artifacts/holdout_test.csv
        artifact_dir=ARTIFACT_DIR,
        report_dir=REPORT_DIR,
    )
    metrics = report["metrics"]
    cm = report["confusion_matrix"]

    print(f"Model       : {report['model']}")
    print(f"Threshold   : {report['threshold']:.3f}")
    print(f"Rows eval.  : {report['evaluated_rows']}")
    print()
    print(f"Recall      : {metrics['recall']:.4f}")
    print(f"Precision   : {metrics['precision']:.4f}")
    print(f"F1          : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC     : {metrics['roc_auc']:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  TN={cm['tn']}  FP={cm['fp']}")
    print(f"  FN={cm['fn']}  TP={cm['tp']}")
    print()
    print("Reports written to:", REPORT_DIR)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
