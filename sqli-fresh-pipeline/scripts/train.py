from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqli_pipeline.config import ARTIFACT_DIR, DATA_PATH, REPORT_DIR  # noqa: E402
from sqli_pipeline.trainer import train_all_models  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fresh SQL injection detection pipeline.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="CSV with Query and Label columns.")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR), help="Where model artifacts are saved.")
    parser.add_argument("--report-dir", default=str(REPORT_DIR), help="Where EDA and metric reports are saved.")
    args = parser.parse_args()

    metadata = train_all_models(args.data_path, args.artifact_dir, args.report_dir)
    print(json.dumps({"best_model": metadata["best_model"], "threshold": metadata["threshold"]}, indent=2))
    print(f"Saved artifacts to: {args.artifact_dir}")
    print(f"Saved reports to: {args.report_dir}")


if __name__ == "__main__":
    main()

