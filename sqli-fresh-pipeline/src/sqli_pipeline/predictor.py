from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, QUERY_COLUMN
from .evaluator import load_model_bundle


class SQLiPredictor:
    def __init__(self, model, threshold: float, model_name: str):
        self.model = model
        self.threshold = threshold
        self.model_name = model_name

    @classmethod
    def load(cls, artifact_dir: str | Path = ARTIFACT_DIR) -> "SQLiPredictor":
        model, metadata = load_model_bundle(artifact_dir)
        return cls(model=model, threshold=float(metadata["threshold"]), model_name=metadata["best_model"])

    def predict_many(self, queries: Iterable[str]) -> pd.DataFrame:
        query_list = [str(query) for query in queries]
        probabilities = self.model.predict_proba(query_list)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        risk_labels = np.where(predictions == 1, "Attack", "Benign")

        return pd.DataFrame(
            {
                QUERY_COLUMN: query_list,
                "malicious_probability": np.round(probabilities, 6),
                "prediction": predictions,
                "risk_label": risk_labels,
                "threshold": self.threshold,
                "model": self.model_name,
            }
        )

    def predict_one(self, query: str) -> dict:
        return self.predict_many([query]).iloc[0].to_dict()

