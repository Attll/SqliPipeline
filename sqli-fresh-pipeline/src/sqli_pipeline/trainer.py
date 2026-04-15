from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .config import (
    ARTIFACT_DIR,
    BEST_MODEL_NAME,
    LABEL_COLUMN,
    MODEL_METADATA_NAME,
    QUERY_COLUMN,
    RANDOM_STATE,
    REPORT_DIR,
)
from .data import build_eda_summary, load_dataset
from .preprocessor import build_preprocessor, get_preprocessor_feature_names


def build_model(model_name: str) -> Pipeline:
    if model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_name == "logistic_regression":
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    elif model_name == "naive_bayes":
        classifier = MultinomialNB(alpha=0.5)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(model_name)),
            ("classifier", classifier),
        ]
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    return metrics


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Choose a probability threshold that maximises F1 score."""
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {}
    best_f1 = -1.0

    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (y_score >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_score)
        if metrics["f1_score"] > best_f1:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_f1 = metrics["f1_score"]

    return best_threshold, best_metrics


def feature_importance_frame(model: Pipeline, limit: int = 40) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = get_preprocessor_feature_names(preprocessor)

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    frame = pd.DataFrame({"feature": feature_names, "importance": values})
    return frame.sort_values("importance", ascending=False).head(limit).reset_index(drop=True)


def train_all_models(
    data_path: str | Path,
    artifact_dir: str | Path = ARTIFACT_DIR,
    report_dir: str | Path = REPORT_DIR,
    test_size: float = 0.2,
) -> Dict:
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    model_dir = artifact_dir / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    eda_summary, eda_df = build_eda_summary(df)
    (report_dir / "eda_summary.json").write_text(json.dumps(eda_summary, indent=2), encoding="utf-8")
    eda_df[[QUERY_COLUMN, LABEL_COLUMN, "query_length", "word_count", "special_char_count"]].to_csv(
        report_dir / "eda_rows.csv", index=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df[QUERY_COLUMN],
        df[LABEL_COLUMN],
        test_size=test_size,
        stratify=df[LABEL_COLUMN],
        random_state=RANDOM_STATE,
    )
    holdout = pd.DataFrame({QUERY_COLUMN: X_test, LABEL_COLUMN: y_test})
    holdout.to_csv(artifact_dir / "holdout_test.csv", index=False)

    results = {}
    trained_models = {}

    for model_name in ("random_forest", "logistic_regression", "naive_bayes"):
        model = build_model(model_name)
        model.fit(X_train.tolist(), y_train.to_numpy())

        y_score = model.predict_proba(X_test.tolist())[:, 1]
        threshold, metrics = find_best_threshold(y_test.to_numpy(), y_score)
        y_pred = (y_score >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        trained_models[model_name] = model

        results[model_name] = {
            "model_name": model_name,
            "model_path": str(model_path),
            "threshold": threshold,
            "metrics": metrics,
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
        }

    best_model_name = max(
        results,
        key=lambda name: (
            results[name]["metrics"]["f1_score"],
            results[name]["metrics"]["recall"],
            results[name]["metrics"]["precision"],
        ),
    )
    best_model = trained_models[best_model_name]
    best_result = results[best_model_name]
    best_result["is_best"] = True

    joblib.dump(best_model, artifact_dir / BEST_MODEL_NAME)
    metadata = {
        "best_model": best_model_name,
        "threshold": best_result["threshold"],
        "selection_metric": "f1_score",
        "positive_label": 1,
        "data_path": str(Path(data_path)),
        "results": results,
    }
    (artifact_dir / MODEL_METADATA_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (report_dir / "model_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics_rows = []
    for model_name, result in results.items():
        row = {"model": model_name, "threshold": result["threshold"], **result["metrics"]}
        row.update(result["confusion_matrix"])
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).sort_values("f1_score", ascending=False).to_csv(
        report_dir / "model_metrics.csv", index=False
    )

    feature_importance_frame(best_model).to_csv(report_dir / "feature_importance.csv", index=False)
    return metadata

