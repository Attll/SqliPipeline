from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import LABEL_COLUMN, QUERY_COLUMN


def _resolve_column(df: pd.DataFrame, expected: str) -> str:
    matches = [col for col in df.columns if col.lower() == expected.lower()]
    if not matches:
        raise ValueError(f"Dataset must contain a '{expected}' column. Found: {list(df.columns)}")
    return matches[0]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the SQL injection dataset."""
    df = pd.read_csv(path)
    query_col = _resolve_column(df, QUERY_COLUMN)
    label_col = _resolve_column(df, LABEL_COLUMN)

    df = df.rename(columns={query_col: QUERY_COLUMN, label_col: LABEL_COLUMN})
    df = df[[QUERY_COLUMN, LABEL_COLUMN]].copy()
    df[QUERY_COLUMN] = df[QUERY_COLUMN].fillna("").astype(str)
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    df = df.dropna(subset=[LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    valid_labels = {0, 1}
    seen_labels = set(df[LABEL_COLUMN].unique())
    if not seen_labels.issubset(valid_labels):
        raise ValueError(f"Labels must be binary 0/1. Found: {sorted(seen_labels)}")
    if len(seen_labels) < 2:
        raise ValueError("Dataset must contain both benign label 0 and attack label 1.")

    return df.reset_index(drop=True)


def build_eda_summary(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """Return a compact EDA summary and row-level fields useful for charts."""
    eda_df = df.copy()
    eda_df["query_length"] = eda_df[QUERY_COLUMN].str.len()
    eda_df["word_count"] = eda_df[QUERY_COLUMN].str.split().str.len().fillna(0)
    eda_df["special_char_count"] = eda_df[QUERY_COLUMN].apply(
        lambda value: sum(not char.isalnum() and not char.isspace() for char in value)
    )

    label_counts = eda_df[LABEL_COLUMN].value_counts().sort_index()
    length_summary = eda_df.groupby(LABEL_COLUMN)["query_length"].describe().round(2)

    summary = {
        "rows": int(len(eda_df)),
        "duplicates": int(eda_df.duplicated(subset=[QUERY_COLUMN, LABEL_COLUMN]).sum()),
        "empty_queries": int((eda_df[QUERY_COLUMN].str.len() == 0).sum()),
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "label_percent": {
            str(key): round(float(value / len(eda_df)), 4) for key, value in label_counts.items()
        },
        "query_length": {
            "min": int(eda_df["query_length"].min()),
            "max": int(eda_df["query_length"].max()),
            "mean": round(float(eda_df["query_length"].mean()), 2),
            "median": round(float(eda_df["query_length"].median()), 2),
        },
        "word_count": {
            "min": int(eda_df["word_count"].min()),
            "max": int(eda_df["word_count"].max()),
            "mean": round(float(eda_df["word_count"].mean()), 2),
        },
        "length_by_label": length_summary.to_dict(),
    }
    return summary, eda_df

