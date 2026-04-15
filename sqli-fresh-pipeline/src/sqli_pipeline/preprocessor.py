from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .features import SQLiFeatureEngineer


def build_preprocessor(model_name: str) -> FeatureUnion:
    """Build the shared text + structural feature preprocessor."""
    scaler = MinMaxScaler() if model_name == "naive_bayes" else StandardScaler()

    numeric_pipeline = Pipeline(
        steps=[
            ("features", SQLiFeatureEngineer()),
            ("scaler", scaler),
        ]
    )

    return FeatureUnion(
        transformer_list=[
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    max_features=2500,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            ("numeric", numeric_pipeline),
        ]
    )


def get_preprocessor_feature_names(preprocessor: FeatureUnion) -> list[str]:
    tfidf = preprocessor.transformer_list[0][1]
    numeric_pipeline = preprocessor.transformer_list[1][1]
    numeric_features = numeric_pipeline.named_steps["features"].get_feature_names_out()

    names = [f"char:{name}" for name in tfidf.get_feature_names_out()]
    names.extend(f"num:{name}" for name in numeric_features)
    return names

