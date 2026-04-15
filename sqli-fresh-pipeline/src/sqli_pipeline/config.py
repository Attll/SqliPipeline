from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sql_injection_dataset.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

QUERY_COLUMN = "Query"
LABEL_COLUMN = "Label"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
RANDOM_STATE = 42

MODEL_NAMES = ("random_forest", "logistic_regression", "naive_bayes")
BEST_MODEL_NAME = "best_model.joblib"
MODEL_METADATA_NAME = "model_metadata.json"

