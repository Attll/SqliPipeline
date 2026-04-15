"""Centralized configuration for the SQLi detection pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths
    base_dir: Path = Path(__file__).resolve().parents[3]
    raw_data_path: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    # Data
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "label"
    text_column: str = "query"

    # Model
    model_name: str = "random_forest"
    n_estimators: int = 100
    max_depth: int = None

    def __post_init__(self):
        self.raw_data_path = self.base_dir / "data" / "raw" / "sql_injection_dataset.csv"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.reports_dir = self.base_dir / "reports"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
