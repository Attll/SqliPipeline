from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqli_pipeline.features import SQLiFeatureEngineer


def test_feature_engineer_shape():
    transformer = SQLiFeatureEngineer()
    output = transformer.transform(["' OR 1=1 --", "select name from users"])
    assert output.shape[0] == 2
    assert output.shape[1] == len(transformer.get_feature_names_out())


def test_feature_engineer_detects_common_patterns():
    transformer = SQLiFeatureEngineer()
    names = list(transformer.get_feature_names_out())
    row = transformer.transform(["' OR 1=1 UNION SELECT @@version --"])[0]
    assert row[names.index("has_tautology")] == 1
    assert row[names.index("has_union_select")] == 1
    assert row[names.index("has_version_probe")] == 1

