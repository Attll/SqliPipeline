from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SQLiFeatureEngineer(BaseEstimator, TransformerMixin):
    """Small, explainable feature block for SQL injection patterns."""

    sql_keywords = (
        "select",
        "union",
        "insert",
        "update",
        "delete",
        "drop",
        "create",
        "alter",
        "where",
        "from",
        "join",
        "having",
        "information_schema",
        "sysobjects",
        "syscolumns",
        "all_tables",
    )

    feature_names = (
        "query_length",
        "word_count",
        "avg_token_length",
        "quote_count",
        "semicolon_count",
        "inline_comment_count",
        "block_comment_count",
        "equals_count",
        "operator_count",
        "parenthesis_count",
        "wildcard_count",
        "comma_count",
        "dot_count",
        "at_count",
        "pipe_count",
        "slash_count",
        "backslash_count",
        "percent_count",
        "digit_ratio",
        "uppercase_ratio",
        "special_char_ratio",
        "or_count",
        "and_count",
        "has_tautology",
        "has_union_select",
        "has_stacked_query",
        "has_time_delay",
        "has_error_probe",
        "has_file_access",
        "has_encoded_char",
        "has_hex_literal",
        "has_char_function",
        "has_concat",
        "has_system_proc",
        "has_version_probe",
    ) + tuple(f"{keyword}_count" for keyword in sql_keywords)

    tautology_pattern = re.compile(
        r"""(?ix)
        (
            \b(?:or|and)\b\s+
            (?:
                ['"]?\s*\d+\s*['"]?\s*=\s*['"]?\s*\d+
                |
                ['"]?[a-z_][\w]*['"]?\s*=\s*['"]?[a-z_][\w]*['"]?
            )
        )
        """
    )
    union_select_pattern = re.compile(r"(?is)\bunion\b.{0,80}\bselect\b")
    time_delay_pattern = re.compile(r"(?i)\b(sleep|pg_sleep|benchmark|waitfor\s+delay|dbms_lock\.sleep)\b")
    error_probe_pattern = re.compile(r"(?i)\b(utl_inaddr|extractvalue|updatexml|dbms_pipe|convert\s*\()\b")
    file_access_pattern = re.compile(r"(?i)\b(load_file|into\s+outfile|into\s+dumpfile|/etc/passwd)\b")
    encoded_char_pattern = re.compile(r"(?i)(%[0-9a-f]{2}|\\x[0-9a-f]{2}|char\s*\()")
    hex_literal_pattern = re.compile(r"(?i)\b0x[0-9a-f]+\b")
    concat_pattern = re.compile(r"(?i)(\bconcat\s*\(|\|\||\+)")
    system_proc_pattern = re.compile(r"(?i)\b(sp_|xp_|exec\s+|execute\s+)\b")
    version_probe_pattern = re.compile(r"(?i)(@@version|\bversion\s*\(|sqlite_version\s*\()")
    operator_pattern = re.compile(r"(=|<>|!=|<=|>=|<|>)")

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        rows = [self._extract_one(str(query)) for query in X]
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names, dtype=object)

    def _extract_one(self, query: str) -> List[float]:
        lowered = query.lower()
        length = max(len(query), 1)
        tokens = query.split()
        word_count = len(tokens)
        special_char_count = sum(not char.isalnum() and not char.isspace() for char in query)

        features: List[float] = [
            len(query),
            word_count,
            (sum(len(token) for token in tokens) / word_count) if word_count else 0.0,
            query.count("'") + query.count('"'),
            query.count(";"),
            query.count("--") + query.count("#"),
            query.count("/*") + query.count("*/"),
            query.count("="),
            len(self.operator_pattern.findall(query)),
            query.count("(") + query.count(")"),
            query.count("*"),
            query.count(","),
            query.count("."),
            query.count("@"),
            query.count("|"),
            query.count("/"),
            query.count("\\"),
            query.count("%"),
            sum(char.isdigit() for char in query) / length,
            sum(char.isupper() for char in query) / length,
            special_char_count / length,
            len(re.findall(r"\bor\b", lowered)),
            len(re.findall(r"\band\b", lowered)),
            float(bool(self.tautology_pattern.search(query))),
            float(bool(self.union_select_pattern.search(query))),
            float(";" in query and bool(re.search(r"(?i)\b(select|drop|insert|delete|update|create|alter)\b", query))),
            float(bool(self.time_delay_pattern.search(query))),
            float(bool(self.error_probe_pattern.search(query))),
            float(bool(self.file_access_pattern.search(query))),
            float(bool(self.encoded_char_pattern.search(query))),
            float(bool(self.hex_literal_pattern.search(query))),
            float(bool(re.search(r"(?i)\bchar\s*\(", query))),
            float(bool(self.concat_pattern.search(query))),
            float(bool(self.system_proc_pattern.search(query))),
            float(bool(self.version_probe_pattern.search(query))),
        ]

        features.extend(lowered.count(keyword) for keyword in self.sql_keywords)
        return features

