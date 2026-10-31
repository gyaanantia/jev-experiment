"""Column profiles: the text Jev (and Haiku) see for each column.

Modes (spec section 5.2):
  schema   - name, inferred type, names of the other columns in the table
  instance - sampled distinct values only (names hidden)
  hybrid   - everything above
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

MODES = ("schema", "instance", "hybrid")

_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_BOOL = {"true", "false", "yes", "no", "t", "f", "y", "n"}


def read_table(path) -> pd.DataFrame:
    """Read a CSV as strings, so values keep their original text."""
    return pd.read_csv(path, dtype=str, keep_default_na=True, low_memory=False)


def _non_empty(series: pd.Series) -> pd.Series:
    s = series.dropna().astype(str).str.strip()
    return s[s != ""]


def infer_type(series: pd.Series, probe: int = 500) -> str:
    vals = _non_empty(series)
    if vals.empty:
        return "empty"
    vals = vals.head(probe)
    if vals.str.lower().isin(_BOOL).all():
        return "boolean"
    if vals.str.match(_INT).all():
        return "integer"
    if vals.str.match(_FLOAT).all():
        return "float"
    try:
        parsed = pd.to_datetime(vals, errors="coerce", format="mixed")
    except (ValueError, TypeError, OverflowError):  # e.g. mixed time zones in noisy dates
        try:
            parsed = pd.to_datetime(vals, errors="coerce", format="mixed", utc=True)
        except (ValueError, TypeError, OverflowError):
            return "string"
    if parsed.notna().mean() >= 0.9 and not vals.str.match(_FLOAT).any():
        return "date"
    return "string"


def _seed(seed: int, col: str) -> int:
    return seed ^ int(hashlib.sha256(col.encode()).hexdigest()[:8], 16)


def sample_values(series: pd.Series, n: int, max_len: int, seed: int, col: str) -> list[str]:
    uniq = _non_empty(series).drop_duplicates()
    if len(uniq) > n:
        idx = np.random.default_rng(_seed(seed, col)).choice(len(uniq), size=n, replace=False)
        uniq = uniq.iloc[np.sort(idx)]
    return [v if len(v) <= max_len else v[: max_len - 1] + "…" for v in uniq.tolist()]


def column_profile(
    df: pd.DataFrame, col: str, mode: str, *, n_values: int, max_len: int, seed: int = 42
) -> dict:
    prof: dict = {}
    if mode in ("schema", "hybrid"):
        prof["name"] = col
        prof["inferred_type"] = infer_type(df[col])
    if mode in ("instance", "hybrid"):
        prof["sample_values"] = sample_values(df[col], n_values, max_len, seed, col)
    return prof


def table_profiles(df: pd.DataFrame, mode: str, **kw) -> dict[str, dict]:
    return {c: column_profile(df, c, mode, **kw) for c in df.columns}


def table_context(df: pd.DataFrame, mode: str) -> list[str] | None:
    """Names of all columns in a table; hidden in instance mode."""
    return list(df.columns) if mode in ("schema", "hybrid") else None
