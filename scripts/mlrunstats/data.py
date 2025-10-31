# mlrunstats/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import math
import re

# Try to use pandas if available (recommended). If not, we fall back to a tiny reader.
try:
    import pandas as pd  # type: ignore
    _PANDAS = True
except Exception:
    pd = None  # type: ignore
    _PANDAS = False


# ----------------------------
# Paths & run discovery
# ----------------------------
@dataclass
class RunPaths:
    name: str            # e.g., "run_1"
    dir: Path            # .../weights/checkpoints/run_1
    spearman: Path       # .../weights/checkpoints/run_1/tsplice_spearman_by_tissue.tsv


def find_runs(exp_dir: str, runs_glob: str, spearman_file: str) -> List[RunPaths]:
    """
    Discover run directories under exp_dir matching runs_glob (e.g., 'run_*')
    and return only those that contain the required spearman_file.
    """
    root = Path(exp_dir)
    if not root.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    runs: List[RunPaths] = []
    for d in sorted([p for p in root.glob(runs_glob) if p.is_dir()]):
        sp = d / spearman_file
        if sp.exists():
            runs.append(RunPaths(name=d.name, dir=d, spearman=sp))
    return runs


# ----------------------------
# Reading helpers
# ----------------------------
_SNAKE_RE = re.compile(r"[^0-9a-zA-Z]+")

def _to_snake(s: str) -> str:
    s = s.strip().lower()
    s = _SNAKE_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _normalize_columns_pandas(df: "pd.DataFrame") -> "pd.DataFrame":
    df.columns = [_to_snake(str(c)) for c in df.columns]
    return df


def _read_table_fallback(path: Path):
    """
    Very small TSV/whitespace reader when pandas is unavailable.
    Returns (rows, columns) where rows is a list of dicts keyed by normalized column names.
    """
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if not lines:
        return [], []

    # Detect delimiter: prefer tab if present in header; else split on whitespace
    header = lines[0]
    if "\t" in header:
        split = lambda s: s.split("\t")
    else:
        split = lambda s: re.split(r"\s+", s.strip())

    cols_raw = split(header)
    cols = [_to_snake(c) for c in cols_raw]

    rows = []
    for ln in lines[1:]:
        toks = split(ln)
        # pad/truncate safely
        toks = toks + [""] * (len(cols) - len(toks))
        toks = toks[: len(cols)]
        row = dict(zip(cols, toks))
        rows.append(row)
    return rows, cols


def read_spearman(path: Path):
    """
    Read a per-run 'tsplice_spearman_by_tissue.tsv' (or similarly named) table.
    - With pandas: robustly read TSV or whitespace-delimited, normalize columns.
    - Without pandas: return a lightweight 'list of dicts' + provide a tiny adapter.
    Returns:
      If pandas available: a pandas.DataFrame
      Else: (rows, cols) where rows is list[dict] keyed by normalized column names.
    """
    if _PANDAS:
        # Try standard TSV; fall back to whitespace with the Python engine.
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            df = pd.read_csv(path, sep=r"\s+", engine="python")
        return _normalize_columns_pandas(df)
    else:
        return _read_table_fallback(path)


# ----------------------------
# Aggregation
# ----------------------------
def _to_float_safe(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def _aggregate_fallback(rows: List[dict], metric_col: str,
                        count_col: Optional[str], weighted: bool) -> float:
    """
    Aggregate without pandas: compute mean or weighted mean of 'metric_col'.
    """
    if not rows:
        return math.nan

    if metric_col not in rows[0]:
        raise KeyError(f"Column '{metric_col}' not found in table.")

    xs = [_to_float_safe(r.get(metric_col, "")) for r in rows]
    xs = [v for v in xs if not math.isnan(v)]
    if not xs:
        return math.nan

    if weighted and count_col:
        if count_col not in rows[0]:
            # fall back to unweighted if count column missing
            return sum(xs) / len(xs)
        ws = [_to_float_safe(r.get(count_col, "")) for r in rows]
        # Align lengths and mask simultaneously
        pairs = [(x, w if (w is not None) else 0.0) for x, w in zip(xs, ws)]
        xs2, ws2 = [], []
        for (x, w) in pairs:
            if math.isnan(x) or math.isnan(w) or w <= 0:
                continue
            xs2.append(x); ws2.append(w)
        if not xs2 or sum(ws2) <= 0:
            return sum(xs) / len(xs)
        return sum(x * w for x, w in zip(xs2, ws2)) / sum(ws2)
    else:
        return sum(xs) / len(xs)


def aggregate_run(df_or_rows,
                  metric_col: str,
                  count_col: Optional[str],
                  weighted: bool) -> float:
    """
    Convert a per-run table to a single scalar (mean or weighted mean of metric_col).
    Works with either a pandas.DataFrame or the fallback (rows, cols).
    """
    metric_col = _to_snake(metric_col)
    count_col = _to_snake(count_col) if count_col else None

    if _PANDAS:
        df = df_or_rows
        if metric_col not in df.columns:
            raise KeyError(f"Column '{metric_col}' not found. Available: {list(df.columns)}")

        # mask NaNs
        vals = df[metric_col].astype(float)
        mask = ~vals.isna()
        vals = vals[mask]

        if len(vals) == 0:
            return float("nan")

        if weighted and count_col and (count_col in df.columns):
            w = df.loc[mask, count_col].astype(float).clip(lower=0.0)
            sw = float(w.sum())
            if sw <= 0:
                return float(vals.mean())
            return float((vals * w).sum() / sw)
        else:
            return float(vals.mean())

    else:
        # Fallback: df_or_rows is (rows, cols) or just rows
        if isinstance(df_or_rows, tuple):
            rows, _ = df_or_rows
        else:
            rows = df_or_rows
        return _aggregate_fallback(rows, metric_col, count_col, weighted)


# def collect_values(exp_dir: str,
#                    runs_glob: str,
#                    spearman_file: str,
#                    metric_col: str,
#                    count_col: Optional[str],
#                    weighted: bool) -> List[float]:
#     """
#     High-level helper:
#       1) find runs,
#       2) read per-run spearman table,
#       3) aggregate to one number per run,
#       4) return list of floats (order = sorted run folders).
#     """
#     values: List[float] = []
#     runs = find_runs(exp_dir=exp_dir, runs_glob=runs_glob, spearman_file=spearman_file)

#     for rp in runs:
#         table = read_spearman(rp.spearman)
#         val = aggregate_run(table, metric_col=metric_col, count_col=count_col, weighted=weighted)
#         if val == val:  # not NaN
#             values.append(float(val))

#     return values

# mlrunstats/data.py  (add/replace collect_values with this version)

def collect_values(exp_dir: str,
                   runs_glob: str,
                   spearman_file: str,
                   metric_col: str,
                   count_col: Optional[str],
                   weighted: bool,
                   tissue: Optional[str] = None) -> List[float]:
    """
    If `tissue` is provided, return the metric value for that tissue for each run.
    Otherwise, return the (weighted) mean across tissues (previous behavior).

    Notes:
      - Matching of tissue is case/space-insensitive via snake_case normalization.
      - If a run doesn't have that tissue, it's skipped (no value appended).
    """
    values: List[float] = []
    runs = find_runs(exp_dir=exp_dir, runs_glob=runs_glob, spearman_file=spearman_file)

    tnorm = _to_snake(tissue) if tissue else None
    mcol  = _to_snake(metric_col)

    for rp in runs:
        table = read_spearman(rp.spearman)

        if tissue is None:
            # old behavior: aggregate across tissues (may use weighting)
            val = aggregate_run(table, metric_col=mcol, count_col=count_col, weighted=weighted)
        else:
            # per-tissue value (no weighting)
            if _PANDAS:
                df = table
                if "tissue" not in df.columns:
                    raise KeyError("Column 'tissue' not found in spearman table.")
                if mcol not in df.columns:
                    raise KeyError(f"Column '{mcol}' not found. Available: {list(df.columns)}")

                tcol = df["tissue"].astype(str).map(_to_snake)
                sub = df.loc[tcol == tnorm, mcol]
                if sub.empty:
                    # skip this run if tissue is missing
                    continue
                val = float(sub.astype(float).iloc[0])
            else:
                # fallback reader
                rows, _ = table if isinstance(table, tuple) else (table, None)
                row = next((r for r in rows if _to_snake(str(r.get("tissue", ""))) == tnorm), None)
                if not row or mcol not in row:
                    continue
                val = _to_float_safe(row.get(mcol, ""))

        if val == val:   # not NaN
            values.append(float(val))

    return values

