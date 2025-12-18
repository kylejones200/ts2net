from __future__ import annotations

"""
rcompat.py - R compatibility utilities.

This module provides functions for working with R data files and converting them to
pandas DataFrames or panel data structures. It includes support for reading RData
files and converting them to more Python-friendly formats.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

try:
    import pyreadr  # type: ignore
except Exception:
    pyreadr = None


def read_r(path: str) -> Dict[str, pd.DataFrame]:
    if pyreadr is None:
        raise RuntimeError("Install pyreadr: pip install pyreadr")
    res = pyreadr.read_r(path)
    out: Dict[str, pd.DataFrame] = {}
    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            out[k] = v.copy()
        else:
            out[k] = pd.DataFrame(v)
    return out


def as_panel(
    df: pd.DataFrame, name_col: Optional[str] = None
) -> Tuple[list[str], np.ndarray]:
    if name_col and name_col in df.columns:
        names = df[name_col].astype(str).tolist()
        X = df.drop(columns=[name_col]).to_numpy(float)
        return names, X
    if df.index.name and df.index.is_monotonic_increasing:
        names = df.columns.astype(str).tolist()
        X = df.to_numpy(float).T
        return names, X
    if "name" in df.columns:
        names = df["name"].astype(str).tolist()
        X = df.drop(columns=["name"]).to_numpy(float)
        return names, X
    names = df.columns.astype(str).tolist()
    X = df.to_numpy(float).T
    return names, X


def r_to_panel_csv(
    r_path: str, obj: str, out_csv: str, name_col: Optional[str] = None
) -> str:
    tables = read_r(r_path)
    if obj not in tables:
        raise KeyError(f"Object '{obj}' not found. Available: {list(tables)}")
    df = tables[obj]
    names, X = as_panel(df, name_col=name_col)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(X, index=names)
    frame.index.name = "name"
    frame.reset_index().to_csv(out, index=False)
    return str(out)


def list_r_objects(r_path: str) -> list[str]:
    if pyreadr is None:
        raise RuntimeError("Install pyreadr: pip install pyreadr")
    res = pyreadr.read_r(r_path)
    return list(res.keys())
