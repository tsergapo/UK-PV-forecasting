"""One split generator for every model: expanding-window validation folds followed by a
held-out test period. Half-open intervals; all boundaries are midnight-aligned."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import SplitConfig


@dataclass(frozen=True)
class Fold:
    kind: str                # "val" or "test"
    name: str
    train_end: pd.Timestamp  # model is fit on rows with time < train_end
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp   # exclusive


def make_folds(index: pd.DatetimeIndex, cfg: SplitConfig) -> list[Fold]:
    end = index.max() + pd.Timedelta(hours=1)  # exclusive end of the data
    boundary = (end - pd.DateOffset(years=cfg.test_years)).ceil("D")
    y = cfg.fold_years
    folds: list[Fold] = []

    for k in range(cfg.n_val_folds):
        vs = boundary - pd.DateOffset(years=(cfg.n_val_folds - k) * y)
        ve = vs + pd.DateOffset(years=y)
        if vs <= index.min():
            raise ValueError("Not enough data before the test period for the requested folds")
        folds.append(Fold("val", f"val_{k + 1}", vs, vs, ve))

    n_test = -(-cfg.test_years // y)  # ceil
    for k in range(n_test):
        ts = boundary + pd.DateOffset(years=k * y)
        te = min(boundary + pd.DateOffset(years=(k + 1) * y), end)
        if ts >= end:
            break
        folds.append(Fold("test", f"test_{k + 1}", boundary, ts, te))
    return folds


def boundary_of(folds: list[Fold]) -> pd.Timestamp:
    """Start of the test period == end of the development data."""
    return next(f.train_end for f in folds if f.kind == "test")
