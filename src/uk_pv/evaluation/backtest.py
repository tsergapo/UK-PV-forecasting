"""Block-wise forecast evaluation shared by ALL models.

A forecast block of `horizon` hours is issued at its first hour: the model sees the target
only up to the block start, plus known covariates inside the block. Every model is scored on
exactly the same set of blocks (blocks with a clean `context_hours` history and a clean target),
so coverage differences cannot bias the comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..splits import Fold


def valid_block_starts(
    df: pd.DataFrame, fold: Fold, bt: BacktestConfig, target: str, exog_cols: list[str]
) -> tuple[np.ndarray, int]:
    """Integer positions of valid block starts inside the fold's eval window, and the number
    of hours in the window (to report coverage)."""
    lo = int(df.index.searchsorted(fold.eval_start))
    hi = int(df.index.searchsorted(fold.eval_end))
    H = bt.horizon
    starts = np.arange(lo, hi - H + 1, H)  # full blocks only

    ok = df[[target] + list(exog_cols)].notna().all(axis=1).to_numpy()
    bad_cs = np.concatenate(([0], np.cumsum(~ok)))
    a = starts - bt.context_hours
    b = starts + H
    keep = (a >= 0) & (bad_cs[b] - bad_cs[np.maximum(a, 0)] == 0)
    return starts[keep], hi - lo


def backtest(
    model, df: pd.DataFrame, fold: Fold, bt: BacktestConfig, target: str, exog_cols: list[str]
) -> tuple[pd.DataFrame, float]:
    need = model.required_context(bt.horizon)
    if need > bt.context_hours:
        raise ValueError(
            f"{model.name} needs {need}h of context but backtest.context_hours={bt.context_hours}"
        )
    starts, n_hours = valid_block_starts(df, fold, bt, target, exog_cols)
    if len(starts) == 0:
        return pd.DataFrame(columns=["y_true", "y_pred", "daylight", "block_start", "step"]), 0.0

    preds = np.asarray(model.forecast_blocks(df, starts, bt.horizon), dtype=float)
    if preds.shape != (len(starts), bt.horizon) or not np.isfinite(preds).all():
        raise RuntimeError(f"{model.name}: bad forecast array shape/values {preds.shape}")

    pos = (starts[:, None] + np.arange(bt.horizon)[None, :]).ravel()
    out = pd.DataFrame(
        {
            "y_true": df[target].to_numpy()[pos],
            "y_pred": preds.ravel(),
            "daylight": df["daylight"].to_numpy()[pos] if "daylight" in df else np.nan,
            "block_start": np.repeat(df.index[starts].to_numpy(), bt.horizon),
            "step": np.tile(np.arange(bt.horizon), len(starts)),
        },
        index=df.index[pos],
    )
    return out, len(out) / max(n_hours, 1)
