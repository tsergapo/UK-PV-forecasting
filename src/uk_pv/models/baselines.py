from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class SeasonalNaive(Forecaster):
    """Value one period earlier. period_hours=24 is 'persistence' (same hour yesterday); the
    lag grows in whole periods so it never reaches into the block being forecast."""

    def __init__(self, target, exog_cols, period_hours: int = 24):
        super().__init__(target, exog_cols)
        self.period = int(period_hours)
        self.name = f"seasonal_naive_{self.period}h"

    def required_context(self, horizon: int) -> int:
        return self.period * (-(-horizon // self.period))

    def forecast_blocks(self, df, starts, horizon):
        y = df[self.target].to_numpy(dtype=float)
        steps = np.arange(horizon)
        lags = self.period * (steps // self.period + 1)
        src = starts[:, None] + steps[None, :] - lags[None, :]
        return y[src]


class Climatology(Forecaster):
    """Mean generation per (month, hour) over the training data."""

    name = "climatology"

    def __init__(self, target, exog_cols):
        super().__init__(target, exog_cols)
        self.table_: pd.Series | None = None

    def fit(self, train):
        y = train[self.target].dropna()
        self.table_ = y.groupby([y.index.month, y.index.hour]).mean()
        return self

    def forecast_blocks(self, df, starts, horizon):
        pos = starts[:, None] + np.arange(horizon)[None, :]
        idx = df.index[pos.ravel()]
        keys = pd.MultiIndex.from_arrays([idx.month, idx.hour])
        vals = self.table_.reindex(keys).to_numpy(dtype=float)
        return np.nan_to_num(vals, nan=0.0).reshape(pos.shape)
