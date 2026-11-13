"""Data-driven sunrise/sunset flags (the notebook's 'dominant sun hours' feature),
as a fit/transform object so that it is fitted on training data only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _month_third(index: pd.DatetimeIndex) -> np.ndarray:
    return np.minimum((index.day.to_numpy() - 1) // 10, 2)


class SunHoursEncoder:
    """For each (month, third-of-month) learn the most common first/last productive hour."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.table_: pd.DataFrame | None = None

    def fit(self, index: pd.DatetimeIndex, y: pd.Series) -> "SunHoursEncoder":
        y = pd.Series(np.asarray(y, dtype=float), index=index)
        dates = pd.DatetimeIndex(index.normalize())
        complete = y.notna().groupby(dates).all()  # ignore days with gaps
        prod = y[(y > self.threshold)]
        hours = pd.Series(prod.index.hour, index=prod.index)
        g = hours.groupby(pd.DatetimeIndex(prod.index.normalize()))
        daily = pd.DataFrame({"start": g.min(), "end": g.max()})
        daily = daily[complete.reindex(daily.index).fillna(False).to_numpy()]
        daily["month"] = daily.index.month
        daily["third"] = _month_third(daily.index)
        counts = daily.groupby(["month", "third", "start", "end"]).size().rename("n").reset_index()
        counts = counts.sort_values(["n", "start", "end"], ascending=[False, True, True])
        self.table_ = counts.drop_duplicates(["month", "third"]).set_index(["month", "third"])[
            ["start", "end"]
        ]
        return self

    def transform(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        if self.table_ is None:
            raise RuntimeError("SunHoursEncoder is not fitted")
        keys = pd.MultiIndex.from_arrays([index.month.to_numpy(), _month_third(index)])
        looked = self.table_.reindex(keys)
        hour = index.hour.to_numpy()
        start = looked["start"].to_numpy()
        end = looked["end"].to_numpy()
        return pd.DataFrame(
            {
                "sun_start_flag": (hour == start).astype("float32"),
                "sun_end_flag": (hour == end).astype("float32"),
            },
            index=index,
        )
