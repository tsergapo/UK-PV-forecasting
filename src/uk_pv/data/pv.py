"""PV cleaning: raw half-hourly Wh (period-ending stamps) -> hourly kWh on a continuous index."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.solar import solar_elevation_deg


def short_gap_mask(isna: np.ndarray, max_len: int) -> np.ndarray:
    """True for NaNs that belong to an interior run of length <= max_len."""
    n = len(isna)
    mask = np.zeros(n, dtype=bool)
    if n == 0 or max_len <= 0:
        return mask
    padded = np.concatenate(([False], isna.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for a, b in zip(edges[0::2], edges[1::2]):  # run is [a, b)
        if (b - a) <= max_len and a > 0 and b < n:
            mask[a:b] = True
    return mask


def clean_pv_to_hourly(
    raw: pd.DataFrame,
    lat: float,
    lon: float,
    max_interp_halfhours: int = 1,
    time_col: str = "datetime_GMT",
    value_col: str = "generation_Wh",
) -> pd.Series:
    """Return hourly generation in kWh indexed by the START of each hour (UTC, tz-naive).

    Convention (inherited from the original notebook, verify against the dataset card):
    raw timestamps mark the END of each half-hour period. Each stamp is moved to its period
    start BEFORE any gap handling, and an hour is the sum of its two half-hours.

    * missing slots while the sun is below the horizon -> 0 (geometry-based, all seasons)
    * isolated daytime gaps (<= max_interp_halfhours) -> linear interpolation
    * all other gaps stay NaN; the index stays continuous (no days are dropped), so row
      offsets always equal time offsets and downstream windows can detect gaps.
    """
    ts = pd.to_datetime(raw[time_col])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    s = pd.Series(raw[value_col].to_numpy(dtype="float64"), index=pd.DatetimeIndex(ts))
    s = s.sort_index()
    s.index = s.index.round("30min") - pd.Timedelta(minutes=30)  # stamp(end) -> period start
    s = s[~s.index.duplicated(keep="first")]

    grid = pd.date_range(
        s.index.min().floor("D"),
        s.index.max().floor("D") + pd.Timedelta(days=1) - pd.Timedelta(minutes=30),
        freq="30min",
    )
    s = s.reindex(grid)
    vals = s.to_numpy().copy()

    night = solar_elevation_deg(grid, lat, lon, offset_minutes=15) < 0
    vals[np.isnan(vals) & night] = 0.0

    interp = pd.Series(vals, index=grid).interpolate(method="linear", limit_area="inside")
    fill = short_gap_mask(np.isnan(vals), max_interp_halfhours)
    vals[fill] = interp.to_numpy()[fill]

    hourly = pd.Series(vals, index=grid).resample("1h", closed="left", label="left").sum(min_count=2)
    first, last = hourly.first_valid_index(), hourly.last_valid_index()
    hourly = hourly.loc[first:last] / 1000.0
    hourly.name = "generation_kwh"
    return hourly
