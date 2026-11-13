from __future__ import annotations

import numpy as np
import pandas as pd

from .solar import solar_elevation_deg

NON_INPUT_COLUMNS = ("generation_kwh", "daylight")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical hour-of-day and day-of-year (bucket start, UTC). Raw year/month/hour are NOT
    provided: a raw year cannot extrapolate to the test period."""
    out = df.copy()
    hour = out.index.hour.to_numpy()
    doy = out.index.dayofyear.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def add_solar_features(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Solar elevation at the middle of each hourly bucket, plus a daylight mask used for
    metrics (derived from geometry, not from the target)."""
    out = df.copy()
    elev = solar_elevation_deg(out.index, lat, lon, offset_minutes=30)
    out["solar_elevation"] = elev
    out["daylight"] = (elev > 0).astype("float32")
    return out


def add_features(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    return add_solar_features(add_calendar_features(df), lat, lon)
