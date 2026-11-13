"""Deterministic solar geometry (NOAA approximation). No target information is used,
so these features can be computed for any future hour without leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd


def solar_elevation_deg(
    index: pd.DatetimeIndex, lat: float, lon: float, offset_minutes: float = 0.0
) -> np.ndarray:
    """Solar elevation angle (degrees) for UTC timestamps.

    offset_minutes shifts the evaluation instant, e.g. 30 to evaluate at the middle of an
    hourly bucket that starts at the index timestamp.
    """
    idx = pd.DatetimeIndex(index) + pd.Timedelta(minutes=offset_minutes)
    doy = idx.dayofyear.to_numpy().astype(float)
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0 + idx.second.to_numpy() / 3600.0

    g = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)  # fractional year (rad)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g)
    )  # minutes
    decl = (
        0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    )  # rad
    true_solar_min = hour * 60.0 + eqtime + 4.0 * lon  # UTC input -> no timezone term
    hour_angle = np.deg2rad(true_solar_min / 4.0 - 180.0)
    lat_r = np.deg2rad(lat)
    cos_zen = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle)
    return 90.0 - np.degrees(np.arccos(np.clip(cos_zen, -1.0, 1.0)))
