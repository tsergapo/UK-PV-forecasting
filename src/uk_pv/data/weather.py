"""ERA5 loading and alignment to the PV hour buckets."""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


_KEYS = ["latitude", "longitude", "valid_time"]


def expand_globs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in sorted(glob.glob(pat)))
    return paths


def _nearest_point(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    pts = df[["latitude", "longitude"]].drop_duplicates()
    d2 = (pts["latitude"] - lat) ** 2 + (pts["longitude"] - lon) ** 2
    best = pts.loc[d2.idxmin()]
    mask = (df["latitude"] == best["latitude"]) & (df["longitude"] == best["longitude"])
    return df.loc[mask]


def load_era5_csvs(paths: Sequence[str | Path], lat: float, lon: float) -> pd.DataFrame:
    """Merge ERA5 time-series CSVs on (latitude, longitude, valid_time) and keep the grid
    point nearest to the site. Returns a frame indexed by valid_time."""
    if not paths:
        raise FileNotFoundError("No ERA5 CSV files found; check paths.era5_csv in configs/data.yaml")
    # Files are grouped by their variable columns. Files with the SAME variables are time chunks
    # (concatenated); groups with DIFFERENT variables are joined on (lat, lon, time).
    groups: dict[tuple, list[pd.DataFrame]] = {}
    for p in paths:
        df = pd.read_csv(p)
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        cols = tuple(sorted(c for c in df.columns if c not in _KEYS))
        groups.setdefault(cols, []).append(df)

    seen: set[str] = set()
    merged: pd.DataFrame | None = None
    for cols, frames in groups.items():
        dup = seen & set(cols)
        if dup:
            raise ValueError(f"ERA5 variables {sorted(dup)} appear in more than one file group")
        seen |= set(cols)
        g = pd.concat(frames, ignore_index=True).drop_duplicates(_KEYS)
        merged = g if merged is None else merged.merge(g, on=_KEYS, how="inner")
    merged = _nearest_point(merged, lat, lon)
    merged = merged.drop(columns=["latitude", "longitude"]).set_index("valid_time").sort_index()
    return merged[~merged.index.duplicated(keep="first")]


def load_era5_grib(
    paths: Sequence[str | Path], lat: float, lon: float, variables=("hcc", "lcc", "mcc")
) -> pd.DataFrame:
    """Nearest-grid-point extraction from one or more GRIB files (requires xarray + cfgrib)."""
    import xarray as xr

    if not paths:
        raise FileNotFoundError("No GRIB files found; check paths.era5_grib in configs/data.yaml")
    frames = []
    for p in paths:
        ds = xr.open_dataset(p, engine="cfgrib")
        try:
            lon_q = lon % 360 if float(ds["longitude"].max()) > 180 else lon
            pt = ds[list(variables)].sel(latitude=lat, longitude=lon_q, method="nearest")
            frames.append(
                pd.DataFrame(
                    {v: np.asarray(pt[v].values, dtype=float) for v in variables},
                    index=pd.DatetimeIndex(pt["time"].values),
                )
            )
        finally:
            ds.close()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="first")]


def align_to_hour_buckets(
    w: pd.DataFrame,
    accumulated: Sequence[str] = ("ssrd", "fdir"),
    mode: str = "accumulation_aware",
) -> pd.DataFrame:
    """Express hourly ERA5 values for the PV bucket [t, t+1h).

    ERA5 accumulations (ssrd, fdir; J/m2) cover the hour ENDING at valid_time, so the
    accumulation for [t, t+1h) is the value stamped t+1h. Instantaneous fields are averaged
    over the bucket edges. Accumulations are converted to mean W/m2.

    mode="legacy_mean" reproduces the notebook (mean of t and t+1h for every variable).
    """
    if mode not in ("accumulation_aware", "legacy_mean"):
        raise ValueError(f"unknown alignment mode: {mode}")
    w = w.sort_index().astype(float)
    nxt = w.copy()
    nxt.index = nxt.index - pd.Timedelta(hours=1)
    nxt = nxt.reindex(w.index)  # nxt.loc[t] == w.loc[t + 1h]

    out = pd.DataFrame(index=w.index)
    for col in w.columns:
        is_acc = col in accumulated
        if is_acc and mode == "accumulation_aware":
            out[col] = nxt[col] / 3600.0
        else:
            val = (w[col] + nxt[col]) / 2.0
            out[col] = val / 3600.0 if is_acc else val
        if is_acc:
            out[col] = out[col].clip(lower=0.0)  # ERA5 has tiny negative accumulations at night
    return out
