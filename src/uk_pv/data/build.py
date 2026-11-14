from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import add_features
from .pv import clean_pv_to_hourly
from .weather import align_to_hour_buckets, expand_globs, load_era5_csvs, load_era5_grib


def load_site_metadata(path: str | Path, ss_id: int) -> dict:
    md = pd.read_csv(path)
    row = md.loc[md["ss_id"] == ss_id]
    if row.empty:
        raise ValueError(f"ss_id {ss_id} not found in {path}")
    row = row.iloc[0]

    def num(key):
        v = row.get(key)
        return None if v is None or pd.isna(v) else float(v)

    return {
        "ss_id": int(ss_id),
        "lat": num("latitude_rounded"),
        "lon": num("longitude_rounded"),
        "kwp": num("kWp"),
        "tilt": num("tilt"),
        "orientation": num("orientation"),
    }


def read_site_raw(pv_dir: str | Path, ss_id: int) -> pd.DataFrame:
    return pd.read_parquet(pv_dir, filters=[("ss_id", "==", ss_id)])


def assemble_dataset(
    pv_hourly: pd.Series, weather_raw: pd.DataFrame, meta: dict, weather_cfg: dict
) -> pd.DataFrame:
    """Join hourly PV with bucket-aligned ERA5 on a continuous hourly index and add features."""
    aligned = align_to_hour_buckets(
        weather_raw,
        accumulated=weather_cfg.get("accumulated_vars", ["ssrd", "fdir"]),
        mode=weather_cfg.get("alignment", "accumulation_aware"),
    )
    valid_w = aligned.dropna()
    start = max(pv_hourly.index.min(), valid_w.index.min())
    end = min(pv_hourly.index.max(), valid_w.index.max())
    idx = pd.date_range(start, end, freq="1h")

    df = aligned.reindex(idx)
    df.insert(0, "generation_kwh", pv_hourly.reindex(idx))
    df["t2m_c"] = df.pop("t2m") - 273.15
    df["ws10"] = np.hypot(df["u10"], df["v10"])
    df.index.name = "time"
    return add_features(df, meta["lat"], meta["lon"])


def build_site_dataset(cfg: dict) -> tuple[pd.DataFrame, dict]:
    paths = cfg["paths"]
    meta = load_site_metadata(paths["metadata_csv"], cfg["ss_id"])
    raw = read_site_raw(paths["pv_parquet_dir"], cfg["ss_id"])
    pv = clean_pv_to_hourly(
        raw, meta["lat"], meta["lon"],
        max_interp_halfhours=cfg.get("cleaning", {}).get("max_interp_halfhours", 1),
    )
    csv_w = load_era5_csvs(expand_globs(paths["era5_csv"]), meta["lat"], meta["lon"])
    grib_w = load_era5_grib(expand_globs(paths["era5_grib"]), meta["lat"], meta["lon"])
    weather = csv_w.join(grib_w, how="inner")
    df = assemble_dataset(pv, weather, meta, cfg.get("weather", {}))
    return df, meta


def save_dataset(df: pd.DataFrame, meta: dict, out_dir: str | Path, ss_id: int) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"site_{ss_id}.parquet"
    df.to_parquet(path)
    (out / f"site_{ss_id}.meta.json").write_text(json.dumps(meta, indent=2))
    return path


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    df = pd.read_parquet(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    return df, meta
