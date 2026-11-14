"""Small synthetic site used by the tests (no external data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from uk_pv.features.build import add_features
from uk_pv.features.solar import solar_elevation_deg

LAT, LON, KWP = 51.5, -1.8, 4.0
EXOG = ["ssrd", "tcc", "t2m_c", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "solar_elevation"]
DATA_CFG = {"target": "generation_kwh", "exog_columns": EXOG}


def make_site(start="2020-01-01", end="2023-01-01", seed=0, gaps=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1h", inclusive="left", name="time")
    elev = solar_elevation_deg(idx, LAT, LON, offset_minutes=30)
    clear = np.clip(np.sin(np.deg2rad(elev)), 0, None)
    cloud = np.clip(pd.Series(rng.normal(0.4, 0.4, len(idx))).rolling(6, min_periods=1).mean(), 0, 1)
    ssrd = 900 * clear * (1 - 0.7 * cloud.to_numpy())
    y = np.clip(KWP * 0.8 * ssrd / 1000 + rng.normal(0, 0.02, len(idx)), 0, None) * (elev > 0)
    df = pd.DataFrame(
        {"generation_kwh": y, "ssrd": ssrd, "tcc": cloud.to_numpy(),
         "t2m_c": 10 + 8 * np.sin(2 * np.pi * (idx.dayofyear - 110) / 365.25)}, index=idx,
    )
    df = add_features(df, LAT, LON)
    if gaps:
        for a, b in [("2020-06-03 00:00", "2020-06-04 12:00"), ("2021-09-10 06:00", "2021-09-10 09:00"),
                     ("2022-04-20 00:00", "2022-04-22 00:00")]:
            df.loc[a:b, "generation_kwh"] = np.nan
    return df
