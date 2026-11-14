import numpy as np
import pandas as pd

from uk_pv.data.weather import align_to_hour_buckets


def _w():
    idx = pd.date_range("2021-01-01", periods=4, freq="1h")
    return pd.DataFrame({"ssrd": 3600.0 * np.array([1, 2, 3, 4]), "t2m": [10.0, 20, 30, 40]}, index=idx)


def test_accumulation_aware_uses_hour_ending_value():
    out = align_to_hour_buckets(_w(), accumulated=["ssrd"], mode="accumulation_aware")
    assert out["ssrd"].iloc[0] == 2.0 and out["ssrd"].iloc[1] == 3.0
    assert out["t2m"].iloc[0] == 15.0
    assert np.isnan(out["ssrd"].iloc[-1]) and np.isnan(out["t2m"].iloc[-1])


def test_legacy_mean_reproduces_notebook_smoothing():
    out = align_to_hour_buckets(_w(), accumulated=["ssrd"], mode="legacy_mean")
    assert out["ssrd"].iloc[0] == 1.5


def test_handles_gaps_by_time_not_position():
    w = _w().drop(_w().index[1])
    out = align_to_hour_buckets(w, accumulated=["ssrd"])
    assert np.isnan(out["t2m"].iloc[0])  # t+1h is missing -> NaN, not the next row's value
