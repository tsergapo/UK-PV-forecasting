import numpy as np
import pandas as pd

from uk_pv.data.pv import clean_pv_to_hourly, short_gap_mask

LAT, LON = 51.5, -1.8


def _raw(values: dict | None = None, default=0.0, start="2021-01-10", days=5, drop=()):
    stamps = pd.date_range(start, periods=48 * days, freq="30min") + pd.Timedelta(minutes=30)  # period END
    vals = np.full(len(stamps), default)
    raw = pd.DataFrame({"datetime_GMT": stamps, "generation_Wh": vals})
    for ts, v in (values or {}).items():
        raw.loc[raw.datetime_GMT == pd.Timestamp(ts), "generation_Wh"] = v
    return raw[~raw.datetime_GMT.isin([pd.Timestamp(d) for d in drop])]


def test_short_gap_mask():
    isna = np.array([0, 1, 0, 1, 1, 0, 1], dtype=bool)
    assert short_gap_mask(isna, 1).tolist() == [False, True, False, False, False, False, False]
    assert short_gap_mask(isna, 2).tolist() == [False, True, False, True, True, False, False]


def test_period_ending_convention():
    # stamp 12:30 = energy of 12:00-12:30 -> belongs to the 12:00 hour with stamp 13:00
    raw = _raw({"2021-01-12 12:30": 1000, "2021-01-12 13:00": 500})
    h = clean_pv_to_hourly(raw, LAT, LON)
    assert abs(h.loc["2021-01-12 12:00"] - 1.5) < 1e-9
    assert h.loc["2021-01-12 11:00"] == 0 and h.loc["2021-01-12 13:00"] == 0


def test_isolated_daytime_gap_interpolated_long_gap_kept():
    base = {f"2021-01-12 {h:02d}:{m:02d}": 100 for h in range(9, 15) for m in (0, 30)}
    one = clean_pv_to_hourly(_raw(base, drop=["2021-01-12 11:00"]), LAT, LON)
    assert one.loc["2021-01-12 10:00":"2021-01-12 11:00"].notna().all()
    long = clean_pv_to_hourly(
        _raw(base, drop=["2021-01-12 11:00", "2021-01-12 11:30", "2021-01-12 12:00"]), LAT, LON)
    assert long.loc["2021-01-12 11:00":"2021-01-12 12:00"].isna().any()


def test_night_gap_filled_with_zero_and_index_continuous():
    h = clean_pv_to_hourly(_raw(drop=["2021-01-12 02:00", "2021-01-12 02:30"]), LAT, LON)
    assert h.loc["2021-01-12 01:00":"2021-01-12 02:00"].notna().all()
    assert (h.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
