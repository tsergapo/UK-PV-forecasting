import numpy as np
import pandas as pd

from uk_pv.features.solar import solar_elevation_deg

LAT, LON = 51.5, -1.8


def test_summer_and_winter_noon():
    idx = pd.DatetimeIndex(["2021-06-21 12:00", "2021-12-21 12:00"])
    e = solar_elevation_deg(idx, LAT, LON)
    assert abs(e[0] - (90 - LAT + 23.44)) < 1.5
    assert abs(e[1] - (90 - LAT - 23.44)) < 1.5


def test_night_is_negative_and_offset_works():
    idx = pd.DatetimeIndex(["2021-06-21 00:00"])
    assert solar_elevation_deg(idx, LAT, LON)[0] < 0
    a = solar_elevation_deg(pd.DatetimeIndex(["2021-03-20 06:00"]), LAT, LON, offset_minutes=60)
    b = solar_elevation_deg(pd.DatetimeIndex(["2021-03-20 07:00"]), LAT, LON)
    assert np.allclose(a, b)
