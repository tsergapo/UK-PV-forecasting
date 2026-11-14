import tempfile
from pathlib import Path

import pandas as pd

from uk_pv.data.weather import load_era5_csvs


def _write(path, times, cols, lat_lon=((51.5, -1.75), (51.5, -2.0))):
    rows = []
    for t in times:
        for la, lo in lat_lon:
            rows.append({"valid_time": t, "latitude": la, "longitude": lo, **{c: float(i) for i, c in enumerate(cols)}})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_time_chunks_are_concatenated_and_variable_groups_joined():
    t1 = pd.date_range("2010-01-01", periods=48, freq="1h")
    t2 = pd.date_range("2010-01-03", periods=48, freq="1h")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write(d / "a1.csv", t1, ["u10", "v10"])
        _write(d / "a2.csv", t2, ["u10", "v10"])
        _write(d / "b1.csv", t1, ["t2m"])
        _write(d / "b2.csv", t2, ["t2m"])
        w = load_era5_csvs(sorted(d.glob("*.csv")), 51.5, -1.78)
    assert list(w.columns) == ["u10", "v10", "t2m"] or set(w.columns) == {"u10", "v10", "t2m"}
    assert len(w) == 96 and w.index.is_monotonic_increasing and not w.isna().any().any()


def test_picks_nearest_grid_point():
    t = pd.date_range("2010-01-01", periods=3, freq="1h")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.csv"
        rows = [{"valid_time": x, "latitude": la, "longitude": lo, "t2m": v}
                for x in t for la, lo, v in [(51.5, -1.75, 1.0), (51.5, -2.0, 2.0), (51.75, -2.0, 3.0)]]
        pd.DataFrame(rows).to_csv(p, index=False)
        w = load_era5_csvs([p], 51.56, -1.78)
    assert (w["t2m"] == 1.0).all()


def test_same_variable_in_two_groups_is_an_error():
    t = pd.date_range("2010-01-01", periods=3, freq="1h")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write(d / "a.csv", t, ["t2m", "u10"])
        _write(d / "b.csv", t, ["t2m"])
        try:
            load_era5_csvs(sorted(d.glob("*.csv")), 51.5, -1.75)
        except ValueError as e:
            assert "t2m" in str(e)
            return
    raise AssertionError("expected ValueError")
