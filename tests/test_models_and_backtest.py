import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from uk_pv.config import BacktestConfig, SplitConfig
from uk_pv.evaluation.backtest import backtest, valid_block_starts
from uk_pv.features.sun_hours import SunHoursEncoder
from uk_pv.models.base import Forecaster
from uk_pv.models.baselines import Climatology, SeasonalNaive
from uk_pv.models.trees import TreeForecaster
from uk_pv.splits import make_folds
from synthetic import EXOG, make_site

T = "generation_kwh"
BT = BacktestConfig(horizon=24, context_hours=168)


def _tree():
    return TreeForecaster(T, EXOG, lambda: HistGradientBoostingRegressor(max_iter=30, random_state=0),
                          use_sun_hours=True, name="tree")


def _folds(df):
    return make_folds(df.index, SplitConfig(test_years=1, n_val_folds=1, fold_years=1))


def test_seasonal_naive_values():
    df = make_site(gaps=False)
    starts = np.array([500, 1000])
    out = SeasonalNaive(T, EXOG, 24).forecast_blocks(df, starts, 24)
    y = df[T].to_numpy()
    assert np.allclose(out[0], y[500 - 24 : 500]) and np.allclose(out[1], y[1000 - 24 : 1000])
    out48 = SeasonalNaive(T, EXOG, 24).forecast_blocks(df, starts, 48)
    assert np.allclose(out48[0, 24:], y[500 - 48 + 24 : 500])  # lag 48 for steps 24..47


def test_blocks_with_gaps_are_excluded_for_all_models():
    df = make_site()
    fold = [f for f in _folds(df) if f.kind == "test"][0]
    starts, n_hours = valid_block_starts(df, fold, BT, T, EXOG)
    assert 0 < len(starts) <= n_hours // 24
    ok = df[[T] + EXOG].notna().all(axis=1).to_numpy()
    for p in starts:
        assert ok[p - 168 : p + 24].all()


def test_same_scored_hours_for_every_model():
    df = make_site()
    train = df[df.index < "2022-01-01"]
    fold = [f for f in _folds(df) if f.kind == "test"][0]
    idxs = []
    for m in [SeasonalNaive(T, EXOG, 24), Climatology(T, EXOG), _tree()]:
        m.fit(train)
        pred, cov = backtest(m, df, fold, BT, T, EXOG)
        assert 0 < cov <= 1 and np.isfinite(pred["y_pred"]).all() and (pred["y_pred"] >= 0).all()
        idxs.append(pred.index)
    assert idxs[0].equals(idxs[1]) and idxs[1].equals(idxs[2])


def test_tree_beats_climatology_on_synthetic_weather():
    df = make_site()
    train = df[df.index < "2022-01-01"]
    fold = [f for f in _folds(df) if f.kind == "test"][0]
    res = {}
    for m in [Climatology(T, EXOG), _tree()]:
        m.fit(train)
        pred, _ = backtest(m, df, fold, BT, T, EXOG)
        res[m.name] = float(np.sqrt(((pred.y_pred - pred.y_true) ** 2).mean()))
    assert res["tree"] < res["climatology"]


class _Leaky(Forecaster):
    name = "leaky"

    def forecast_blocks(self, df, starts, horizon):
        pos = starts[:, None] + np.arange(horizon)[None, :]
        return df[self.target].to_numpy()[pos]


def _perturb_after(df, p):
    d = df.copy()
    d.iloc[p:, d.columns.get_loc(T)] = np.random.default_rng(1).normal(5, 1, len(d) - p)
    return d


def test_no_target_leakage_in_registered_models():
    df = make_site(gaps=False)
    train = df[df.index < "2022-01-01"]
    p = int(df.index.searchsorted(pd.Timestamp("2022-06-01")))
    d2 = _perturb_after(df, p)
    for m in [SeasonalNaive(T, EXOG, 24), SeasonalNaive(T, EXOG, 168), Climatology(T, EXOG), _tree()]:
        m.fit(train)
        a = m.forecast_blocks(df, np.array([p]), 24)
        b = m.forecast_blocks(d2, np.array([p]), 24)
        assert np.allclose(a, b), f"{m.name} uses target values at or after the block start"


def test_leak_canary_detects_leaky_model():
    df = make_site(gaps=False)
    p = 20000
    m = _Leaky(T, EXOG)
    assert not np.allclose(m.forecast_blocks(df, np.array([p]), 24),
                           m.forecast_blocks(_perturb_after(df, p), np.array([p]), 24))


def test_sun_hours_encoder_learns_productive_window():
    idx = pd.date_range("2021-05-01", "2021-05-31 23:00", freq="1h")
    y = pd.Series(np.where((idx.hour >= 6) & (idx.hour <= 18), 1.0, 0.0), index=idx)
    enc = SunHoursEncoder().fit(idx, y)
    f = enc.transform(idx)
    assert (f.loc[f.sun_start_flag == 1].index.hour == 6).all()
    assert (f.loc[f.sun_end_flag == 1].index.hour == 18).all()
    assert f.sun_start_flag.sum() == 31


class _Nan(Forecaster):
    name = "nanmodel"

    def forecast_blocks(self, df, starts, horizon):
        out = np.zeros((len(starts), horizon))
        out[0, 0] = np.nan
        return out


def test_backtest_reports_nonfinite_forecasts_clearly():
    df = make_site(gaps=False)
    fold = [f for f in _folds(df) if f.kind == "test"][0]
    try:
        backtest(_Nan(T, EXOG), df, fold, BT, T, EXOG)
    except RuntimeError as e:
        assert "NaN/inf" in str(e) and "1 of" in str(e)
        return
    raise AssertionError("expected RuntimeError")
