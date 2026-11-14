"""Neural-model smoke tests; skipped automatically when torch is not installed."""
import numpy as np
import pandas as pd
import pytest

from uk_pv.models.registry import build_model
from synthetic import EXOG, make_site

T = "generation_kwh"
PARAMS = {"seq_len": 24, "epochs": 2, "batch_size": 128, "refit_full": False}
SPECS = [
    {"name": "lstm", "type": "lstm", "params": {**PARAMS, "hidden_dim": 8}},
    {"name": "tcn", "type": "tcn", "params": {**PARAMS, "num_channels": [8, 8], "kernel_size": 3}},
]


def test_nn_fit_forecast_shape_and_no_leakage():
    pytest.importorskip("torch")
    df = make_site(start="2020-01-01", end="2021-07-01", gaps=False)
    train = df[df.index < "2021-03-01"]
    p = int(df.index.searchsorted(pd.Timestamp("2021-05-01")))
    d2 = df.copy()
    d2.iloc[p:, d2.columns.get_loc(T)] = 7.0
    for spec in SPECS:
        m = build_model(spec, T, EXOG, seed=0).fit(train)
        a = m.forecast_blocks(df, np.array([p, p + 24]), 24)
        assert a.shape == (2, 24) and np.isfinite(a).all() and (a >= 0).all()
        b = m.forecast_blocks(d2, np.array([p]), 24)
        assert np.allclose(a[0], b[0], atol=1e-5), f"{spec['name']} leaks future target values"


def test_nn_training_is_reproducible():
    pytest.importorskip("torch")
    df = make_site(start="2020-01-01", end="2020-09-01", gaps=False)
    train = df[df.index < "2020-07-01"]
    starts = np.array([int(df.index.searchsorted(pd.Timestamp("2020-07-15")))])
    outs = [build_model(SPECS[0], T, EXOG, seed=3).fit(train).forecast_blocks(df, starts, 24)
            for _ in range(2)]
    assert np.allclose(outs[0], outs[1], atol=1e-6)
