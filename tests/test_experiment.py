import numpy as np

from uk_pv.config import BacktestConfig, SplitConfig
from uk_pv.experiment import run_experiment
from synthetic import DATA_CFG, make_site

SPECS = [
    {"name": "persistence_24h", "type": "seasonal_naive", "params": {"period_hours": 24}},
    {"name": "climatology", "type": "climatology", "params": {}},
]


def test_run_experiment_end_to_end_on_baselines():
    df = make_site()
    meta = {"kwp": 4.0, "lat": 51.5, "lon": -1.8}
    res = run_experiment(df, meta, DATA_CFG, SplitConfig(1, 1, 1), BacktestConfig(24, 168),
                         SPECS, seeds=[0, 1])
    # deterministic baselines run once even with two seeds
    assert set(res["seed"]) == {0}
    for model in ("persistence_24h", "climatology"):
        for split in ("val", "test"):
            sub = res[(res.model == model) & (res.split == split)]
            assert "pooled" in set(sub.fold)
            pooled = sub[sub.fold == "pooled"].iloc[0]
            assert np.isfinite(pooled.rmse) and 0 < pooled.coverage <= 1
            assert pooled.n_scored > 1000
    assert {"nrmse_kwp", "rmse_day", "git_hash", "config_hash"} <= set(res.columns)
