import numpy as np

from uk_pv.evaluation.metrics import regression_metrics, skill_score


def test_perfect_forecast():
    y = np.array([0.0, 1, 2, 3])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0 and m["r2"] == 1 and m["bias"] == 0


def test_known_values_daytime_and_nrmse():
    y = np.array([0.0, 0.0, 2.0, 4.0])
    p = np.array([0.0, 0.0, 3.0, 3.0])
    m = regression_metrics(y, p, daylight=np.array([0, 0, 1, 1]), capacity_kwp=4.0)
    assert abs(m["rmse"] - np.sqrt(0.5)) < 1e-12
    assert abs(m["rmse_day"] - 1.0) < 1e-12 and abs(m["mae_day"] - 1.0) < 1e-12
    assert abs(m["nrmse_kwp"] - np.sqrt(0.5) / 4) < 1e-12


def test_skill_score():
    assert skill_score(0.5, 1.0) == 0.5
    assert np.isnan(skill_score(0.5, 0.0))
