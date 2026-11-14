from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..features.sun_hours import SunHoursEncoder
from .base import Forecaster


class TreeForecaster(Forecaster):
    """Gradient-boosted trees on same-hour covariates (no target lags). The estimator factory
    makes the wrapper testable without xgboost (any sklearn-style regressor works)."""

    stochastic = True

    def __init__(
        self, target, exog_cols, estimator_factory: Callable[[], object],
        use_sun_hours: bool = False, name: str = "tree",
    ):
        super().__init__(target, exog_cols)
        self.estimator_factory = estimator_factory
        self.use_sun_hours = use_sun_hours
        self.name = name
        self.sun_: SunHoursEncoder | None = None
        self.est_ = None

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.exog_cols]
        if self.sun_ is not None:
            X = pd.concat([X, self.sun_.transform(df.index)], axis=1)
        return X

    def fit(self, train):
        if self.use_sun_hours:
            self.sun_ = SunHoursEncoder().fit(train.index, train[self.target])
        X, y = self._features(train), train[self.target]
        m = y.notna().to_numpy()
        self.est_ = self.estimator_factory()
        self.est_.fit(X[m].to_numpy(), y[m].to_numpy())
        return self

    def forecast_blocks(self, df, starts, horizon):
        pos = starts[:, None] + np.arange(horizon)[None, :]
        X = self._features(df).to_numpy()[pos.ravel()]
        pred = np.clip(np.asarray(self.est_.predict(X), dtype=float), 0.0, None)
        return pred.reshape(pos.shape)


def make_xgb(target, exog_cols, seed: int = 0, use_sun_hours: bool = True, **params):
    from xgboost import XGBRegressor

    p = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3, "tree_method": "hist"}
    p.update(params)
    return TreeForecaster(
        target, exog_cols,
        estimator_factory=lambda: XGBRegressor(random_state=seed, n_jobs=1, **p),
        use_sun_hours=use_sun_hours, name="xgb",
    )
