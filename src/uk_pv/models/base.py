from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """Common interface for baselines, tree models, neural nets and (later) foundation models.

    `df` is always the full continuous hourly frame. `starts` are integer row positions of
    block starts. Contract: forecasts for the block starting at row p may use
      * target values at rows < p only, and
      * covariates (exog_cols) at any row in [p - history, p + horizon).
    tests/test_models_and_backtest.py enforces this for every registered model.
    """

    name: str = "forecaster"
    stochastic: bool = False  # True -> run once per seed

    def __init__(self, target: str, exog_cols: list[str]):
        self.target = target
        self.exog_cols = list(exog_cols)

    def required_context(self, horizon: int) -> int:
        """Hours of clean history needed before a block start."""
        return 0

    def fit(self, train: pd.DataFrame) -> "Forecaster":
        return self

    @abstractmethod
    def forecast_blocks(self, df: pd.DataFrame, starts: np.ndarray, horizon: int) -> np.ndarray:
        """Return array [len(starts), horizon] of kWh forecasts."""
