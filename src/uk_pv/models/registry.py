from __future__ import annotations

from .base import Forecaster
from .baselines import Climatology, SeasonalNaive


def build_model(spec: dict, target: str, exog_cols: list[str], seed: int = 0) -> Forecaster:
    """spec = {"name": ..., "type": ..., "params": {...}} (see configs/models/*.yaml).
    Heavy dependencies (torch, xgboost) are imported only when the model is requested."""
    kind, params = spec["type"], dict(spec.get("params") or {})
    if kind == "seasonal_naive":
        return SeasonalNaive(target, exog_cols, **params)
    if kind == "climatology":
        return Climatology(target, exog_cols)
    if kind == "xgb":
        from .trees import make_xgb
        return make_xgb(target, exog_cols, seed=seed, **params)
    if kind == "lstm":
        from .sequence import make_lstm
        return make_lstm(target, exog_cols, seed=seed, **params)
    if kind == "tcn":
        from .sequence import make_tcn
        return make_tcn(target, exog_cols, seed=seed, **params)
    raise ValueError(f"unknown model type: {kind}")
