from __future__ import annotations

import numpy as np
import pandas as pd


def _rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e**2))) if e.size else float("nan")


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    daylight: np.ndarray | None = None,
    capacity_kwp: float | None = None,
) -> dict:
    """All-hours metrics plus daytime-only metrics (daylight from solar geometry, not from
    the target) and capacity-normalised RMSE. Night hours are trivially zero and inflate R2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    sst = float(np.sum((y_true - y_true.mean()) ** 2)) if y_true.size else 0.0
    out = {
        "n_scored": int(y_true.size),
        "rmse": _rmse(err),
        "mae": float(np.mean(np.abs(err))) if err.size else float("nan"),
        "r2": 1.0 - float(np.sum(err**2)) / sst if sst > 0 else float("nan"),
        "bias": float(np.mean(err)) if err.size else float("nan"),
    }
    if daylight is not None:
        m = np.asarray(daylight).astype(bool)
        out["rmse_day"] = _rmse(err[m])
        out["mae_day"] = float(np.mean(np.abs(err[m]))) if m.any() else float("nan")
    if capacity_kwp:
        out["nrmse_kwp"] = out["rmse"] / capacity_kwp
    return out


def skill_score(rmse: float, rmse_ref: float) -> float:
    """1 - RMSE/RMSE_ref (positive = better than the reference forecast)."""
    return 1.0 - rmse / rmse_ref if rmse_ref and np.isfinite(rmse_ref) else float("nan")


def summarize(pred: pd.DataFrame, capacity_kwp: float | None = None) -> dict:
    return regression_metrics(
        pred["y_true"].to_numpy(), pred["y_pred"].to_numpy(),
        pred["daylight"].to_numpy() if "daylight" in pred else None, capacity_kwp,
    )
