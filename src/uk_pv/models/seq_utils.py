"""Pure-numpy helpers for autoregressive sequence models (testable without torch).

Design (fixes a misalignment in the original notebook): a window row at time s holds
[y_{s-1}, covariates_s]. The window ending at t therefore contains the covariates of the
hour being predicted (y_t) but only the target up to t-1. In the notebook the window ended at
t-1 and so the same-hour covariates were never seen."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=float)
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=np.atleast_1d(mean), std=np.atleast_1d(std))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        return cls(np.asarray(d["mean"]), np.asarray(d["std"]))


def build_design(y_scaled: np.ndarray, exog_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A[s] = [y_{s-1}, exog_s] (float32, NaN where unavailable); T[s] = y_s."""
    n = len(y_scaled)
    lag = np.full(n, np.nan)
    lag[1:] = y_scaled[:-1]
    A = np.column_stack([lag, exog_scaled]).astype(np.float32)
    return A, np.asarray(y_scaled, dtype=np.float32)


def valid_positions(A: np.ndarray, T: np.ndarray, seq_len: int) -> np.ndarray:
    """Positions t whose window rows [t-L+1, t] are NaN-free and whose target is finite.
    The array is a continuous hourly series, so windows can never silently straddle a gap."""
    n = len(A)
    if n < seq_len:
        return np.empty(0, dtype=np.int64)
    bad_cs = np.concatenate(([0], np.cumsum((~np.isfinite(A)).any(axis=1))))
    t = np.arange(seq_len - 1, n)
    ok = (bad_cs[t + 1] - bad_cs[t + 1 - seq_len] == 0) & np.isfinite(T[t])
    return t[ok]


def rollout(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    A: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    horizon: int,
    batch_size: int = 512,
) -> np.ndarray:
    """Vectorised recursive forecasting of many blocks at once.

    For a block starting at row b, the window rows are b-L+1 .. b+H-1. Target-lag entries of
    rows > b are zeroed (no target information beyond b-1) and are overwritten with the
    model's own predictions as the rollout proceeds. Covariates of future rows are real
    (known-in-advance) inputs."""
    starts = np.asarray(starts, dtype=np.int64)
    out = np.empty((len(starts), horizon), dtype=np.float32)
    L, H = seq_len, horizon
    offsets = np.arange(L + H - 1)
    for s in range(0, len(starts), batch_size):
        st = starts[s : s + batch_size]
        rows = st[:, None] - (L - 1) + offsets[None, :]
        X = A[rows].copy()
        X[:, L:, 0] = 0.0
        for k in range(H):
            p = np.asarray(predict_fn(X[:, k : k + L, :]), dtype=np.float32)
            out[s : s + len(st), k] = p
            if k < H - 1:
                X[:, L + k, 0] = p
    return out
