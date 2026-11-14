"""Run models through the SAME rolling-origin validation + held-out test protocol."""
from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import BacktestConfig, SplitConfig, config_hash
from .evaluation.backtest import backtest
from .evaluation.metrics import summarize
from .models.registry import build_model
from .splits import boundary_of, make_folds
from .utils.provenance import git_hash
from .utils.seed import seed_everything


def _rows_for(pred, cov, hours, fold_name, split, base, cap):
    if pred.empty:
        return []
    m = summarize(pred, cap)
    return [{**base, "split": split, "fold": fold_name, "coverage": cov, "n_hours": hours, **m}]


def run_experiment(
    df: pd.DataFrame, meta: dict, data_cfg: dict, split_cfg: SplitConfig, bt_cfg: BacktestConfig,
    specs: list[dict], seeds: list[int], pred_dir: str | Path | None = None,
) -> pd.DataFrame:
    target = data_cfg.get("target", "generation_kwh")
    exog = list(data_cfg["exog_columns"])
    cap = meta.get("kwp")
    folds = make_folds(df.index, split_cfg)
    boundary = boundary_of(folds)
    dev = df.iloc[: int(df.index.searchsorted(boundary))]
    chash = config_hash(specs, asdict(split_cfg), asdict(bt_cfg), exog)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commit = git_hash()
    rows: list[dict] = []

    def n_hours(f):
        return int(df.index.searchsorted(f.eval_end) - df.index.searchsorted(f.eval_start))

    for spec in specs:
        for si, seed in enumerate(seeds):
            probe = build_model(spec, target, exog, seed)
            if si > 0 and not probe.stochastic:
                break
            base = {"model": spec["name"], "seed": seed, "config_hash": chash,
                    "git_hash": commit, "timestamp": stamp}
            pooled = {"val": [], "test": []}

            def fit_model(train):
                seed_everything(seed)
                model = build_model(spec, target, exog, seed)
                t0 = time.time()
                model.fit(train)
                return model, time.time() - t0

            for fold in (f for f in folds if f.kind == "val"):
                train = df.iloc[: int(df.index.searchsorted(fold.train_end))]
                model, secs = fit_model(train)
                pred, cov = backtest(model, df, fold, bt_cfg, target, exog)
                rows += _rows_for(pred, cov, n_hours(fold), fold.name, "val",
                                  {**base, "fit_seconds": secs}, cap)
                pooled["val"].append(pred)

            model, secs = fit_model(dev)  # final model: all development data
            for fold in (f for f in folds if f.kind == "test"):
                pred, cov = backtest(model, df, fold, bt_cfg, target, exog)
                rows += _rows_for(pred, cov, n_hours(fold), fold.name, "test",
                                  {**base, "fit_seconds": secs}, cap)
                pooled["test"].append(pred)

            for split in ("val", "test"):
                parts = [p for p in pooled[split] if not p.empty]
                if not parts:
                    continue
                allp = pd.concat(parts)
                hrs = sum(n_hours(f) for f in folds if f.kind == split)
                rows += _rows_for(allp, len(allp) / hrs, hrs, "pooled", split,
                                  {**base, "fit_seconds": secs}, cap)
                if pred_dir:
                    out = Path(pred_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    allp.to_parquet(out / f"{spec['name']}_seed{seed}_{split}.parquet")
    return pd.DataFrame(rows)
