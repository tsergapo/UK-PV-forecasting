from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class SplitConfig:
    test_years: int = 2
    n_val_folds: int = 2
    fold_years: int = 1


@dataclass(frozen=True)
class BacktestConfig:
    horizon: int = 24
    context_hours: int = 168


def split_configs_from_yaml(path: str | Path) -> tuple[SplitConfig, BacktestConfig]:
    cfg = load_yaml(path)
    return SplitConfig(**cfg.get("split", {})), BacktestConfig(**cfg.get("backtest", {}))


def config_hash(*objs) -> str:
    blob = json.dumps(objs, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]
