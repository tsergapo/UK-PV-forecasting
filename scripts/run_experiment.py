"""Evaluate models with identical folds/metrics; append results to results/metrics.csv."""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch/CUDA is touched

import argparse
from pathlib import Path

import pandas as pd

from uk_pv.config import load_yaml, split_configs_from_yaml
from uk_pv.data.build import load_dataset
from uk_pv.experiment import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--split-config", default="configs/split.yaml")
    ap.add_argument("--models", nargs="+", required=True, help="names of configs/models/<name>.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--results", default="results/metrics.csv")
    ap.add_argument("--pred-dir", default="results/predictions")
    args = ap.parse_args()

    data_cfg = load_yaml(args.data_config)
    df, meta = load_dataset(Path(data_cfg["paths"]["out_dir"]) / f"site_{data_cfg['ss_id']}.parquet")
    split_cfg, bt_cfg = split_configs_from_yaml(args.split_config)
    specs = [{"name": m, **load_yaml(f"configs/models/{m}.yaml")} for m in args.models]

    res = run_experiment(df, meta, data_cfg, split_cfg, bt_cfg, specs, args.seeds, args.pred_dir)
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, mode="a", header=not out.exists(), index=False)
    show = res[(res["split"] == "test") & (res["fold"] == "pooled")]
    cols = ["model", "seed", "rmse", "mae", "r2", "rmse_day", "nrmse_kwp", "coverage"]
    print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
