"""Summarise results/metrics.csv: test-period metrics, mean +/- std over seeds, skill vs persistence."""
from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/metrics.csv")
    ap.add_argument("--out", default="results/summary.txt")
    ap.add_argument("--reference", default="persistence_24h")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df[(df["split"] == "test") & (df["fold"] == "pooled")]
    latest = df.sort_values("timestamp").groupby("model")["config_hash"].last()
    df = df[df.apply(lambda r: r["config_hash"] == latest[r["model"]], axis=1)]

    metrics = [c for c in ["rmse", "mae", "r2", "rmse_day", "nrmse_kwp", "coverage"] if c in df]
    g = df.groupby("model")[metrics]
    mean, std, n = g.mean(), g.std().fillna(0.0), g.size()
    table = pd.DataFrame(index=mean.index)
    for c in metrics:
        table[c] = [f"{m:.4f} ± {s:.4f}" if n[i] > 1 else f"{m:.4f}" for i, m, s in
                    zip(mean.index, mean[c], std[c])]
    table["seeds"] = n
    if args.reference in mean.index:
        ref = mean.loc[args.reference, "rmse"]
        table[f"skill_vs_{args.reference}"] = (1 - mean["rmse"] / ref).round(3)
    table = table.loc[mean["rmse"].sort_values().index]
    text = table.reset_index().to_string(index=False)
    open(args.out, "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
