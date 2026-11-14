"""Raw PV + ERA5 -> data/processed/site_<ss_id>.parquet (+ .meta.json)."""
from __future__ import annotations

import argparse

from uk_pv.config import load_yaml
from uk_pv.data.build import build_site_dataset, save_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    df, meta = build_site_dataset(cfg)
    path = save_dataset(df, meta, cfg["paths"]["out_dir"], cfg["ss_id"])
    n_gap = int(df["generation_kwh"].isna().sum())
    print(f"saved {path}: {len(df)} hourly rows {df.index.min()} -> {df.index.max()}, "
          f"{n_gap} hours with missing target kept as NaN")
    print("meta:", meta)


if __name__ == "__main__":
    main()
