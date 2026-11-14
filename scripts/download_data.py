"""Download the OCF UK PV dataset (30-minute files + metadata) from Hugging Face.
Reads HF_TOKEN from the environment / .env if set (the dataset may not need one)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/uk_pv")
    args = ap.parse_args()
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id="openclimatefix/uk_pv", repo_type="dataset",
        allow_patterns=["30_minutely/*", "metadata.csv"],
        local_dir=args.out, token=os.environ.get("HF_TOKEN") or None,
    )
    print("downloaded to", path)
    print("Check that configs/data.yaml paths.pv_parquet_dir / metadata_csv match this layout.")


if __name__ == "__main__":
    main()
