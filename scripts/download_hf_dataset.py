from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Download a Hugging Face dataset and save it locally."
    )
    p.add_argument(
        "--dataset",
        default="Bose345/sp500_earnings_transcripts",
        help="Hugging Face dataset id.",
    )
    p.add_argument(
        "--output",
        default="data/raw/sp500_earnings_transcripts_hf",
        help="Local output directory for save_to_disk() format.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
    if isinstance(ds, DatasetDict):
        split_info = {k: len(v) for k, v in ds.items()}
    else:
        split_info = {"train": len(ds)}
    print(f"Loaded splits: {split_info}")

    print(f"Saving dataset to: {out_dir}")
    ds.save_to_disk(str(out_dir))
    print("Done.")


if __name__ == "__main__":
    main()

