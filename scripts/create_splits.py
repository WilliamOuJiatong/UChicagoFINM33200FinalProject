from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qa_benchmark.data import load_csv, validate_gold_df
from qa_benchmark.splits import create_three_way_split, write_splits


def parse_args():
    p = argparse.ArgumentParser(description="Create reproducible train/val/test CSV splits.")
    p.add_argument("--input", required=True, help="Path to labeled gold CSV.")
    p.add_argument("--output", default="data/processed/splits", help="Output split directory.")
    p.add_argument("--train-size", type=float, default=0.7, help="Train split fraction.")
    p.add_argument("--val-size", type=float, default=0.15, help="Validation split fraction.")
    p.add_argument("--test-size", type=float, default=0.15, help="Test split fraction.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def main():
    args = parse_args()
    df = validate_gold_df(load_csv(args.input))
    train_df, val_df, test_df = create_three_way_split(
        df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
    )
    write_splits(train_df, val_df, test_df, args.output)

    print(f"Saved splits to: {args.output}")
    print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("Label distributions:")
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        print(f"\n{name}:")
        print(split_df["label"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()

