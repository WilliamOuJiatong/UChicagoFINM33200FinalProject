from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qa_benchmark.data import load_csv, validate_raw_df


def parse_args():
    p = argparse.ArgumentParser(description="Create a Q&A labeling sheet from raw data.")
    p.add_argument("--input", required=True, help="Path to raw CSV.")
    p.add_argument("--output", required=True, help="Path to output label sheet CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    raw_df = load_csv(input_path)
    validate_raw_df(raw_df)

    df = raw_df.copy()
    if "q_id" not in df.columns:
        df.insert(0, "q_id", [f"Q{i:04d}" for i in range(1, len(df) + 1)])

    for col in ("label", "annotator_notes"):
        if col not in df.columns:
            df[col] = ""

    ordered = [
        "q_id",
        "company",
        "ticker",
        "date",
        "question",
        "answer",
        "source_url",
        "label",
        "annotator_notes",
    ]
    df = df[ordered]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote label sheet: {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()

