from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "JPM",
    "XOM",
    "WMT",
    "UNH",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Create a compact subset from local HF earnings transcript dataset."
    )
    p.add_argument(
        "--input",
        default="data/raw/sp500_earnings_transcripts_hf",
        help="Path to full local dataset saved via save_to_disk().",
    )
    p.add_argument(
        "--output",
        default="data/raw/sp500_subset_hf",
        help="Path for subset dataset (save_to_disk format).",
    )
    p.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated ticker list. Example: AAPL,MSFT,NVDA",
    )
    p.add_argument(
        "--year-start",
        type=int,
        default=2021,
        help="Start year (inclusive).",
    )
    p.add_argument(
        "--year-end",
        type=int,
        default=2024,
        help="End year (inclusive).",
    )
    p.add_argument(
        "--max-calls-per-company",
        type=int,
        default=8,
        help="Max transcript rows to keep per ticker, using most recent by date.",
    )
    return p.parse_args()


def parse_ticker_list(raw: str) -> list[str]:
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        raise ValueError("No tickers provided.")
    return tickers


def load_train_split(path: str) -> Dataset:
    ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        if "train" not in ds:
            raise ValueError("DatasetDict does not contain a 'train' split.")
        return ds["train"]
    return ds


def main():
    args = parse_args()
    tickers = parse_ticker_list(args.tickers)
    ticker_set = set(tickers)
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = load_train_split(str(in_path))
    bucket: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for idx, row in enumerate(train_ds):
        year = int(row["year"])
        if year < args.year_start or year > args.year_end:
            continue
        symbol = str(row["symbol"]).upper()
        if symbol not in ticker_set:
            continue
        dt = str(row["date"])
        bucket[symbol].append((dt, idx))

    selected_indices: list[int] = []
    for symbol in tickers:
        rows = bucket.get(symbol, [])
        rows.sort(key=lambda x: x[0], reverse=True)
        kept = rows[: args.max_calls_per_company]
        selected_indices.extend(idx for _, idx in kept)

    if not selected_indices:
        raise ValueError("No rows selected. Check ticker names and year range.")

    subset = train_ds.select(selected_indices)
    subset.save_to_disk(str(out_path))

    print("Subset created.")
    print(f"Input rows: {len(train_ds)}")
    print(f"Output rows: {len(subset)}")
    print(f"Tickers requested: {tickers}")
    print(
        f"Filter: years {args.year_start}-{args.year_end}, "
        f"max {args.max_calls_per_company} calls/company"
    )
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

