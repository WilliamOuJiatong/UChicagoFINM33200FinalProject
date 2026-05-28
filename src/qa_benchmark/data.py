from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import GOLD_REQUIRED_COLUMNS, LABELS, RAW_REQUIRED_COLUMNS


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_raw_df(df: pd.DataFrame) -> None:
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")
    if df.empty:
        raise ValueError("Raw dataframe is empty.")


def validate_gold_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in GOLD_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required gold columns: {missing}")

    if df.empty:
        raise ValueError("Gold dataframe is empty.")

    cleaned = df.copy()
    cleaned["question"] = cleaned["question"].astype(str).str.strip()
    cleaned["answer"] = cleaned["answer"].astype(str).str.strip()
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()

    if (cleaned["question"] == "").any() or (cleaned["answer"] == "").any():
        raise ValueError("Found empty question/answer rows in gold data.")

    invalid_labels = sorted(set(cleaned["label"]) - set(LABELS))
    if invalid_labels:
        raise ValueError(
            f"Invalid labels: {invalid_labels}. Allowed labels: {list(LABELS)}"
        )

    return cleaned


def build_text_features(df: pd.DataFrame) -> pd.Series:
    return (
        "QUESTION: "
        + df["question"].astype(str).str.strip()
        + "\nANSWER: "
        + df["answer"].astype(str).str.strip()
    )


def split_gold(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    y = df["label"]
    counts = y.value_counts()
    can_stratify = (counts.min() >= 2) and (len(counts) > 1)

    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )

