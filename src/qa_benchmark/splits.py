from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_three_way_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    y = df["label"]
    counts = y.value_counts()
    can_stratify = (counts.min() >= 3) and (len(counts) > 1)
    strat = y if can_stratify else None

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=strat,
    )

    # Split remaining pool into val/test with relative ratio.
    rel_val = val_size / (val_size + test_size)
    temp_y = temp_df["label"]
    temp_counts = temp_y.value_counts()
    temp_can_stratify = (temp_counts.min() >= 2) and (len(temp_counts) > 1)
    temp_strat = temp_y if temp_can_stratify else None

    val_df, test_df = train_test_split(
        temp_df,
        train_size=rel_val,
        random_state=random_state,
        stratify=temp_strat,
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


def write_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str | Path
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)

