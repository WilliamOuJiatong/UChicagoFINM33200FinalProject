from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .constants import LABELS


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": report,
    }


def confusion_as_df(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    return pd.DataFrame(cm, index=[f"true_{c}" for c in LABELS], columns=list(LABELS))


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

