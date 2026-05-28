from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qa_benchmark.data import load_csv, split_gold, validate_gold_df
from qa_benchmark.eval import compute_metrics, confusion_as_df, write_json
from qa_benchmark.models import predict_heuristic, predict_tfidf_logreg, train_tfidf_logreg


def parse_args():
    p = argparse.ArgumentParser(description="Run Q&A credibility benchmark baselines.")
    p.add_argument("--input", required=True, help="Path to gold labeled CSV.")
    p.add_argument("--output", default="outputs", help="Output directory root.")
    p.add_argument("--test-size", type=float, default=0.25, help="Test split ratio.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_root = Path(args.output)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = validate_gold_df(load_csv(input_path))
    if len(df) < 30:
        print(
            f"Warning: only {len(df)} labeled rows. "
            "Metrics will be noisy. Target at least 80-120 rows."
        )

    train_df, test_df = split_gold(df, test_size=args.test_size, random_state=args.seed)

    y_test = list(test_df["label"])

    heuristic_preds = predict_heuristic(test_df)
    heuristic_metrics = compute_metrics(y_test, heuristic_preds)
    heuristic_cm = confusion_as_df(y_test, heuristic_preds)

    logreg_model = train_tfidf_logreg(train_df)
    logreg_preds = predict_tfidf_logreg(logreg_model, test_df)
    logreg_metrics = compute_metrics(y_test, logreg_preds)
    logreg_cm = confusion_as_df(y_test, logreg_preds)

    summary = {
        "run_id": run_id,
        "input_file": str(input_path),
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "models": {
            "heuristic": heuristic_metrics,
            "tfidf_logreg": logreg_metrics,
        },
    }
    write_json(out_dir / "metrics_summary.json", summary)

    heuristic_cm.to_csv(out_dir / "heuristic_confusion_matrix.csv")
    logreg_cm.to_csv(out_dir / "tfidf_logreg_confusion_matrix.csv")

    predictions = pd.DataFrame(
        {
            "q_id": test_df["q_id"].values,
            "question": test_df["question"].values,
            "answer": test_df["answer"].values,
            "gold_label": y_test,
            "heuristic_pred": heuristic_preds,
            "tfidf_logreg_pred": logreg_preds,
        }
    )
    predictions.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"Saved run artifacts to: {out_dir}")
    print("Macro F1:")
    print(f"  heuristic:   {heuristic_metrics['macro_f1']:.4f}")
    print(f"  tfidf_logreg:{logreg_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

