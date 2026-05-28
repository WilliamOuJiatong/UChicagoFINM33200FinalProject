from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build concise error reports from test predictions.")
    p.add_argument(
        "--predictions",
        required=True,
        help="Path to test_predictions.csv from a benchmark run.",
    )
    p.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for error report files.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Max error rows to export per model.",
    )
    return p.parse_args()


def _export_model_errors(df: pd.DataFrame, model_col: str, top_n: int, output_path: Path):
    errors = df[df["gold_label"] != df[model_col]].copy()
    if errors.empty:
        errors.to_csv(output_path, index=False)
        return 0

    keep_cols = [
        "q_id",
        "company",
        "ticker",
        "date",
        "question_speaker",
        "answer_speakers",
        "gold_label",
        model_col,
        "question",
        "answer",
    ]
    final_cols = [c for c in keep_cols if c in errors.columns]
    errors = errors[final_cols].head(top_n)
    errors.to_csv(output_path, index=False)
    return len(errors)


def main():
    args = parse_args()
    pred_path = Path(args.predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    expected = {"gold_label", "heuristic_pred", "tfidf_logreg_pred"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions file: {sorted(missing)}")

    summary_rows = []
    for model_col in ("heuristic_pred", "tfidf_logreg_pred"):
        out_file = out_dir / f"errors_{model_col}.csv"
        exported = _export_model_errors(df, model_col, args.top_n, out_file)
        total_errors = int((df["gold_label"] != df[model_col]).sum())
        summary_rows.append(
            {
                "model": model_col,
                "total_errors": total_errors,
                "exported_rows": exported,
                "file": str(out_file),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "error_report_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

