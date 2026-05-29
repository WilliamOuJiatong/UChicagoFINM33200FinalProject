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
from qa_benchmark.models import (
    predict_heuristic,
    predict_tfidf_linear_svc,
    predict_tfidf_logreg,
    train_tfidf_linear_svc,
    train_tfidf_logreg,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run Q&A credibility benchmark baselines.")
    p.add_argument("--input", help="Path to gold labeled CSV.")
    p.add_argument(
        "--split-dir",
        help="Directory containing train.csv, val.csv, test.csv. If set, --input is ignored.",
    )
    p.add_argument("--output", default="outputs", help="Output directory root.")
    p.add_argument("--test-size", type=float, default=0.25, help="Test split ratio.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def load_eval_splits(args):
    if args.split_dir:
        split_dir = Path(args.split_dir)
        train_df = validate_gold_df(load_csv(split_dir / "train.csv"))
        val_path = split_dir / "val.csv"
        test_path = split_dir / "test.csv"
        if not test_path.exists():
            raise ValueError(f"Missing required split file: {test_path}")
        val_df = validate_gold_df(load_csv(val_path)) if val_path.exists() else None
        test_df = validate_gold_df(load_csv(test_path))
        return train_df, val_df, test_df, str(split_dir)

    if not args.input:
        raise ValueError("Provide either --input or --split-dir.")

    input_path = Path(args.input)
    df = validate_gold_df(load_csv(input_path))
    if len(df) < 30:
        print(
            f"Warning: only {len(df)} labeled rows. "
            "Metrics will be noisy. Target at least 80-120 rows."
        )
    train_df, test_df = split_gold(df, test_size=args.test_size, random_state=args.seed)
    return train_df, None, test_df, str(input_path)


def evaluate_model(name: str, y_true: list[str], y_pred: list[str], out_dir: Path) -> dict:
    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_as_df(y_true, y_pred)
    cm.to_csv(out_dir / f"{name}_confusion_matrix.csv")
    return metrics


def select_best_non_heuristic_model(train_df, val_df, out_dir: Path):
    c_grid = [0.25, 1.0, 4.0]
    y_val = list(val_df["label"])
    candidates = []

    for c in c_grid:
        lr_name = f"tfidf_logreg_c{c}"
        lr_model = train_tfidf_logreg(train_df, c=c)
        lr_preds = predict_tfidf_logreg(lr_model, val_df)
        lr_metrics = evaluate_model(f"val_{lr_name}", y_val, lr_preds, out_dir)
        candidates.append((lr_name, lr_model, lr_metrics, "logreg"))

        svc_name = f"tfidf_linear_svc_c{c}"
        svc_model = train_tfidf_linear_svc(train_df, c=c)
        svc_preds = predict_tfidf_linear_svc(svc_model, val_df)
        svc_metrics = evaluate_model(f"val_{svc_name}", y_val, svc_preds, out_dir)
        candidates.append((svc_name, svc_model, svc_metrics, "linear_svc"))

    candidates_sorted = sorted(
        candidates,
        key=lambda x: (x[2]["macro_f1"], x[2]["weighted_f1"], x[2]["accuracy"]),
        reverse=True,
    )
    best_name, best_model, best_metrics, best_family = candidates_sorted[0]

    all_val = {
        name: metrics
        for name, _, metrics, _ in candidates
    }
    all_val["best_model"] = {
        "name": best_name,
        "family": best_family,
    }
    return best_name, best_model, best_metrics, all_val


def main():
    args = parse_args()
    out_root = Path(args.output)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, source_ref = load_eval_splits(args)

    # Select best non-heuristic model on validation when available.
    val_results = None
    if val_df is not None and not val_df.empty:
        y_val = list(val_df["label"])
        val_heuristic_preds = predict_heuristic(val_df)
        heuristic_val_metrics = evaluate_model(
            "val_heuristic", y_val, val_heuristic_preds, out_dir
        )
        best_model_name, best_model, best_val_metrics, non_heuristic_val = (
            select_best_non_heuristic_model(train_df, val_df, out_dir)
        )
        val_results = {
            "heuristic": heuristic_val_metrics,
            **non_heuristic_val,
            "selected_non_heuristic_metrics": best_val_metrics,
        }
    else:
        best_model_name = "tfidf_logreg_c1.0"
        best_model = train_tfidf_logreg(train_df, c=1.0)

    y_test = list(test_df["label"])
    heuristic_preds = predict_heuristic(test_df)
    heuristic_metrics = evaluate_model("test_heuristic", y_test, heuristic_preds, out_dir)
    if best_model_name.startswith("tfidf_linear_svc"):
        best_preds = predict_tfidf_linear_svc(best_model, test_df)
    else:
        best_preds = predict_tfidf_logreg(best_model, test_df)
    best_metrics = evaluate_model(f"test_{best_model_name}", y_test, best_preds, out_dir)

    summary = {
        "run_id": run_id,
        "input_source": source_ref,
        "n_total": int(len(train_df) + len(test_df) + (len(val_df) if val_df is not None else 0)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)) if val_df is not None else 0,
        "n_test": int(len(test_df)),
        "selected_non_heuristic_model": best_model_name,
        "models": {
            "test": {
                "heuristic": heuristic_metrics,
                best_model_name: best_metrics,
            },
        },
    }
    if val_results is not None:
        summary["models"]["validation"] = val_results

    write_json(out_dir / "metrics_summary.json", summary)

    pred_payload = {
        "q_id": test_df["q_id"].values,
        "question": test_df["question"].values,
        "answer": test_df["answer"].values,
        "gold_label": y_test,
        "heuristic_pred": heuristic_preds,
        f"{best_model_name}_pred": best_preds,
    }
    for optional_col in (
        "company",
        "ticker",
        "date",
        "question_speaker",
        "question_role",
        "answer_speakers",
        "answer_role",
    ):
        if optional_col in test_df.columns:
            pred_payload[optional_col] = test_df[optional_col].values

    predictions = pd.DataFrame(pred_payload)
    predictions.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"Saved run artifacts to: {out_dir}")
    if val_results is not None:
        print("Validation Macro F1:")
        print(f"  heuristic:               {val_results['heuristic']['macro_f1']:.4f}")
        print(
            f"  selected_non_heuristic: {val_results['selected_non_heuristic_metrics']['macro_f1']:.4f} "
            f"({best_model_name})"
        )
    print("Test Macro F1:")
    print(f"  heuristic:   {heuristic_metrics['macro_f1']:.4f}")
    print(f"  {best_model_name}:{best_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
