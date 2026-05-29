from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate benchmark run metrics into presentation-ready tables."
    )
    p.add_argument(
        "--runs-root",
        default="outputs",
        help="Directory containing run_* folders with metrics_summary.json.",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/summary",
        help="Directory to write consolidated summary files.",
    )
    return p.parse_args()


def _read_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _frame_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return f"```\n{df.to_string(index=False)}\n```"


def _extract_row(run_dir: Path, payload: dict, split_name: str, model_name: str, model_metrics: dict) -> dict:
    return {
        "run_id": payload.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "input_source": payload.get("input_source", payload.get("input_file", "")),
        "n_total": payload.get("n_total"),
        "n_train": payload.get("n_train"),
        "n_val": payload.get("n_val", 0),
        "n_test": payload.get("n_test"),
        "split": split_name,
        "model": model_name,
        "accuracy": model_metrics.get("accuracy"),
        "macro_f1": model_metrics.get("macro_f1"),
        "weighted_f1": model_metrics.get("weighted_f1"),
    }


def _is_metric_dict(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    return {"accuracy", "macro_f1", "weighted_f1"}.issubset(set(obj.keys()))


def collect_rows(runs_root: Path) -> list[dict]:
    rows: list[dict] = []
    metrics_paths = sorted(runs_root.rglob("metrics_summary.json"))
    for metrics_path in metrics_paths:
        run_dir = metrics_path.parent
        payload = _read_metrics(metrics_path)
        models = payload.get("models", {})

        test_models = models.get("test", {})
        for model_name, m in test_models.items():
            rows.append(_extract_row(run_dir, payload, "test", model_name, m))

        val_models = models.get("validation", {})
        if _is_metric_dict(val_models.get("heuristic")):
            rows.append(
                _extract_row(run_dir, payload, "validation", "heuristic", val_models["heuristic"])
            )
        if _is_metric_dict(val_models.get("selected_non_heuristic_metrics")):
            rows.append(
                _extract_row(
                    run_dir,
                    payload,
                    "validation",
                    "selected_non_heuristic",
                    val_models["selected_non_heuristic_metrics"],
                )
            )
        for model_name, m in val_models.items():
            if model_name in {"heuristic", "selected_non_heuristic_metrics", "best_model"}:
                continue
            if _is_metric_dict(m):
                rows.append(_extract_row(run_dir, payload, "validation", model_name, m))
    return rows


def build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        return test_df
    leaderboard = test_df.sort_values(by=["macro_f1", "accuracy"], ascending=False)
    return leaderboard[
        [
            "run_id",
            "model",
            "macro_f1",
            "accuracy",
            "weighted_f1",
            "n_train",
            "n_test",
            "input_source",
            "run_dir",
        ]
    ]


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(runs_root)
    if not rows:
        raise ValueError(f"No run metrics found under: {runs_root}")

    all_df = pd.DataFrame(rows)
    all_df = all_df.sort_values(by=["run_id", "split", "model"])
    all_path = out_dir / "all_run_metrics.csv"
    all_df.to_csv(all_path, index=False)

    leaderboard_df = build_leaderboard(all_df)
    leaderboard_csv = out_dir / "leaderboard_test.csv"
    leaderboard_df.to_csv(leaderboard_csv, index=False)

    leaderboard_md = out_dir / "leaderboard_test.md"
    with open(leaderboard_md, "w", encoding="utf-8") as f:
        f.write("# Test Leaderboard\n\n")
        f.write(_frame_to_markdown(leaderboard_df))
        f.write("\n")

    print(f"Wrote: {all_path}")
    print(f"Wrote: {leaderboard_csv}")
    print(f"Wrote: {leaderboard_md}")
    print("\nTop test rows:")
    print(leaderboard_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
