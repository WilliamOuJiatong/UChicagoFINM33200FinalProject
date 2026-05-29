from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run AI-vs-manual comparison multiple times with different split seeds, "
            "then write a stability report."
        )
    )
    p.add_argument(
        "--ai-dataset",
        default="data/processed/qa_gold_ai_dataset.csv",
        help="Path to AI dataset CSV.",
    )
    p.add_argument(
        "--manual-dataset",
        default="data/processed/qa_gold_manual_dataset.csv",
        help="Path to manual dataset CSV.",
    )
    p.add_argument(
        "--output-root",
        default="outputs/stability_5x",
        help="Output root for repeated-run artifacts.",
    )
    p.add_argument(
        "--seeds",
        default="41,42,43,44,45",
        help="Comma-separated list of seeds (5 recommended).",
    )
    return p.parse_args()


def frame_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return f"```\n{df.to_string(index=False)}\n```"


def run_compare_once(cwd: Path, ai_dataset: str, manual_dataset: str, out_root: Path, seed: int):
    cmd = [
        "python3",
        "scripts/compare_ai_manual_datasets.py",
        "--ai-dataset",
        ai_dataset,
        "--manual-dataset",
        manual_dataset,
        "--output-root",
        str(out_root),
        "--seed",
        str(seed),
    ]
    subprocess.run(cmd, cwd=cwd, check=True, text=True)


def load_run_metrics(run_root: Path, seed: int) -> dict:
    report_csv = run_root / "reports" / "ai_vs_manual_metrics.csv"
    if not report_csv.exists():
        raise FileNotFoundError(f"Missing report file: {report_csv}")
    df = pd.read_csv(report_csv)
    ai_rows = df[df["dataset"] == "ai"]
    manual_rows = df[df["dataset"] == "manual"]
    if ai_rows.empty or manual_rows.empty:
        raise ValueError(f"Expected both ai and manual rows in: {report_csv}")
    ai = ai_rows.iloc[0]
    manual = manual_rows.iloc[0]

    return {
        "seed": seed,
        "run_root": str(run_root),
        "ai_selected_model": ai["selected_non_heuristic_model"],
        "manual_selected_model": manual["selected_non_heuristic_model"],
        "ai_selected_test_accuracy": float(ai["selected_test_accuracy"]),
        "ai_selected_test_macro_f1": float(ai["selected_test_macro_f1"]),
        "ai_selected_test_weighted_f1": float(ai["selected_test_weighted_f1"]),
        "manual_selected_test_accuracy": float(manual["selected_test_accuracy"]),
        "manual_selected_test_macro_f1": float(manual["selected_test_macro_f1"]),
        "manual_selected_test_weighted_f1": float(manual["selected_test_weighted_f1"]),
        "manual_minus_ai_accuracy": float(manual["selected_test_accuracy"]) - float(ai["selected_test_accuracy"]),
        "manual_minus_ai_macro_f1": float(manual["selected_test_macro_f1"]) - float(ai["selected_test_macro_f1"]),
        "manual_minus_ai_weighted_f1": float(manual["selected_test_weighted_f1"])
        - float(ai["selected_test_weighted_f1"]),
    }


def metric_summary(series: pd.Series) -> dict:
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
        "n_unique": int(series.nunique()),
    }


def main():
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) != 5:
        raise ValueError("Please provide exactly 5 seeds for this analysis.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = Path(args.output_root) / ts
    base_root.mkdir(parents=True, exist_ok=True)

    cwd = Path(__file__).resolve().parents[1]
    rows = []
    for seed in seeds:
        seed_root = base_root / f"seed_{seed}"
        run_compare_once(cwd, args.ai_dataset, args.manual_dataset, seed_root, seed)

        # compare script creates subfolder under output root; pick latest
        compare_roots = sorted(seed_root.glob("*"), key=lambda p: p.stat().st_mtime)
        if not compare_roots:
            raise ValueError(f"No compare run produced under {seed_root}")
        compare_root = compare_roots[-1]
        rows.append(load_run_metrics(compare_root, seed))

    results_df = pd.DataFrame(rows).sort_values("seed")
    results_csv = base_root / "five_run_metrics.csv"
    results_df.to_csv(results_csv, index=False)

    tracked = [
        "ai_selected_test_macro_f1",
        "manual_selected_test_macro_f1",
        "manual_minus_ai_macro_f1",
        "ai_selected_test_accuracy",
        "manual_selected_test_accuracy",
        "manual_minus_ai_accuracy",
    ]
    summaries = {m: metric_summary(results_df[m]) for m in tracked}
    summary_df = pd.DataFrame(
        [{"metric": m, **s} for m, s in summaries.items()]
    ).sort_values("metric")
    summary_csv = base_root / "five_run_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    any_identical = any(v["n_unique"] == 1 for v in summaries.values())
    all_identical = all(v["n_unique"] == 1 for v in summaries.values())

    if all_identical:
        verdict = (
            "All tracked metrics are identical across 5 runs. "
            "That is unusual for split-variance testing and should be investigated."
        )
    elif any_identical:
        verdict = (
            "Some metrics are identical while others vary. "
            "This can happen, but the fixed metrics should be checked for ceiling effects or overlap."
        )
    else:
        verdict = (
            "Tracked metrics vary across all 5 runs. "
            "This is expected and indicates normal split sensitivity."
        )

    report_md = base_root / "five_run_interpretation_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Five-Run Stability Report\n\n")
        f.write(f"Seeds: `{', '.join(str(s) for s in seeds)}`\n\n")
        f.write(f"## Verdict\n\n**{verdict}**\n\n")
        f.write("## Per-Run Metrics\n\n")
        f.write(frame_to_markdown(results_df))
        f.write("\n\n## Metric Stability Summary\n\n")
        f.write(frame_to_markdown(summary_df))
        f.write("\n\n## Interpretation\n\n")
        f.write(
            "- When `manual_minus_ai_* > 0`, the manual dataset is producing stronger selected-model test performance than the AI dataset.\n"
        )
        f.write(
            "- Non-zero standard deviation across runs confirms that results move with split composition, which is normal.\n"
        )
        f.write(
            "- If heuristic stays near-perfect on AI-labeled data, treat that as label overlap leakage rather than true model quality.\n"
        )

    print(f"Wrote: {results_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {report_md}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
