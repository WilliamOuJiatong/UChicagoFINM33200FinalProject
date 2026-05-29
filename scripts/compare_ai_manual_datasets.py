from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CORE_REQUIRED_COLUMNS = [
    "q_id",
    "question",
    "answer",
    "label",
]
METADATA_COLUMNS = [
    "company",
    "ticker",
    "date",
    "source_url",
    "question_speaker",
    "question_role",
    "answer_speakers",
    "answer_role",
]
VALID_LABELS = {"direct", "partial", "evasive"}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run synchronized train/val/test experiments for AI-labeled and "
            "manual-weighted datasets, then generate a comparison report."
        )
    )
    p.add_argument(
        "--ai-dataset",
        default="data/processed/qa_gold_ai_dataset.csv",
        help="Path to AI-judgment gold dataset CSV.",
    )
    p.add_argument(
        "--manual-dataset",
        default="data/processed/qa_gold_manual_dataset.csv",
        help="Path to manual-weighted gold dataset CSV.",
    )
    p.add_argument(
        "--output-root",
        default="outputs/dual_dataset_compare",
        help="Root directory for comparison artifacts.",
    )
    p.add_argument("--train-size", type=float, default=0.7)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def frame_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return f"```\n{df.to_string(index=False)}\n```"


def sanitize_dataset(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in CORE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")

    keep_cols = CORE_REQUIRED_COLUMNS + [c for c in METADATA_COLUMNS if c in df.columns]
    df = df[keep_cols].copy()

    for c in ["question", "answer", "label", "q_id"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["label"] = df["label"].str.lower()
    df = df[df["label"].isin(VALID_LABELS)].copy()
    df = df[(df["q_id"] != "") & (df["question"] != "") & (df["answer"] != "")]

    duplicated = df[df["q_id"].duplicated(keep=False)]
    if not duplicated.empty:
        sample_qids = duplicated["q_id"].head(10).tolist()
        raise ValueError(
            f"{name}: found duplicate q_id values (showing up to 10): {sample_qids}"
        )

    if df.empty:
        raise ValueError(f"{name}: no valid rows after cleaning.")

    missing_labels = sorted(VALID_LABELS.difference(set(df["label"])))
    if missing_labels:
        print(f"Warning: {name} has no examples for labels: {missing_labels}")

    return df


def can_stratify(labels: list[str]) -> bool:
    counts = pd.Series(labels).value_counts()
    return (len(counts) > 1) and (counts.min() >= 2)


def build_split_strata(ai_df: pd.DataFrame, manual_df: pd.DataFrame) -> dict[str, str]:
    ai_map = ai_df.set_index("q_id")["label"].to_dict()
    manual_map = manual_df.set_index("q_id")["label"].to_dict()
    strata: dict[str, str] = {}
    for q_id, ai_label in ai_map.items():
        manual_label = manual_map[q_id]
        if ai_label == manual_label:
            strata[q_id] = f"agree:{ai_label}"
        else:
            strata[q_id] = "disagree"
    return strata


def build_synchronized_split_map(
    q_ids: list[str],
    strata_by_qid: dict[str, str],
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, str]:
    if abs((train_size + val_size + test_size) - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    ids = sorted(q_ids)
    strata = [strata_by_qid[q_id] for q_id in ids]
    strat = strata if can_stratify(strata) else None

    train_ids, temp_ids, _, temp_strata = train_test_split(
        ids,
        strata,
        train_size=train_size,
        random_state=seed,
        stratify=strat,
    )

    rel_val = val_size / (val_size + test_size)
    temp_strat = temp_strata if can_stratify(temp_strata) else None
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=rel_val,
        random_state=seed,
        stratify=temp_strat,
    )

    split_map: dict[str, str] = {}
    for q_id in train_ids:
        split_map[q_id] = "train"
    for q_id in val_ids:
        split_map[q_id] = "val"
    for q_id in test_ids:
        split_map[q_id] = "test"
    return split_map


def write_split_files(df: pd.DataFrame, split_map: dict[str, str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = df["q_id"].map(split_map)
    train_df = df[tag == "train"].copy().sort_values("q_id")
    val_df = df[tag == "val"].copy().sort_values("q_id")
    test_df = df[tag == "test"].copy().sort_values("q_id")
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


def run_cmd(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)
    return proc.stdout


def latest_run_dir(run_root: Path) -> Path:
    candidates = sorted(run_root.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise ValueError(f"No run_* directory found under {run_root}")
    return candidates[-1]


def load_metrics(run_dir: Path) -> dict:
    with open(run_dir / "metrics_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def extract_result_row(dataset_name: str, run_dir: Path, metrics: dict) -> dict:
    selected = metrics.get("selected_non_heuristic_model", "")
    test_models = metrics.get("models", {}).get("test", {})
    heuristic = test_models.get("heuristic", {})
    selected_metrics = test_models.get(selected, {})
    val_models = metrics.get("models", {}).get("validation", {})
    selected_val = val_models.get("selected_non_heuristic_metrics", {})

    return {
        "dataset": dataset_name,
        "run_id": metrics.get("run_id"),
        "run_dir": str(run_dir),
        "n_train": metrics.get("n_train"),
        "n_val": metrics.get("n_val"),
        "n_test": metrics.get("n_test"),
        "selected_non_heuristic_model": selected,
        "heuristic_test_accuracy": heuristic.get("accuracy"),
        "heuristic_test_macro_f1": heuristic.get("macro_f1"),
        "heuristic_test_weighted_f1": heuristic.get("weighted_f1"),
        "selected_val_macro_f1": selected_val.get("macro_f1"),
        "selected_test_accuracy": selected_metrics.get("accuracy"),
        "selected_test_macro_f1": selected_metrics.get("macro_f1"),
        "selected_test_weighted_f1": selected_metrics.get("weighted_f1"),
        "selected_minus_heuristic_accuracy": (
            (selected_metrics.get("accuracy") or 0.0) - (heuristic.get("accuracy") or 0.0)
        ),
        "selected_minus_heuristic_macro_f1": (
            (selected_metrics.get("macro_f1") or 0.0) - (heuristic.get("macro_f1") or 0.0)
        ),
        "selected_minus_heuristic_weighted_f1": (
            (selected_metrics.get("weighted_f1") or 0.0)
            - (heuristic.get("weighted_f1") or 0.0)
        ),
    }


def build_single_run_rows(compare_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in compare_df.iterrows():
        dataset = row["dataset"]
        selected_model = row["selected_non_heuristic_model"]
        rows.append(
            {
                "dataset": dataset,
                "model_slot": "heuristic",
                "model_name": "heuristic",
                "accuracy": row["heuristic_test_accuracy"],
                "macro_f1": row["heuristic_test_macro_f1"],
                "weighted_f1": row["heuristic_test_weighted_f1"],
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "model_slot": "selected_non_heuristic",
                "model_name": selected_model,
                "accuracy": row["selected_test_accuracy"],
                "macro_f1": row["selected_test_macro_f1"],
                "weighted_f1": row["selected_test_weighted_f1"],
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    root = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    splits_root = root / "splits"
    runs_root = root / "runs"
    reports_root = root / "reports"
    cleaned_root = root / "cleaned"
    for p in (splits_root, runs_root, reports_root, cleaned_root):
        p.mkdir(parents=True, exist_ok=True)

    ai_df = sanitize_dataset(Path(args.ai_dataset), "ai_dataset")
    manual_df = sanitize_dataset(Path(args.manual_dataset), "manual_dataset")

    # Ensure both datasets are aligned on the same q_id universe.
    ai_ids = set(ai_df["q_id"])
    manual_ids = set(manual_df["q_id"])
    common_ids = sorted(ai_ids.intersection(manual_ids))
    if not common_ids:
        raise ValueError("No overlapping q_id values between datasets.")
    if len(common_ids) < min(len(ai_ids), len(manual_ids)):
        print(
            f"Warning: using overlap only (ai={len(ai_ids)}, manual={len(manual_ids)}, common={len(common_ids)})."
        )

    ai_df = ai_df[ai_df["q_id"].isin(common_ids)].copy()
    manual_df = manual_df[manual_df["q_id"].isin(common_ids)].copy()
    strata_by_qid = build_split_strata(ai_df, manual_df)

    split_map = build_synchronized_split_map(
        common_ids,
        strata_by_qid=strata_by_qid,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    datasets = {"ai": ai_df, "manual": manual_df}
    result_rows = []
    cwd = Path(__file__).resolve().parents[1]

    for name, df in datasets.items():
        # Save cleaned dataset copy for traceability.
        cleaned_path = cleaned_root / f"qa_gold_{name}_clean.csv"
        df.to_csv(cleaned_path, index=False)

        split_dir = splits_root / name
        write_split_files(df, split_map, split_dir)

        dataset_run_root = runs_root / name
        dataset_run_root.mkdir(parents=True, exist_ok=True)

        run_cmd(
            [
                "python3",
                "scripts/run_benchmark.py",
                "--split-dir",
                str(split_dir),
                "--output",
                str(dataset_run_root),
            ],
            cwd=cwd,
        )
        run_dir = latest_run_dir(dataset_run_root)

        run_cmd(
            [
                "python3",
                "scripts/build_error_report.py",
                "--predictions",
                str(run_dir / "test_predictions.csv"),
                "--output-dir",
                str(run_dir),
                "--top-n",
                "25",
            ],
            cwd=cwd,
        )

        metrics = load_metrics(run_dir)
        result_rows.append(extract_result_row(name, run_dir, metrics))

    compare_df = pd.DataFrame(result_rows).sort_values("dataset").reset_index(drop=True)
    single_run_df = build_single_run_rows(compare_df)
    single_run_csv = reports_root / "single_run_metrics.csv"
    single_run_df.to_csv(single_run_csv, index=False)
    compare_csv = reports_root / "ai_vs_manual_metrics.csv"
    compare_df.to_csv(compare_csv, index=False)

    # Simple delta table: manual - ai on selected model metrics.
    if {"ai", "manual"}.issubset(set(compare_df["dataset"])):
        ai_row = compare_df[compare_df["dataset"] == "ai"].iloc[0]
        manual_row = compare_df[compare_df["dataset"] == "manual"].iloc[0]
        delta_df = pd.DataFrame(
            [
                {
                    "metric": "selected_test_accuracy",
                    "manual_minus_ai": manual_row["selected_test_accuracy"]
                    - ai_row["selected_test_accuracy"],
                },
                {
                    "metric": "selected_test_macro_f1",
                    "manual_minus_ai": manual_row["selected_test_macro_f1"]
                    - ai_row["selected_test_macro_f1"],
                },
                {
                    "metric": "selected_test_weighted_f1",
                    "manual_minus_ai": manual_row["selected_test_weighted_f1"]
                    - ai_row["selected_test_weighted_f1"],
                },
            ]
        )
        key_findings = [
            (
                "selected_test_macro_f1 (manual - ai)",
                manual_row["selected_test_macro_f1"] - ai_row["selected_test_macro_f1"],
            ),
            (
                "selected_test_accuracy (manual - ai)",
                manual_row["selected_test_accuracy"] - ai_row["selected_test_accuracy"],
            ),
            (
                "selected-vs-heuristic macro_f1 gap on manual",
                manual_row["selected_minus_heuristic_macro_f1"],
            ),
            (
                "selected-vs-heuristic macro_f1 gap on ai",
                ai_row["selected_minus_heuristic_macro_f1"],
            ),
        ]
    else:
        delta_df = pd.DataFrame(columns=["metric", "manual_minus_ai"])
        key_findings = []

    delta_csv = reports_root / "ai_vs_manual_deltas.csv"
    delta_df.to_csv(delta_csv, index=False)

    md_path = reports_root / "ai_vs_manual_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AI vs Manual Dataset Comparison\n\n")
        f.write("## Key Differences\n\n")
        if key_findings:
            for name, value in key_findings:
                sign = "+" if value >= 0 else ""
                f.write(f"- **{name}: {sign}{value:.4f}**\n")
        else:
            f.write("- **Key gap calculation unavailable.**\n")
        f.write("\n")

        f.write("## Single-Run Metrics\n\n")
        f.write(frame_to_markdown(compare_df))
        f.write("\n\n")

        f.write("## Interpretation\n\n")
        f.write(
            "- Positive `manual_minus_ai` means the manually labeled set yielded stronger test metrics on this split.\n"
        )
        f.write(
            "- If heuristic performance is near-perfect on one label set, treat that as a label-overlap warning, not evidence of model generalization.\n"
        )
        f.write(
            "- These numbers are split-sensitive; confirm with repeated seeds before final claims.\n"
        )
        f.write("\n\n## Delta (manual - ai)\n\n")
        if len(delta_df) > 0:
            f.write(frame_to_markdown(delta_df))
        else:
            f.write("Not available.\n")
        f.write("\n")

    summary_json = reports_root / "comparison_summary.json"
    payload = {
        "comparison_root": str(root),
        "single_run_metrics_csv": str(single_run_csv),
        "ai_vs_manual_metrics_csv": str(compare_csv),
        "ai_vs_manual_deltas_csv": str(delta_csv),
        "report_md": str(md_path),
        "key_findings": [
            {"metric": k, "value": float(v)} for k, v in key_findings
        ],
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Comparison root: {root}")
    print(f"Wrote: {single_run_csv}")
    print(f"Wrote: {compare_csv}")
    print(f"Wrote: {delta_csv}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {summary_json}")
    print("\nSummary:")
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()
