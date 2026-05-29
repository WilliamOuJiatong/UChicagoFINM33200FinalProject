# FINM 33200 Final Project

## Project: Earnings Call Q&A Credibility Benchmark

This repo builds a benchmark that classifies management answers in earnings-call Q&A as:

- `direct`
- `partial`
- `evasive`

The goal is to measure how reliably different approaches detect answer quality.

## Repository Layout

- `data/raw/`: raw Q&A data you collect
- `data/processed/`: cleaned and labeled benchmark files
- `data/templates/`: schema templates
- `scripts/`: runnable pipeline scripts
- `src/qa_benchmark/`: core benchmark logic
- `outputs/`: model outputs and metrics
- `docs/`: annotation guidelines

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Build Labeling Sheet from Raw Data

Prepare `data/raw/qa_raw.csv` with columns:

- `company`
- `ticker`
- `date`
- `question`
- `answer`
- `source_url`
- `question_speaker` (recommended)
- `question_role` (recommended, e.g. `analyst`)
- `answer_speakers` (recommended)
- `answer_role` (recommended, e.g. `management`)

Then run:

```bash
python scripts/make_label_sheet.py \
  --input data/raw/qa_raw.csv \
  --output data/processed/qa_label_sheet.csv
```

Optional: create a readable Excel copy for manual annotation:

```bash
python scripts/make_label_xlsx.py \
  --input data/processed/qa_label_sheet.csv \
  --output data/processed/qa_label_sheet_readable.xlsx
```

## Optional: Create a Compact Transcript Subset

If the full transcript dataset is too large, create a filtered local subset:

```bash
python scripts/create_hf_subset.py \
  --input data/raw/sp500_earnings_transcripts_hf \
  --output data/raw/sp500_subset_hf \
  --tickers AAPL,MSFT,AMZN,GOOGL,META,NVDA,JPM,XOM,WMT,UNH \
  --year-start 2021 \
  --year-end 2024 \
  --max-calls-per-company 8
```

## Extract Q&A Pairs from HF Dataset

Use the subset (or full HF dataset) to build `qa_raw.csv`:

```bash
python scripts/extract_qa_from_hf.py \
  --input data/raw/sp500_subset_hf \
  --output data/raw/qa_raw.csv
```

## 2) Create Gold Labels

Open `data/processed/qa_label_sheet.csv` and fill `label` using:

- `direct`: directly addresses the asked question with clear substance.
- `partial`: addresses part of the question but leaves key parts unresolved.
- `evasive`: avoids answering the question materially.

Guidelines are in `docs/annotation_guidelines.md`.

After labeling, save as:

- `data/processed/qa_gold.csv`

## 3) Run Baselines + Evaluation

```bash
python scripts/run_benchmark.py \
  --input data/processed/qa_gold.csv \
  --output outputs
```

Or, if you have fixed train/val/test files:

```bash
python scripts/run_benchmark.py \
  --split-dir data/processed/splits \
  --output outputs
```

Model selection is validation-based for non-heuristic models:

- candidates: `tfidf_logreg` and `tfidf_linear_svc`
- hyperparameter grid: `C = [0.25, 1.0, 4.0]`
- selection metric: validation macro-F1

The benchmark evaluates:

1. Heuristic baseline
2. TF-IDF + Logistic Regression baseline
3. TF-IDF + Linear SVC baseline

Outputs include:

- metrics JSON
- validation/test confusion matrices CSV
- test predictions CSV

## Create Fixed Train/Val/Test Splits

After labels are ready, create fixed split files:

```bash
python scripts/create_splits.py \
  --input data/processed/qa_gold.csv \
  --output data/processed/splits \
  --train-size 0.7 \
  --val-size 0.15 \
  --test-size 0.15 \
  --seed 42
```

## Generate Error-Case Reports

From a benchmark run directory, generate error reports:

```bash
python scripts/build_error_report.py \
  --predictions outputs/run_YYYYMMDD_HHMMSS/test_predictions.csv \
  --output-dir outputs/run_YYYYMMDD_HHMMSS \
  --top-n 25
```

## Build Cross-Run Leaderboard Tables

Aggregate all `outputs/run_*` metrics into one CSV and one Markdown leaderboard:

```bash
python scripts/summarize_runs.py \
  --runs-root outputs \
  --output-dir outputs/summary
```

## Compare AI vs Manual Gold Sets (One Command)

Run synchronized train/val/test experiments on both datasets and generate a side-by-side comparison:

```bash
python scripts/compare_ai_manual_datasets.py \
  --ai-dataset data/processed/qa_gold_ai_dataset.csv \
  --manual-dataset data/processed/qa_gold_manual_dataset.csv \
  --output-root outputs/dual_dataset_compare \
  --seed 42
```

Outputs:

- `.../reports/single_run_metrics.csv` (long format: dataset x model slot)
- `.../reports/ai_vs_manual_metrics.csv`
- `.../reports/ai_vs_manual_deltas.csv`
- `.../reports/ai_vs_manual_report.md`
- `.../reports/comparison_summary.json`

## Stability Check (5 Seeds)

Run 5 repeated AI-vs-manual comparisons and produce one interpretation report:

```bash
python scripts/run_stability_5x.py \
  --ai-dataset data/processed/qa_gold_ai_dataset.csv \
  --manual-dataset data/processed/qa_gold_manual_dataset.csv \
  --output-root outputs/stability_5x \
  --seeds 41,42,43,44,45
```

Outputs:

- `.../five_run_metrics.csv`
- `.../five_run_summary.csv`
- `.../five_run_interpretation_report.md`

## Notes for Course Deliverable

Five repeated runs with different split seeds (`41, 42, 43, 44, 45`) produced non-identical results. That is expected.

General interpretation from the tests:

- The manual-rule dataset is usually stronger than the AI-labeled dataset on selected non-heuristic models (higher average test macro-F1 and accuracy).
- The manual-vs-AI gap changes by split, so report mean and variance, not a single run.
- Heuristic scores on AI-labeled data can be inflated by label overlap; use those as diagnostics, not final evidence.
- The selected non-heuristic model (`tfidf_logreg` vs `tfidf_linear_svc`) can change by seed, so multi-run reporting is important.
