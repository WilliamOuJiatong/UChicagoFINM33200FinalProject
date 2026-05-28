# FINM 33200 Final Project

## Project: Earnings Call Q&A Credibility Benchmark

This repo builds a benchmark that classifies management answers in earnings-call Q&A as:

- `direct`
- `partial`
- `evasive`

The goal is to evaluate how reliably different approaches can detect answer quality.

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

The script runs:

1. Heuristic baseline
2. TF-IDF + Logistic Regression baseline

Outputs include:

- metrics JSON
- validation/test confusion matrices CSV
- test predictions CSV

## Create Fixed Train/Val/Test Splits

After labels are ready, generate reproducible split files:

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

From a benchmark run directory:

```bash
python scripts/build_error_report.py \
  --predictions outputs/run_YYYYMMDD_HHMMSS/test_predictions.csv \
  --output-dir outputs/run_YYYYMMDD_HHMMSS \
  --top-n 25
```

## Build Cross-Run Leaderboard Tables

Aggregate all `outputs/run_*` metrics into one CSV and a Markdown leaderboard:

```bash
python scripts/summarize_runs.py \
  --runs-root outputs \
  --output-dir outputs/summary
```

## Notes for Course Deliverable

- Use real transcripts and keep source URLs.
- Document label policy and adjudication choices.
- Include failure-case analysis in the final write-up.
