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

Then run:

```bash
python scripts/make_label_sheet.py \
  --input data/raw/qa_raw.csv \
  --output data/processed/qa_label_sheet.csv
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

The script runs:

1. Heuristic baseline
2. TF-IDF + Logistic Regression baseline

Outputs include:

- metrics JSON
- confusion matrices CSV
- test predictions CSV

## Notes for Course Deliverable

- Use real transcripts and keep source URLs.
- Document label policy and adjudication choices.
- Include failure-case analysis in the final write-up.
