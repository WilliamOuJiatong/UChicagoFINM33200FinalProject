# FINM 33200 Final Report

## Project Title
Earnings-Call Q&A Credibility Benchmark: Direct vs. Partial vs. Evasive Management Answers

## Executive Summary
This project builds a practical benchmark to classify management answers in earnings-call Q&A into `direct`, `partial`, or `evasive`.

The core finding is that label quality is the main performance driver. Across repeated runs, models trained/evaluated on the manual-rule dataset consistently outperform the AI-only label set on accuracy, while macro-F1 improves modestly on average.

The benchmark is operational and reproducible. It includes synchronized data splits, multiple baseline models, side-by-side dataset comparisons, and repeated-seed stability testing.

## Problem and Motivation
Investors and analysts rely on earnings-call Q&A to evaluate management transparency. Manual reading is slow and subjective. The project goal is to build a repeatable system that scores answer quality and highlights potential evasiveness at scale.

## Data and Labeling
- Source: S&P 500 earnings-call transcripts (Hugging Face dataset, locally subsetted and processed).
- Unit of analysis: one analyst question paired with one management response block.
- Labels: `direct`, `partial`, `evasive`.
- Gold sets used in experiments:
  - `qa_gold_ai_dataset.csv` (AI-judgment labels)
  - `qa_gold_manual_dataset.csv` (manual-rule labels aligned to explicit annotation principles)

## Modeling Approach
Three baselines are evaluated:
1. Heuristic rule-based classifier.
2. TF-IDF + Logistic Regression.
3. TF-IDF + Linear SVC.

Model selection is validation-driven:
- Candidates: Logistic Regression and Linear SVC.
- Hyperparameter grid: `C in {0.25, 1.0, 4.0}`.
- Selection metric: validation macro-F1.

## Experimental Design
- Synchronized split protocol across AI and manual datasets so both use the same train/val/test membership.
- Split sizes: 70% train / 15% validation / 15% test.
- Single-run comparison plus five-run stability analysis (`seeds = 41, 42, 43, 44, 45`).

## Main Results

### 1) Single-Run Comparison (seed 42)
From `outputs/dual_dataset_compare/20260529_152137/reports/`:

| Dataset | Selected Model | Test Accuracy | Test Macro-F1 |
|---|---|---:|---:|
| AI labels | `tfidf_linear_svc_c4.0` | 0.4923 | 0.4938 |
| Manual labels | `tfidf_logreg_c0.25` | 0.7692 | 0.4938 |

Key point: macro-F1 is nearly identical in this run, but manual labels produce much higher accuracy.

### 2) Five-Run Stability Summary
From `outputs/stability_5x/20260529_151843/five_run_summary.csv`:

| Metric | Mean | Std |
|---|---:|---:|
| AI selected model test accuracy | 0.4492 | 0.0461 |
| Manual selected model test accuracy | 0.7662 | 0.0265 |
| Manual - AI accuracy gap | +0.3169 | 0.0671 |
| AI selected model test macro-F1 | 0.4467 | 0.0466 |
| Manual selected model test macro-F1 | 0.4859 | 0.0278 |
| Manual - AI macro-F1 gap | +0.0392 | 0.0687 |

Interpretation:
- Results are not identical across seeds, which is expected and healthy.
- Manual-rule labels are consistently stronger on accuracy and modestly stronger on macro-F1 on average.

### 3) Heuristic Baseline Behavior
On the AI-labeled set, heuristic scores can become unrealistically high (including perfect values in some runs). This is treated as overlap leakage between heuristic rules and label construction, not as evidence of true generalization.

## What This Means
For an audience using this tool in practice:
- The pipeline can reliably produce a credibility signal for management answers.
- The quality of labeling policy matters more than model complexity at this stage.
- A simple linear text model with disciplined labels is a defensible baseline for this task.

## Limitations
- Dataset size is still moderate for three-class text classification.
- Current features are lexical (TF-IDF), so deeper discourse logic is only partially captured.
- Annotation uncertainty remains for borderline `partial` vs. `evasive` cases.

## Next Build Priorities
1. Expand the manually reviewed gold set.
2. Add confidence calibration and thresholding for analyst-facing use.
3. Add error taxonomy tags (topic pivot, numeric non-answer, hedge-heavy response) for interpretability.
4. Evaluate a lightweight transformer baseline against the current linear baselines.

## Reproducibility and Artifacts
Core runnable assets in repo:
- Datasets:
  - `data/processed/qa_gold_ai_dataset.csv`
  - `data/processed/qa_gold_manual_dataset.csv`
- Comparison outputs:
  - `outputs/dual_dataset_compare/20260529_152137/reports/`
- Stability outputs:
  - `outputs/stability_5x/20260529_151843/`
- Aggregated summary:
  - `outputs/summary/`

This report reflects the current committed code and outputs on `main` as of May 29, 2026.
