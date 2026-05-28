# Final Deliverables Checklist

## Engineering

- [ ] `scripts/create_splits.py` used to create fixed `train/val/test` files.
- [ ] `scripts/run_benchmark.py` run on fixed splits, with run artifacts saved under `outputs/run_*`.
- [ ] `scripts/build_error_report.py` run to export top model failures.

## Quantitative Evidence

- [ ] Final test metrics table (accuracy, macro-F1, weighted-F1) for each model.
- [ ] Confusion matrix plots/tables for test split.
- [ ] Error taxonomy from misclassified cases (at least 3 recurring patterns).

## Reproducibility

- [ ] README commands work from clean environment.
- [ ] Input schema and label policy documented.
- [ ] Data provenance and source links included.

## Communication

- [ ] 1-page project summary (problem, method, results, limits, next step).
- [ ] Slide with benchmark design choices and tradeoffs.
- [ ] Slide with failure cases and why errors happen.

