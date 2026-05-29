# AI vs Manual Dataset Comparison

## Key Differences

- **selected_test_macro_f1 (manual - ai): -0.0000**
- **selected_test_accuracy (manual - ai): +0.2769**
- **selected-vs-heuristic macro_f1 gap on manual: +0.0500**
- **selected-vs-heuristic macro_f1 gap on ai: -0.5062**

## Single-Run Metrics

| dataset   |          run_id | run_dir                                                                      |   n_train |   n_val |   n_test | selected_non_heuristic_model   |   heuristic_test_accuracy |   heuristic_test_macro_f1 |   heuristic_test_weighted_f1 |   selected_val_macro_f1 |   selected_test_accuracy |   selected_test_macro_f1 |   selected_test_weighted_f1 |   selected_minus_heuristic_accuracy |   selected_minus_heuristic_macro_f1 |   selected_minus_heuristic_weighted_f1 |
|:----------|----------------:|:-----------------------------------------------------------------------------|----------:|--------:|---------:|:-------------------------------|--------------------------:|--------------------------:|-----------------------------:|------------------------:|-------------------------:|-------------------------:|----------------------------:|------------------------------------:|------------------------------------:|---------------------------------------:|
| ai        | 20260529_152141 | outputs/dual_dataset_compare/20260529_152137/runs/ai/run_20260529_152141     |       298 |      64 |       65 | tfidf_linear_svc_c4.0          |                  1        |                  1        |                     1        |                0.487179 |                 0.492308 |                 0.493839 |                    0.48524  |                           -0.507692 |                          -0.506161  |                              -0.51476  |
| manual    | 20260529_152146 | outputs/dual_dataset_compare/20260529_152137/runs/manual/run_20260529_152146 |       298 |      64 |       65 | tfidf_logreg_c0.25             |                  0.507692 |                  0.443792 |                     0.599285 |                0.424894 |                 0.769231 |                 0.49382  |                    0.757252 |                            0.261538 |                           0.0500287 |                               0.157966 |

## Interpretation

- Positive `manual_minus_ai` means the manually labeled set yielded stronger test metrics on this split.
- If heuristic performance is near-perfect on one label set, treat that as a label-overlap warning, not evidence of model generalization.
- These numbers are split-sensitive; confirm with repeated seeds before final claims.


## Delta (manual - ai)

| metric                    |   manual_minus_ai |
|:--------------------------|------------------:|
| selected_test_accuracy    |       0.276923    |
| selected_test_macro_f1    |      -1.84175e-05 |
| selected_test_weighted_f1 |       0.272012    |
