# Five-Run Stability Report

Seeds: `41, 42, 43, 44, 45`

## Verdict

**Tracked metrics vary across all 5 runs. This is expected and indicates normal split sensitivity.**

## Per-Run Metrics

|   seed | run_root                                                     | ai_selected_model     | manual_selected_model   |   ai_selected_test_accuracy |   ai_selected_test_macro_f1 |   ai_selected_test_weighted_f1 |   manual_selected_test_accuracy |   manual_selected_test_macro_f1 |   manual_selected_test_weighted_f1 |   manual_minus_ai_accuracy |   manual_minus_ai_macro_f1 |   manual_minus_ai_weighted_f1 |
|-------:|:-------------------------------------------------------------|:----------------------|:------------------------|----------------------------:|----------------------------:|-------------------------------:|--------------------------------:|--------------------------------:|-----------------------------------:|---------------------------:|---------------------------:|------------------------------:|
|     41 | outputs/stability_5x/20260529_151843/seed_41/20260529_151845 | tfidf_linear_svc_c1.0 | tfidf_logreg_c0.25      |                    0.507692 |                    0.508778 |                       0.499102 |                        0.723077 |                        0.44383  |                           0.716086 |                   0.215385 |               -0.0649485   |                      0.216985 |
|     42 | outputs/stability_5x/20260529_151843/seed_42/20260529_151859 | tfidf_linear_svc_c4.0 | tfidf_logreg_c0.25      |                    0.492308 |                    0.493839 |                       0.48524  |                        0.769231 |                        0.49382  |                           0.757252 |                   0.276923 |               -1.84175e-05 |                      0.272012 |
|     43 | outputs/stability_5x/20260529_151843/seed_43/20260529_151910 | tfidf_logreg_c1.0     | tfidf_logreg_c0.25      |                    0.384615 |                    0.391506 |                       0.360507 |                        0.8      |                        0.530415 |                           0.781429 |                   0.415385 |                0.138909    |                      0.420922 |
|     44 | outputs/stability_5x/20260529_151843/seed_44/20260529_151921 | tfidf_logreg_c1.0     | tfidf_logreg_c0.25      |                    0.446154 |                    0.431548 |                       0.425    |                        0.784615 |                        0.479263 |                           0.754626 |                   0.338462 |                0.0477151   |                      0.329626 |
|     45 | outputs/stability_5x/20260529_151843/seed_45/20260529_151933 | tfidf_linear_svc_c4.0 | tfidf_logreg_c0.25      |                    0.415385 |                    0.407912 |                       0.408365 |                        0.753846 |                        0.482092 |                           0.753377 |                   0.338462 |                0.0741797   |                      0.345012 |

## Metric Stability Summary

| metric                        |      mean |       std |        min |      max |   n_unique |
|:------------------------------|----------:|----------:|-----------:|---------:|-----------:|
| ai_selected_test_accuracy     | 0.449231  | 0.0460512 |  0.384615  | 0.507692 |          5 |
| ai_selected_test_macro_f1     | 0.446717  | 0.0465967 |  0.391506  | 0.508778 |          5 |
| manual_minus_ai_accuracy      | 0.316923  | 0.0671305 |  0.215385  | 0.415385 |          4 |
| manual_minus_ai_macro_f1      | 0.0391674 | 0.0687357 | -0.0649485 | 0.138909 |          5 |
| manual_selected_test_accuracy | 0.766154  | 0.0264687 |  0.723077  | 0.8      |          5 |
| manual_selected_test_macro_f1 | 0.485884  | 0.0278307 |  0.44383   | 0.530415 |          5 |

## Interpretation

- When `manual_minus_ai_* > 0`, the manual dataset is producing stronger selected-model test performance than the AI dataset.
- Non-zero standard deviation across runs confirms that results move with split composition, which is normal.
- If heuristic stays near-perfect on AI-labeled data, treat that as label overlap leakage rather than true model quality.
