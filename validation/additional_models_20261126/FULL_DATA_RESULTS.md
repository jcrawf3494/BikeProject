# Full-data supplemental model results

All twelve runs completed: three seeds for each of FM, quantile-binned Naive Bayes, EBM and CatBoost. Each final fit used all 536,406 original training trips and evaluated the same 133,553 test trips with the same 39 source inputs. Independent saved-prediction checks passed all 12 outcomes. Input hashes and cross-family calibration memberships were also checked.

| Model | Mean Customer F0.5 | Sample SD | Precision | Recall | Difference from P1 mean |
|---|---:|---:|---:|---:|---:|
| P4 FM | 0.874431 | 0.000085 | 95.73% | 64.95% | -0.017386 |
| P5 Quantile-binned Naive Bayes | 0.773744 | 0.000623 | 89.55% | 50.20% | -0.118074 |
| P6 EBM | 0.886242 | 0.000179 | 95.61% | 68.57% | -0.005575 |
| P7 CatBoost | 0.888252 | 0.001177 | 95.96% | 68.53% | -0.003566 |

The existing P1 reference mean is 0.891818; its three saved outcomes were independently rechecked against the same full split, feature list and calibration memberships. P1 remains the highest observed score in this comparison. CatBoost is the strongest supplemental preset. These results do not establish that an optimally tuned model from either family would win.

These are predeclared preset runs, not new training-CV selections. FM used factor size 8, regularization 0.001, step size 0.05 and 100 iterations, without fitting weights. NB used 16-quantile numeric binning and smoothing 1. EBM used 8 interactions, 500 rounds, learning rate 0.03, 128 bins and minimum 20 samples per leaf. CatBoost used 400 trees, depth 6, learning rate 0.05 and L2 leaf regularization 3. NB, EBM and CatBoost used inverse-frequency fitting weights. Every cutoff was selected on reserved training calibration rows before the final full-data refit.

The pasted execution log contains Windows NativeIO path warnings, locked temporary-JAR cleanup errors, and joblib/loky resource-tracker KeyErrors. The main execution continued and completed all model records. Saved memberships, predictions, cutoffs, confusion counts, F0.5, inverse weights and persistence attestations passed the independent audit. The results need not be discarded because of those log messages. Repeated quantiles are expected for binary or low-cardinality inputs and produce fewer distinct NB bins.

The next performance experiment is a bounded training-only search for CatBoost and EBM, followed by frozen-parameter three-seed evaluation. The existing test has already been examined, so later scores remain benchmark evaluations. Raw timing is retained, but no controlled speedup or fresh confirmation is claimed. Historical V1 weather remains in use.

The persisted CatBoost models for all three seeds use bootstrap_type=MVS and subsample approximately 0.8. All 536,406 original training rows were supplied to fitting; MVS samples internally during boosting iterations. P1 uses subsamplingRate=1.0. These are not identical per-iteration sampling settings. See catboost_effective_sampling.json for the saved-model parameter check.
