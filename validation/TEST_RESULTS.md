# Validation and measured results

November 1 update: the primary 10,354-row reconstruction is complete. See [RECONSTRUCTION_10354_RESULTS.md](RECONSTRUCTION_10354_RESULTS.md). Its compressed reference mean is 0.815863; its original-project-test calibrated similarity mean is 0.802624. The earlier 60,810-row result is exploratory. Expanded full-data Spark validation completed for all three models: P1 0.717141, P2 0.714662, P3 0.686073; final test evaluation of those new configurations remains pending.

Updated October 28, 2026. Runtime: Windows, Spark 3.5.9, Scala 2.12.18, JDK 11.0.20.1, local master with two worker threads. The recall experiment used a 2 GB driver heap.

## Latest full-data development update

The October 29 extended search aborted during P1 task serialization with a StackOverflowError, before any model metrics or selected settings were completed. It adds no valid scores to the tables in this report. A runtime correction enables GBT checkpoints and a 4 MB Java thread stack; the full-data search must restart in a fresh process. See `stack_overflow_fix_check.json` for the separate small-fixture runtime verification and FULL_DATA_TUNING.md for recovery instructions.

See [FULL_DATA_TUNING.md](../docs/FULL_DATA_TUNING.md) for the full-data audit, new options, commands, and completed validation comparisons. Native tree categories improved P1 internal-validation F0.5 from 0.714537 to 0.715517 on exactly the same trips; both selected depth 5 and 80 iterations. A supplemental XGBoost control retaining all fitting rows scored 0.711456 on those same validation trips. These are validation results, not replacements for the fixed-test table below. The expanded standard-feature full-data grids have completed for all three models; interaction-feature full-data searches remain pending.

Independent checks verified every supplied trip against raw CSV features, weather city/date joins, fold membership and disjoint roles, unchanged labels, prediction/confusion arithmetic, native category metadata, and model reloads. See `full_data_revision_audit.json` and `feature_semantics_audit.json`. Native-search runtime includes long system pauses and must not be used as a clean speed benchmark.

## Data and model integrity

The enriched feature builder preserves all 536,406 original training trips and 133,553 original test trips, their IDs, labels, and split membership. All trips match station metadata and daily city weather. K-means uses 70 public station coordinates, k=5, seed 42, and no trip labels. Numeric weather imputation is fitted in each model-training fold. Original input hashes remain in `data_checksums.json`.

All three initial full-feature model families completed smoke checks and native PipelineModel save/reload checks. The current full-data final experiment is complete for P1. Updated P2/P3 full-data validation searches have completed; final evaluations of the new configurations remain pending. Legacy files named `pipeline_check_*`, `tuning_check_*`, and the unprefixed classifier CV files describe earlier core-feature fixture checks, not the updated final experiments.

## Completed P1 tuning and final evaluation

The completed full-training-pool search tested four GBT candidates with three-fold CrossValidator, using inverse-frequency weights and preprocessing fitted inside each fold. It selected depth 5 and 80 boosting iterations, step size 0.1, maxBins 128, and minimum 20 instances per node. The mean CV selection score was 0.715969. CV uses the best achievable fold cutoff as a ranking diagnostic, not as an independent performance estimate.

The tuning run reserved 178,803 original training trips for internal validation, with 357,603 fitting trips. Within fitting, 54,043 were reserved for cutoff selection and 303,560 for model selection. After refitting on all fitting rows, independent internal-validation F0.5 was 0.714537. Its selected parameters are copied into `P1_selected_settings.json`.

The final experiment reused those parameters for seeds 42, 43, and 44, selected a fresh cutoff using each seed's training calibration rows, refitted on all 536,406 original training trips, and evaluated the unchanged 133,553 original test trips. Each model passed native-model reload checks. Results:

| Seed | Customer F0.5 | Precision | Recall | Accuracy |
|---:|---:|---:|---:|---:|
| 42 | 0.717689 | 0.816141 | 0.484100 | 0.903027 |
| 43 | 0.715727 | 0.847172 | 0.441635 | 0.900998 |
| 44 | 0.718365 | 0.821467 | 0.478261 | 0.902915 |

Mean Customer F0.5 is **0.717260**, with sample standard deviation **0.001370** across the three seeds. This describes seed sensitivity on a common fixed test set, not uncertainty from three independent test samples.

The earlier full-feature depth-3/20 preset obtained fixed-test F0.5 0.708569. The preceding core-feature P1 obtained 0.580092 on the same project split. These changes combine features, preprocessing, hyperparameters, and cutoff selection; no single contribution is isolated by these comparisons.

## Recall operating-point experiment

`CompareCustomerRecall` selects the highest validation F0.5 subject to a declared minimum recall, using saved training-validation predictions. It saves the policy before reading final-test results. It then changes only the native classifier cutoff in copies of the existing fitted models, preserving original models, probabilities, labels, and test IDs. No new trees or coefficients are fitted.

The example minimum validation recall is 60%. The selected cutoff was **0.785792636364**, producing validation recall 0.600175 and validation F0.5 0.688951. Those validation metrics describe cutoff selection. A validation target does not guarantee the same recall on models refitted on more data.

| Policy | Mean precision | Mean recall | Mean F0.5 | Mean accuracy |
|---|---:|---:|---:|---:|
| Original F0.5-selected cutoffs | 0.828260 | 0.467999 | 0.717260 | 0.902313 |
| Validation recall-target cutoff | 0.718446 | 0.595023 | 0.689828 | 0.900978 |

The operating-point experiment increases recall but lowers precision and F0.5. Retain the original results as the primary F0.5 comparison. These are additional analyses on a previously examined project test set, not a new independent confirmation experiment.

The independent NumPy/Arrow audit verified the constrained validation optimum, disjoint validation/test IDs, exact equality of old/new probabilities, unchanged test IDs and labels, Customer-prediction superset behavior, and confusion-count arithmetic for every test row in all three seeds. Native-model reload probes verified unchanged probabilities and the revised cutoff predictions. The audit is in `recall_operating_point_audit.json`.

## Paper-guided training-validation error analysis

The original P1 internal-validation model missed 13,672 Customer trips. Of these, **12,834 (93.9%) lasted at most 20 minutes**. Customer recall was about 19.0% for 10-20 minute trips, 83.0% for 20-30 minute trips, and above 98% for trips longer than 30 minutes. These are training-validation diagnostics, not results from selecting features on project test labels. The full grouped profile is in `P1_validation_error_profile.json`.

## Baseline and original-paper limits

The B0 protocol approximation obtained fixed-test F0.5 0.557954, retaining 60,810 training rows. It uses min-max preprocessing, width-0.03 class-aware feature cells, unweighted XGBoost, and cutoff 0.5. Its versions, sampling assumptions, and metrics remain in `B0_fixed_test_metrics.json`. The original paper reports 10,354 sampled rows and does not provide sufficient sampler, split, class-distribution, or final hyperparameter details for an exact reproduction. Quantized cells do not merge nearby points across cell boundaries; the current sampler remains an approximation.

The published chart suggests F0.5 around 0.81; that is a visual estimate from a different evaluation setup. A numerical difference from that chart cannot establish a controlled improvement. The proposal's main comparison is P1 versus a reproduced B0 on the same project test set. Baseline fidelity and its three-seed evaluation remain necessary before treating that conjecture as fully tested. Spark PR area uses default curve binning; the Python baseline uses a complete trapezoidal curve. F0.5 is the primary metric.

## Remaining experiments and runtime notes

- Final three-seed evaluation of the newly selected P1/P2/P3 configurations.
- Baseline sampler/feature fidelity review and B0 repeated-seed evaluation.
- P1's smaller-data runs at frozen settings; retain the completed 100% reference.
- Controlled feature/parameter ablations, only when justified by training-validation analysis.
- Final model comparison, runtime boundaries, and reporting.

Versioned JARs permit compilation while an earlier shell retains its JAR. Launchers read `target/current-models-jar.txt`; restart the shell after an update. A Windows temporary-JAR deletion warning after `COMPLETE` does not invalidate results that have passed persistence checks. The recall helper stops its Spark session after completion, as does the training application.

Testing uses isolated Spark 3.5.9 and Hadoop 3.3.5 helpers; system installations are unchanged. A local two-thread run does not establish a multi-machine speedup. Run histories, fitted models, older JARs, local runtime installations, and compiler scratch directories are excluded from the shareable ZIP. Full experiment outputs remain on the original computer.
