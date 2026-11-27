# Scope and comparison methods

Execution examples run from the repository root. Restore private data and local execution artifacts as described in [data preparation](../data/README.md); row-level inputs and fitted binaries are not committed.


The research question and model families are unchanged: Customer-versus-Subscriber classification with Spark GBT, random forest, and logistic regression, compared with the original XGBoost baseline protocol.

| Proposal or original method | Implementation |
|---|---|
| Full trip dataset and fixed stratified 80/20 split | Preserve the supplied training/test IDs and labels exactly |
| Daily weather matched to start-station city and date | Five-city ZIP mapping, case-normalized headers/events, numeric weather columns, trace-precipitation indicator |
| Five K-means station clusters | Static station latitude/longitude only, k=5, seed 42, start/end cluster IDs |
| Trip/time and round-trip features | Existing duration/hour/weekday/month and calendar flags, plus derived business-day and between-cluster flags |
| Original weather feature families | Temperature, humidity, dew point, pressure, visibility, wind, precipitation, cloud cover, wind direction, and event indicators |
| Original min-max scaling | B0 uses min-max scaling; proposed one-hot pipelines fit their scaler within each fold, while native tree pipelines retain unscaled integer category codes and metadata |
| Inverse-frequency class weighting | N/(2*N_class), recalculated inside each fitting fold |
| Three-fold training-only selection | Spark CrossValidator on model-selection rows; class-stratified internal folds |
| Customer F0.5 as the objective | Cutoff-aware validation scoring and a separately reserved training calibration set |
| Practical runtime comparison | CV, cutoff selection, and final fitting measured separately |
| Three seeds and training-size comparison | Full-data primary fits; separately authorized saved 10,354-row controls |

Logarithmic duration is preprocessing of an existing predictor, not an additional data source. It can be disabled for an ablation. Business-day and between-cluster indicators are derived from already-proposed calendar and cluster inputs. Weather flags are derived from the existing events field. The original P1-P3 profile does not introduce another prediction task or external outcome data. Separately authorized supplemental FM, quantile-binned Naive Bayes, EBM and CatBoost implementations are documented in [additional experiments](../supplemental/README.md). They preserve P1-P3 and are evaluated separately; FM is unweighted under the installed Spark optimizer, and CatBoost remains part of the boosted-tree family.

An optional interactions variant derives hour-by-business-day and station-cluster-pair categories, fixed duration bands, a duration-by-business-day term, and cyclic hour/month/wind-direction components from those same proposed inputs. These are explicitly recorded preprocessing ablations. Native categorical encoding is available for GBT/RF; LR retains one-hot encoding. Legacy one-hot/standard runs remain reproducible, and imported selected settings must match their encoding, feature variant, duration transform, and cap.

The supplemental full-data XGBoost control disables B0's sampler while preserving its other classifier/preprocessing/cutoff choices. A shared saved-fold option provides the same internal-validation IDs as Spark. It is an additional control, not a replacement for the proposed downsampled B0 or the three Spark model families.

The original proposal-only Spark input profile excludes rider home ZIP/locality and the early-period flag. At the user's direction, a separately versioned profile adds `rider_locality` and `before_oct_2013`, bringing the classifier input count from 37 to 39. Both additions preserve every training/test row and original feature value. The locality categories match the declared assumption used in the 10,354-row reconstruction; the exact authors' definition is unavailable. The date flag uses the verified local trip date. This is an explicit revision of the original predictor scope. Station weather ZIP remains a join key, and raw rider ZIP, trip ID and subscription type are never classifier features. See [the revised evaluation protocol](FINAL_EVALUATION.md).

The revised primary experiment retains all 536,406 training trips. Supplemental controls reuse the exact saved similarity and stratified reconstruction IDs: 10,354 training trips per cohort, the same 39 inputs, and all 133,553 original test trips. The same fixed hyperparameters and training-only calibration procedure are used across sizes. Similarity sampling changes class proportions, so its comparison includes a sampler effect; the stratified control more closely isolates row count. No sample-specific parameter retuning is claimed. A new weighted XGBoost control uses the same inputs, cohort IDs and Spark calibration memberships. Its results remain distinct from the original paper-style reconstruction. The abandoned 10/25/50% volume plan produced no model outcomes.

## Selection and test boundaries

Development mode holds out one third of the original training data for internal evaluation. Within the remaining fitting cohort, 15% are reserved for cutoff calibration. Cross-validation uses the other rows and freshly assigned three-fold strata. The final pipeline is refitted on the whole fitting cohort after selection. In final test mode the fitting cohort is the entire original training set, and the original project test set is scored only after training-only model and cutoff selection.

The threshold-aware CV score is the best F0.5 within each validation fold and can be optimistic as a generalization estimate. It is used to rank candidates only. Independent internal-validation or final-test scores are the performance results. Calibration scores are also selection diagnostics, not final performance estimates.

An optional recall operating-point analysis reuses the same fitted Spark classifiers, features, and probabilities. A minimum validation recall is declared, and the cutoff maximizing validation F0.5 subject to that target is selected from saved training-validation predictions. The policy is saved before final-test scores are read. It is then transferred to copies of the existing final models, with no additional fitting. Increased recall can reduce precision and F0.5; the alternative is reported separately from the original F0.5-optimized experiment. The target is not guaranteed on test data, and previously examined project test results are not presented as a fresh independent experiment.

## Original-paper benchmark and baseline limits

The published chart suggests an XGBoost F0.5 around 0.81; this is a visual estimate, not an exact tabulated score. Its overall ranking value 11.465292 is not F0.5. Its 10,354-row experiment has a different sampling/evaluation setup. A numeric comparison to the chart alone cannot establish a controlled improvement.

The included B0 implementation reproduces the stated library family, min-max scaling, and width-0.03 similarity idea, with declared choices for unspecified details. It keeps one seeded representative per feature cell and class. Nearby points on opposite cell boundaries are not merged. The retained sample count is reported and is not forced to 10,354. It uses explicit modern XGBoost defaults, which may differ from the original software and settings. Results are labeled a protocol approximation rather than an exact replication.

That legacy 60,810-row result is exploratory. The primary reconstruction now enforces exactly 10,354 sampled rows with a documented size-adjustment step, preserves the original project test distribution, and separately evaluates a 10,354-total-row compressed holdout. Reference cutoff-0.5 results are kept separate from training-only tuned/calibrated extensions. See RECONSTRUCT_10354.md and the independent reconstruction audit; no exact replication of undocumented authors' code is claimed.

Final reports should compare the common fixed-test results, disclose the approximation and added preprocessing/cutoff selection, retain all three seeds, and distinguish a better score from a statistically supported improvement. Test rows never enter classifier fitting, class-weight calculation, imputation fitting or cutoff calibration. This is a previously examined benchmark; later feature revisions were motivated by earlier findings and are not presented as an untouched confirmatory evaluation.
