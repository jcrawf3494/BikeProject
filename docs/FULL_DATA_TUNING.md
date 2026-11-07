# Full-data development and tuning

Execution examples run from the repository root. Restore private data and local execution artifacts as described in [data preparation](../data/README.md); row-level inputs and fitted binaries are not committed.


The objective remains Customer F0.5, with accuracy, precision, recall, PR area, and training cost recorded. No sampling study is performed by the commands below. The full original training pool is 536,406 trips; the original 133,553 test trips remain outside fitting. Cross-validation and internal validation divide the training pool rather than discard rows. A final test run refits on the entire training pool after training-only selection.

## Recovery from the October 29 stack-overflow failure

The extended search started on October 29 aborted inside P1's gradient-boosted tree fit with `Task serialization failed: java.lang.StackOverflowError`. The failed run has a manifest and fold assignments but no completed model scores or selected-settings file. P2 and P3 were scheduled after P1 and were not reached. Preserve that folder as an incomplete run; do not import settings from it.

The updated application enables a run-specific Spark checkpoint directory and explicitly checkpoints GBT iteration state every 10 rounds. Both launchers now start the driver with `-Xss4m` and configure the same stack size for executors. These are execution controls; the full-data split, predictors, candidate grids, class weighting, and cutoff-selection rules are unchanged. A fresh process is required to load the new JAR and Java stack option. Setting stack options inside an already-running shell is insufficient.

At the Windows command prompt, start `start-models-spark-shell.ps1` again, then repeat the training-validation tuning command below. The failed search cannot resume intermediate CV fits with the current implementation. A new timestamped result folder preserves earlier experiments. Keep `--fraction 1.0`; no downsampling is needed for this correction.

The patched spark-shell path completed a depth-7, 240-round technical run on the existing small fixture, including both the calibration fit and final refit, model reload, and independent ID/label/cutoff/confusion checks. The monitor observed 46 materialized checkpoint directories across those fits. This verifies the execution correction on the fixture; the full-data extended search remains pending. Details are in `../validation/stack_overflow_fix_check.json`. Windows reported a temporary JAR cleanup warning after the completed shell run; model artifacts and verification passed before exit.

## Completed full-data development comparison

These scores use the same 178,803 internal-validation trips, held out from the original training pool. Every one of the remaining 357,603 fitting trips is retained for refitting. Selection uses three folds within fitting, with separate training calibration rows for the final cutoff.

| Procedure | Customer F0.5 on internal validation |
|---|---:|
| P1, earlier one-hot categories, standard grid | 0.714537 |
| P1, native categories, same standard grid | 0.715517 |
| XGBoost full-fitting-data supplemental control | 0.711456 |

Both P1 searches selected depth 5, 80 iterations, step size 0.1, and minimum 20 rows per node. The native search mean CV ranking score was 0.720239. Its validation precision was 0.797291 and recall 0.507364. The validation gain is small: 0.000980 F0.5, or about 0.098 percentage points. It is not evidence of a large or statistically established improvement.

The XGBoost control retains the earlier baseline classifier, preprocessing, and cutoff while fitting all available fitting rows. Its result does not isolate an algorithm effect relative to P1, which uses different weights, preprocessing, and a calibrated cutoff. It also does not directly measure the effect of sampling relative to B0's previously recorded test score.

The subsequent extended standard-feature full-data search completed for all three models: P1 validation F0.5 0.717141, P2 0.714662, and P3 0.686073. No new fixed-test score was produced for those Spark configurations. The earlier three-seed P1 test mean remains 0.717260. The extended standard-feature grids have now completed; the interaction variants remain technically checked and await full-data searches.

The completed native CV job experienced long system pauses and memory pressure. Its recorded wall-clock time is unsuitable for clean runtime comparisons. The independent audit is in `../validation/full_data_revision_audit.json`; the matching selected parameters are in `../validation/P1_native_selected_settings.json`.

## Data correctness checks completed

`../validation/feature_semantics_audit.json` independently checks every supplied trip against the original raw CSV and reconstructs daily weather matches. It confirms:

- Original IDs, labels, split membership, durations in seconds, and start/end station IDs are preserved.
- Customer is label 1; Subscriber is label 0. ID, label text, rider ZIP, and raw timestamps are excluded from predictors.
- Hour, date, month, weekday coding, weekend, observed US federal holidays, business day, round trip, and between-cluster flags are consistent.
- Every added weather value and event flag agrees with an independent city/date join, including trace precipitation handling.
- Five station clusters remain present. Locations are public static metadata and clustering uses no trip labels.

The stored `start_ts` values have a legacy timezone encoding: their UTC representation is four or five hours ahead of the original local-clock CSV values. The actual hour/date features used for classification match the original CSV. New derived time features use those verified local-clock columns, not a reinterpretation of `start_ts`.

Median imputation, category indexing, and scaling where applicable are fitted within each CV training fold. Inverse-frequency class weights are recomputed from each fitting fold. The separately reserved training calibration rows determine the operating cutoff. Test scores are not used by these selection steps. The test set has already been examined during earlier experiments, so later results must be disclosed as additional benchmark experiments rather than a fresh independent confirmation.

## Optional preprocessing improvements

Existing behavior remains reproducible with `--tree-encoding onehot --feature-variant standard`.

`--tree-encoding native` lets GBT and random forest use indexed categorical attributes directly. StringIndexer metadata is retained through VectorAssembler, and integer category codes are not scaled. Native category count and maxBins compatibility are checked before recording completion. Logistic regression continues to use one-hot encoding and min-max scaling regardless of the tree option. The standard grid is useful for measuring this representation change before expanding the search.

`--feature-variant interactions` adds fixed transformations of existing proposed inputs:

- Hour-by-business-day and start/end-cluster-pair categories.
- Duration bands at 5, 10, 20, 30, and 60 minutes, plus a missing-value category.
- Duration-by-business-day numeric interaction.
- Sine/cosine representations of hour, month, and wind direction.

These are generated inside the saved pipeline. They introduce no external data, target encodings, or rider information. Duration transformations and an explicitly requested duration cap are respected; no cap is enabled by default. Missing numeric values remain subject to fold-fitted imputation. Feature variant, encoding, and input-column lists are saved with selected settings and must match when settings are imported.

## Run full-data searches

From PowerShell in the repository root:

```powershell
.\run-full-data-tuning.ps1 -Models 'P1' -SearchBudget extended -TreeEncoding native -FeatureVariant standard
```

For all three models, one after another in separate Spark processes and result folders:

```powershell
.\run-full-data-tuning.ps1 -Models 'P1,P2,P3' -SearchBudget extended -TreeEncoding native -FeatureVariant standard
```

After a standard-feature comparison, use `-FeatureVariant interactions` for a separately recorded feature ablation. Do not attribute a gain to interactions if the encoding or classifier settings changed at the same time.

For spark-shell, quit an earlier session and restart `start-models-spark-shell.ps1` to load the updated JAR. At the new prompt:

```scala
val project = sys.env("BIKESHARE_PROJECT")
TrainBikeShareModels.main(Array(
  "--data-dir", project + "/data",
  "--output-dir", project + "/results",
  "--mode", "tune",
  "--evaluation", "validation",
  "--models", "P1",
  "--fraction", "1.0",
  "--seeds", "42",
  "--search-budget", "extended",
  "--tree-encoding", "native",
  "--feature-variant", "standard"
))
```

Use `P2` or `P3` for the other models. Begin selection with one seed, then freeze the chosen procedure before repeated-seed test evaluation. The prior P1 test mean 0.7173 remains the recorded reference until a new frozen final experiment is completed; new development scores are validation scores.

## Extended grids

| Model | Candidates |
|---|---|
| P1 | Depth/iterations/step/minimum-node-rows: 5/80/0.1/20; 5/160/0.05/20; 7/160/0.05/20; 7/240/0.03/40 |
| P2 | Depth/trees/minimum-node-rows: 9/120/10; 13/160/10; 17/160/20; 13/240/5 |
| P3 | Regularization 0.0001, 0.001, 0.01, 0.1 crossed with elastic-net mixing 0 and 0.5; maximum 200 iterations |

Each candidate uses three training folds. These are targeted searches, not guarantees of a global optimum. Larger grids and deeper ensembles may run for hours on a constrained laptop. The default driver heap is 2 GB and jobs run sequentially. Avoid simultaneous model jobs and keep the computer awake; a run affected by sleep, long pauses, or heavy paging should not be used for clean runtime comparisons.

The extended tree grids are focused follow-ups, not exhaustive supersets of every standard candidate. Retain earlier completed searches, compare their training-validation results, and keep the best tested configuration. A newly completed search does not automatically replace an earlier stronger result.

`--gbt-step-size` also sets the GBT step size for a controlled `train` run. Extended CV candidates specify their own step sizes. `--search-budget check` is only for technical validation; its settings cannot be imported as final experiment settings.

## Refit selected settings

After a completed native-category search, use its actual run name:

```scala
val project = sys.env("BIKESHARE_PROJECT")
TrainBikeShareModels.main(Array(
  "--data-dir", project + "/data",
  "--output-dir", project + "/results",
  "--mode", "train", "--evaluation", "test",
  "--models", "P1", "--fraction", "1.0", "--seeds", "42,43,44",
  "--tree-encoding", "native", "--feature-variant", "standard",
  "--settings-file", project + "/results/YOUR-COMPLETED-RUN/selected_settings.json"
))
```

Match `--feature-variant` to the source run. Legacy settings without encoding/variant fields are interpreted as one-hot/standard; they are intentionally rejected when imported with different preprocessing. Earlier result folders and fitted models are never overwritten.

## Full-data XGBoost control

The original B0 default remains similarity sampling. The additional `--sampling-mode none` control retains every fitting row while keeping B0's raw-duration preprocessing, features, unweighted classifier settings, and cutoff 0.5 unchanged.

```powershell
python .\baseline\run_xgboost_baseline.py --data-dir .\data --output-dir .\results --sampling-mode none --evaluation validation --validation-folds .\results\YOUR-REFERENCE-RUN\folds\seed-42\fraction-10000
```

The shared-fold option requires the whole original training pool and holds fold 0 out, matching Spark's internal validation IDs exactly. Full-data XGBoost is a supplemental control and does not replace the three proposed Spark families. A difference from Spark cannot be attributed solely to the algorithm when preprocessing, weights, and cutoff policies differ. The project test set remains unscored during this development comparison.

## Reproducibility and current checks

The small fixture check completed three-fold technical searches and native-model reload checks for P1, P2, and P3 with generated features. The fixture's scores are code checks, not project findings. Its native tree vectors contained the expected nine nominal attributes and 47 total features. The cap-zero interaction path was exercised; the optional nonzero cap branch is not an exhaustive tested experiment.

The full-data audit and current measured development results are recorded in `../validation/`. A completed model result is written only after score arithmetic, row preservation, native-category metadata checks when requested, and saved-model reload checks pass. JSON/CSV outputs distinguish validation from fixed-test scores and record preprocessing choices.

References: [Spark feature preprocessing](https://spark.apache.org/docs/3.5.6/ml-features.html), [Spark model selection](https://spark.apache.org/docs/3.5.6/ml-tuning.html), [Spark JVM startup configuration](https://spark.apache.org/docs/3.5.9/configuration.html#runtime-environment).
