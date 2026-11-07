# Bike Share Model Training

Execution examples run from the repository root. Restore private data and local execution artifacts as described in [data preparation](../data/README.md); row-level inputs and fitted binaries are not committed.


For the full-data audit, native tree-category option, interaction features, and expanded search commands, see [FULL_DATA_TUNING.md](FULL_DATA_TUNING.md). Existing one-hot/standard runs remain reproducible; the new options are recorded separately.

For the primary 10,354-row paper reconstruction, documented sampling/feature assumptions, three-seed XGBoost experiments, and separate fixed-project-test versus compressed-dataset evaluation, see [RECONSTRUCT_10354.md](RECONSTRUCT_10354.md). The earlier 60,810-row baseline is retained as an exploratory approximation.

The October 29 runtime update enables run-specific GBT checkpoints every 10 iterations and starts Java with a 4 MB thread stack in both launchers. Restart spark-shell to load this update after a task-serialization StackOverflowError. The data, split, and model-search grids are unchanged.

This project classifies bike-share trips as Customer (positive label 1) or Subscriber (label 0). The proposal's Spark MLlib models remain gradient-boosted trees (P1), random forest (P2), and logistic regression (P3), implemented in Scala with inverse-frequency class weights.

The original Parquet training/test pair is preserved under `data/`. The prepared pair under `data/enriched/` adds daily weather and five station clusters without changing any trip IDs, labels, or split membership. Full-feature runs automatically select the enriched pair.

## Start the Scala shell

From the repository root in Command Prompt or PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-models-spark-shell.ps1
```

This launcher uses configured runtime paths and loads the current compiled JAR. It defaults to two local threads and 2 GB of driver heap; pass `-DriverMemory 4g` if more RAM is available. Run one Spark training job at a time on a memory-constrained computer. On another computer, pass `-SparkHome`, `-JavaHome`, and `-HadoopHome` explicitly. Use Spark 3.5.x with Scala 2.12 and a supported JDK, preferably Java 11 or 17. Spark 3.0.x is incompatible. Windows needs matching Hadoop native helpers.

At the `scala>` prompt, first run the quick check:

```scala
val project = sys.env("BIKESHARE_PROJECT")
TrainBikeShareModels.main(Array(
  "--data-dir", project + "/data",
  "--output-dir", project + "/results",
  "--mode", "smoke"
))
```

The application stops its Spark session after each run. Enter `:quit` and restart the launcher before another run. Always restart after compiling an update; an already-open shell retains its earlier classes.

## Develop using training validation

The updated default is `--evaluation validation`. It reserves one third of the selected training rows as an internal validation set and does not score the project's original test set. The remaining rows are the fitting cohort. Within that cohort, 15% are reserved for decision-cutoff calibration; the other 85% supply the model-selection folds.

For full-feature development with the preset model settings:

```scala
val project = sys.env("BIKESHARE_PROJECT")
TrainBikeShareModels.main(Array(
  "--data-dir", project + "/data",
  "--output-dir", project + "/results",
  "--mode", "train",
  "--evaluation", "validation"
))
```

For three-fold hyperparameter tuning, replace `"train"` with `"tune"`. Select models individually with `"--models", "P1"`, or pass `"P3,P2,P1"` for all three. `--fraction 0.05` is available for preliminary tuning on a deterministic training subset. Report the actual fraction used. Every tuning run writes `selected_settings.json`, which can be reused for full-data training without repeating the search.

The standard search has four GBT candidates, four random-forest candidates, and six logistic-regression candidates. The extended search has four, four, and eight respectively; see FULL_DATA_TUNING.md for the grids and full-data commands. When cutoff calibration is enabled, candidates are ranked by the mean best-cutoff Customer F0.5 attainable within each validation fold. The actual deployment cutoff is then chosen on the separately reserved calibration rows. Neither step uses the project test set. A `--search-budget check` option uses tiny parameter grids solely for verifying the code; it is not the final experiment search.

After settings and a cutoff have been selected, the model is refitted on the entire fitting cohort. The selected cutoff is saved inside the classifier. Calibration scores describe cutoff selection, not independent predictive performance; use the untouched internal validation scores to assess development changes.

## Final held-out test experiment

Once the feature and training procedure is fixed, explicitly request the project test set. This command runs the GBT using the completed full-training-pool CV settings, copied into `../validation/P1_selected_settings.json`:

```scala
val project = sys.env("BIKESHARE_PROJECT")
TrainBikeShareModels.main(Array(
  "--data-dir", project + "/data",
  "--output-dir", project + "/results",
  "--mode", "train",
  "--evaluation", "test",
  "--models", "P1",
  "--settings-file", project + "/../validation/P1_selected_settings.json",
  "--seeds", "42"
))
```

This fits the final model on all 536,406 original training rows and evaluates the same 133,553 original test rows. Use `"--models", "P3,P2,P1"` for all three models, and `"--seeds", "42,43,44"` for the proposal's repeated experiment. Each run selects a fresh cutoff using only its own training calibration rows. The outer split does not change across seeds.

To reuse a completed training-only search, add `"--settings-file", project + "/results/YOUR-TUNING-RUN/selected_settings.json"`, replacing the folder name. Imported settings must come from a standard- or extended-budget cross-validation run with matching input features, duration preprocessing and cap, tree encoding, and feature variant. Legacy settings imply one-hot encoding and standard features. Matching seed settings are used when present; otherwise the first saved settings for that model are reused. Alternatively, use `--mode tune --evaluation test` to repeat the entire training-only search at full data size. Repeated tuning across all models and seeds can take substantially longer than a preliminary subset search. A subset search is a development choice and must be disclosed in the final report.

## Explore higher Customer recall without retraining

F0.5 emphasizes precision. A lower Customer probability cutoff can increase recall while reducing precision and F0.5. `CompareCustomerRecall` performs a separate operating-point experiment with already fitted models. Earlier models and result files are preserved.

The tool selects the highest validation F0.5 among cutoffs meeting a requested minimum recall, using saved **training-validation predictions only**. It freezes and saves that policy before reading final-test results. It then applies the same cutoff to the existing final models, saves adjusted model copies and predictions, and reports the actual precision, recall, and F0.5. It does not fit new trees or change predicted probabilities.

Restart the shell after updating the JAR. For the completed P1 runs on this computer:

```scala
val project = sys.env("BIKESHARE_PROJECT")
CompareCustomerRecall.main(Array(
  "--validation-run", project + "/results/tune-2026-10-28T03-20-22-656784500Z-f65328f7",
  "--final-run", project + "/results/train-2026-10-28T04-02-57-528292100Z-8d2ae312",
  "--data-dir", project + "/data",
  "--output-dir", project + "/results",
  "--minimum-recall", "0.60",
  "--model", "P1"
))
```

On another computer, replace the two run paths with matching completed training-validation and final-test runs. Run histories and fitted models are not included in the shareable ZIP. The same entry point supports P2/P3 when corresponding runs exist. Hyperparameters, feature profile, and duration preprocessing must match between the source validation and final models.

The 60% target applies to the validation selection data. Actual test recall can differ when the cutoff is transferred to a model refitted on more data. Validation scores after cutoff selection are selection diagnostics. Final results are an additional operating-point analysis on the existing project test set, whose earlier results have already been examined; they are not a fresh independent test experiment. Retain the original F0.5-optimized results for the primary comparison, and report this alternative policy separately.

Outputs go to a new `results/recall-...` folder, including `selected_recall_policy.json`, `validation_cutoff_comparison.json`, `results.json`, `summary.csv/`, adjusted native PipelineModels, and revised predictions. Model reload checks verify unchanged probabilities and the new cutoff decisions on a probe. The original test IDs and labels are checked before scoring.

## Features and original-paper methods

Default predictors include trip duration; hour, weekday, and month; weekend, federal-holiday, business-day, and round-trip flags; start/end station clusters; a between-cluster flag; daily weather; and weather-event indicators. Weather includes temperature, dew point, humidity, pressure, visibility, wind, precipitation, cloud cover, and wind direction. These remain within the proposal's trip/time, station-cluster, and daily-weather feature families. See `SCOPE_AND_METHODS.md` for the mapping.

Duration uses `log1p` by default to reduce the effect of extremely long trips while retaining all trips. `--duration-transform raw` provides the raw-duration comparison. There is no default duration cap; an optional fixed cap is available through `--duration-cap-seconds`.

Numeric imputation, category indexing, one-hot encoding, and min-max scaling are fitted inside each training fold. The cutoff defaults to training calibration; `--threshold-mode fixed` restores 0.5 for an ablation. `--feature-profile core` omits the added weather/cluster-derived features for a corresponding ablation. Explicit `--numeric-features` and `--categorical-features` lists are also supported.

Trip ID, text subscription type, the target label, rider home ZIP, and raw timestamps are excluded as predictors. Weather ZIP is used only to map station city to daily weather. Customer remains the positive class in every mode. Evaluation metrics are unweighted, even though model fitting uses class weights.

This is retrospective trip classification: duration and end-station features become available after a trip ends. These results do not demonstrate prediction before a ride starts.

## Rebuild feature tables if needed

The package already contains prepared full-feature data. To rebuild from `data/raw/station.csv` and `weather.csv`, start the shell and call:

```scala
val project = sys.env("BIKESHARE_PROJECT")
PrepareBikeShareFeatures.main(Array(
  "--data-dir", project + "/data",
  "--raw-dir", project + "/data/raw",
  "--output-dir", project + "/data/enriched-new"
))
```

The destination must be new. For a subsequent model run, pass that new directory directly as `--data-dir`. Original inputs and earlier outputs are never overwritten. The feature builder checks station metadata, unique weather city/date keys, complete joins, exact split membership, and label preservation. Remaining numeric weather nulls are imputed later inside model-training folds.

## Original XGBoost baseline B0

The proposal defines B0 using the scikit-learn XGBoost interface, so the separate baseline uses Python while P1-P3 stay in Scala. Install its dependencies and run:

```powershell
python -m pip install -r .\baseline\requirements.txt
python .\baseline\run_xgboost_baseline.py --data-dir .\data --output-dir .\results --evaluation validation
```

Use `--evaluation test` for the fixed project test pair after the baseline procedure is fixed. The script fits min-max preprocessing on training data, keeps a seeded representative per class within width-0.03 feature cells, and fits unweighted XGBoost at cutoff 0.5.

This is explicitly labeled **B0 protocol approximation**. The paper does not fully specify its sampling order, merging procedure, split, or XGBoost settings. The script does not force a 10,354-row result, discard test rows, or claim an exact reproduction. It records the retained training count, class balance, assumptions, settings, and versions. Baseline and Spark internal validation splits use different implementations; compare final results only on the common project test IDs. Its PR metric uses the trapezoidal precision-recall area, rather than substituting average precision.

## Data-volume experiment

Use P1 at 10%, 25%, 50%, and 100% of the saved training data. This command keeps the selected P1 settings fixed:

```scala
val project = sys.env("BIKESHARE_PROJECT")
TrainBikeShareModels.main(Array(
  "--data-dir", project + "/data",
  "--output-dir", project + "/results",
  "--mode", "volume", "--evaluation", "test",
  "--settings-file", project + "/../validation/P1_selected_settings.json",
  "--seeds", "42,43,44"
))
```

Subsets are deterministic and nested within a seed. Each cutoff is selected using only that subset's training calibration rows. The fixed test set remains unchanged. Freeze settings before viewing volume-test results. To reuse a completed tuning run, add its `--settings-file` option; the volume study then keeps those hyperparameters fixed across fractions.

## Saved outputs and timing

Each invocation creates a new timestamped run folder. Each model saves its standard PipelineModel, test or validation predictions, `metrics.json`, optional `cross_validation.json`, and `threshold_selection.json`, plus `feature_metadata.json` describing the fitted feature vector and native categories when requested. The overall run has `manifest.json`, `results.json`, `summary.csv/`, saved outer/internal fold IDs, model-selection fold IDs, and calibration-role IDs. These permit split and leakage audits.

`cv_total_seconds` covers CrossValidator, including its winner refit. `threshold_selection_seconds` covers calibration-model fitting when required and cutoff selection/evaluation. `final_fit_seconds` covers the separate final pipeline fit and class-weight calculation, excluding scoring and saving. Compare B0 timing boundaries explicitly; its fit timing includes sampling as well. A local Spark run uses one computer, so it does not establish a multi-machine speedup.

`COMPLETE` is printed only after all requested work finishes. A Windows temporary-JAR cleanup warning after completion does not by itself invalidate saved models. Model reload and probability-cutoff checks are performed before recording a completed model result.

## Compile or run without the Scala shell

```powershell
.\compile-models.ps1 -SparkHome 'C:\tools\spark-3.5.9-bin-hadoop3' -JavaHome 'C:\tools\jdk-17'
.\run-local-models.ps1 -Mode smoke
.\run-local-models.ps1 -Mode train -AppArgs @('--evaluation','test')
```

Compilation uses all Scala files under `src/` and writes a source-versioned JAR. Launchers read `target/current-models-jar.txt` and select the current JAR automatically. Existing open shells can retain their previous JAR; restart before using an update. For other computers, `run-models.ps1` accepts explicit runtime paths. With sbt, use `sbt package` and submit its resulting JAR with `--class TrainBikeShareModels`. Linux/macOS users can use `spark-submit` directly with that JAR and the same application options.

The ZIP excludes run history, local dependency installations, and old JAR versions. Earlier results remain on the original computer. See `../validation/TEST_RESULTS.md` and the associated audit files for checks actually completed.

## References

- [Original paper](https://doi.org/10.1145/3335656.3335704)
- [Dataset](https://www.kaggle.com/datasets/benhamner/sf-bay-area-bike-share)
- [Spark cross-validation](https://spark.apache.org/docs/3.5.6/ml-tuning.html)
- [Spark F-beta threshold metrics](https://spark.apache.org/docs/3.5.6/api/scala/org/apache/spark/mllib/evaluation/BinaryClassificationMetrics.html)
- [Spark K-means](https://spark.apache.org/docs/3.5.6/ml-clustering.html)
- [XGBoost scikit-learn interface](https://xgboost.readthedocs.io/en/release_3.1.0/python/python_api.html)
