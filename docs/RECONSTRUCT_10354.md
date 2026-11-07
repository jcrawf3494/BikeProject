# Reconstruction of the paper's 10,354-row experiment

Execution examples run from the repository root. Restore private data and local execution artifacts as described in [data preparation](../data/README.md); row-level inputs and fitted binaries are not committed.


The primary reconstruction target is the 10,354-row sampled experiment described in Section 4.2.1 of the original paper. The earlier 60,810-row experiment remains an exploratory approximation. It is not the paper's result and is not substituted for this reconstruction. Figure 8 suggests F0.5 around 0.81; no exact XGBoost F0.5, complete split procedure, classifier configuration, or sampler implementation is supplied.

Completed measured results and per-seed scores are in [the reconstruction report](../validation/RECONSTRUCTION_10354_RESULTS.md). The compressed reference mean is 0.815863; the shared-test calibrated similarity mean is 0.802624. The highest size-matched stratified control mean is 0.879851 and uses a different sampling procedure.

## Two evaluation interpretations

1. **Shared project test:** draw exactly 10,354 fitting rows from the project's training pool, then score the unchanged 133,553 original project test trips. Development holds the same original training fold out as earlier Spark runs. The final baseline uses all 10,354 sampled training rows after training-only selection.
2. **Compressed dataset holdout:** form a dataset containing exactly 10,354 rows, then make a stratified 80/20 split: 8,283 fitting rows and 2,071 holdout rows. The compressed dataset is drawn entirely from the original project training pool, protecting the original project test set. Its holdout has the compressed sample's distribution, which can differ substantially from the full dataset.

These two scores must remain separate. Reaching 0.81 on the compressed holdout does not establish 0.81 on the original project test set. Neither procedure establishes the authors' undocumented original split exactly.

There is a numerical clue favoring the training-sample interpretation: reducing 536,406 rows to 10,354 removes 98.07%, which rounds to the paper's reported 98.1% compression. Reducing all 669,959 trips to 10,354 removes 98.45%. This supports investigating 10,354 fitting rows as the primary shared-test reconstruction, but does not establish the authors' actual split or retained IDs.

## Declared reconstruction assumptions

- Use the scikit-learn XGBoost estimator interface, raw-duration min-max preprocessing, median numeric imputation, one-hot categories, unweighted logistic loss, and a fixed 0.5 cutoff for the reference.
- The reference classifier has 100 trees, depth 3, learning rate 0.1, minimum child weight 1, and L2 regularization 1. These are stated assumptions, not parameters claimed to be given in the paper.
- The similarity sampler retains one seeded representative per width-0.03 min-max feature cell and class. Neighboring cells are not merged. It then selects exactly 10,354 representatives, preserving the representative pool's class proportion. The exact-size adjustment is an explicit addition needed to match the reported count.
- A separate 10,354-row stratified random sample provides a size-matched sensitivity control. It preserves the original fitting pool's class proportion.
- Feature-profile and sampling-method comparisons are predetermined. Their effects are reported separately; no claim of matching the paper rests on row count alone.
- `paper_visible` maps Figure 9's duration, hour, weekday/month, business-day, weather, city, early-period, and locality variables to the available data. Mean daily values stand in for the abbreviated pressure, humidity, temperature, and dew features; the precise original aggregations are unavailable. Existing k=5 between-cluster status stands in for `inter_group`.
- The `before_oct_2013` flag uses the verified local trip date. The `local_zip` assumption is a valid five-digit US rider ZIP beginning with 94 or 95, with missing recorded separately. The original locality definition is unknown. City/weather ZIP and rider locality are different variables.
- `paper_without_zip` repeats that feature definition without locality. `proposal_full` retains the predictors used by the proposed Spark models. The proposed Spark models continue to exclude rider locality and the early-period flag.
- The locality file stores only trip ID and the derived local/outside/missing category; raw rider ZIP values are not copied into the package.
- The reconstructed features use the existing independently audited weather version. They must not be described as using a later team weather file without verifying that version and refitting affected experiments.

## Training-only tuning toward the plotted score

Each feature/sampling recipe gets the same ten predefined XGBoost configurations and three-fold cross-validation. The grid varies depth (3, 5, 7, 9), tree count (100, 300, 400, 800), learning rate (0.03, 0.05, 0.1), minimum child weight, and L2 regularization. The complete candidate list is stored before scoring in the run manifest.

Within each fitting sample, 15% is reserved for cutoff calibration. Imputation, category vocabulary, and scaling for each CV model are fitted only on its CV fitting rows. The sampler's geometry is fitted on its original fitting pool, excluding the outer validation and project test data. Sampling defines the cohort before CV; it is not claimed to reproduce the authors' algorithm exactly.

Three procedures are recorded: the reference configuration at cutoff 0.5; the configuration selected by mean CV F0.5 at cutoff 0.5; and an extension selected by cutoff-aware CV F0.5 with its deployment cutoff chosen on the separate calibration rows. The final model refits on all available sampled fitting rows.

The strongest size-matched shared-test recipe and parameters are frozen using original-training validation results before project test scoring. Independently, the primary paper-style recipe is declared as `paper_visible` plus similarity sampling and is always evaluated on the shared project test, even if a different sampler wins development. The main compressed-dataset procedure uses that same declared feature/sampling recipe; its tuned parameters come from training CV, never its holdout scores. Final evaluation repeats seeds 42, 43, and 44. Averages describe seed sensitivity, not three independent test datasets.

The target is to investigate the plotted 0.81 result, not to force a score or select settings from final-test outcomes. All declared recipes, reference results, tuned results, sample counts, and class rates are retained.

## Reproduce

Install `baseline/requirements.txt` in a suitable Python environment. The baseline uses Python because the original comparison uses the XGBoost estimator interface; the proposed P1-P3 models remain Scala/Spark.

Prepare the locality metadata locally from the original archive. Row mappings are excluded from this public repository:

```powershell
python .\baseline\prepare_paper_locality.py --trip-archive 'PATH-TO-ORIGINAL-DATA.zip' --output .\data\raw\paper_locality.parquet
```

Development, using an existing saved Spark fold directory:

```powershell
python .\baseline\reconstruct_10354.py --mode develop --data-dir .\data --output-dir .\results --locality-metadata .\data\raw\paper_locality.parquet --validation-folds .\data\validation_folds.parquet
```

After development completes and writes `frozen_project_test_settings.json`:

```powershell
python .\baseline\reconstruct_10354.py --mode final --data-dir .\data --output-dir .\results --locality-metadata .\data\raw\paper_locality.parquet --frozen-settings .\results\YOUR-RECONSTRUCTION-RUN\frozen_project_test_settings.json
```

Models, sample/fitting/scoring IDs, calibration predictions, CV candidates, and evaluation predictions are saved in new result folders. Existing experiments are preserved.

A locally restored `data/validation_folds.parquet` preserves the original Spark training fold assignments. Fold membership rows are excluded from this public repository. Development uses it by default if `--validation-folds` is omitted. A final control was added before project test scoring to ensure the paper-style similarity procedure is evaluated separately from the training-selected sampler; code hashes and the frozen-settings hash record this additive control.
