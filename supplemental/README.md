# Additional model experiments

These are supplemental experiments. The original P1-P3 source files, compiled
artifact pointer and historical results are preserved.

The first full-data three-seed comparison is complete and independently audited.
See [measured results](../validation/additional_models_20261126/FULL_DATA_RESULTS.md):
FM mean F0.5 **0.874431**, Naive Bayes **0.773744**, EBM **0.886242**, CatBoost
**0.888252**. These initial presets did not exceed the P1 reference **0.891818**.
The commands below reproduce experiments in fresh output folders; optional
training-only tuning is now running in a separate
[matched full-data study](matched/README.md), which adds LightGBM, XGBoost and
a cross-fitted stack. The preset results below remain a separate experiment.

| ID | Model | Runtime | Fitting weights |
|---|---|---|---|
| P4 | Factorization machine | Scala / Spark 3.5 | Unweighted; the installed FM optimizer does not consume instance weights |
| P5 | Quantile-binned Naive Bayes | Scala / Spark 3.5 | Inverse class frequency |
| P6 | Explainable Boosting Machine | Python / InterpretML | Inverse class frequency |
| P7 | CatBoost | Python / CatBoost | Inverse class frequency |

FM and Naive Bayes add families absent from the reference paper's classifier
table. EBM uses an additive structure with selected interactions, learned using
boosted trees. CatBoost is another boosted-tree implementation, not a new broad
algorithm family. No score improvement is assumed.

## Inputs and split

Use the audited historical V1 inputs plus `rider_locality` and `before_oct_2013`
under `data/paper_flags_full/enriched`. Final training requires all **536,406**
original training trips and scores the unchanged **133,553** test trips. The 39
source columns match `validation/P1_P2_P3_with_paper_inputs.json`. No scientific
training-fraction option is provided. Technical smoke checks use separate
synthetic data and never produce project-performance claims.

Every model uses log1p duration and training-fitted numeric medians. FM uses
one-hot categories and scaling. Naive Bayes fits 16-quantile numeric bins within
each fit, then one-hot encodes each binned/categorical input. Its multinomial
estimator receives one categorical event per source variable; raw continuous
weather values are not treated as event counts. EBM and CatBoost receive numeric
columns and explicitly identified categorical strings. EBM uses one outer bag,
no internal validation exclusion and a fixed round budget so the final fit
includes every supplied training row.

Python models share the exact saved P3 calibration roles and model-selection CV
folds. Restore the private P3 outputs, then prepare their role files:

```powershell
python -m venv .venv-additional
.\.venv-additional\Scripts\python.exe -m pip install -r supplemental/requirements.lock.txt
.\.venv-additional\Scripts\python.exe supplemental/prepare_roles.py --source-package PATH-TO-PRIVATE-MODEL-PACKAGE --output-dir data/supplemental_roles
```

`--source-package` contains the saved `validation/paper_flags_full_20261108`
states and their private `results` directories. A different state location can
be passed with `--source-state-dir`. The Scala trainer regenerates the same
deterministic roles and folds using the preserved ID-hash rule.

## Run the four models

From the repository root, with SPARK_HOME, JAVA_HOME and HADOOP_HOME configured:

```powershell
.\compile-supplemental-models.ps1
.\run-additional-models.ps1 -Mode train -Seeds '42,43,44'
```

This creates **12 full-data outcomes** sequentially, using predeclared presets.
They are explicitly labeled **not newly tuned**. All output directories are new
and timestamped. P1-P3 are not retrained or overwritten. To run one seed first:

```powershell
.\run-additional-models.ps1 -Mode train -Seeds 42
```

The Python runtime path, data directory and role directory can be overridden
with `-Python`, `-DataDir` and `-RolesDir`. Select models with `-Models P4,P5`
or `-Models P6,P7`. The Spark launchers use two local worker threads and 2 GB
driver memory. These settings do not establish distributed speedup.

For FM and Naive Bayes in `spark-shell`, load the independent JAR:

```powershell
.\start-supplemental-spark-shell.ps1
```

At `scala>`, enter `:paste`, paste the block, and finish with Ctrl+D:

```scala
val project = sys.env("BIKESHARE_PROJECT")
TrainSupplementalSparkModels.main(Array(
  "--data-dir", project + "/data/paper_flags_full",
  "--output-dir", project + "/results/additional-spark",
  "--mode", "train", "--evaluation", "test",
  "--models", "P4,P5", "--seeds", "42,43,44",
  "--fraction", "1.0", "--tree-encoding", "onehot"
))
```

The application stops its Spark session on completion. Restart the shell for
another application. EBM and CatBoost run through the Python launcher rather
than inside `spark-shell`.

## Optional training-only parameter selection

Tune on training data with one declared seed, then freeze the two resulting
settings files before final scoring:

```powershell
.\run-additional-models.ps1 -Mode tune -Seeds 42
.\run-additional-models.ps1 -Mode train -Seeds '42,43,44' -SparkSettingsFile results/additional-spark/YOUR-TUNE-RUN/selected_settings.json -PythonSettingsFile results/additional-python/YOUR-TUNE-RUN/selected_settings.json
```

The Spark grids compare FM factor sizes 8/16 and regularization .001/.01, and NB
smoothing .1/1/2. Python compares EBM with 0/8 interactions and CatBoost depth
6/8. EBM uses 500 rounds and CatBoost 400 trees in these bounded grids. All grids
use three fitting-only folds and cutoff-aware fold F0.5 for ranking. Fold-best
scores are selection diagnostics, not generalization estimates. Calibration is
reserved separately, and the final classifier refits the full training set.
Technical-check grids cannot be imported for scientific final runs.

The inherited Spark tuning workflow also reports an outer training-validation
fold and therefore searches within a smaller development fitting cohort. The
Python tuning workflow uses CV on the saved full-pool model-selection roles.
Their search pools and candidate counts are not identical. Final `train` runs
share the original full fitting population and seed-specific calibration IDs;
do not describe the tuning budgets as exactly matched.

## Verification and saved artifacts

Run the protocol unit tests with the additional environment:

```powershell
.\.venv-additional\Scripts\python.exe -m unittest discover -s supplemental/tests -v
```

Every completed training run saves its parameters, split/role evidence, cutoff
selection, predictions, confusion counts, F0.5, precision, recall, accuracy,
PR area, timing and a saved model. Reload probes compare probability predictions
before and after persistence. Independently verify the output directories:

```powershell
.\.venv-additional\Scripts\python.exe supplemental/audit_results.py --data-dir data/paper_flags_full --spark-runs results/additional-spark/YOUR-RUN --python-runs results/additional-python/YOUR-RUN --output results/additional-audit.json
```

When Spark settings were imported, also pass `--settings-file` with that frozen
Spark file. The audit checks membership, label/cutoff arithmetic, calibration
curve optima and Python weights. Raw data, role IDs, predictions, model binaries
and machine logs remain private. Export aggregate evidence only after auditing.
Python PR area uses the exact precision-recall curve; Spark uses its default
binned computation. Customer F0.5 is the main comparison. The test set has been
examined previously, so these remain additional benchmark experiments.

The saved CatBoost presets use `bootstrap_type=MVS` and `subsample` approximately
0.8 in every seed, as verified from the persisted models. All 536,406 original
training rows are supplied to fitting; MVS performs internal stochastic row
sampling during boosting. P1 uses `subsamplingRate=1.0`. This is another
implementation difference to disclose. An explicit no-bootstrap CatBoost
configuration can be evaluated through training-only selection.

The dated [implementation verification](../validation/additional_models_20261126/implementation_verification.json)
records the completed synthetic end-to-end checks, independent prediction audits,
and parameter-selection checks. The technical run records are included in the
experiment ledger and explicitly excluded from scientific score claims.
