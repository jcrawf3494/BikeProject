# Matched full-data tuning and stacking

This experiment tunes five base classifiers on the same 39 source features and
adds a cross-fitted stack. It preserves the original P1-P3 application, the four
supplemental presets, and their result directories. The new study has its own
Scala worker, compiler, Python coordinator, audit, and output directory.

| ID | Classifier | Runtime |
|---|---|---|
| P1 | Gradient-boosted trees with native categorical metadata | Scala / Spark |
| P6 | Explainable Boosting Machine | Python / InterpretML |
| P7 | CatBoost | Python / CatBoost |
| P8 | LightGBM | Python / LightGBM |
| P9 | XGBoost with training-fitted one-hot categories | Python / XGBoost |
| P10 | Weighted logistic combiner of P1, P6, and P7 probabilities | Python / scikit-learn |

P8, P9, P6 and P7 are local CPU estimators; only P1 uses Spark. This does not
implement distributed LightGBM, CatBoost, XGBoost or EBM. The stack is a separate
experiment beyond the original three proposed classifier implementations.

## Inputs and selection boundaries

The study uses the preserved historical V1 feature table with audited locality
and early-period additions. It keeps all **536,406 original training trips** and
the fixed **133,553 test trips**. `before_oct_2013` is a predictor, never a row
filter. Raw rider ZIP codes, trip IDs, target text and raw timestamps are not
classifier inputs. Inputs are checked against the saved 39-column feature list
and hashed before fitting. This experiment does not silently replace the weather
table with the later shared V2 revision.

The fixed seed-42 calibration memberships are shared by every model and every
initialization seed in this study. Training is divided as follows:

| Role | Rows | Purpose |
|---|---:|---|
| B | 364,528 | Three-fold base-model search and out-of-fold predictions |
| H | 91,132 | Separate training holdout for stack regularization selection |
| C | 80,746 | Probability-cutoff calibration after parameter selection |
| Total | 536,406 | Final fitting for every base classifier and stack combiner |

Each base model has three predeclared configurations, evaluated on three
stratified folds of B: **45 base CV fits**. Imputers, encoders, scalers, category
vocabularies and inverse-frequency weights are fitted within each fit. The
selection statistic is the mean attainable Customer F0.5 across validation
folds. Fold cutoffs are diagnostics; the final cutoff is learned separately on
C. These validation scores are not reported as test performance.

The configurations vary depth, round budget, learning rate, regularization,
leaf size, bins or interaction count as appropriate. The exact grid is stored
in `protocol.json` and in `train_matched_models.py`. P1's historical preset and
the initial EBM/CatBoost presets are included. CatBoost's initial configuration
uses MVS with subsample 0.8; its other two configurations use no bootstrap.
Every final classifier receives the entire original training set. Internal
per-round sampling in the MVS candidate does not create a smaller input cohort.

## Stack construction

The three stack members are declared before fitting: P1, P6 and P7. P8/P9 test
scores cannot change that membership. Each selected member supplies out-of-fold
predictions for B, then fits all B to predict the disjoint H. A weighted logistic
combiner is fitted on B's out-of-fold predictions. Its regularization parameter
is chosen from C values 0.001, 0.1 and 10 using H alone. Base-model hyperparameter
selection never fits or scores H.

The combiner uses standardized logits of clipped probabilities. It is refitted
on the combined out-of-fit predictions for B and H, then calibrated on C using
base models fitted on B+H. Once the cutoff is fixed, the final combiner is fitted
on all 536,406 training labels and their out-of-fit base predictions, including
C predictions from models fitted on B+H. Every meta-training row was excluded
from its corresponding base-model fit. B-fold and H predictions have different
base fitting sizes; this is a disclosed approximation, not nested CV.

The OOF construction and selected settings are shared across initialization
seeds 42, 43 and 44. Seeds vary estimator initialization, not the data split.
These are repeated benchmark runs, not three independent test populations.

## Run and resume

From the repository root in PowerShell, restore the private audited inputs and
saved calibration roles described in [the supplemental instructions](../README.md).
Configure Spark 3.5 / Scala 2.12, Java 11 and the Windows Hadoop native runtime.
Then install the pinned model libraries and run:

```powershell
python -m venv .venv-additional
.\.venv-additional\Scripts\python.exe -m pip install -r supplemental/matched/requirements.txt
.\run-matched-tuning.ps1 -OutputDir results\matched-tuned39
```

The launcher accepts `-SparkHome`, `-JavaHome`, `-HadoopHome`, `-Python`,
`-DataDir`, `-RolesFile`, `-TempDir` and `-Threads`. It defaults to four local
Spark workers, a 4 GB driver heap, four estimator threads, and one EBM job.
Use `-Stage tune` to stop after training-only selection; `-Stage evaluate`
requires the saved `FROZEN_SETTINGS.json` and runs final evaluation plus audit.
The default `-Stage all` performs both stages and the independent audit.

Run the same command with the same output directory to resume. Completed
`DONE.json` cases are reused. An operating-system lock blocks duplicate
coordinators. Changed inputs, source hashes, dependencies or study settings
require a new output directory. Failed Spark attempts remain available for
inspection. Do not edit scientific source files during an active study.

`STATUS.json` identifies the latest model/case. Spark subprocess progress also
appears in `HEARTBEAT.json` and private logs. `FROZEN_SETTINGS.json` is written
before test features are opened. Successful completion produces 18 final
outcomes, `results.json`, `summary.json`, `COMPLETED.json` and
`independent_audit.json`. A completed fit is not an audited completion until
`independent_audit_pending` is false and the audit passes.

The coordinator owns its Spark subprocesses and cannot be pasted directly into
`spark-shell`. Use the PowerShell launcher for this multi-runtime experiment.

## Verification and interpretation

The independent audit reconstructs all CV memberships, labels, weights,
candidate selections, calibration optima and final metrics. It reconstructs the
stack training matrices and independently refits its small combiners to
reproduce every saved stack test probability. Final base models also have
save/reload prediction probes. Unit checks cover disjoint roles, finite logits
and unseen-category handling. A separate synthetic end-to-end run verifies the
complete orchestration; its scores are never bike-share performance results.

Every final classifier is refitted on the full original training set after
training-only selection and cutoff calibration. Using training subsets for CV
does not replace this full-data final fit. The fixed test has been examined in
prior experiments, so new scores extend an existing benchmark. They are not a
fresh independent confirmation or evidence of a globally optimal configuration.

New fit timings measure estimator fitting and exclude preprocessing and IO.
The resource configuration also differs from older local-two-thread, 2 GB Spark
runs. Do not use these timings to claim a controlled speedup. Aggregate scores,
parameters and hashes may be published; raw IDs, row-level predictions, model
binaries, local paths, data files and execution logs remain private.
