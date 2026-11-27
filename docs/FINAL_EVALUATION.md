# Full-data and matched-input evaluation

The primary experiment trains P1 (Spark GBT), P2 (Spark random forest), and P3
(Spark logistic regression) on every one of the 536,406 supplied training trips.
All evaluations preserve the same 133,553 test trips. The supplementary
10,354-row experiments change training membership while holding the 39 inputs,
classifier parameters, target definition and test population fixed.

## Feature revision

The original enriched input profile had 37 classifier columns. The revised
profile adds `rider_locality` and `before_oct_2013`. Every original row, label,
split and column value is retained. These variables are predictors, not filters.
The locality category uses the same declared assumption as the paper-style
reconstruction: a valid US five-digit rider ZIP, optionally ZIP+4, with a 94 or
95 prefix is local; other valid ZIPs are outside; missing/invalid is separate.
The original authors' exact locality definition is unavailable. Raw ZIP values
are not classifier inputs or public artifacts.

The date flag is 1 when the verified local `trip_date` precedes October 1, 2013,
and 0 otherwise. It does not use the legacy stored timestamp. Weather remains
the historical audited V1 version; V2 requires a separate experiment.

The fixed values in [the expanded settings](../validation/P1_P2_P3_with_paper_inputs.json)
come from the previous training-only search on the original 37 inputs. They
have not been retuned under the expanded input profile:

| Model | Fixed settings |
|---|---|
| P1 | Depth 7, 240 iterations, learning rate 0.03, minimum 40 rows per node |
| P2 | 160 trees, depth 17, minimum 20 rows per node |
| P3 | Regularization 0.001, elastic-net mixing 0.5, maximum 200 iterations, tolerance 0.000001 |

Spark uses logarithmic duration, training-fitted preprocessing, and
inverse-frequency weights N/(2*N_class). Trees use native indexed categories;
LR uses one-hot categories and scaling. A reserved 15% of the supplied training
cohort chooses the Customer F0.5 cutoff, followed by a final refit on the entire
supplied cohort. Calibration does not remove rows from the final fit.

## Prepare the two added columns

Restore the private historical enriched inputs and ID-to-locality mapping as
described in [data preparation](../data/README.md). With PyArrow installed, run
from the repository root:

```powershell
python tools/prepare_full_data_paper_flags.py --data-dir data --output-dir data/paper_flags_full --locality-metadata data/raw/paper_locality.parquet --base-settings validation/P1_P2_P3_frozen_settings.json --validation-dir results/local-input-preparation
```

Keep new preparation attestations under `results`; published historical audit
files are immutable evidence. The script verifies all 669,959 IDs, retains the
original columns, and creates `data/paper_flags_full/enriched/{train,test}.parquet`.

## Run only the primary full-data models

Configure SPARK_HOME, JAVA_HOME and HADOOP_HOME for Spark 3.5 / Scala 2.12 /
Java 11, then run in PowerShell:

```powershell
.\compile-models.ps1
.\run-final-evaluation.ps1
```

This runs P3, P1 and P2 sequentially for seeds 42, 43 and 44, producing nine
full-training results. It uses two local worker threads, a 2 GB driver heap and
a 4 MB Java stack. Existing results are preserved in timestamped folders.

## Prepare and run the matched training-size controls

Restore the exact private sampled IDs from the saved three-seed 10,354-row
reconstruction. A newly reconstructed sample may differ; report that as a new
cohort. Both the similarity and stratified samples come exclusively from the
original training pool. The preparation command below preserves all 39 values
for those IDs and copies the full test data byte for byte:

```powershell
python tools/prepare_sample_comparison_cohorts.py --full-data-dir data/paper_flags_full --baseline-run results/baseline-10354-test-20261101T193451Z-48e5ef1a --output-dir data/pfc --validation-dir results/local-cohort-preparation --settings validation/P1_P2_P3_with_paper_inputs.json
.\run-matched-input-evaluation.ps1
```

The second command runs a new complete comparison: nine full-data Spark cases,
18 sampled Spark cases, and nine matched-input weighted XGBoost cases. It
creates new local state directories used to share the exact calibration IDs
between Spark and XGBoost. Do not run it alongside the full-only command.
Install XGBoost 3.1.2, scikit-learn 1.6.1, NumPy, pandas, PyArrow and joblib for
the Python control. The historical matched-control protocol records exact
runtime dependency versions. XGBoost uses 400 trees, depth 3, learning rate
0.05, minimum child weight 1 and lambda 1, transferred unchanged from an earlier
training-CV selection. It is a new weighted 39-input control, distinct from the
paper-style reconstruction.

## Interpret the comparisons

The expanded-feature full versus original-feature full comparison measures
the combined addition of two variables under fixed settings. It does not
isolate locality alone. A matching original-input final score was not completed
for the newest P1 configuration; the older P1 result used different settings.

Within the expanded profile, full versus stratified 10,354 is the closer
training-size comparison. The similarity samples change the Customer share
from roughly 15% to roughly 43%, so that contrast also includes class mix and
sampling geometry. Settings are held fixed, rather than optimized separately
for each training budget. Three-seed variability is not a confidence interval
over independent test populations.

The original paper's roughly 0.81 is a chart estimate. The earlier compressed
holdout used a different test population and remains separate. The project
test has already been examined; these are further benchmark evaluations with
training-only fitting and cutoff selection. Raw local timing is observational
and cannot establish distributed speedup. The canceled 10/25/50% volume plan
was never executed; the separately authorized 10,354-row controls supplement
the full-data primary experiment.
