# BikeProject

Customer-versus-Subscriber classification using Scala/Spark MLlib and a documented 10,354-row Python/XGBoost reconstruction of the reference experiment.

## Current state

Snapshot: **November 7, 2026**. [PROJECT_STATE.md](PROJECT_STATE.md) gives completed work, measured results, limitations, and next experiments. [The experiment ledger](experiments/LEDGER.md) documents **29 run folders** and **96 saved aggregate model outcomes**, including technical checks and incomplete attempts.

| Procedure | Evaluation population | Mean Customer F0.5 |
|---|---|---:|
| Earlier full-data Spark P1, three seeds | Fixed 133,553 project test trips | 0.717260 |
| 10,354-row similarity reconstruction, calibrated | Same fixed project test | 0.802624 |
| 10,354-row stratified XGBoost control, tuned | Same fixed project test | 0.879851 |
| 10,354-total-row compressed similarity reference | 2,071 compressed holdout rows | 0.815863 |

The compressed reference uses 8,283 fitting rows plus 2,071 holdout rows. Its score is close to the paper's plotted 0.81, but its evaluation population differs from the project test. The paper's exact source, sampler, retained IDs, and feature definitions are unavailable. These are reconstructions with declared assumptions. The stratified control uses a different sampling procedure.

Expanded full-data training-validation searches have completed for all three Spark families: P1 **0.717141**, P2 **0.714662**, P3 **0.686073**. Final three-seed test evaluation of those newly selected configurations remains pending.

## Repository map

- `src/`: Spark feature preparation, weighted model training/CV, and recall operating-point analysis.
- `baseline/`: XGBoost reconstruction, legacy exploratory comparator, locality preparation, and dependencies.
- `scripts/`: shared feature-table ingestion scaffold.
- `docs/`: methodology, full-data training, and reconstruction instructions.
- `experiments/`: complete aggregate registry, CSV/JSON outcomes, per-run records, and historical source snapshots.
- `validation/`: selected settings, independent audit summaries, measured-result reports, and input checksums.
- `tools/`: independent checks for locally held private run outputs and public-artifact checks.
- `tests/`: threshold edge cases and aggregate/registry integrity checks.

## Runtime and private inputs

Use **Spark 3.5.x / Scala 2.12**, preferably the recorded Spark 3.5.9 / Scala 2.12.18 / JDK 11 environment. A supported JDK 17 is also usable. Windows requires compatible Hadoop native helpers. Set `SPARK_HOME`, `JAVA_HOME`, and `HADOOP_HOME`, or pass them explicitly to the launchers. Historical performance was measured with two local worker threads; no multi-machine speedup has been established.

See [data/README.md](data/README.md) for input requirements. Row-level data, locality mappings, predictions, model binaries, logs, and installed runtimes are intentionally excluded from the public tree. Aggregate experiment records and all current model code are included. Existing private input membership must be preserved to reproduce the saved numerical comparisons.

## Scala/Spark

From PowerShell in the repository root:

```powershell
.\compile-models.ps1 -SparkHome $env:SPARK_HOME -JavaHome $env:JAVA_HOME
.\start-models-spark-shell.ps1 -SparkHome $env:SPARK_HOME -JavaHome $env:JAVA_HOME -HadoopHome $env:HADOOP_HOME
```

At `scala>`, use `:paste`, paste the following block, then press Ctrl+D:

```scala
val project = sys.env("BIKESHARE_PROJECT")
TrainBikeShareModels.main(Array(
  "--data-dir", project + "/data",
  "--output-dir", project + "/results",
  "--mode", "tune", "--evaluation", "validation",
  "--models", "P1,P2,P3", "--fraction", "1.0", "--seeds", "42",
  "--search-budget", "extended", "--tree-encoding", "native",
  "--feature-variant", "standard", "--parallelism", "1"
))
```

Restart the shell after each completed application because it stops its Spark session. The launcher sets a 4 MB thread stack, and the application enables run-specific GBT checkpoints every ten iterations. See [model training](docs/MODEL_TRAINING.md) and [full-data tuning](docs/FULL_DATA_TUNING.md).

## Reconstruct the 10,354-row experiments

In a Python environment with local private inputs prepared:

```powershell
python -m pip install -r baseline/requirements.txt
python baseline/reconstruct_10354.py --mode final --data-dir data --output-dir results --locality-metadata data/raw/paper_locality.parquet --frozen-settings validation/reconstruction_10354_frozen_settings.json
```

This repeats the frozen three-seed experiments. Development commands, sampling/feature assumptions, and the two evaluation interpretations are documented in [the reconstruction guide](docs/RECONSTRUCT_10354.md). [Measured reconstruction results](validation/RECONSTRUCTION_10354_RESULTS.md) include every seed and reference/tuned procedure.

The current proposed Spark models exclude rider locality and the early-period flag. The paper-visible reconstruction includes them under declared assumptions. This feature mismatch prevents attributing the score difference solely to training size or framework.

## Verification

```powershell
python -m pip install -r baseline/requirements.txt
python -m unittest discover -s tests -v
python tools/check_public_artifacts.py
```

CI checks source syntax, threshold edge cases, aggregate score arithmetic, registry coverage, and public-artifact privacy rules. It does not retrain the models or independently recheck raw-row membership without private data. Historical private-data audits are included with their recorded scope; local audit scripts support rechecking the restored inputs and run outputs.

Reference publication: https://doi.org/10.1145/3335656.3335704.
