# Local input preparation

Data is excluded from the public repository. Obtain the original bike-share data and the original shared Parquet split through the project's approved data-sharing process. Reference dataset: https://www.kaggle.com/datasets/benhamner/sf-bay-area-bike-share.

Required local layout:

```text
data/
  train.parquet/                  # preserved original 536,406 training rows
  test.parquet/                   # preserved original 133,553 test rows
  enriched/train.parquet/         # same training IDs/labels, weather and k=5 clusters
  enriched/test.parquet/          # same test IDs/labels
  raw/station.csv
  raw/weather.csv
  raw/paper_locality.parquet      # derived trip/locality mapping; private
  validation_folds.parquet        # saved original-training Spark folds; private
```

The audit checksums in `validation/data_checksums.json` identify the original Parquet inputs. The original membership must be preserved to reproduce historical numerical results. Regenerating a nominal seed-42 split can produce different IDs because input ordering/partitioning matters.

`scripts/build-shared-features.scala` is an ingestion scaffold for raw `trip.csv` and `station.csv`. It uses explicit local-clock parsing, declares UTC while interpreting the original clock fields, derives calendar/trip features, and writes a new output directory. Set `BIKESHARE_RAW` and `BIKESHARE_BASE_OUTPUT` to local folders. It refuses to overwrite existing outputs. Freshly generated data is a new dataset version until its IDs and feature semantics have been checked against the preserved split.

`PrepareBikeShareFeatures` adds the independently audited weather and station-cluster features to a supplied original split. It checks station/weather join coverage and split preservation. With a compiled JAR:

```powershell
spark-submit --class PrepareBikeShareFeatures --master 'local[2]' target/YOUR-CURRENT-MODEL-JAR.jar --data-dir data --raw-dir data/raw --output-dir data/enriched
```

Use the launcher's Windows native-helper configuration where required. Do not replace earlier feature tables without versioning the inputs and refitting affected experiments.

To derive the reconstruction's locality mapping from the original trip archive:

```powershell
python baseline/prepare_paper_locality.py --trip-archive PATH-TO-ORIGINAL-DATA.zip --output data/raw/paper_locality.parquet
```

The mapping stores trip ID and the derived local/outside/missing category, not raw postal values. Both the mapping and original archive remain private.

Supply the original saved Spark fold assignments through `--validation-folds`, or place a single Parquet table with `id`, `label`, and `cv_fold` under `data/validation_folds.parquet`. It must cover the whole original training pool exactly once, with fold values 0/1/2 and matching labels. The reconstruction holds fold 0 out during development.

Raw predictions, sample IDs, scoring IDs, and fitted models are local execution artifacts. Use the independent audit tools before exporting aggregate results. CI does not recreate private datasets from aggregate records.
