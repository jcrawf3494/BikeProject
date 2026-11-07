# Project state

Snapshot: **November 7, 2026**. Latest saved model experiments: November 1, 2026. This snapshot publishes existing work and verification records; it does not claim new training results from the publication date.

## Objective and implemented methods

Predict Customer (positive label 1) versus Subscriber (label 0). The proposed families are Scala/Spark gradient-boosted trees (P1), random forests (P2), and logistic regression (P3), with inverse-frequency class weights and training-only three-fold model selection. Customer F0.5 is primary; precision, recall, accuracy, PR area, and runtime provide context.

The preserved supplied split contains **536,406 training trips** and **133,553 test trips** from **669,959 trips**. Training has 82,490 Customers and test has 20,723. Trip/calendar features, daily city weather, k=5 start/end station clusters, and round-trip/between-cluster flags are prepared. IDs, target text, and raw timestamps are excluded from classifier predictors.

Numeric imputation and learned categorical transformations are fitted within classifier training folds. Class weights are recomputed per fit. A distinct training calibration subset chooses the probability cutoff before final refitting. Model persistence probes check saved and restored predictions.

## Completed work

- Data count, label, overlap, station/weather join, raw-feature semantics, and original-input checksum checks.
- Spark P1/P2/P3 implementations, native categorical handling for trees, one-hot LR, and optional interactions derived from existing features.
- Initial core-feature runs, full-feature presets, CV searches, and a full-data P1 three-seed final evaluation.
- Expanded standard-feature full-data CV searches for all three Spark models.
- Recall operating-point experiment using copies of existing P1 models and unchanged probabilities.
- Full-training-pool XGBoost validation control without sampling.
- 10,354-row reconstruction with three feature profiles, two sampling methods, two evaluation interpretations, ten predefined parameter configurations per recipe, and three training CV folds.
- Frozen three-seed reconstruction evaluation: 12 original-project-test model outcomes and six compressed-holdout outcomes.
- Technical fixture checks for native categories/interactions and a depth-7, 240-iteration GBT stack/checkpoint correction.
- Independent result checks and a complete aggregate run inventory.

## Principal measured outcomes

| Experiment | Scope | Customer F0.5 |
|---|---|---:|
| Initial core-feature P1 | Fixed project test, one seed | 0.580092 |
| Full-feature P1 preset, depth 3 / 20 rounds | Fixed project test, one seed | 0.708569 |
| Full-feature tuned P1, depth 5 / 80 rounds | Fixed project test, mean of three seeds | 0.717260 |
| Same P1 models under a validation recall-target cutoff | Fixed project test, mean of three seeds | 0.689828 |
| Expanded native P1, depth 7 / 240 rounds | Original-training validation | 0.717141 |
| Expanded native P2 | Original-training validation | 0.714662 |
| Expanded P3 | Original-training validation | 0.686073 |
| Full-data XGBoost supplemental control | Same original-training validation population | 0.711456 |
| Legacy 60,810-row sampled XGBoost approximation | Fixed project test | 0.557954 |
| 10,354-row similarity reconstruction, reference cutoff 0.5 | Fixed project test, three-seed mean | 0.671597 |
| 10,354-row similarity reconstruction, calibrated cutoff | Fixed project test, three-seed mean | 0.802624 |
| 10,354-row stratified XGBoost control, tuned | Fixed project test, three-seed mean | 0.879851 |
| 10,354-total-row compressed similarity reference | 2,071-row compressed holdout, three-seed mean | 0.815863 |
| Same compressed procedure, calibrated cutoff | Compressed holdout, three-seed mean | 0.827035 |

Do not compare validation scores as though they were final-test scores, or compressed-holdout scores as though they measured the original population. Full per-seed outcomes, all early runs, reference/tuned comparisons, and technical checks are in [experiments/LEDGER.md](experiments/LEDGER.md).

## Interpretation and baseline correction

The reference publication describes 10,354 sampled rows and plots F0.5 near 0.81, without an exact tabulated XGBoost F0.5 or sufficient implementation details for exact replication. Its composite ranking value 11.465292 is a different quantity. The 60,810-row implementation is an exploratory approximation and must not be called the paper's experiment.

The reconstructed compressed reference averages 0.815863 and is consistent with the approximate chart level under the declared assumptions. The original-project-test calibrated similarity reconstruction averages 0.802624 with substantial seed sensitivity (sample SD 0.044910). The stratified control averages 0.879851, but changes the sampling procedure. All procedures remain explicitly labeled.

Figure 9 of the reference study includes a rider-locality feature. The reconstruction assumes a valid five-digit US rider ZIP beginning with 94 or 95, with missing separate; the original definition is unavailable. The proposed Spark models exclude locality and the early-period flag. On identical stratified development IDs with the same reference classifier settings, locality raises validation F0.5 from 0.705868 to 0.864998. This feature mismatch is material. The baseline's higher score does not establish a causal advantage from using fewer rows.

The original conjecture that full-data P1 consistently exceeds the reconstructed baseline is **not established**. Earlier P1 exceeds the assumed untuned similarity reference but scores below its calibrated extension. A fair causal comparison needs matching inputs, disclosed sampling, and comparable selection budgets.

A separately reported full-data Spark XGBoost comparator has F0.5 0.7708427522330429. Its metric formula is arithmetically consistent, but its full local artifacts and exact test IDs are unavailable here. It is documented separately rather than treated as an independently verified local run.

## Incomplete and failed work

The registry contains 23 completed run folders, one partial run, one confirmed failed run, and four incomplete folders without a completion record. Partial results remain usable only for their completed models.

The October 29 extended P1 attempt failed with task-serialization StackOverflowError. Enabling periodic checkpoints and increasing the Java thread stack was verified with a 240-iteration fixture. A subsequent all-model run has only P1 completion evidence; its interruption cause is unspecified. The October 30 all-model validation run completed.

Full-data interaction searches, final three-seed test evaluation of the newest P1/P2/P3 settings, the frozen-setting data-volume study, matched-feature full-data versus sampled comparisons, and clean runtime benchmarks remain pending. Neural networks, factorization machines, and blends were discussed as supplemental possibilities but were not implemented or evaluated.

## Runtime and validity limits

Experiments used a local two-thread Spark runtime. Wall-clock timings affected by long system pauses or memory pressure should not be used to claim a speedup. Python PR area and Spark's default-binned PR area are also not identical computations; F0.5 remains the primary comparison.

The supplied time-of-day/date predictors match original local-clock CSV values. Stored raw timestamps have a legacy timezone encoding and are excluded from predictors. Newly generated datasets must preserve verified local-clock semantics. The original split memberships must be retained for exact comparison; regenerating a split can produce different rows even with the same nominal seed.

The fixed project test has already been examined in earlier work. Later results are additional benchmark experiments using training-only selection, rather than a fresh independent confirmation set. The later shared weather cleaning described elsewhere was not silently substituted into historical results.

## Next experiments

1. Agree on and document a matched feature list for sampled XGBoost and full-data Spark, treating any addition of locality/early-period inputs as an explicit extension of the proposed Spark predictors.
2. Compare sampling size while keeping classifier, preprocessing, target definition, cutoff procedure, and evaluation population consistent.
3. Run full-data interaction-feature ablations within training-only selection.
4. Freeze settings and complete three-seed evaluation of the newly selected P1/P2/P3 configurations.
5. Complete the proposed 10/25/50/100% P1 volume experiment with frozen settings and accurately bounded timing.

## Public artifact scope

Current code, historical scientific source snapshots, aggregate experiment records, selected settings, audit summaries, and reproducibility documentation are published. Private row-level inputs, locality mappings, predictions, fitted model binaries, local runtime bundles, machine logs, and original documents are excluded. See [PRIVACY.md](PRIVACY.md).
