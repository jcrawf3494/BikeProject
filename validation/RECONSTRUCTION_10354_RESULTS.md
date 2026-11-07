# Completed 10,354-row XGBoost reconstruction

Completed November 1, 2026. These results replace the 60,810-row exploratory approximation as the primary documented paper-reconstruction experiment. They do not claim an exact replication of unavailable authors' code.

The paper reports 10,354 sampled rows and plots F0.5 near 0.81. Its original sampler, exact feature definitions, final hyperparameters, cutoff, and split are incompletely specified. The reconstruction follows the visible feature families and min-max/similarity description with explicit assumptions. See [RECONSTRUCT_10354.md](../docs/RECONSTRUCT_10354.md).

## Three-seed measured results

| Procedure | Classifier fitting rows | Evaluation population | Mean Customer F0.5 | Sample SD |
|---|---:|---|---:|---:|
| Similarity reconstruction, reference cutoff 0.5 | 10,354 | Original 133,553 project test | 0.671597 | 0.041172 |
| Similarity reconstruction, training-calibrated cutoff | 10,354 | Original 133,553 project test | 0.802624 | 0.044910 |
| Stratified size-matched control, reference cutoff 0.5 | 10,354 | Original 133,553 project test | 0.869427 | 0.003441 |
| Stratified size-matched control, training-selected parameters/cutoff | 10,354 | Original 133,553 project test | 0.879851 | 0.003556 |
| Compressed similarity dataset, reference cutoff 0.5 | 8,283 | 2,071 compressed-sample holdout | 0.815863 | 0.015020 |
| Compressed similarity dataset, training-calibrated cutoff | 8,283 | 2,071 compressed-sample holdout | 0.827035 | 0.004218 |

All means use seeds 42, 43, and 44. Shared-test results reuse the same fixed project test trips. Compressed holdouts are different 2,071-row test populations drawn from compressed original-training subsets; their metrics are not interchangeable with project-test metrics. The 10,354-total-row interpretation uses 8,283 fitting rows plus 2,071 holdout rows. No project test row is included in any fitting or compressed cohort.

The compressed reference mean of 0.815863 is close to the plotted 0.81, using the assumed reference classifier and cutoff 0.5. On the original project test distribution, the similarity procedure averages 0.671597 at cutoff 0.5 and 0.802624 after training-only cutoff calibration. The stratified control exceeds 0.81 but changes the sampling procedure; it is not presented as reproducing the authors' similarity sampler.

## Scores by seed

| Seed | Shared similarity reference | Shared similarity calibrated | Shared stratified reference | Shared stratified tuned | Compressed reference | Compressed calibrated |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.639873 | 0.751683 | 0.871557 | 0.880342 | 0.815440 | 0.822195 |
| 43 | 0.718124 | 0.819683 | 0.871267 | 0.883136 | 0.831089 | 0.828984 |
| 44 | 0.656793 | 0.836506 | 0.865457 | 0.876075 | 0.801058 | 0.829926 |

## Why the reconstruction differs from the earlier models

Figure 9 includes rider locality and an early-period flag, which the proposed Spark models excluded. The reconstruction includes those variables in a paper-visible feature variant and repeats a no-locality ablation. Rider locality is derived from the original rider postal code, not the station/weather ZIP. The declared local definition is a valid five-digit US ZIP beginning with 94 or 95, with missing separate; the original paper does not define its locality flag precisely.

On exactly the same 10,354 stratified development sample IDs and the same reference classifier settings, adding locality raises original-training validation F0.5 from 0.705868 to 0.864998. This is a training-validation feature ablation, not a project-test-selected feature change. It establishes that matching the input variables matters greatly; the high scores cannot be credited solely to using 10,354 rows.

The similarity sampler changes class proportions substantially. In final full-training sampling, the seed-42 candidate representatives produce a sample with 4,447 Customers out of 10,354 (42.95%), versus 1,592 Customers (15.38%) in the stratified control. The project test has 20,723 Customers out of 133,553 (15.52%). This distribution difference helps explain why a compressed-sample holdout score and a population test score diverge. It does not establish the authors' original class proportions.

The sampler first retains one representative per width-0.03 feature cell and class, then selects exactly 10,354 representatives. Boundary-neighbor merging and the authors' order are unknown. The exact-size adjustment and class-aware cells are explicit assumptions. Stratified sampling is a separate control, not silently substituted for similarity sampling.

## Selection and verification

Development evaluated three feature profiles and two sampling methods under two declared evaluation interpretations. Each used ten predefined XGBoost configurations and three training CV folds. Fifteen percent of fitting rows were reserved for cutoff calibration. Numeric imputation, categorical vocabulary, and model scaling were fitted inside each classifier fitting fold. Sampler geometry was fitted only on the original fitting pool, excluding outer validation and project test rows.

The shared-test variant with the best original-training validation result was frozen before project test scoring. The paper-visible similarity procedure was independently retained as the primary reconstruction control. Its parameters and those of the compressed procedure came from training CV. Reference cutoff-0.5 scores remain separate from tuned/calibrated extensions. Model settings were not selected using final project-test scores.

The independent audit checked all 36 development model results, 12 project-test results, and 6 compressed-holdout results. It verified exact sample sizes, original IDs/labels, disjoint fitting/calibration/scoring roles, CV parameter selections, calibration optima by brute force, all saved prediction/confusion arithmetic, and frozen-settings timing. Each saved model also passed a probability reload probe. See reconstruction_10354_audit.json.

## Comparison with P1 and the conjecture

The earlier full-data Spark P1 mean test F0.5 remains 0.717260. It exceeds the assumed untuned similarity reference, while the training-calibrated similarity reconstruction scores higher than P1 on the same project test. The original proposal's superiority conjecture is not established by these results. P1 and the paper-visible reconstruction also use different feature sets, so the difference does not isolate training volume, execution framework, or classifier family.

The newer extended Spark validation scores are P1 0.717141, P2 0.714662, and P3 0.686073. Final test evaluation of those newly selected Spark configurations remains pending. The paper's approximate chart score, the reproduced project-test scores, and compressed-holdout scores must retain their separate labels.

The project test set had been viewed in earlier work. These are additional benchmark experiments using training-only selection, not a fresh independent confirmation set. Full-data versus sampled effects require matched features, sampling disclosure, and comparable tuning budgets.

## Reproducibility files

- Development run: `reconstruct-10354-20261101T192542Z-f8094711`.
- Final run: `baseline-10354-test-20261101T193451Z-48e5ef1a`.
- `reconstruction_10354_frozen_settings.json`: settings frozen before final project-test evaluation.
- `reconstruction_10354_development_results.json`: every declared development result.
- `reconstruction_10354_fixed_test_results.json`: all fixed-project-test results.
- `reconstruction_10354_compressed_results.json`: all compressed-holdout results.
- `reconstruction_10354_audit.json`: independent verification and mean/SD calculations.
- `reconstruction_10354_code_versions.json`: phase-specific source and frozen-settings hashes.
- Full samples, fitting/scoring IDs, calibration predictions, models, and predictions remain in the original result folders. This public repository includes code and compact aggregate records. Derived locality mappings, fold membership rows, and full private run folders remain excluded; restore them locally before execution.
