# Original paper: implementation comparison and useful next experiments

Execution examples run from the repository root. Restore private data and local execution artifacts as described in [data preparation](../data/README.md); row-level inputs and fitted binaries are not committed.


November 1 update: the 10,354-row reconstruction and its original-project-test/compressed-holdout evaluations are now complete. See [RECONSTRUCT_10354.md](RECONSTRUCT_10354.md) and [the measured results](../validation/RECONSTRUCTION_10354_RESULTS.md). The historical experiment notes below describe earlier implementation stages.

Sources reviewed: the supplied six-page paper, DOI https://doi.org/10.1145/3335656.3335704, especially Sections 2.2, 4.1, 4.2.1-4.2.3 and Figures 7-9; and the supplied project proposal.

## What the paper establishes

- The task is Customer/non-Subscriber versus Subscriber, with F0.5 primary and accuracy secondary (Section 4.1).
- It scales features to a zero-to-one range and describes removing similar training examples with maximum feature deviation 0.03. The reported compressed dataset contains 10,354 rows (Section 4.2.1).
- It compares classifier scores and runtime, ranking XGBoost first by an overall score. The value 11.465292 is a composite ranking score, not F0.5. Figure 8 suggests F0.5 near 0.81 but supplies no exact tabulated score.
- The Figure 9 importance chart counts feature appearances in trees. Duration and hour are the top two entries, followed by wind direction and weather variables including pressure, humidity, temperature, dew point, and gust. Importance counts are not evidence that copying a feature guarantees a score improvement.
- Section 2.2 highlights different hour-of-day patterns on business and non-business days. This motivates testing interactions of existing time/calendar inputs.
- It discusses k=3 and k=5 station clustering; the proposal fixes k=5, which the implementation preserves.

The paper does not provide a reproducible complete classifier script, exact final hyperparameters, an exact precision/recall breakdown, or enough split/sampler details to match its chart reliably. It does not report a special recall-improvement procedure. Its XGBoost explanation is not a specification of Spark GBT hyperparameters.

## Comparison with this project

| Item | Current implementation | Implication |
|---|---|---|
| Duration, hour, weekday/month, daily weather | Present, including pressure/dew point/gust/wind direction | Most of the paper's main feature families have already been incorporated |
| Business-day versus non-business-day behavior | Separate hour/calendar predictors, business flag; no explicit hour-by-business category | A derived interaction is a reasonable ablation, especially for logistic regression; any benefit remains untested |
| Similarity down-sampling | B0 retains 60,810 rows using class-aware quantized cells | Baseline fidelity is the clearest reproduction gap; inspect features, scaling, representative choice, and boundary-neighbor handling |
| Model families | Spark GBT/RF/LR; XGBoost as B0 | Keep the three proposed Spark families; replacing P1 with XGBoost would change the proposed comparison |
| Class weights | Full-data Spark models use N/(2*N_class) | Keep these specified weights and full training data for the primary experiments; down-sampling belongs to B0 |
| Recall operating point | Increasing mean test recall from 46.8% to 59.5% reduced F0.5 from 0.7173 to 0.6898 | Threshold adjustment alone has not improved the primary metric |

Figure 9 includes local-ZIP and early-period variables outside the declared proposed predictor list. They remain excluded. The paper does not establish how much of its score is attributable to these variables, so they cannot be assumed to explain the performance gap.

## Specific clues from our validation data

Among 13,672 missed validation Customers, 12,834 (93.9%) took trips of at most 20 minutes. In the 10-20 minute band the model found 1,663 of 8,745 Customers, recall 19.0%; longer trips were much easier. The next analysis should focus on short/medium trips and their hour, business-day, round-trip, weather, and station-cluster combinations. This uses existing proposed inputs and training-validation labels only.

The following are hypotheses to test, not settings claimed to appear in the paper:

1. Audit B0's feature/scaling/sampling parity before interpreting the score gap. Keep the project's original test distribution unchanged and report the actual retained training count; do not force the sample to 10,354 merely to match a reported number.
2. Complete P2/P3 validation to establish which proposed model is strongest. Retain P1's frozen three-seed result as the current reference.
3. Test explicit hour-by-business-day and duration-by-calendar interactions derived from the existing features, most directly for the additive logistic model. Apply the same feature version to all models when making matched-feature comparisons.
4. Test a compact, paper-inspired weather subset against the existing full set. Its abbreviated feature labels do not specify every raw weather aggregation, so declare the chosen mapping. Evaluate changes through training-only CV/validation rather than importance counts alone.
5. If further P1 tuning is justified, the current search selected its largest tested depth/iteration combination (5/80). Targeted candidates could pair step size 0.05 with 160 trees at depth 5, and compare depth 7 under the same training-validation protocol. These are proposed experiments, not implemented defaults or guaranteed improvements.

The proposal defines success as P1 exceeding a reproduced B0 on the shared test set consistently across three seeds. The current P1 result exceeds the implemented approximate B0, but exact-reproduction and repeated-baseline limitations must remain explicit. The chart estimate near 0.81 is context, not a precisely specified passing threshold for the proposal.
