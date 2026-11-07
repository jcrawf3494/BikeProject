import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

work = Path(__file__).resolve().parent
import numpy as np
import pyarrow.parquet as pq
from audit_improved_runs import read_table, labels


def load(path):
    return json.loads(path.read_text())


def verify_ids(path, original):
    record = labels(path)
    assert record.keys() <= original.keys()
    assert all(original[i] == y for i, y in record.items())
    return record


def verify_prediction(path, record, original):
    frame = read_table(path)
    ids = frame['id']
    assert len(ids) == len(set(ids)) == record['evaluation_rows']
    assert set(ids) == original.keys()
    truth = np.array(frame['label'], dtype=int)
    probability = np.array(frame['customer_probability'], dtype=float)
    prediction = np.array(frame['prediction'], dtype=int)
    assert all(original[i] == y for i, y in zip(ids, truth))
    assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
    assert np.array_equal(prediction, (probability > record['decision_threshold']).astype(int))
    counts = dict(tp=int(np.sum((truth == 1) & (prediction == 1))),
                  fp=int(np.sum((truth == 0) & (prediction == 1))),
                  fn=int(np.sum((truth == 1) & (prediction == 0))),
                  tn=int(np.sum((truth == 0) & (prediction == 0))))
    assert all(record[key] == value for key, value in counts.items())
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    denominator = 1.25 * tp + fp + 0.25 * fn
    score = 1.25 * tp / denominator if denominator else 0
    assert math.isclose(score, record['customer_f0_5'], abs_tol=1e-12)
    assert record['model_reload_verified']


def verify_calibration(path, cutoff):
    frame = read_table(path)
    truth = np.array(frame['label'], dtype=int)
    probability = np.array(frame['customer_probability'], dtype=float)

    def score(threshold):
        pred = probability > threshold
        tp = np.sum(pred & (truth == 1))
        fp = np.sum(pred & (truth == 0))
        fn = np.sum(~pred & (truth == 1))
        den = 1.25 * tp + fp + 0.25 * fn
        return 1.25 * tp / den if den else 0
    cutoffs = np.r_[np.nextafter(probability.min(), -np.inf), np.unique(probability)]
    maximum = max(score(threshold) for threshold in cutoffs)
    assert math.isclose(score(cutoff), maximum, abs_tol=1e-12)
    return set(frame['id'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--final', type=Path)
    args = parser.parse_args()
    package = work.parent
    train = labels(package / 'data/enriched/train.parquet')
    test = labels(package / 'data/enriched/test.parquet')
    assert len(train) == 536406 and len(test) == 133553 and train.keys().isdisjoint(test)
    development = args.development
    manifest = load(development / 'manifest.json')
    outer = verify_ids(development / 'outer_validation_ids.parquet', train)
    assert len(outer) == 178803
    development_records = load(development / 'development_results.json')
    assert len(development_records) == 36
    model_checks = 0
    assert labels(development / 'paper_visible__stratified/sampled_ids.parquet') == labels(
        development / 'paper_without_zip__stratified/sampled_ids.parquet')
    for recipe in sorted(p for p in development.iterdir() if p.is_dir()):
        sampled = verify_ids(recipe / 'sampled_ids.parquet', train)
        info = load(recipe / 'sample.json')
        assert len(sampled) == info['sampled_rows'] == 10354
        assert int(sum(sampled.values())) == info['sampled_customer_rows']
        assert sampled.keys().isdisjoint(outer) and sampled.keys().isdisjoint(test)
        for protocol in ('shared_split', 'compressed_holdout'):
            root = recipe / protocol
            fitting = verify_ids(root / 'fitting_ids.parquet', train)
            scoring = verify_ids(root / 'scoring_ids.parquet', train)
            assert fitting.keys().isdisjoint(scoring)
            if protocol == 'shared_split':
                assert fitting == sampled and scoring == outer
            else:
                assert len(fitting) == 8283 and len(scoring) == 2071
                assert fitting.keys() | scoring.keys() == sampled.keys()
            roles = read_table(root / 'selection_roles.parquet')
            assert set(roles['id']) == fitting.keys()
            calibration = {i for i, role in zip(roles['id'], roles['role']) if role == 'cutoff_calibration'}
            selection = {i for i, role in zip(roles['id'], roles['role']) if role == 'model_selection'}
            assert calibration.isdisjoint(selection) and calibration | selection == fitting.keys()
            cv = load(root / 'cross_validation_progress.json')
            assert len(cv) == 10 and [c['parameters'] for c in cv] == manifest['grid']
            fixed = max(cv, key=lambda r: (r['mean_fixed_f05'], -r['index']))
            calibrated = max(cv, key=lambda r: (r['mean_attainable_f05'], -r['index']))
            settings = load(root / 'selected_settings.json')
            assert settings['fixed_parameters'] == fixed['parameters']
            assert settings['calibrated_parameters'] == calibrated['parameters']
            assert verify_calibration(root / 'calibration_predictions.parquet', settings['cutoff']) == calibration
            assert (root / 'selected_settings.json').stat().st_mtime <= (root / 'tuned_calibrated/metrics.json').stat().st_mtime
            for policy in ('reference_fixed', 'tuned_fixed', 'tuned_calibrated'):
                record = load(root / policy / 'metrics.json')
                assert record['training_rows'] == len(fitting)
                verify_prediction(root / policy / 'predictions.parquet', record, scoring)
                model_checks += 1
    frozen = load(development / 'frozen_project_test_settings.json')
    eligible = [r for r in development_records if r['protocol'] == 'shared_split' and r['policy'] != 'reference_fixed']
    winner = max(eligible, key=lambda r: r['customer_f0_5'])
    assert frozen['source_recipe'] == winner['recipe'] and frozen['parameters'] == winner['parameters']
    assert frozen['policy'] == winner['policy'] and frozen['sampler'] == winner['sampler']
    report = dict(development_run=development.name, development_models_verified=model_checks,
                  all_samples_exactly_10354=True, sampled_holdout_counts_verified=True,
                  original_split_ids_labels_preserved=True, no_project_test_rows_in_development=True,
                  disjoint_calibration_selection_and_scoring_ids=True,
                  cutoff_optima_independently_bruteforced=True,
                  cross_validation_selected_parameter_records_verified=True,
                  saved_predictions_and_confusion_metrics_verified=True,
                  locality_ablation_uses_identical_stratified_sample_ids=True,
                  locality_definition_is_assumed=True, exact_original_experiment_reproduction_claimed=False)
    if args.final:
        final = args.final
        assert (final / 'complete.json').exists()
        final_frozen = load(final / 'frozen_settings_copy.json')
        assert final_frozen['source_recipe'] == winner['recipe']
        assert final_frozen['parameters'] == winner['parameters']
        assert final_frozen['shared_similarity_primary']['parameters'] == load(
            development / 'paper_visible__similarity/shared_split/selected_settings.json')['calibrated_parameters']
        assert final_frozen['compressed_primary']['parameters'] == load(
            development / 'paper_visible__similarity/compressed_holdout/selected_settings.json')['calibrated_parameters']
        summaries = {}
        final_records = load(final / 'results.json')
        assert len(final_records) == 12
        for record in final_records:
            if record.get('family') == 'paper_similarity_primary':
                root = final / 'paper-similarity' / ('seed-' + str(record['seed']))
                policy_folder = record['policy'].replace('similarity_', '')
            else:
                root = final / ('seed-' + str(record['seed']))
                policy_folder = record['policy']
            sampled = verify_ids(root / 'sampled_ids.parquet', train)
            assert len(sampled) == record['training_rows'] == 10354
            assert sampled.keys().isdisjoint(test)
            verify_prediction(root / policy_folder / 'predictions.parquet', record, test)
            expected_params = (final_frozen['reference_parameters'] if 'reference_fixed' in record['policy'] else
                               final_frozen['shared_similarity_primary']['parameters'] if record.get('family') == 'paper_similarity_primary' else
                               final_frozen['parameters'])
            assert record['parameters'] == expected_params
            if 'training_selected' in record['policy'] and (root / 'calibration_predictions.parquet').exists():
                cal_ids = verify_calibration(root / 'calibration_predictions.parquet', record['decision_threshold'])
                assert cal_ids <= sampled.keys()
            assert (final / 'frozen_settings_copy.json').stat().st_mtime <= (root / policy_folder / 'predictions.parquet').stat().st_mtime
            summaries.setdefault(record['policy'], []).append(record['customer_f0_5'])
        compressed_records = load(final / 'compressed_results.json')
        assert len(compressed_records) == 6
        for record in compressed_records:
            root = final / 'compressed' / ('seed-' + str(record['seed']))
            fitting = verify_ids(root / 'fitting_ids.parquet', train)
            scoring = verify_ids(root / 'scoring_ids.parquet', train)
            assert len(fitting) == 8283 and len(scoring) == 2071
            assert fitting.keys().isdisjoint(scoring) and (fitting.keys() | scoring.keys()).isdisjoint(test)
            assert record['training_rows'] == 8283 and record['compressed_total_rows'] == 10354
            verify_prediction(root / record['policy'] / 'predictions.parquet', record, scoring)
            summaries.setdefault('compressed_' + record['policy'], []).append(record['customer_f0_5'])
            if record['policy'] == 'training_selected':
                assert verify_calibration(root / 'calibration_predictions.parquet', record['decision_threshold']) <= fitting.keys()
        report.update(final_run=final.name, final_project_test_predictions_verified=len(final_records),
                      final_compressed_holdout_predictions_verified=len(compressed_records),
                      frozen_settings_copied_before_project_test_predictions=True,
                      summaries={name: dict(seeds=3, mean_f0_5=statistics.mean(values),
                                           sample_sd_f0_5=statistics.stdev(values), scores=values)
                                 for name, values in summaries.items()})
    output = package / 'validation/reconstruction_10354_audit.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
