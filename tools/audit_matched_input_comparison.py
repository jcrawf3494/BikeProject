"""Independently verify saved cohorts, predictions and the matched-input matrix."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

WORK = Path(__file__).resolve().parent
sys.path[:0] = [str(WORK / 'baseline-runtime'), str(WORK / 'vendor')]
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def table(path, columns=None):
    parts = sorted(path.glob('*.parquet')) if path.is_dir() else [path]
    assert parts, 'Missing Parquet input: ' + str(path)
    sources = [('\\\\?\\' + str(p.resolve())) if os.name == 'nt' else str(p.resolve()) for p in parts]
    return pa.concat_tables([pq.ParquetFile(p).read(columns=columns) for p in sources])


def mapping(path, column='label'):
    data = table(path, ['id', column]).to_pydict()
    result = dict(zip(data['id'], data[column]))
    assert len(result) == len(data['id'])
    return result


def metrics(y, probability, threshold):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    predicted = probability > threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                customer_f0_5=1.25 * tp / (1.25 * tp + fp + .25 * fn),
                customer_precision=tp / (tp + fp) if tp + fp else 0.,
                customer_recall=tp / (tp + fn), accuracy=(tp + tn) / len(y))


def compare_metrics(actual, expected):
    for name, value in expected.items():
        assert abs(actual[name] - value) < 1e-12, 'Metric differs: ' + name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=WORK.parent / 'outputs/BikeShareModelTraining')
    parser.add_argument('--cohorts-only', action='store_true')
    parser.add_argument('--spark-only', action='store_true')
    args = parser.parse_args()
    package = args.package.resolve()
    primary = package / 'validation/paper_flags_full_20261108'
    samples = package / 'validation/paper_input_sample_controls_20261108'
    xgb = package / 'validation/matched_feature_xgboost_20261108'
    out = package / 'validation/matched_input_comparison_20261108'
    protocol = load(samples / 'protocol.json')
    settings = load(samples / 'frozen_selected_settings.json')
    assert sha(samples / 'frozen_selected_settings.json') == protocol['settings_sha256']
    assert sha(primary / 'frozen_selected_settings.json') == protocol['settings_sha256']
    assert sha(samples / 'cohort_preparation_audit.json') == protocol['cohort_preparation_audit_sha256']
    assert sha(WORK / 'prepare_sample_comparison_cohorts.py') == protocol['cohort_preparation_source_sha256']
    for name, fingerprint in protocol['scala_source_sha256'].items():
        assert sha(package / 'src' / name) == fingerprint
    assert sha(package / 'target' / protocol['compiled_jar']) == protocol['compiled_jar_sha256']
    features = settings['numeric_features'] + settings['categorical_features']
    assert len(features) == len(set(features)) == 39
    train_path = package / 'data/paper_flags_full/enriched/train.parquet'
    test_path = package / 'data/paper_flags_full/enriched/test.parquet'
    full = table(train_path)
    train = mapping(train_path)
    test = mapping(test_path)
    assert len(train) == 536406 and len(test) == 133553
    assert sum(train.values()) == 82490 and sum(test.values()) == 20723
    assert train.keys().isdisjoint(test)
    preparation = load(samples / 'cohort_preparation_audit.json')
    baseline = package / 'results' / preparation['source_baseline_run']
    cohorts = {('full', seed): train for seed in (42, 43, 44)}
    checks = []
    for case in preparation['cases']:
        cohort, seed = case['sampler'], case['seed']
        saved = baseline / case['saved_ids_path']
        assert sha(saved) == case['saved_ids_sha256']
        ids = mapping(saved)
        assert len(ids) == 10354 and ids.keys() <= train.keys()
        assert all(train[i] == y for i, y in ids.items())
        folder = package / 'data/pfc' / case['cohort_relative_path']
        part = folder / 'enriched/train.parquet/part-00000.parquet'
        assert sha(part) == case['training_parquet_sha256']
        assert mapping(part) == ids
        selected = table(part)
        original = full.filter(pc.is_in(full['id'], value_set=selected['id']))
        assert selected.equals(original, check_metadata=False)
        for relative, fingerprint in case['full_input_hashes'].items():
            assert sha(package / 'data/paper_flags_full' / relative) == fingerprint
        for copied in case['test_copies']:
            assert sha(folder / copied['copy']) == copied['sha256']
            assert sha(test_path / copied['source']) == copied['sha256']
        assert mapping(folder / 'enriched/test.parquet') == test
        cohorts[cohort, seed] = ids
        checks.append(dict(cohort=cohort, seed=seed, training_rows=len(ids),
                           customer_rows=int(sum(ids.values())), customer_rate=sum(ids.values()) / len(ids),
                           saved_reconstruction_ids_exact=True, all_original_values_equal=True,
                           same_test_bytes=True))
    assert len(checks) == 6
    cohort_report = dict(passed=True, audited_utc=datetime.now(timezone.utc).isoformat(),
                         feature_count=39, test_rows=len(test), cases=checks)
    write(out / 'cohort_independent_audit.json', cohort_report)
    if args.cohorts_only:
        print(json.dumps(dict(cohorts_verified=6, feature_count=39, same_test_rows=133553)))
        return
    module = importlib.util.spec_from_file_location('saved_run_audit', WORK / 'audit_improved_runs.py')
    audit = importlib.util.module_from_spec(module)
    module.loader.exec_module(audit)
    records, spark_checks, role_maps = [], [], {}
    for cohort in ('full', 'similarity', 'stratified'):
        for seed in (42, 43, 44):
            for model in ('P1', 'P2', 'P3'):
                status_path = primary / f'{model}_seed_{seed}.json' if cohort == 'full' else samples / f'{cohort}_{model}_seed_{seed}.json'
                status = load(status_path)
                assert status['status'] == 'completed'
                run = package / status['run']
                selected = cohorts[cohort, seed]
                check = audit.audit_run(run, selected, test, settings)
                manifest = load(run / 'manifest.json')
                assert manifest['numeric_features'] == settings['numeric_features']
                assert manifest['categorical_features'] == settings['categorical_features']
                assert manifest['compiled_artifact'] == protocol['compiled_jar']
                assert manifest['evaluation'] == 'test' and manifest['master'] == 'local[2]'
                assert manifest['test_scores_used_for_selection'] is False and manifest['test_split_recreated'] is False
                row, = load(run / 'results.json')
                assert (row['model'], row['seed']) == (model, seed)
                assert row['training_rows'] == row['sample_rows'] == len(selected)
                assert row['evaluation_rows'] == len(test) and row['requested_fraction'] == 1.0
                suffix = f'seed-{seed}/fraction-10000'
                assert mapping(run / 'folds' / suffix) == selected
                roles = mapping(run / 'training_roles' / suffix, 'training_role')
                assert roles.keys() == selected.keys()
                if (cohort, seed) in role_maps:
                    assert roles == role_maps[cohort, seed]
                else:
                    role_maps[cohort, seed] = roles
                check.update(cohort=cohort, seed=seed, model=model)
                spark_checks.append(check)
                records.append(dict(row, cohort=cohort, run_id=run.name,
                                    training_pool_relation='all_original_training' if cohort == 'full' else 'externally_selected_10354',
                                    supplied_cohort_fraction=1.0))
    assert len(spark_checks) == 27
    write(out / 'spark_independent_audit.json', dict(passed=True, model_records=27, runs=spark_checks,
                                                    same_calibration_ids_across_families=True))
    if args.spark_only:
        print(json.dumps(dict(spark_records_verified=27)))
        return
    from sklearn.metrics import precision_recall_curve
    xp = load(xgb / 'protocol.json')
    assert xp['settings_sha256'] == protocol['settings_sha256']
    assert sha(WORK / 'run_matched_feature_xgboost.py') == xp['source_sha256']
    reference = package / 'validation/reconstruction_10354_frozen_settings.json'
    assert sha(reference) == xp['source_baseline_settings_sha256']
    assert xp['parameters'] == load(reference)['parameters']
    assert xp['parameters'] == dict(max_depth=3, n_estimators=400, learning_rate=.05, min_child_weight=1, reg_lambda=1)
    for relative, fingerprint in xp['input_parquet_sha256'].items():
        if relative.startswith('common_test/'):
            path = test_path / relative.removeprefix('common_test/')
        else:
            prefix, filename = relative.rsplit('/train/', 1)
            base = package / 'data/paper_flags_full' if prefix == 'full' else package / 'data/pfc' / prefix
            path = base / 'enriched/train.parquet' / filename
        assert sha(path) == fingerprint
    xgb_checks = []
    for cohort in ('full', 'similarity', 'stratified'):
        for seed in (42, 43, 44):
            state = load(xgb / f'{cohort}_seed_{seed}.json')
            assert state['status'] == 'completed'
            row = load(package / state['metrics'])
            selected = cohorts[cohort, seed]
            artifact = package / row['raw_artifact_directory']
            assert mapping(artifact / 'fitting_ids.parquet') == selected
            assert mapping(artifact / 'training_roles.parquet', 'training_role') == role_maps[cohort, seed]
            pred = table(artifact / 'predictions.parquet').to_pydict()
            assert dict(zip(pred['id'], pred['label'])) == test
            assert len(pred['id']) == len(test)
            threshold = row['decision_threshold']
            assert np.array_equal(np.asarray(pred['prediction']), np.asarray(pred['customer_probability']) > threshold)
            compare_metrics(row, metrics(pred['label'], pred['customer_probability'], threshold))
            cal = table(artifact / 'calibration_predictions.parquet').to_pydict()
            expected_cal = {i: selected[i] for i, role in role_maps[cohort, seed].items() if role == 'calibration'}
            assert dict(zip(cal['id'], cal['label'])) == expected_cal and len(cal['id']) == len(expected_cal)
            precision, recall, _ = precision_recall_curve(cal['label'], cal['customer_probability'])
            denominator = .25 * precision + recall
            values = np.divide(1.25 * precision * recall, denominator, out=np.zeros_like(precision), where=denominator != 0)
            policy = load(artifact / 'threshold_selection.json')
            assert policy['test_used'] is False and policy['threshold'] == threshold
            actual = metrics(cal['label'], cal['customer_probability'], threshold)
            compare_metrics(policy['selected_threshold_metrics'], actual)
            assert abs(actual['customer_f0_5'] - float(values.max())) < 1e-12
            compare_metrics(policy['default_threshold_metrics'], metrics(cal['label'], cal['customer_probability'], .5))
            assert row['training_rows'] == len(selected) and row['evaluation_rows'] == len(test)
            assert row['calibration_rows'] == len(expected_cal)
            assert row['model_selection_rows'] + row['calibration_rows'] == len(selected)
            for key, data in [('final_fit_weights', selected), ('calibration_fit_weights', {i: y for i, y in selected.items() if i not in expected_cal})]:
                count = int(sum(data.values()))
                expected_weights = dict(Customer=len(data) / (2 * count), Subscriber=len(data) / (2 * (len(data) - count)))
                for name, value in expected_weights.items():
                    assert abs(row[key][name] - value) < 1e-12
            assert row['parameters'] == xp['parameters'] and row['model_reload_verified'] is True
            assert row['input_numeric_features'] == settings['numeric_features']
            assert row['input_categorical_features'] == settings['categorical_features']
            xgb_checks.append(dict(cohort=cohort, seed=seed, predictions_and_confusion_verified=True,
                                   calibration_optimum_independently_verified=True,
                                   exact_spark_calibration_ids=True, inverse_weights_verified=True))
            records.append(dict(row, run_id=Path(row['raw_artifact_directory']).parts[1],
                                training_pool_relation='all_original_training' if cohort == 'full' else 'externally_selected_10354'))
    assert len(records) == 36
    summary = []
    fields = ('customer_f0_5', 'customer_precision', 'customer_recall', 'accuracy', 'final_fit_seconds', 'threshold_selection_seconds')
    models = ('P1', 'P2', 'P3', 'XGBoost_matched_inputs_weighted')
    for model in models:
        for cohort in ('full', 'stratified', 'similarity'):
            rows = [r for r in records if r['model'] == model and r['cohort'] == cohort]
            assert sorted(r['seed'] for r in rows) == [42, 43, 44]
            summary.append(dict(model=model, cohort=cohort, training_rows=rows[0]['training_rows'],
                                metrics={f: dict(mean=statistics.mean(r[f] for r in rows),
                                                 sample_sd=statistics.stdev(r[f] for r in rows)) for f in fields}))
    deltas = []
    for model in models:
        for cohort in ('stratified', 'similarity'):
            for seed in (42, 43, 44):
                full_row = next(r for r in records if (r['model'], r['cohort'], r['seed']) == (model, 'full', seed))
                small_row = next(r for r in records if (r['model'], r['cohort'], r['seed']) == (model, cohort, seed))
                deltas.append(dict(model=model, seed=seed, sampled_cohort=cohort,
                                   full_minus_sample_f0_5=full_row['customer_f0_5'] - small_row['customer_f0_5']))
    write(out / 'all_results.json', records)
    write(out / 'summary.json', summary)
    write(out / 'paired_training_size_differences.json', deltas)
    report = dict(passed=True, audited_utc=datetime.now(timezone.utc).isoformat(), model_records=36,
                  spark_model_records=27, xgboost_model_records=9, feature_count=39, test_rows=133553,
                  cohort_checks=checks, spark_checks=spark_checks, xgboost_checks=xgb_checks,
                  limitation='Fixed configurations transferred across training sizes; similarity changes class mix; test previously examined; original paper remains an approximate reconstruction')
    write(out / 'independent_audit.json', report)
    columns = ['model', 'cohort', 'seed', 'training_rows', 'evaluation_rows', 'customer_f0_5', 'customer_precision',
               'customer_recall', 'accuracy', 'decision_threshold', 'final_fit_seconds', 'threshold_selection_seconds', 'run_id']
    with (out / 'all_results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    text = ['# Matched-input training-size comparison', '',
            'All cases use the same 39 input columns, fixed 133,553 test trips, and seeds 42, 43, 44. '
            'The full cohort has 536,406 training trips; each saved sampled cohort has 10,354. '
            'Cutoffs use calibration rows within the supplied training cohort, followed by a refit on its entire membership.', '',
            '| Model | Training cohort | Rows | Mean F0.5 | Sample SD | Precision | Recall |',
            '|---|---|---:|---:|---:|---:|---:|']
    for item in summary:
        m = item['metrics']
        text.append(f"| {item['model']} | {item['cohort']} | {item['training_rows']:,} | {m['customer_f0_5']['mean']:.6f} | {m['customer_f0_5']['sample_sd']:.6f} | {m['customer_precision']['mean']:.2%} | {m['customer_recall']['mean']:.2%} |")
    text += ['', 'The same hyperparameters are held fixed across training sizes. They were selected in previous training-only searches on different input profiles; these results do not establish the best possible model for each size. '
             'Calibration memberships match across model families within each cohort and seed. Stratified samples approximately preserve the original class mix, while similarity samples change it.', '',
             'The weighted 39-input XGBoost control is a new experiment. The earlier paper-style reconstruction, its assumed locality definition, and the approximate Figure 8 score remain separate references. '
             'The fixed project test was examined previously; these are benchmark comparisons. Weather remains historical V1. Local timings are observational and do not demonstrate cluster speedup.']
    (out / 'RESULTS.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, model_records=36, comparison_cells=12)))


if __name__ == '__main__':
    main()
