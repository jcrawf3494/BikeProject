import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import pyarrow.parquet as pq


def read_table(path, columns=None):
    # Read physical parts, avoiding directory partition inference for seed/fraction names.
    path = Path(path)
    files = sorted(path.glob('*.parquet')) if path.is_dir() else [path]
    import pyarrow as pa
    sources = [('\\\\?\\' + str(f.resolve())) if os.name == 'nt' else str(f) for f in files]
    return pa.concat_tables([pq.ParquetFile(f).read(columns=columns) for f in sources]).to_pydict()


def labels(path):
    rows = read_table(path, ['id', 'label'])
    result = dict(zip(rows['id'], rows['label']))
    assert len(result) == len(rows['id'])
    return result


def audit_run(run, train, test, imported_settings=None):
    manifest = json.loads((run / 'manifest.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    checks = []
    for fold_path in sorted((run / 'folds').glob('seed-*/fraction-*')):
        relative = fold_path.relative_to(run / 'folds')
        outer = read_table(fold_path)
        outer_ids = set(outer['id'])
        assert outer_ids <= train.keys() and outer_ids.isdisjoint(test)
        assert all(train[i] == y for i, y in zip(outer['id'], outer['label']))
        roles = read_table(run / 'training_roles' / relative)
        calibration = {i for i, role in zip(roles['id'], roles['training_role']) if role == 'calibration'}
        selection = {i for i, role in zip(roles['id'], roles['training_role']) if role == 'model_selection'}
        assert calibration.isdisjoint(selection)
        assert len(calibration | selection) == len(roles['id'])
        cv = read_table(run / 'model_selection_folds' / relative)
        assert set(cv['id']) == selection
        fold_counts = {}
        for i, y, fold in zip(cv['id'], cv['label'], cv['cv_fold']):
            assert train[i] == y
            fold_counts[y, fold] = fold_counts.get((y, fold), 0) + 1
        assert set(fold_counts) == {(y, f) for y in (0.0, 1.0) for f in (0, 1, 2)}
        for y in (0.0, 1.0):
            assert max(fold_counts[y, f] for f in (0, 1, 2)) - min(fold_counts[y, f] for f in (0, 1, 2)) <= 1
        validation_ids = {i for i, fold in zip(outer['id'], outer['cv_fold']) if fold == 0}
        expected_fit = outer_ids - validation_ids if manifest['evaluation'] == 'validation' else outer_ids
        assert calibration | selection == expected_fit
        checks.append({'partition': str(relative).replace('\\', '/'), 'fitting_rows': len(expected_fit),
                       'calibration_rows': len(calibration), 'model_selection_rows': len(selection),
                       'disjoint_roles': True, 'balanced_three_folds': True})
    cv_parameter_checks = 0
    imported_parameter_checks = 0
    for result in results:
        model_dir = run / result['model'] / f"seed-{result['seed']}" / f"fraction-{round(result['requested_fraction'] * 10000)}"
        pred = read_table(model_dir / 'predictions.parquet')
        ids = set(pred['id'])
        assert len(ids) == len(pred['id']) == result['evaluation_rows']
        if result['evaluation_scope'] == 'fixed_test':
            expected_labels = test
        else:
            outer = read_table(run / 'folds' / model_dir.relative_to(run / result['model']))
            expected_labels = {i: y for i, y, f in zip(outer['id'], outer['label'], outer['cv_fold']) if f == 0}
        assert ids == expected_labels.keys()
        confusion = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        for i, y, p, probability in zip(pred['id'], pred['label'], pred['prediction'], pred['customer_probability']):
            assert expected_labels[i] == y
            assert math.isfinite(probability) and 0 <= probability <= 1
            assert p == float(probability > result['decision_threshold'])
            confusion[('t' if y == p else 'f') + ('p' if p == 1 else 'n')] += 1
        assert all(confusion[k] == result[k] for k in confusion)
        tp, fp, fn = (confusion[k] for k in ('tp', 'fp', 'fn'))
        denominator = 1.25 * tp + fp + 0.25 * fn
        score = 1.25 * tp / denominator if denominator else 0
        assert abs(score - result['customer_f0_5']) < 1e-12
        cv_file = model_dir / 'cross_validation.json'
        if cv_file.exists():
            candidates = json.loads(cv_file.read_text())
            chosen = [c for c in candidates if c['selected']]
            assert len(chosen) == 1
            assert chosen[0]['mean_customer_f0_5'] == max(c['mean_customer_f0_5'] for c in candidates)
            for key, value in chosen[0]['parameters'].items():
                assert result['classifier_parameters'][key] == value
            cv_parameter_checks += 1
        if manifest.get('settings_source', 'none') != 'none':
            assert imported_settings is not None, 'Provide the imported settings for independent verification'
            choices = [r for r in imported_settings['models'] if r['model'] == result['model']]
            chosen = next((r for r in choices if r['seed'] == result['seed']), choices[0])
            for key, value in chosen['parameters'].items():
                assert result['classifier_parameters'][key] == value
            imported_parameter_checks += 1
        if result['calibration_rows']:
            cutoff = json.loads((model_dir / 'threshold_selection.json').read_text())
            assert cutoff['test_used'] is False
            assert cutoff['rows'] == result['calibration_rows']
            assert cutoff['threshold'] == result['decision_threshold']
        assert result['model_reload_verified'] is True
    return {'run': run.name, 'evaluation': manifest['evaluation'], 'models_verified': len(results),
            'role_checks': checks, 'prediction_ids_labels_cutoffs_and_confusion_metrics_verified': True,
            'cv_selected_parameter_records_verified': cv_parameter_checks,
            'imported_parameter_records_verified': imported_parameter_checks,
            'model_reload_checks_passed': True}


def audit_baseline(run, train, test):
    result = json.loads((run / 'metrics.json').read_text())
    assert result['evaluation_scope'] == 'fixed_test'
    pred = read_table(run / 'predictions.parquet')
    assert len(pred['id']) == len(set(pred['id'])) == len(test)
    assert set(pred['id']) == test.keys()
    confusion = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    for i, y, p, probability in zip(pred['id'], pred['label'], pred['prediction'], pred['customer_probability']):
        assert test[i] == y and p == float(probability > result['threshold'])
        assert math.isfinite(probability) and 0 <= probability <= 1
        confusion[('t' if y == p else 'f') + ('p' if p == 1 else 'n')] += 1
    assert all(result[key] == value for key, value in confusion.items())
    tp, fp, fn = (confusion[key] for key in ('tp', 'fp', 'fn'))
    assert abs(1.25 * tp / (1.25 * tp + fp + 0.25 * fn) - result['customer_f0_5']) < 1e-12
    sampled = labels(run / 'sampled_training_ids.parquet')
    assert len(sampled) == result['sampled_training_rows']
    assert sampled.keys() <= train.keys() and sampled.keys().isdisjoint(test)
    assert all(train[key] == value for key, value in sampled.items())
    assert result['model_reload_verified'] is True
    return {'run': run.name, 'fixed_test_ids_labels_cutoff_and_confusion_metrics_verified': True,
            'sampled_ids_exclusively_from_training': True, 'sampled_training_rows': len(sampled),
            'protocol_is_approximation': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    train = labels(args.package / 'data/train.parquet')
    test = labels(args.package / 'data/test.parquet')
    assert train.keys().isdisjoint(test)
    assert labels(args.package / 'data/enriched/train.parquet') == train
    assert labels(args.package / 'data/enriched/test.parquet') == test
    report = {'train_rows': len(train), 'test_rows': len(test),
              'enriched_ids_labels_and_split_preserved': True,
              'runs': [audit_run(run, train, test, json.loads(args.settings.read_text()) if args.settings else None)
                       for run in args.runs]}
    if args.baseline:
        report['baseline'] = audit_baseline(args.baseline, train, test)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
