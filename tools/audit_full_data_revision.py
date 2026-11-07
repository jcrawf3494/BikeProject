import argparse
import importlib.util
import json
from pathlib import Path
import sys

work = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('audit', work / 'audit_improved_runs.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
parser = argparse.ArgumentParser()
parser.add_argument('--native-run', type=Path, required=True)
args = parser.parse_args()
package = work.parent
load = lambda path: json.loads(path.read_text())
train = audit.labels(package / 'data/train.parquet')
test = audit.labels(package / 'data/test.parquet')
old_tune = package / 'results/tune-2026-10-28T03-20-22-656784500Z-f65328f7'
old_final = package / 'results/train-2026-10-28T04-02-57-528292100Z-8d2ae312'
technical = package / 'results/technical/full-tuning-checks/tune-2026-10-28T19-06-23-941768200Z-ae552820'
control = package / 'results/xgboost-full-20261028T192432Z-c24bc870'
report = {'data_semantics': load(package / 'validation/feature_semantics_audit.json'),
          'runs': [audit.audit_run(old_tune, train, test),
                   audit.audit_run(old_final, train, test, load(old_tune / 'selected_settings.json')),
                   audit.audit_run(technical, train, test), audit.audit_run(args.native_run, train, test)]}

native = load(args.native_run / 'results.json')
for result in native:
    assert result['sample_rows'] == len(train) and result['requested_fraction'] == 1.0
    assert result['training_pool_downsampled'] is False
    assert result['evaluation_scope'] == 'training_validation_only'
    assert result['tree_encoding'] == 'native' and result['feature_variant'] == 'standard'
    path = args.native_run / f"{result['model']}/seed-{result['seed']}/fraction-10000/feature_metadata.json"
    metadata = load(path)
    assert metadata['feature_vector_size'] == 37
    assert len(metadata['native_categorical_attributes']) == 6 and metadata['metadata_preserved'] is True
for name in ['P1', 'P2', 'P3']:
    metadata = load(technical / f'{name}/seed-42/fraction-10000/feature_metadata.json')
    assert metadata['feature_variant'] == 'interactions'
    if name != 'P3':
        assert metadata['tree_encoding'] == 'native' and metadata['feature_vector_size'] == 47
        assert len(metadata['native_categorical_attributes']) == 9
    else:
        assert metadata['tree_encoding'] == 'onehot' and metadata['feature_vector_size'] > 47

folds = audit.read_table(args.native_run / 'folds/seed-42/fraction-10000')
validation = {i: y for i, y, fold in zip(folds['id'], folds['label'], folds['cv_fold']) if fold == 0}
old_folds = audit.read_table(old_tune / 'folds/seed-42/fraction-10000')
assert dict(zip(folds['id'], folds['cv_fold'])) == dict(zip(old_folds['id'], old_folds['cv_fold']))
result = load(control / 'metrics.json')
assert result['sampling_mode'] == 'none' and result['input_training_pool_rows'] == len(train)
assert result['input_training_rows'] == result['sampled_training_rows'] == len(train) - len(validation)
sampled = audit.labels(control / 'sampled_training_ids.parquet')
assert sampled.keys() == train.keys() - validation.keys()
assert all(train[i] == y for i, y in sampled.items())
predictions = audit.read_table(control / 'predictions.parquet')
assert dict(zip(predictions['id'], predictions['label'])) == validation
assert len(predictions['id']) == len(validation)
confusion = dict(tp=0, fp=0, tn=0, fn=0)
for y, p, score in zip(predictions['label'], predictions['prediction'], predictions['customer_probability']):
    assert p == float(score > result['threshold'])
    confusion[('t' if y == p else 'f') + ('p' if p else 'n')] += 1
assert all(result[k] == v for k, v in confusion.items())
f05 = 1.25 * confusion['tp'] / (1.25 * confusion['tp'] + confusion['fp'] + .25 * confusion['fn'])
assert abs(f05 - result['customer_f0_5']) < 1e-12
reference = load(package / 'validation/B0_fixed_test_metrics.json')
for field in ['numeric_features', 'categorical_features', 'duration_transform', 'threshold', 'class_weighting', 'versions', 'classifier_parameters']:
    assert result[field] == reference[field], 'XGBoost control changed an additional protocol setting: ' + field
report['full_xgboost_control'] = {'all_fitting_rows_retained': True, 'same_validation_ids_as_spark': True,
                                 'no_test_ids_in_fitting_or_validation': sampled.keys().isdisjoint(test) and validation.keys().isdisjoint(test),
                                 'confusion_and_f05_verified': True, 'original_baseline_settings_preserved': True,
                                 'customer_f0_5': f05, 'scope': 'training_validation_only'}
report['native_feature_metadata_verified'] = True
report['generated_feature_metadata_verified_all_three_models'] = True
report['test_performance_not_used_for_this_development_search'] = True
(package / 'validation/full_data_revision_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
files = [(args.native_run / 'results.json', 'native_tree_validation_results.json'),
         (args.native_run / 'manifest.json', 'native_tree_validation_manifest.json'),
         (args.native_run / 'selected_settings.json', 'P1_native_selected_settings.json'),
         (args.native_run / 'P1/seed-42/fraction-10000/cross_validation.json', 'P1_native_cross_validation.json'),
         (args.native_run / 'P1/seed-42/fraction-10000/feature_metadata.json', 'P1_native_feature_metadata.json'),
         (technical / 'results.json', 'native_interactions_technical_results.json'),
         (technical / 'manifest.json', 'native_interactions_technical_manifest.json'),
         (control / 'metrics.json', 'XGBoost_full_validation_control.json')]
for source, name in files:
    (package / 'validation' / name).write_text(json.dumps(load(source), indent=2, allow_nan=False) + '\n')
print(json.dumps({'native_validation_f0_5': native[0]['customer_f0_5'], 'old_onehot_validation_f0_5': load(old_tune / 'results.json')[0]['customer_f0_5'],
                  'full_xgboost_validation_f0_5': f05, 'checks_passed': True}, indent=2))
