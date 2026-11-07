import importlib.util
import json
from pathlib import Path
import statistics
import sys
import shutil

work = Path(__file__).resolve().parent
import numpy as np

spec = importlib.util.spec_from_file_location('run_audit', work / 'audit_improved_runs.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
package = work.parent
source_run = package / 'results/train-2026-10-28T04-02-57-528292100Z-8d2ae312'
validation_run = package / 'results/tune-2026-10-28T03-20-22-656784500Z-f65328f7'
recall_run = package / 'results/recall-2026-10-28T17-40-19-824301100Z-2f7b98df'
load = lambda path: json.loads(path.read_text())
policy = load(recall_run / 'selected_recall_policy.json')
results = load(recall_run / 'results.json')
previous = load(source_run / 'results.json')
assert policy['test_used_for_selection'] is False

validation = audit.read_table(validation_run / 'P1/seed-42/fraction-10000/predictions.parquet')
probability = np.asarray(validation['customer_probability'])
truth = np.asarray(validation['label'])
order = np.argsort(-probability, kind='stable')
ordered = probability[order]
ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(ordered) - 1]
tp = np.cumsum(truth[order])[ends]
fp = ends + 1 - tp
fn = truth.sum() - tp
recall = tp / truth.sum()
f05 = 1.25 * tp / (1.25 * tp + fp + 0.25 * fn)
eligible = (recall >= policy['minimum_validation_recall']) & (ordered[ends] > 0)
assert eligible.any()
assert abs(float(f05[eligible].max()) - policy['validation_selection_metrics']['customer_f0_5']) < 1e-12
assert policy['validation_selection_metrics']['customer_recall'] >= policy['minimum_validation_recall']

test = audit.labels(package / 'data/enriched/test.parquet')
assert set(validation['id']).isdisjoint(test)
checks = []
for result in results:
    seed = result['seed']
    relative = Path(f'P1/seed-{seed}/fraction-10000')
    original = audit.read_table(source_run / relative / 'predictions.parquet')
    updated = audit.read_table(recall_run / relative / 'predictions.parquet')
    old_by_id = {i: (y, p, s) for i, y, p, s in zip(original['id'], original['label'], original['prediction'], original['customer_probability'])}
    assert len(updated['id']) == len(set(updated['id'])) == len(test)
    assert set(updated['id']) == test.keys() == old_by_id.keys()
    confusion = dict(tp=0, fp=0, tn=0, fn=0)
    for i, y, p, score in zip(updated['id'], updated['label'], updated['prediction'], updated['customer_probability']):
        oy, op, os = old_by_id[i]
        assert y == oy == test[i] and score == os
        assert p == float(score > policy['selected_threshold'])
        assert p >= op, 'Lower cutoff unexpectedly removed a Customer prediction'
        confusion[('t' if y == p else 'f') + ('p' if p else 'n')] += 1
    assert all(result[key] == value for key, value in confusion.items())
    assert result['model_reload_verified'] is True
    score = 1.25 * confusion['tp'] / (1.25 * confusion['tp'] + confusion['fp'] + .25 * confusion['fn'])
    assert abs(score - result['customer_f0_5']) < 1e-12
    checks.append({'seed': seed, 'test_ids_and_labels_unchanged': True, 'probabilities_unchanged_exactly': True,
                   'customer_predictions_are_superset': True, 'cutoff_and_confusion_metrics_verified': True,
                   'model_reload_probe_passed': True})

fields = ['customer_precision', 'customer_recall', 'customer_f0_5', 'accuracy']
summarize = lambda rows: {field: {'mean': statistics.mean(r[field] for r in rows),
                                  'sample_sd': statistics.stdev(r[field] for r in rows)} for field in fields}
report = {'policy': policy, 'original_three_seed_summary': summarize(previous),
          'recall_three_seed_summary': summarize(results), 'seed_checks': checks,
          'validation_constrained_optimum_independently_verified': True,
          'threshold_selection_test_ids_disjoint': True,
          'interpretation': 'Alternative operating-point experiment on an already examined project test set; target recall applies to validation only.'}
validation_dir = package / 'validation'
(validation_dir / 'recall_operating_point_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
for source, destination in [(recall_run / 'selected_recall_policy.json', 'selected_recall_policy.json'),
                            (recall_run / 'results.json', 'recall_operating_point_results.json'),
                            (recall_run / 'validation_cutoff_comparison.json', 'recall_validation_cutoff_comparison.json'),
                            (source_run / 'results.json', 'P1_tuned_three_seed_test_results.json'),
                            (validation_run / 'results.json', 'P1_full_tuning_validation_results.json'),
                            (validation_run / 'selected_settings.json', 'P1_selected_settings.json'),
                            (validation_run / 'P1/seed-42/fraction-10000/cross_validation.json', 'P1_full_cross_validation.json')]:
    (validation_dir / destination).write_text(json.dumps(load(source), indent=2, allow_nan=False) + '\n')
print(json.dumps({key: report[key] for key in ['original_three_seed_summary', 'recall_three_seed_summary', 'seed_checks']}, indent=2))
