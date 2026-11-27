"""Publish audited aggregate records without row-level or machine-path artifacts."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
MODELS = ['P1', 'P6', 'P7', 'P8', 'P9', 'P10']
PRIVATE_FIELDS = {'predictions', 'calibration_predictions', 'fitting_ids', 'attempt_directory'}
NAMES = dict(P1='Spark GBT', P6='EBM', P7='CatBoost', P8='LightGBM',
             P9='XGBoost', P10='P1/EBM/CatBoost stack')
FAMILIES = dict(P1='spark_gradient_boosted_trees', P6='explainable_boosting_machine',
                P7='catboost_gradient_boosted_trees', P8='lightgbm_gradient_boosted_trees',
                P9='xgboost_gradient_boosted_trees', P10='cross_fitted_logistic_stack')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def aggregate(value):
    if isinstance(value, dict):
        return {key: aggregate(child) for key, child in value.items() if key not in PRIVATE_FIELDS}
    if isinstance(value, list):
        return [aggregate(child) for child in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    source = args.run.resolve()
    protocol = load(source/'protocol.json')
    audit = load(source/'independent_audit.json')
    complete = load(source/'COMPLETED.json')
    frozen = load(source/'FROZEN_SETTINGS.json')
    technical = protocol['technical_verification']
    assert audit['passed'] is True and complete['independent_audit_pending'] is False
    assert audit['technical_verification'] == technical
    assert frozen['protocol_sha256'] == hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest()
    rows = aggregate(load(source/'results.json'))
    expected = {(model, seed) for model in MODELS for seed in protocol['seeds']}
    assert len(rows) == len(expected) == audit['model_records'] == complete['model_outcomes']
    assert {(row['model'], row['seed']) for row in rows} == expected
    assert all(row['technical_verification'] == technical for row in rows)
    if not technical:
        assert len(rows) == 18 and all(row['training_rows'] == 536406 and row['evaluation_rows'] == 133553 for row in rows)
    run_id = source.name
    scope = 'technical/matched-tuning' if technical else 'project/matched-tuning'
    destination = REPO/'experiments/records'/scope/run_id
    destination.mkdir(parents=True, exist_ok=True)
    # An explicit file allowlist excludes configuration paths, data, logs and models.
    names = ['protocol.json', 'FROZEN_SETTINGS.json', 'COMPLETED.json', 'independent_audit.json', 'summary.json']
    if (source/'COMPILED_ARTIFACT.json').exists():
        names.append('COMPILED_ARTIFACT.json')
    if (source/'EXECUTION_NOTES.json').exists():
        names.append('EXECUTION_NOTES.json')
    for model in MODELS:
        if model == 'P10':
            names.append('P10/selection.json')
        else:
            names.append(model+'/cross_validation.json')
            for candidate in range(3):
                for fold in range(3):
                    names.append(f'{model}/cv/candidate-{candidate}/fold-{fold}/DONE.json')
        for seed in protocol['seeds']:
            names.extend([f'{model}/seed-{seed}/metrics.json', f'{model}/seed-{seed}/threshold_selection.json'])
    for name in names:
        assert (source/name).is_file(), name
        write(destination/name, aggregate(load(source/name)))
    normalized = [dict(row, run_id=run_id, input_revision='synthetic_matched_tuning' if technical else 'full_data_matched_tuning_39',
                       parameter_selection='training_only_three_configuration_three_fold_search', model_family=FAMILIES[row['model']]) for row in rows]
    write(destination/'results.json', normalized)
    summary = load(source/'summary.json')
    for cell in summary:
        actual = [row['customer_f0_5'] for row in rows if row['model'] == cell['model']]
        assert abs(statistics.mean(actual)-cell['mean_f0_5']) < 1e-12
    title = 'Synthetic matched-study verification' if technical else 'Matched full-data tuning results'
    lines = ['# '+title, '']
    if technical:
        lines += ['This is a synthetic implementation check. Its scores do not measure bike-share performance.', '']
    else:
        lines += ['All 18 final outcomes completed and passed independent saved-prediction checks. Every base classifier and final stack combiner used all 536,406 original training rows; evaluation used the unchanged 133,553-row test.', '',
                  '| Model | Mean Customer F0.5 | Sample SD | Precision | Recall |', '|---|---:|---:|---:|---:|']
        for cell in summary:
            lines.append(f"| {cell['model']} {NAMES[cell['model']]} | {cell['mean_f0_5']:.6f} | {cell['sample_sd']:.6f} | {cell['mean_precision']:.2%} | {cell['mean_recall']:.2%} |")
        best = max(summary, key=lambda cell: cell['mean_f0_5'])
        lines += ['', f"The highest observed mean in this study is {best['model']} ({NAMES[best['model']]}) at {best['mean_f0_5']:.6f}. All settings and stack membership were frozen using training-only selection before these test predictions.", '',
                  'The prior P1 reference mean is 0.891818, EBM preset mean is 0.886242, and CatBoost preset mean is 0.888252. The new study uses a fixed seed-42 calibration split for every initialization, whereas historical repeats varied calibration memberships with the seed. The old and new means are benchmark comparisons, not paired split-controlled improvements.', '',
                  'The paper plots F0.5 near 0.81 without an exact tabulated value or complete reproduction code. These full-data results use a different evaluation population and explicit feature assumptions; numerical differences are not an exact replication claim.', '']
    lines += ['The search uses 45 base CV fits and three stack regularization candidates. B/H/C training roles separate base selection, stack selection and cutoff calibration. Seeds 42/43/44 vary estimator initialization only; the stack shares its seed-42 OOF construction. Different OOF base fitting sizes are disclosed. The test was already examined in prior work, so this is an extension of an existing benchmark. Fit timings exclude preprocessing and IO and do not establish a controlled speedup.', '',
              'See supplemental/matched/README.md for the complete selection and reproduction method. Raw data, row IDs, predictions, models and machine logs are private.']
    (destination/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    record = dict(run_id=run_id, source_scope=scope, status='completed',
                  kind='technical_verification' if technical else 'experiment', models=MODELS,
                  training_pool_rows=protocol['training_rows'], feature_scope='39-column synthetic fixture' if technical else '39 paper-input features; historical V1',
                  feature_variant='standard', search_budget='three_configurations_three_folds_per_base_and_three_stack_C_values',
                  aggregate_record_directory=destination.relative_to(REPO).as_posix(),
                  aggregate_record_count=len(names)+3, measured_result_count=len(rows), measured_results=normalized,
                  notes='Audited matched-model tuning and cross-fitted stacking. Fixed calibration membership across initialization seeds. Synthetic scores are technical checks only.' if technical else
                        'Audited full-data matched-model tuning and cross-fitted stacking. All final fits retain the full original training cohort. Previously examined fixed test; seed-42 calibration membership and shared seed-42 OOF construction.')
    write(destination/'RUN_STATUS.json', {key: value for key, value in record.items() if key != 'measured_results'})
    registry = load(REPO/'experiments/registry.json')
    existing = [run for run in registry['runs'] if run['run_id'] == run_id]
    assert not existing or existing == [record], 'A previous published run differs; preserve it.'
    if not existing:
        registry['runs'].append(record)
    registry.update(run_count=len(registry['runs']), as_of=datetime.now(ZoneInfo('America/New_York')).date().isoformat())
    flat = [row for run in registry['runs'] for row in run['measured_results']]
    write(REPO/'experiments/registry.json', registry)
    write(REPO/'experiments/all_model_results.json', flat)
    csv_path = REPO/'experiments/all_model_results.csv'
    with csv_path.open(encoding='utf-8', newline='') as handle:
        fields = csv.DictReader(handle).fieldnames
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(flat)
    ledger = REPO/'experiments/LEDGER.md'
    old = ledger.read_text(encoding='utf-8'); tail = old[old.index('## Reading the results'):]
    lines = ['# Experiment ledger', '', 'Snapshot: '+registry['as_of']+'.', '',
             f"Archive: {len(registry['runs'])} run folders and {len(flat)} saved outcomes, including explicitly labeled synthetic verification. Matched tuning is a separate study from the incomplete older P1-P3 completion batch.", '',
             '| Run | Type | Status | Models | Saved results |', '|---|---|---|---|---:|']
    for run in registry['runs']:
        link = run['aggregate_record_directory'].removeprefix('experiments/')+'/RUN_STATUS.json'
        lines.append(f"| [{run['run_id']}]({link}) | {run['kind']} | {run['status']} | {', '.join(run['models'])} | {run['measured_result_count']} |")
    ledger.write_text('\n'.join(lines)+'\n\n'+tail, encoding='utf-8')
    print(json.dumps(dict(exported_run=run_id, technical_verification=technical, model_outcomes=len(rows),
                          archive_runs=len(registry['runs']), archive_outcomes=len(flat))))


if __name__ == '__main__':
    main()
