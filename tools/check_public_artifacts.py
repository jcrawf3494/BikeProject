"""Check aggregate result integrity and public-tree privacy without private data."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {'.parquet', '.joblib', '.pickle', '.pkl', '.jar', '.class',
                      '.zip', '.pdf', '.docx', '.log'}
EMAIL = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)
PROFILE = re.compile(r'(?i)(?:[a-z]:[\\/]Users[\\/]|/home/|/Users/)[A-Za-z0-9_.-]+')
SECRET = re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b')


def load(path):
    def reject(value):
        raise ValueError('Non-standard JSON numeric value: ' + value)
    return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=reject)


def check_metrics(value, context, findings, counter):
    if isinstance(value, dict):
        if 'id' in value and ('label' in value or 'customer_probability' in value):
            findings.append(context + ': row-shaped trip/prediction record')
        keys = {'tp', 'fp', 'fn', 'tn', 'customer_f0_5'}
        if keys <= value.keys() and value['customer_f0_5'] is not None:
            tp, fp, fn, tn = (value[k] for k in ['tp', 'fp', 'fn', 'tn'])
            if not all(isinstance(n, (int, float)) and math.isfinite(n) and n >= 0 for n in [tp, fp, fn, tn]):
                findings.append(context + ': invalid confusion counts')
            else:
                den = 1.25 * tp + fp + 0.25 * fn
                expected = 1.25 * tp / den if den else 0.0
                if not math.isclose(expected, value['customer_f0_5'], abs_tol=1e-10):
                    findings.append(context + ': F0.5 arithmetic mismatch')
                if isinstance(value.get('evaluation_rows'), (int, float)) and tp + fp + fn + tn != value['evaluation_rows']:
                    findings.append(context + ': confusion counts disagree with evaluation rows')
                counter[0] += 1
        for key, child in value.items():
            check_metrics(child, context + '/' + key, findings, counter)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_metrics(child, context + '/' + str(index), findings, counter)


def inspect_tree(root=ROOT):
    findings, metric_counter = [], [0]
    files = []
    try:
        tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0'))
    except (OSError, subprocess.CalledProcessError):
        tracked = set()
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if any(p in {'.git', '__pycache__', '.venv', 'target'} for p in relative.parts):
            if relative.as_posix() in tracked:
                findings.append(relative.as_posix() + ': generated/private artifact is tracked')
            continue
        if not path.is_file():
            continue
        if relative.parts[0] in {'results', 'logs', 'vendor'}:
            if relative.as_posix() in tracked:
                findings.append(relative.as_posix() + ': private execution artifact is tracked')
            continue  # Private locally restored execution artifacts are not public exports.
        if relative.parts[0] == 'data' and relative.as_posix() not in {'data/README.md', 'data/SCHEMA.md'}:
            if relative.as_posix() in tracked:
                findings.append(relative.as_posix() + ': row-level data path is tracked')
            continue
        files.append(path)
        context = relative.as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(context + ': prohibited data/binary/log type')
        if path.name == '.env' or path.name.startswith('.env.'):
            findings.append(context + ': private environment configuration')
        try:
            text = path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            findings.append(context + ': opaque non-text artifact')
            continue
        if EMAIL.search(text):
            findings.append(context + ': contact-address pattern')
        if PROFILE.search(text):
            findings.append(context + ': machine profile path')
        if SECRET.search(text):
            findings.append(context + ': credential pattern')
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=context)
            except SyntaxError as exc:
                findings.append(context + ': Python syntax error: ' + str(exc))
        if path.suffix == '.json':
            try:
                check_metrics(load(path), context, findings, metric_counter)
            except (ValueError, TypeError) as exc:
                findings.append(context + ': invalid JSON: ' + str(exc))

    registry = load(root / 'experiments/registry.json')
    rows = load(root / 'experiments/all_model_results.json')
    if registry['run_count'] != len(registry['runs']):
        findings.append('Registry run count mismatch')
    if len({r['run_id'] for r in registry['runs']}) != len(registry['runs']):
        findings.append('Duplicate registry run IDs')
    for run in registry['runs']:
        directory = root / run['aggregate_record_directory']
        if not (directory / 'RUN_STATUS.json').exists():
            findings.append('Missing status record for ' + run['run_id'])
        if run['measured_result_count'] != len(run['measured_results']):
            findings.append('Measured result count mismatch for ' + run['run_id'])
        for result in run['measured_results']:
            if run['kind'] == 'technical_verification' and not result['technical_verification']:
                findings.append('Fixture result lacks technical label: ' + run['run_id'])
    if sum(r['measured_result_count'] for r in registry['runs']) != len(rows):
        findings.append('Flattened model results do not cover the registry')
    with (root / 'experiments/all_model_results.csv').open(encoding='utf-8', newline='') as stream:
        if len(list(csv.DictReader(stream))) != len(rows):
            findings.append('CSV result count mismatch')
    versions = load(root / 'validation/reconstruction_10354_code_versions.json')
    actual = hashlib.sha256((root / 'baseline/reconstruct_10354.py').read_bytes()).hexdigest()
    if actual != versions['final_source_sha256']:
        findings.append('Current reconstruction source differs from the audited final version')
    development_source = root / 'experiments/source_snapshots/reconstruction_development.py'
    if hashlib.sha256(development_source.read_bytes()).hexdigest() != versions['development_source_sha256']:
        findings.append('Historical reconstruction source differs from the audited development version')
    frozen = root / 'validation/reconstruction_10354_frozen_settings.json'
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != versions['frozen_settings_sha256']:
        findings.append('Frozen reconstruction settings differ from the audited version')
    scala_signature = '\n'.join(p.name + ':' + hashlib.sha256(p.read_bytes()).hexdigest().upper()
                                for p in sorted((root / 'src').glob('*.scala'), key=lambda p:p.name.lower()))
    expected_jar = 'bikeshare-models-' + hashlib.sha256(scala_signature.encode()).hexdigest()[:12] + '.jar'
    if expected_jar != load(root / 'validation/package_integrity.json')['current_jar']:
        findings.append('Current Scala sources differ from the recorded compiled artifact version')
    completion = root / 'validation/completion_20261108'
    if (completion / 'protocol.json').exists():
        protocol = load(completion / 'protocol.json')
        for frozen in [completion / 'frozen_selected_settings.json',
                       root / 'validation/P1_P2_P3_frozen_settings.json']:
            if hashlib.sha256(frozen.read_bytes()).hexdigest() != protocol['settings_sha256']:
                findings.append('Frozen Spark completion settings checksum mismatch: ' + frozen.name)
        for filename, fingerprint in protocol['source_sha256'].items():
            if hashlib.sha256((root / 'src' / filename).read_bytes()).hexdigest() != fingerprint:
                findings.append('Spark completion source checksum mismatch: ' + filename)
        if protocol['compiled_jar'] != expected_jar:
            findings.append('Spark completion compiled artifact differs from its source signature')
        completion_rows = load(completion / 'all_results.json')
        if len(completion_rows) != protocol['expected_model_records']:
            findings.append('Spark completion record count differs from its frozen protocol')
        if load(completion / 'independent_audit.json').get('passed') is not True:
            findings.append('Spark completion lacks a passing private-data audit attestation')
    paper_inputs = root / 'validation/paper_flags_full_20261108'
    if (paper_inputs / 'protocol.json').exists():
        protocol = load(paper_inputs / 'protocol.json')
        for frozen in [paper_inputs / 'frozen_selected_settings.json',
                       paper_inputs / 'frozen_paper_input_settings.json',
                       root / 'validation/P1_P2_P3_with_paper_inputs.json']:
            if hashlib.sha256(frozen.read_bytes()).hexdigest() != protocol['settings_sha256']:
                findings.append('Paper-input frozen settings checksum mismatch: ' + frozen.name)
        settings = load(paper_inputs / 'frozen_selected_settings.json')
        base = root / 'validation/P1_P2_P3_frozen_settings.json'
        if hashlib.sha256(base.read_bytes()).hexdigest() != protocol['base_selected_settings_sha256']:
            findings.append('Paper-input base parameter settings checksum mismatch')
        if settings['models'] != load(base)['models'] or settings.get('retuned_with_added_features') is not False:
            findings.append('Paper-input fixed-parameter ablation provenance mismatch')
        for filename, fingerprint in protocol['source_sha256'].items():
            if hashlib.sha256((root / 'src' / filename).read_bytes()).hexdigest() != fingerprint:
                findings.append('Paper-input Scala source checksum mismatch: ' + filename)
        if protocol['compiled_jar'] != expected_jar:
            findings.append('Paper-input compiled artifact differs from its Scala source signature')
        if hashlib.sha256((paper_inputs / 'feature_preparation_audit.json').read_bytes()).hexdigest() != protocol['feature_preparation_audit_sha256']:
            findings.append('Paper-input preparation audit checksum mismatch')
        if hashlib.sha256((root / 'tools/prepare_full_data_paper_flags.py').read_bytes()).hexdigest() != protocol['feature_preparation_source_sha256']:
            findings.append('Paper-input preparation source checksum mismatch')
        preparation = load(paper_inputs / 'feature_preparation_audit.json')
        if preparation['rows_filtered'] != 0 or preparation['split_recreated'] or not preparation['all_original_rows_and_columns_preserved']:
            findings.append('Paper-input preparation does not attest full row/feature preservation')
        paper_rows = load(paper_inputs / 'all_results.json')
        expected = {(model, seed) for model in ['P1','P2','P3'] for seed in [42,43,44]}
        if len(paper_rows) != 9 or {(r['model'],r['seed']) for r in paper_rows} != expected:
            findings.append('Paper-input results do not cover all nine model/seed combinations')
        if any(r['training_rows'] != 536406 or r['evaluation_rows'] != 133553 or r['requested_fraction'] != 1.0 for r in paper_rows):
            findings.append('Paper-input results did not retain the full training/test cohorts')
        if load(paper_inputs / 'independent_audit.json').get('passed') is not True:
            findings.append('Paper-input results lack passing private-data verification')
    sample_inputs = root / 'validation/paper_input_sample_controls_20261108'
    comparison = root / 'validation/matched_input_comparison_20261108'
    matched_xgb = root / 'validation/matched_feature_xgboost_20261108'
    if (sample_inputs / 'protocol.json').exists():
        protocol = load(sample_inputs / 'protocol.json')
        frozen = sample_inputs / 'frozen_selected_settings.json'
        if hashlib.sha256(frozen.read_bytes()).hexdigest() != protocol['settings_sha256']:
            findings.append('Sample-control frozen settings checksum mismatch')
        if frozen.read_bytes() != (paper_inputs / 'frozen_selected_settings.json').read_bytes():
            findings.append('Sample controls did not preserve the primary feature/parameter settings')
        if hashlib.sha256((sample_inputs / 'cohort_preparation_audit.json').read_bytes()).hexdigest() != protocol['cohort_preparation_audit_sha256']:
            findings.append('Sample-control preparation attestation checksum mismatch')
        if hashlib.sha256((root / 'tools/prepare_sample_comparison_cohorts.py').read_bytes()).hexdigest() != protocol['cohort_preparation_source_sha256']:
            findings.append('Sample-control preparation source checksum mismatch')
        for name, fingerprint in protocol['scala_source_sha256'].items():
            if hashlib.sha256((root / 'src' / name).read_bytes()).hexdigest() != fingerprint:
                findings.append('Sample-control Scala source checksum mismatch: ' + name)
        if protocol['compiled_jar'] != expected_jar or protocol['expected_model_records'] != 18:
            findings.append('Sample-control source version or expected count differs')
        prepared = load(sample_inputs / 'cohort_preparation_audit.json')
        if len(prepared['cases']) != 6 or any(c['training_rows'] != 10354 or not c['same_original_rows_labels_and_39_inputs'] for c in prepared['cases']):
            findings.append('Sample-control prepared cohorts do not match their declared scope')
        if load(comparison / 'independent_audit.json').get('passed') is not True:
            findings.append('Matched comparison lacks a passing private-data audit attestation')
        matrix = load(comparison / 'all_results.json')
        expected = {(m,c,s) for m in ['P1','P2','P3','XGBoost_matched_inputs_weighted']
                    for c in ['full','similarity','stratified'] for s in [42,43,44]}
        if len(matrix) != 36 or {(r['model'],r['cohort'],r['seed']) for r in matrix} != expected:
            findings.append('Matched comparison does not cover all 36 declared cases')
        for row in matrix:
            expected_rows = 536406 if row['cohort'] == 'full' else 10354
            if row['training_rows'] != expected_rows or row['evaluation_rows'] != 133553:
                findings.append('Matched comparison changed the declared training/test cohort size')
            if row.get('model_reload_verified') is not True:
                findings.append('Matched comparison lacks a successful persistence probe')
        summaries = load(comparison / 'summary.json')
        if len(summaries) != 12:
            findings.append('Matched comparison summary has an unexpected number of cells')
        for summary in summaries:
            values = [r['customer_f0_5'] for r in matrix if (r['model'],r['cohort']) == (summary['model'],summary['cohort'])]
            if len(values) != 3:
                findings.append('Matched comparison cell lacks three seeds')
                continue
            metrics = summary['metrics']['customer_f0_5']
            if not math.isclose(statistics.mean(values), metrics['mean'], abs_tol=1e-12) or not math.isclose(statistics.stdev(values), metrics['sample_sd'], abs_tol=1e-12):
                findings.append('Matched comparison summary arithmetic differs')
        xp = load(matched_xgb / 'protocol.json')
        if hashlib.sha256((root / 'tools/run_matched_feature_xgboost.py').read_bytes()).hexdigest() != xp['source_sha256']:
            findings.append('Matched XGBoost source checksum mismatch')
        if hashlib.sha256((root / 'validation/reconstruction_10354_frozen_settings.json').read_bytes()).hexdigest() != xp['source_baseline_settings_sha256']:
            findings.append('Matched XGBoost parameter-origin checksum mismatch')
        if xp['settings_sha256'] != protocol['settings_sha256'] or xp['parameters'] != load(root / 'validation/reconstruction_10354_frozen_settings.json')['parameters']:
            findings.append('Matched XGBoost inputs or transferred parameters differ')
        if len([r for r in matrix if r['model'] == 'XGBoost_matched_inputs_weighted']) != 9:
            findings.append('Matched XGBoost expected nine outcomes')
    additional = root / 'validation/additional_models_20261126/implementation_verification.json'
    if additional.exists():
        verified = load(additional)
        for path, field in [(root / 'supplemental/scala/TrainSupplementalSparkModels.scala','current_scala_source_sha256'),
                            (root / 'supplemental/run_python_models.py','python_source_sha256'),
                            (root / 'experiments/source_snapshots/supplemental_smoke.scala','smoke_scala_source_sha256')]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != verified[field]:
                findings.append('Supplemental implementation checksum differs: ' + path.name)
        if set(verified['models']) != {'P4','P5','P6','P7'} or verified['scientific_full_data_outcomes'] != 0:
            findings.append('Supplemental implementation verification misstates its tested scope')
        for run_id in verified['verification_runs']:
            matched = [run for run in registry['runs'] if run['run_id'] == run_id]
            if len(matched) != 1 or matched[0]['kind'] != 'technical_verification':
                findings.append('Supplemental synthetic check is missing or lacks its technical label')
    supplemental_results = root / 'validation/additional_models_20261126/full_data_results.json'
    if supplemental_results.exists():
        full_rows = load(supplemental_results)
        full_summary = load(supplemental_results.parent / 'full_data_summary.json')
        expected = {(m,s) for m in ['P4','P5','P6','P7'] for s in [42,43,44]}
        if len(full_rows) != 12 or {(r['model'],r['seed']) for r in full_rows} != expected:
            findings.append('Full-data supplemental comparison lacks its twelve declared cases')
        if any(r['training_rows'] != 536406 or r['evaluation_rows'] != 133553 for r in full_rows):
            findings.append('Full-data supplemental comparison changed the declared split sizes')
        if load(supplemental_results.parent / 'full_data_independent_audit.json').get('passed') is not True:
            findings.append('Full-data supplemental comparison lacks passing saved-prediction verification')
        for summary in full_summary['models']:
            values = [r['customer_f0_5'] for r in full_rows if r['model'] == summary['model']]
            if len(values) != 3 or not math.isclose(statistics.mean(values),summary['metrics']['customer_f0_5']['mean'],abs_tol=1e-12):
                findings.append('Full-data supplemental mean arithmetic differs')
    for run in registry['runs']:
        if run['source_scope'] not in {'technical/matched-tuning', 'project/matched-tuning'}:
            continue
        directory = root / run['aggregate_record_directory']
        protocol = load(directory / 'protocol.json')
        frozen = load(directory / 'FROZEN_SETTINGS.json')
        audit = load(directory / 'independent_audit.json')
        technical = run['kind'] == 'technical_verification'
        if audit.get('passed') is not True or audit['technical_verification'] != technical:
            findings.append('Matched tuning lacks a passing independent audit: ' + run['run_id'])
        if load(directory / 'COMPLETED.json')['independent_audit_pending'] is not False:
            findings.append('Matched tuning audit remains pending: ' + run['run_id'])
        if frozen['protocol_sha256'] != hashlib.sha256((directory / 'protocol.json').read_bytes()).hexdigest():
            findings.append('Matched tuning frozen protocol checksum differs: ' + run['run_id'])
        for name, fingerprint in protocol['source_sha256'].items():
            source = root / 'supplemental/matched' / name
            if not source.exists():
                source = root / 'supplemental' / name
            if hashlib.sha256(source.read_bytes()).hexdigest() != fingerprint:
                findings.append('Matched tuning scientific source checksum differs: ' + name)
        expected = {(model, seed) for model in ['P1','P6','P7','P8','P9','P10'] for seed in protocol['seeds']}
        outcomes = load(directory / 'results.json')
        if len(outcomes) != len(expected) or {(row['model'], row['seed']) for row in outcomes} != expected:
            findings.append('Matched tuning final outcomes differ from declared model/seed cases')
        if not technical and (len(outcomes) != 18 or any(row['training_rows'] != 536406 or row['evaluation_rows'] != 133553 for row in outcomes)):
            findings.append('Matched tuning changed the full-data final cohorts')
        if protocol['stack_bases'] != ['P1','P6','P7'] or frozen['stack_bases'] != protocol['stack_bases']:
            findings.append('Matched tuning changed the predeclared stack membership')
        for model in ['P1','P6','P7','P8','P9']:
            candidates = load(directory / model / 'cross_validation.json')
            selected = [candidate for candidate in candidates if candidate['selected']]
            if len(candidates) != 3 or len(selected) != 1 or selected[0]['parameters'] != frozen['models'][model]:
                findings.append('Matched tuning base selection differs: ' + model)
            for candidate in candidates:
                if len(candidate['fold_f0_5']) != 3 or not math.isclose(statistics.mean(candidate['fold_f0_5']),candidate['mean_f0_5'],abs_tol=1e-12):
                    findings.append('Matched tuning CV mean arithmetic differs: ' + model)
            if selected and selected[0]['mean_f0_5'] != max(candidate['mean_f0_5'] for candidate in candidates):
                findings.append('Matched tuning did not select the best declared CV candidate: ' + model)
        for summary in load(directory / 'summary.json'):
            values = [row['customer_f0_5'] for row in outcomes if row['model'] == summary['model']]
            if not math.isclose(statistics.mean(values),summary['mean_f0_5'],abs_tol=1e-12):
                findings.append('Matched tuning test mean arithmetic differs')
            if len(values) > 1 and not math.isclose(statistics.stdev(values),summary['sample_sd'],abs_tol=1e-12):
                findings.append('Matched tuning test SD arithmetic differs')
        for row in outcomes:
            threshold = load(directory / row['model'] / f"seed-{row['seed']}" / 'threshold_selection.json')
            if threshold['test_used'] or threshold['threshold'] != row['decision_threshold'] or row['calibration_split_seed'] != 42:
                findings.append('Matched tuning cutoff selection boundary differs')
            expected_features = 3 if row['model'] == 'P10' else 39
            if row['feature_count'] != expected_features or row['training_pool_downsampled'] or not row['model_reload_verified']:
                findings.append('Matched tuning feature/full-fit/persistence declaration differs')
    return dict(public_text_files=len(files), run_folders=len(registry['runs']), model_results=len(rows),
                aggregate_metric_records_checked=metric_counter[0], findings=findings,
                scope='Public source/aggregate checks; private raw-row audit attestations are separate')


if __name__ == '__main__':
    report = inspect_tree()
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report['findings'] else 0)
