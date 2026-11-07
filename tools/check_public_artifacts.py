"""Check aggregate result integrity and public-tree privacy without private data."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
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
    return dict(public_text_files=len(files), run_folders=len(registry['runs']), model_results=len(rows),
                aggregate_metric_records_checked=metric_counter[0], findings=findings,
                scope='Public source/aggregate checks; private raw-row audit attestations are separate')


if __name__ == '__main__':
    report = inspect_tree()
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report['findings'] else 0)
