import importlib.util
import json
from pathlib import Path
import sys

work = Path(__file__).resolve().parent
import pandas as pd

spec = importlib.util.spec_from_file_location('audit', work / 'audit_improved_runs.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
package = work.parent
run = package / 'results/tune-2026-10-28T03-20-22-656784500Z-f65328f7'
features = pd.DataFrame(audit.read_table(package / 'data/enriched/train.parquet',
    ['id', 'duration_sec', 'start_hour', 'is_business_day', 'is_round_trip', 'start_cluster', 'end_cluster']))
predictions = pd.DataFrame(audit.read_table(run / 'P1/seed-42/fraction-10000/predictions.parquet'))
data = predictions.merge(features, on='id', validate='one_to_one')
assert len(data) == 178803 and data.label.sum() == 27497
data['duration_band'] = pd.cut(data.duration_sec / 60, [-1, 5, 10, 20, 30, 60, float('inf')],
                              labels=['0-5 minutes', '5-10 minutes', '10-20 minutes', '20-30 minutes', '30-60 minutes', '60+ minutes'])
data['hour_band'] = pd.cut(data.start_hour, [-1, 6, 9, 16, 19, 23],
                          labels=['00-06', '07-09', '10-16', '17-19', '20-23'])

def summarize(frame, columns):
    rows = []
    for keys, group in frame.groupby(columns, observed=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        customer = group.label == 1
        positive = group.prediction == 1
        tp = int((customer & positive).sum())
        fn = int((customer & ~positive).sum())
        fp = int((~customer & positive).sum())
        rows.append(dict(zip(columns, map(str, keys))) | {'rows': len(group), 'customers': int(customer.sum()),
                    'tp': tp, 'fn': fn, 'fp': fp, 'customer_recall': tp / (tp + fn) if tp + fn else None,
                    'customer_precision': tp / (tp + fp) if tp + fp else None})
    return sorted(rows, key=lambda r: r['fn'], reverse=True)

missed = data[(data.label == 1) & (data.prediction == 0)]
report = {'source_run': run.name, 'scope': 'Training-validation error exploration; no project test data used',
          'rows': len(data), 'missed_customers': len(missed),
          'missed_customers_with_duration_at_most_10_minutes': int((missed.duration_sec <= 600).sum()),
          'by_duration': summarize(data, ['duration_band']),
          'by_business_day_and_hour': summarize(data, ['is_business_day', 'hour_band']),
          'by_round_trip': summarize(data, ['is_round_trip']),
          'by_start_cluster': summarize(data, ['start_cluster'])}
(package / 'validation/P1_validation_error_profile.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: report[k] for k in ['missed_customers', 'missed_customers_with_duration_at_most_10_minutes', 'by_duration']}, indent=2))
