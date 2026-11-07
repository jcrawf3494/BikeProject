import argparse
import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile

work = Path(__file__).resolve().parent
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

spec = importlib.util.spec_from_file_location('audit', work / 'audit_improved_runs.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
package = work.parent
parser = argparse.ArgumentParser()
parser.add_argument('--trip-archive', type=Path, required=True)
args = parser.parse_args()
with ZipFile(args.trip_archive) as archive:
    names = [name for name in archive.namelist() if name.rsplit('/', 1)[-1] == 'trip.csv']
    assert len(names) == 1
    with archive.open(names[0]) as source:
        raw_trips = pd.read_csv(source, usecols=['id', 'start_date', 'duration', 'start_station_id', 'end_station_id', 'subscription_type']).set_index('id')
raw_trips['local_start'] = pd.to_datetime(raw_trips.start_date, format='%m/%d/%Y %H:%M')
assert len(raw_trips) == 669959 and raw_trips.index.is_unique
weather = pd.read_csv(package / 'data/raw/weather.csv')
weather.columns = weather.columns.str.lower()
weather['trip_date'] = pd.to_datetime(weather.date, format='%m/%d/%Y').dt.normalize()
weather['start_city'] = weather.zip_code.map({94107: 'San Francisco', 95113: 'San Jose', 94041: 'Mountain View',
                                           94301: 'Palo Alto', 94063: 'Redwood City'})
assert weather.start_city.notna().all() and not weather.duplicated(['start_city', 'trip_date']).any()
weather_columns = [name for name in weather if name not in ['date', 'zip_code', 'events', 'trip_date', 'start_city']]
trace = weather.precipitation_inches.astype('string').str.upper().str.strip().eq('T').fillna(False)
weather['precipitation_trace'] = trace.astype(float)
weather.loc[trace, 'precipitation_inches'] = '0'
for name in weather_columns:
    weather[name] = pd.to_numeric(weather[name], errors='raise')
weather['weather_events'] = weather.events.fillna('none').str.strip().str.lower().replace('', 'none').str.replace(r'\s*-\s*', '-', regex=True)
for token in ['fog', 'rain', 'thunder']:
    weather['weather_' + token] = weather.weather_events.str.contains(token).astype(float)
numeric_weather = weather_columns + ['precipitation_trace', 'weather_fog', 'weather_rain', 'weather_thunder']
weather = weather[['start_city', 'trip_date', 'weather_events'] + numeric_weather]
stations = pd.read_csv(package / 'data/raw/station.csv').set_index('id')
calendar = USFederalHolidayCalendar().holidays('2013-01-01', '2016-01-01')
reports = []
seen_ids = set()
for split in ['train', 'test']:
    enriched = pd.DataFrame(audit.read_table(package / f'data/enriched/{split}.parquet'))
    original = audit.labels(package / f'data/{split}.parquet')
    assert dict(zip(enriched.id, enriched.label)) == original
    assert enriched.id.is_unique and set(enriched.id).isdisjoint(seen_ids)
    seen_ids.update(enriched.id)
    assert (enriched.label == (enriched.subscription_type == 'Customer').astype(float)).all()
    original_trips = raw_trips.loc[enriched.id].reset_index()
    assert (enriched.duration_sec.to_numpy() == original_trips.duration.to_numpy()).all()
    for column in ['start_station_id', 'end_station_id', 'subscription_type']:
        assert (enriched[column].to_numpy() == original_trips[column].to_numpy()).all()
    timestamp = original_trips.local_start
    stored_timestamp_offset = ((pd.to_datetime(enriched.start_ts) - timestamp).dt.total_seconds() / 3600).value_counts().sort_index()
    dates = timestamp.dt.normalize()
    assert (pd.to_datetime(enriched.trip_date).dt.normalize() == dates).all()
    assert (enriched.start_hour.astype(int) == timestamp.dt.hour).all()
    assert (enriched.month.astype(int) == timestamp.dt.month).all()
    weekday_map = pd.DataFrame({'actual_weekday_monday_zero': timestamp.dt.dayofweek,
                                'saved_day_of_week': enriched.day_of_week}).drop_duplicates()
    assert len(weekday_map) == 7 and weekday_map.saved_day_of_week.nunique() == 7
    assert (enriched.is_weekend.astype(bool) == (timestamp.dt.dayofweek >= 5)).all()
    assert (enriched.is_round_trip.astype(bool) == (enriched.start_station_id == enriched.end_station_id)).all()
    assert (enriched.is_business_day.astype(bool) == ((enriched.is_weekend == 0) & (enriched.is_holiday == 0))).all()
    assert (enriched.is_inter_cluster.astype(bool) == (enriched.start_cluster != enriched.end_cluster)).all()
    for prefix in ['start', 'end']:
        station_ids = enriched[prefix + '_station_id']
        assert station_ids.isin(stations.index).all()
    assert (enriched.start_city == enriched.start_station_id.map(stations.city)).all()
    assert np.isfinite(enriched.duration_sec).all() and (enriched.duration_sec >= 0).all()
    enriched['trip_date'] = pd.to_datetime(enriched.trip_date).dt.normalize()
    joined = enriched.merge(weather, on=['start_city', 'trip_date'], suffixes=('', '__raw'), validate='many_to_one', indicator=True)
    assert joined._merge.eq('both').all()
    for column in numeric_weather:
        assert np.isclose(joined[column], joined[column + '__raw'], equal_nan=True).all(), column
    assert (joined.weather_events == joined.weather_events__raw).all()
    expected_holiday = dates.isin(calendar)
    holiday_diff = pd.DataFrame({'date': dates, 'saved_holiday': enriched.is_holiday.astype(bool), 'observed_federal_holiday': expected_holiday})
    holiday_diff = holiday_diff[holiday_diff.saved_holiday != holiday_diff.observed_federal_holiday].drop_duplicates()
    reports.append({'split': split, 'rows': len(enriched), 'customer_rows': int(enriched.label.sum()),
                    'ids_labels_and_split_preserved': True, 'date_hour_month_consistent': True,
                    'trip_features_match_original_csv_by_id': True,
                    'stored_timestamp_minus_original_local_clock_hours': {str(k): int(v) for k, v in stored_timestamp_offset.items()},
                    'weekday_encoding': weekday_map.sort_values('actual_weekday_monday_zero').to_dict('records'),
                    'weekend_round_trip_business_day_and_inter_cluster_flags_consistent': True,
                    'all_weather_values_match_independent_city_date_join': True,
                    'observed_holiday_definition_differences': [dict(date=str(row.date.date()), saved_holiday=bool(row.saved_holiday),
                                                                 observed_federal_holiday=bool(row.observed_federal_holiday))
                                                               for row in holiday_diff.itertuples()],
                    'duration_seconds_quantiles': {str(q): float(enriched.duration_sec.quantile(q)) for q in [0, .25, .5, .75, .99, 1]},
                    'start_clusters': int(enriched.start_cluster.nunique()), 'end_clusters': int(enriched.end_cluster.nunique())})
    del enriched, joined
report = {'audit_scope': 'Independent feature semantics and preservation checks; no model selection using test outcomes',
          'splits': reports, 'trace_weather_source_rows': int(trace.sum()),
          'note': 'Time features are verified against original local-clock CSV values. Stored timestamps have a legacy timezone encoding and are excluded from predictors. Holiday differences are reported for review; original feature values are preserved.'}
(package / 'validation/feature_semantics_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
