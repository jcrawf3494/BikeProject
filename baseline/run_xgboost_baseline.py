"""B0 comparison using the original library family and a documented sampling approximation.

The paper does not specify enough sampling/tuning details for an exact reproduction.
No attempt is made to force its 10,354-row count or alter the fixed project test set.
"""
import argparse
import json
import time
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, auc, confusion_matrix, fbeta_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier

CORE_NUMERIC = ['duration_sec', 'is_weekend', 'is_holiday', 'is_round_trip']
CORE_CATEGORICAL = ['start_hour', 'day_of_week', 'month']
WEATHER = ['max_temperature_f', 'mean_temperature_f', 'min_temperature_f',
           'max_dew_point_f', 'mean_dew_point_f', 'min_dew_point_f',
           'max_humidity', 'mean_humidity', 'min_humidity',
           'max_sea_level_pressure_inches', 'mean_sea_level_pressure_inches', 'min_sea_level_pressure_inches',
           'max_visibility_miles', 'mean_visibility_miles', 'min_visibility_miles',
           'max_wind_speed_mph', 'mean_wind_speed_mph', 'max_gust_speed_mph',
           'precipitation_inches', 'cloud_cover', 'wind_dir_degrees', 'precipitation_trace',
           'weather_fog', 'weather_rain', 'weather_thunder', 'is_business_day', 'is_inter_cluster']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    parser.add_argument('--evaluation', choices=['validation', 'test'], default='validation')
    parser.add_argument('--feature-profile', choices=['core', 'full'], default='full')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-fraction', type=float, default=1.0)
    parser.add_argument('--similarity-width', type=float, default=0.03)
    parser.add_argument('--sampling-mode', choices=['similarity', 'none'], default='similarity',
                        help='Keep similarity for B0; none is the supplemental full-training-data control.')
    parser.add_argument('--validation-folds', type=Path,
                        help='Saved Spark folds directory for a common full-pool internal validation split (fold 0 held out).')
    args = parser.parse_args()
    assert 0 < args.sample_fraction <= 1 and 0 < args.similarity_width <= 1
    if args.validation_folds and (args.evaluation != 'validation' or args.sample_fraction != 1.0):
        parser.error('--validation-folds requires --evaluation validation and --sample-fraction 1.0')
    source = args.data_dir
    if args.feature_profile == 'full' and (source / 'enriched/train.parquet').exists():
        source = source / 'enriched'
    train = pd.read_parquet(source / 'train.parquet')
    test = pd.read_parquet(source / 'test.parquet')
    assert train.id.is_unique and test.id.is_unique
    assert not set(train.id).intersection(test.id), 'Training and test IDs overlap'
    for frame in (train, test):
        assert frame.id.notna().all() and frame.label.notna().all()
        assert set(frame.label.unique()) == {0.0, 1.0}
        assert (frame.label == (frame.subscription_type == 'Customer').astype(float)).all()
    if args.sample_fraction < 1:
        train = train.groupby('label', group_keys=False).sample(frac=args.sample_fraction, random_state=args.seed)
    numeric = CORE_NUMERIC + (WEATHER if args.feature_profile == 'full' else [])
    categorical = CORE_CATEGORICAL + (['start_cluster', 'end_cluster', 'weather_events'] if args.feature_profile == 'full' else [])
    features = numeric + categorical
    assert set(features).issubset(train.columns) and set(features).issubset(test.columns), 'Prepare the full features first'
    for frame in (train, test):
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
        assert not np.isinf(frame[numeric].to_numpy(dtype=float)).any()
        for column in categorical:
            frame[column] = frame[column].astype('string').fillna('__MISSING__').astype(str)
    if args.evaluation == 'validation':
        if args.validation_folds:
            import pyarrow as pa
            import pyarrow.parquet as pq
            fold_parts = sorted(args.validation_folds.glob('*.parquet'))
            assert fold_parts, 'No saved fold Parquet parts found'
            fold_tables = [pq.ParquetFile(('\\\\?\\' + str(part.resolve())) if os.name == 'nt' else str(part))
                           .read(columns=['id', 'label', 'cv_fold']) for part in fold_parts]
            folds = pa.concat_tables(fold_tables).to_pandas()
            assert folds.id.is_unique and set(folds.id) == set(train.id)
            assert set(folds.cv_fold.unique()) == {0, 1, 2}
            aligned = train[['id', 'label']].merge(folds, on='id', suffixes=('', '_fold'), validate='one_to_one')
            assert (aligned.label == aligned.label_fold).all(), 'Fold labels disagree with original training labels'
            membership = folds.set_index('id').cv_fold
            holdout = train.id.map(membership).eq(0)
            fitting, scoring = train.loc[~holdout], train.loc[holdout]
        else:
            fit_indices, score_indices = next(StratifiedKFold(3, shuffle=True, random_state=args.seed).split(train, train.label))
            fitting, scoring = train.iloc[fit_indices], train.iloc[score_indices]
    else:
        fitting, scoring = train, test
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', MinMaxScaler())]), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
    ], remainder='drop')
    started = time.perf_counter()
    matrix = preprocessing.fit_transform(fitting[features]).astype(np.float32)
    labels = fitting.label.to_numpy(dtype=np.int32)
    # Width-.03 cells guarantee a maximum within-cell feature deviation below .03.
    # Boundary-neighbor merging and the original representative order are unspecified.
    # Keep one seed-shuffled example per cell AND class; never use a test row here.
    order = np.random.default_rng(args.seed).permutation(len(fitting))
    if args.sampling_mode == 'similarity':
        bins = np.floor(np.clip(matrix[order], 0.0, 1.0) / args.similarity_width).astype(np.int16)
        keys = np.column_stack([bins, labels[order].astype(np.int16)])
        _, retained = np.unique(keys, axis=0, return_index=True)
        indices = order[np.sort(retained)]
    else:
        indices = order
        assert len(indices) == len(fitting) and len(np.unique(indices)) == len(fitting)
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                          objective='binary:logistic', eval_metric='logloss',
                          tree_method='hist', n_jobs=2, random_state=args.seed)
    model.fit(matrix[indices], labels[indices])
    fit_seconds = time.perf_counter() - started
    probability = model.predict_proba(preprocessing.transform(scoring[features]).astype(np.float32))[:, 1]
    prediction = (probability > 0.5).astype(int)
    truth = scoring.label.to_numpy(dtype=int)
    precision_curve, recall_curve, _ = precision_recall_curve(truth, probability)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = 'baseline-' if args.sampling_mode == 'similarity' else 'xgboost-full-'
    output = args.output_dir / (prefix + timestamp + '-' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    bundle = {'preprocessing': preprocessing, 'classifier': model, 'features': features, 'threshold': 0.5}
    joblib.dump(bundle, output / 'xgboost_baseline.joblib')
    restored = joblib.load(output / 'xgboost_baseline.joblib')
    probe = scoring[features].iloc[:25]
    check = restored['classifier'].predict_proba(restored['preprocessing'].transform(probe).astype(np.float32))[:, 1]
    assert np.array_equal(check, probability[:25]), 'Reloaded baseline predictions differ'
    pd.DataFrame({'id': scoring.id.to_numpy(), 'label': truth, 'prediction': prediction,
                  'customer_probability': probability}).to_parquet(output / 'predictions.parquet', index=False)
    fitting[['id', 'label']].iloc[indices].to_parquet(output / 'sampled_training_ids.parquet', index=False)
    record = {
        'model': 'B0_protocol_approximation' if args.sampling_mode == 'similarity' else 'XGBoost_full_training_control',
        'reproduction_status': ('Approximation: source does not fully specify sampler, split, or XGBoost hyperparameters'
                                if args.sampling_mode == 'similarity' else
                                'Supplemental control: unchanged B0 features, preprocessing, classifier settings, and cutoff; sampling disabled'),
        'evaluation_scope': 'fixed_test' if args.evaluation == 'test' else 'training_validation_only',
        'feature_profile': args.feature_profile, 'numeric_features': numeric, 'categorical_features': categorical,
        'seed': args.seed, 'input_training_rows': len(fitting), 'sampled_training_rows': len(indices),
        'input_training_pool_rows': len(train),
        'validation_fold_source': args.validation_folds.name if args.validation_folds else 'Python stratified split' if args.evaluation == 'validation' else 'fixed project test',
        'sampled_customer_rows': int(labels[indices].sum()), 'evaluation_rows': len(scoring),
        'paper_sample_rows': 10354, 'paper_sample_count_forced': False,
        'sampling_mode': args.sampling_mode,
        'sampler': ('Seeded class-aware min-max feature cells; one representative per configured-width cell and label'
                    if args.sampling_mode == 'similarity' else 'None; all fitting rows retained in seeded order'),
        'similarity_width': args.similarity_width if args.sampling_mode == 'similarity' else None,
        'duration_transform': 'raw', 'threshold': 0.5,
        'class_weighting': 'none; original baseline protocol',
        'customer_f0_5': float(fbeta_score(truth, prediction, beta=0.5, zero_division=0)),
        'customer_precision': float(precision_score(truth, prediction, zero_division=0)),
        'customer_recall': float(recall_score(truth, prediction, zero_division=0)),
        'accuracy': float(accuracy_score(truth, prediction)),
        'area_under_pr': float(auc(recall_curve, precision_curve)),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn), 'fit_seconds': fit_seconds,
        'model_reload_verified': True,
        'versions': {'xgboost': xgboost.__version__, 'sklearn': sklearn.__version__, 'numpy': np.__version__},
        'classifier_parameters': {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                  for k, v in model.get_params().items()},
        'timing_definition': 'Preprocessing, similarity sampling, and classifier fit; excludes input reading and scoring'
    }
    (output / 'metrics.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: record[k] for k in ['model', 'evaluation_scope', 'sampled_training_rows', 'customer_f0_5']}, indent=2))
    print('BASELINE COMPLETE:', output)


if __name__ == '__main__':
    main()
