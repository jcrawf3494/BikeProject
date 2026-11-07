"""Documented 10,354-row XGBoost reconstruction and training-only search.

The original source code, precise sampler, split, and hyperparameters are unavailable.
This experiment records assumptions and keeps the project test set outside selection.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier

TARGET = 10354
PAPER_NUMERIC = [
    'duration_sec', 'start_hour', 'day_of_week', 'month', 'is_business_day',
    'is_inter_cluster', 'before_oct_2013', 'wind_dir_degrees',
    'mean_sea_level_pressure_inches', 'mean_humidity', 'mean_temperature_f',
    'mean_dew_point_f', 'max_gust_speed_mph', 'max_wind_speed_mph',
    'mean_wind_speed_mph', 'cloud_cover', 'min_visibility_miles',
    'mean_visibility_miles', 'max_visibility_miles', 'precipitation_inches',
    'weather_fog', 'weather_rain', 'weather_thunder', 'weather_fog_rain']
PROPOSAL_NUMERIC = ['duration_sec', 'is_weekend', 'is_holiday', 'is_round_trip'] + [
    'max_temperature_f', 'mean_temperature_f', 'min_temperature_f',
    'max_dew_point_f', 'mean_dew_point_f', 'min_dew_point_f',
    'max_humidity', 'mean_humidity', 'min_humidity',
    'max_sea_level_pressure_inches', 'mean_sea_level_pressure_inches',
    'min_sea_level_pressure_inches', 'max_visibility_miles', 'mean_visibility_miles',
    'min_visibility_miles', 'max_wind_speed_mph', 'mean_wind_speed_mph',
    'max_gust_speed_mph', 'precipitation_inches', 'cloud_cover', 'wind_dir_degrees',
    'precipitation_trace', 'weather_fog', 'weather_rain', 'weather_thunder',
    'is_business_day', 'is_inter_cluster']
PROPOSAL_CATEGORICAL = ['start_hour', 'day_of_week', 'month', 'start_cluster',
                        'end_cluster', 'weather_events']

# Fixed before scoring. The first configuration is an explicit reference assumption.
GRID = [
    dict(max_depth=3, n_estimators=100, learning_rate=0.1, min_child_weight=1, reg_lambda=1),
    dict(max_depth=3, n_estimators=400, learning_rate=0.05, min_child_weight=1, reg_lambda=1),
    dict(max_depth=5, n_estimators=400, learning_rate=0.05, min_child_weight=1, reg_lambda=1),
    dict(max_depth=7, n_estimators=400, learning_rate=0.05, min_child_weight=3, reg_lambda=5),
    dict(max_depth=9, n_estimators=400, learning_rate=0.05, min_child_weight=5, reg_lambda=10),
    dict(max_depth=5, n_estimators=800, learning_rate=0.03, min_child_weight=3, reg_lambda=5),
    dict(max_depth=7, n_estimators=800, learning_rate=0.03, min_child_weight=3, reg_lambda=5),
    dict(max_depth=5, n_estimators=300, learning_rate=0.1, min_child_weight=1, reg_lambda=1),
    dict(max_depth=7, n_estimators=300, learning_rate=0.1, min_child_weight=3, reg_lambda=5),
    dict(max_depth=3, n_estimators=800, learning_rate=0.03, min_child_weight=1, reg_lambda=5),
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_frame(path, columns=None):
    path = Path(path)
    parts = sorted(path.glob('*.parquet')) if path.is_dir() else [path]
    assert parts, 'No Parquet data found'
    tables = [pq.ParquetFile(('\\\\?\\' + str(p.resolve())) if os.name == 'nt' else str(p))
              .read(columns=columns) for p in parts]
    return pa.concat_tables(tables).to_pandas()


def save_frame(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = ('\\\\?\\' + str(path.resolve())) if os.name == 'nt' else str(path)
    frame.to_parquet(destination, index=False)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(profile):
    if profile == 'proposal_full':
        return PROPOSAL_NUMERIC, PROPOSAL_CATEGORICAL
    return PAPER_NUMERIC, ['start_city'] + (['local_zip'] if profile == 'paper_visible' else [])


def prepare(frame, locality):
    frame = frame.copy()
    assert frame.id.is_unique and frame.id.notna().all()
    if 'label' in frame:
        assert set(frame.label.unique()) == {0.0, 1.0}
        assert (frame.label == frame.subscription_type.eq('Customer').astype(float)).all()
    # Use independently verified local-clock columns, not the legacy stored timestamp.
    frame['before_oct_2013'] = (pd.to_datetime(frame.trip_date) < pd.Timestamp('2013-10-01')).astype(float)
    frame['weather_fog_rain'] = (frame.weather_fog.eq(1) & frame.weather_rain.eq(1)).astype(float)
    frame = frame.merge(locality, on='id', how='left', validate='one_to_one')
    assert frame.local_zip.notna().all(), 'Missing original-trip locality metadata'
    for col in set(PAPER_NUMERIC + PROPOSAL_NUMERIC):
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        assert not np.isinf(frame[col].to_numpy(dtype=float)).any()
    for col in set(PROPOSAL_CATEGORICAL + ['start_city', 'local_zip']):
        frame[col] = frame[col].astype('string').fillna('__MISSING__').astype(str)
    return frame.sort_values('id').reset_index(drop=True)


def preprocessing(profile):
    numeric, categorical = features(profile)
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', MinMaxScaler())]), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
    ], remainder='drop')


def estimator(params, seed):
    return XGBClassifier(**params, objective='binary:logistic', eval_metric='logloss',
                         tree_method='hist', n_jobs=2, random_state=seed,
                         scale_pos_weight=1.0, subsample=1.0, colsample_bytree=1.0)


def matrix_fit(train, profile):
    transformer = preprocessing(profile)
    numeric, categorical = features(profile)
    matrix = transformer.fit_transform(train[numeric + categorical]).astype(np.float32)
    return transformer, matrix


def transform(transformer, frame, profile):
    numeric, categorical = features(profile)
    return transformer.transform(frame[numeric + categorical]).astype(np.float32)


def sample_exact(pool, profile, method, seed):
    order = np.random.default_rng(seed).permutation(len(pool))
    if method == 'similarity':
        # Fit this geometry ONLY on the fitting pool, never outer validation or test.
        _, matrix = matrix_fit(pool, profile)
        bins = np.floor(np.clip(matrix[order], 0, 1) / 0.03).astype(np.int16)
        keys = np.ascontiguousarray(np.column_stack([bins, pool.label.to_numpy(np.int16)[order]]))
        packed = keys.view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()
        _, first = np.unique(packed, return_index=True)
        candidates = order[np.sort(first)]
        del matrix, bins, keys, packed
    else:
        candidates = order
    candidate_labels = pool.label.to_numpy(dtype=int)[candidates]
    assert len(candidates) >= TARGET, 'Similarity candidate pool is smaller than requested sample'
    # Exact stratified count preserves the candidate pool's class proportion.
    positive_count = int(round(TARGET * candidate_labels.mean()))
    rng = np.random.default_rng(seed + 313)
    positive = rng.choice(candidates[candidate_labels == 1], positive_count, replace=False)
    negative = rng.choice(candidates[candidate_labels == 0], TARGET - positive_count, replace=False)
    indices = np.concatenate([positive, negative])
    rng.shuffle(indices)
    selected = pool.iloc[indices].reset_index(drop=True)
    assert len(selected) == TARGET and selected.id.is_unique
    info = dict(method=method, target_rows=TARGET, similarity_width=0.03 if method == 'similarity' else None,
                fitting_pool_rows=len(pool), candidate_rows=len(candidates),
                candidate_customer_rows=int(candidate_labels.sum()),
                sampled_rows=len(selected), sampled_customer_rows=int(selected.label.sum()),
                sampled_customer_rate=float(selected.label.mean()),
                exact_size_adjustment='Seeded stratified selection from candidate pool; class mix follows that pool',
                similarity_assumption='One representative per width-.03 min-max feature cell AND class; cross-cell neighbors not merged')
    return selected, info


def scores(truth, probability, cutoff):
    probability = np.asarray(probability, dtype=np.float64)
    prediction = (probability > cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    denominator = 1.25 * tp + fp + 0.25 * fn
    precision_curve, recall_curve, _ = precision_recall_curve(truth, probability)
    return dict(customer_f0_5=float(1.25 * tp / denominator if denominator else 0),
                customer_precision=float(tp / (tp + fp) if tp + fp else 0),
                customer_recall=float(tp / (tp + fn) if tp + fn else 0),
                accuracy=float(accuracy_score(truth, prediction)),
                area_under_pr=float(auc(recall_curve, precision_curve)),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                evaluation_rows=len(truth), decision_threshold=float(cutoff))


def best_cutoff(truth, probability):
    probability = np.asarray(probability, dtype=np.float64)
    # Exact score groups; prediction uses > cutoff. Prefer higher cutoffs on ties.
    order = np.argsort(-probability, kind='stable')
    sorted_probability, sorted_truth = probability[order], np.asarray(truth)[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_probability) != 0), len(order) - 1]
    tp = np.cumsum(sorted_truth)[ends]
    fp = ends + 1 - tp
    fn = sorted_truth.sum() - tp
    denominator = 1.25 * tp + fp + 0.25 * fn
    values = np.divide(1.25 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    best = int(np.argmax(values))
    endpoint = int(ends[best])
    cutoff = ((float(sorted_probability[endpoint]) + float(sorted_probability[endpoint + 1])) / 2
              if endpoint + 1 < len(order) else float(np.nextafter(sorted_probability[endpoint], -np.inf, dtype=np.float32)))
    result = scores(truth, probability, cutoff)
    assert abs(result['customer_f0_5'] - values[best]) < 1e-12
    return cutoff, result


def fit_model(frame, profile, params, seed):
    transformer, matrix = matrix_fit(frame, profile)
    model = estimator(params, seed)
    model.fit(matrix, frame.label.to_numpy(dtype=int))
    return dict(preprocessing=transformer, classifier=model, profile=profile)


def predict(bundle, frame):
    probability = bundle['classifier'].predict_proba(transform(bundle['preprocessing'], frame, bundle['profile']))[:, 1].astype(np.float64)
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    return probability


def model_run(train, scoring, profile, params, seed, cutoff, output, scope):
    assert train.id.is_unique and scoring.id.is_unique and set(train.id).isdisjoint(scoring.id)
    started = time.perf_counter()
    bundle = fit_model(train, profile, params, seed)
    fit_seconds = time.perf_counter() - started
    probability = predict(bundle, scoring)
    record = scores(scoring.label.to_numpy(dtype=int), probability, cutoff)
    record.update(training_rows=len(train), training_customer_rows=int(train.label.sum()),
                  feature_profile=profile, parameters=params, class_weighting='none',
                  duration_transform='raw', evaluation_scope=scope, fit_seconds=fit_seconds,
                  model_reload_verified=True, seed=seed)
    output.mkdir(parents=True, exist_ok=False)
    bundle['threshold'] = cutoff
    model_path = output / 'model.joblib'
    model_path = ('\\\\?\\' + str(model_path.resolve())) if os.name == 'nt' else str(model_path)
    joblib.dump(bundle, model_path)
    restored = joblib.load(model_path)
    assert np.array_equal(predict(restored, scoring.iloc[:25]), probability[:25])
    save_frame(pd.DataFrame(dict(id=scoring.id.to_numpy(), label=scoring.label.to_numpy(),
                                customer_probability=probability,
                                prediction=(probability > cutoff).astype(int))), output / 'predictions.parquet')
    write_json(output / 'metrics.json', record)
    return record


def choose_parameters(train, profile, seed, output):
    selection_indices, calibration_indices = train_test_split(
        np.arange(len(train)), test_size=0.15, stratify=train.label, random_state=seed + 11)
    selection, calibration = train.iloc[selection_indices], train.iloc[calibration_indices]
    roles = train[['id', 'label']].copy()
    roles['role'] = 'model_selection'
    roles.loc[calibration_indices, 'role'] = 'cutoff_calibration'
    splitter = StratifiedKFold(3, shuffle=True, random_state=seed + 1009)
    cached = []
    fold_ids = []
    for fold, (fit_idx, score_idx) in enumerate(splitter.split(selection, selection.label)):
        fit, score = selection.iloc[fit_idx], selection.iloc[score_idx]
        transformer, matrix = matrix_fit(fit, profile)
        cached.append((matrix, fit.label.to_numpy(dtype=int), transform(transformer, score, profile),
                       score.label.to_numpy(dtype=int)))
        fold_ids.append(pd.DataFrame(dict(id=score.id.to_numpy(), cv_fold=fold)))
    roles = roles.merge(pd.concat(fold_ids), on='id', how='left', validate='one_to_one')
    save_frame(roles, output / 'selection_roles.parquet')
    candidates = []
    for index, params in enumerate(GRID):
        fixed, attainable = [], []
        for fold, (matrix, truth_fit, score_matrix, truth_score) in enumerate(cached):
            model = estimator(params, seed + fold)
            model.fit(matrix, truth_fit)
            probability = model.predict_proba(score_matrix)[:, 1]
            fixed.append(scores(truth_score, probability, 0.5)['customer_f0_5'])
            attainable.append(best_cutoff(truth_score, probability)[1]['customer_f0_5'])
        record = dict(index=index, parameters=params, fold_fixed_f05=fixed,
                      fold_attainable_f05=attainable, mean_fixed_f05=float(np.mean(fixed)),
                      mean_attainable_f05=float(np.mean(attainable)))
        candidates.append(record)
        print('CV', output.name, index + 1, '/', len(GRID),
              'fixed=', round(record['mean_fixed_f05'], 5),
              'cutoff-aware=', round(record['mean_attainable_f05'], 5), flush=True)
        write_json(output / 'cross_validation_progress.json', candidates)
    fixed_choice = max(candidates, key=lambda x: (x['mean_fixed_f05'], -x['index']))
    calibrated_choice = max(candidates, key=lambda x: (x['mean_attainable_f05'], -x['index']))
    calibration_model = fit_model(selection, profile, calibrated_choice['parameters'], seed)
    calibration_probability = predict(calibration_model, calibration)
    cutoff, calibration_scores = best_cutoff(calibration.label.to_numpy(dtype=int), calibration_probability)
    save_frame(pd.DataFrame(dict(id=calibration.id.to_numpy(), label=calibration.label.to_numpy(),
                                customer_probability=calibration_probability)), output / 'calibration_predictions.parquet')
    settings = dict(selected_by='three_fold_training_cross_validation',
                    fixed_parameters=fixed_choice['parameters'],
                    calibrated_parameters=calibrated_choice['parameters'],
                    cutoff=cutoff, calibration_scores=calibration_scores,
                    selection_rows=len(selection), calibration_rows=len(calibration))
    write_json(output / 'selected_settings.json', settings)
    return settings


def develop(args, train, test_ids, locality):
    folder = args.output_dir / ('reconstruct-10354-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    print('RECONSTRUCTION OUTPUT:', folder, flush=True)
    folds = read_frame(args.validation_folds, ['id', 'label', 'cv_fold'])
    assert folds.id.is_unique and set(folds.id) == set(train.id)
    aligned = train[['id', 'label']].merge(folds, on='id', validate='one_to_one', suffixes=('', '_fold'))
    assert aligned.label.eq(aligned.label_fold).all()
    holdout_ids = set(folds.loc[folds.cv_fold.eq(0), 'id'])
    fitting = train.loc[~train.id.isin(holdout_ids)].reset_index(drop=True)
    outer = train.loc[train.id.isin(holdout_ids)].reset_index(drop=True)
    assert set(fitting.id).isdisjoint(outer.id) and set(train.id).isdisjoint(test_ids)
    manifest = dict(target_rows=TARGET, seed=args.seed, grid=GRID,
                    profiles=args.profiles, samplers=args.samplers,
                    protocols=['fixed_project_test_reconstruction', 'compressed_dataset_80_20'],
                    stage='training_only_development', source_training_rows=len(train),
                    fitting_pool_rows=len(fitting), outer_validation_rows=len(outer),
                    project_test_used_for_selection=False, locality_metadata_sha256=sha256(args.locality_metadata),
                    local_zip_assumption='Raw rider postal code begins with 94 or 95, for a valid five-digit US ZIP; missing is separate',
                    paper_feature_mapping='Pressure/humidity/temperature/dew use mean daily values; exact original aggregations unknown',
                    source_code_unavailable=True, similarity_radius_exactly_reproduced=False,
                    reference_parameters=GRID[0], class_weighting='none',
                    dependencies=dict(xgboost=xgboost.__version__, sklearn=sklearn.__version__, numpy=np.__version__))
    write_json(folder / 'manifest.json', manifest)
    save_frame(outer[['id', 'label']], folder / 'outer_validation_ids.parquet')
    all_records = []
    for profile in args.profiles:
        for sampler in args.samplers:
            name = profile + '__' + sampler
            recipe = folder / name
            recipe.mkdir()
            sampled, sample_info = sample_exact(fitting, profile, sampler, args.seed)
            write_json(recipe / 'sample.json', sample_info)
            save_frame(sampled[['id', 'label']], recipe / 'sampled_ids.parquet')
            print('SAMPLE', name, sample_info, flush=True)
            # Two independent evaluation procedures with identical declared feature/sample definitions.
            for protocol in ('shared_split', 'compressed_holdout'):
                output = recipe / protocol
                output.mkdir()
                if protocol == 'shared_split':
                    fit, score = sampled, outer
                else:
                    fit_idx, score_idx = train_test_split(np.arange(TARGET), test_size=0.2,
                                                         stratify=sampled.label, random_state=args.seed + 73)
                    fit, score = sampled.iloc[fit_idx].reset_index(drop=True), sampled.iloc[score_idx].reset_index(drop=True)
                save_frame(fit[['id', 'label']], output / 'fitting_ids.parquet')
                save_frame(score[['id', 'label']], output / 'scoring_ids.parquet')
                settings = choose_parameters(fit, profile, args.seed, output)
                procedures = [('reference_fixed', GRID[0], 0.5),
                              ('tuned_fixed', settings['fixed_parameters'], 0.5),
                              ('tuned_calibrated', settings['calibrated_parameters'], settings['cutoff'])]
                for policy, params, cutoff in procedures:
                    scope = 'original_training_outer_validation' if protocol == 'shared_split' else 'compressed_training_dataset_holdout'
                    record = model_run(fit, score, profile, params, args.seed, cutoff, output / policy, scope)
                    record.update(recipe=name, sampler=sampler, protocol=protocol, policy=policy)
                    all_records.append(record)
                    write_json(folder / 'development_results.json', all_records)
                    print('RESULT', name, protocol, policy, 'F0.5=', round(record['customer_f0_5'], 6), flush=True)
    # The shared project test has not been scored. Freeze recipe/settings using development results only.
    eligible = [r for r in all_records if r['protocol'] == 'shared_split' and r['policy'] != 'reference_fixed']
    winner = max(eligible, key=lambda r: r['customer_f0_5'])
    frozen = dict(selected_by='original_training_outer_validation_after_training_cv',
                  selected_before_project_test_scoring=True, target_training_rows=TARGET,
                  feature_profile=winner['feature_profile'], sampler=winner['sampler'],
                  policy=winner['policy'], parameters=winner['parameters'],
                  development_cutoff=winner['decision_threshold'],
                  development_validation_f05=winner['customer_f0_5'],
                  seeds=[42, 43, 44], source_recipe=winner['recipe'],
                  reference_parameters=GRID[0], project_test_used_for_selection=False,
                  manifest_sha256=sha256(folder / 'manifest.json'))
    compressed_settings = folder / 'paper_visible__similarity/compressed_holdout/selected_settings.json'
    similarity_settings = folder / 'paper_visible__similarity/shared_split/selected_settings.json'
    if similarity_settings.exists():
        primary = json.loads(similarity_settings.read_text())
        frozen['shared_similarity_primary'] = dict(
            recipe_predeclared='paper_visible__similarity', selected_by='training_cross_validation_only',
            feature_profile='paper_visible', sampler='similarity',
            parameters=primary['calibrated_parameters'], reference_parameters=GRID[0])
    if compressed_settings.exists():
        compressed = json.loads(compressed_settings.read_text())
        frozen['compressed_primary'] = dict(
            recipe_predeclared='paper_visible__similarity', selected_by='training_cross_validation_only',
            feature_profile='paper_visible', sampler='similarity',
            parameters=compressed['calibrated_parameters'],
            definition='10,354 total compressed rows; 8,283 fit and 2,071 holdout; independent of project test',
            reference_parameters=GRID[0])
    write_json(folder / 'frozen_project_test_settings.json', frozen)
    write_json(folder / 'development_complete.json', dict(completed=True, recipes=len(args.profiles) * len(args.samplers), results=len(all_records)))
    print('DEVELOPMENT COMPLETE:', folder, flush=True)


def final(args, train, locality):
    assert args.frozen_settings, 'Final mode requires --frozen-settings'
    frozen = json.loads(args.frozen_settings.read_text())
    assert frozen['selected_before_project_test_scoring'] and not frozen['project_test_used_for_selection']
    assert frozen['target_training_rows'] == TARGET
    folder = args.output_dir / ('baseline-10354-test-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'frozen_settings_copy.json', frozen)
    frozen_digest = sha256(args.frozen_settings)
    # Read test predictor values and labels only after the frozen settings have been copied.
    test = prepare(read_frame(args.data_dir / 'enriched/test.parquet'), locality)
    assert set(train.id).isdisjoint(test.id)
    all_records = []
    for seed in frozen['seeds']:
        sampled, sample_info = sample_exact(train, frozen['feature_profile'], frozen['sampler'], seed)
        output = folder / ('seed-' + str(seed))
        output.mkdir()
        write_json(output / 'sample.json', sample_info)
        save_frame(sampled[['id', 'label']], output / 'sampled_ids.parquet')
        cutoff = 0.5
        if frozen['policy'] == 'tuned_calibrated':
            fit_idx, calibration_idx = train_test_split(np.arange(TARGET), test_size=0.15,
                                                       stratify=sampled.label, random_state=seed + 11)
            model = fit_model(sampled.iloc[fit_idx], frozen['feature_profile'], frozen['parameters'], seed)
            calibration = sampled.iloc[calibration_idx]
            probability = predict(model, calibration)
            cutoff, calibration_scores = best_cutoff(calibration.label.to_numpy(dtype=int), probability)
            save_frame(pd.DataFrame(dict(id=calibration.id.to_numpy(), label=calibration.label.to_numpy(),
                                        customer_probability=probability)), output / 'calibration_predictions.parquet')
            write_json(output / 'cutoff_selection.json', dict(cutoff=cutoff, calibration_scores=calibration_scores))
        for policy, params, threshold in [('reference_fixed', GRID[0], 0.5),
                                          ('training_selected', frozen['parameters'], cutoff)]:
            record = model_run(sampled, test, frozen['feature_profile'], params, seed, threshold, output / policy, 'fixed_project_test')
            record.update(policy=policy, sampler=frozen['sampler'], frozen_settings_sha256=frozen_digest)
            all_records.append(record)
            write_json(folder / 'results.json', all_records)
            print('FINAL', seed, policy, 'F0.5=', round(record['customer_f0_5'], 6), flush=True)
    assert sha256(args.frozen_settings) == frozen_digest, 'Frozen settings changed during final evaluation'
    if 'shared_similarity_primary' in frozen:
        primary = frozen['shared_similarity_primary']
        for seed in frozen['seeds']:
            sampled, sample_info = sample_exact(train, primary['feature_profile'], primary['sampler'], seed)
            output = folder / 'paper-similarity' / ('seed-' + str(seed))
            output.mkdir(parents=True)
            write_json(output / 'sample.json', sample_info)
            save_frame(sampled[['id', 'label']], output / 'sampled_ids.parquet')
            train_idx, calibration_idx = train_test_split(np.arange(TARGET), test_size=0.15,
                                                         stratify=sampled.label, random_state=seed + 11)
            model = fit_model(sampled.iloc[train_idx], primary['feature_profile'], primary['parameters'], seed)
            calibration = sampled.iloc[calibration_idx]
            probability = predict(model, calibration)
            cutoff, calibration_scores = best_cutoff(calibration.label.to_numpy(dtype=int), probability)
            save_frame(pd.DataFrame(dict(id=calibration.id.to_numpy(), label=calibration.label.to_numpy(),
                                        customer_probability=probability)), output / 'calibration_predictions.parquet')
            write_json(output / 'cutoff_selection.json', dict(cutoff=cutoff, calibration_scores=calibration_scores))
            for policy, params, threshold in [('reference_fixed', GRID[0], 0.5),
                                              ('training_selected', primary['parameters'], cutoff)]:
                record = model_run(sampled, test, primary['feature_profile'], params, seed, threshold,
                                   output / policy, 'fixed_project_test')
                record.update(policy='similarity_' + policy, sampler='similarity',
                              family='paper_similarity_primary', frozen_settings_sha256=frozen_digest)
                all_records.append(record)
                write_json(folder / 'results.json', all_records)
                print('SIMILARITY FINAL', seed, policy, 'F0.5=', round(record['customer_f0_5'], 6), flush=True)
    if 'compressed_primary' in frozen:
        compressed = frozen['compressed_primary']
        compressed_records = []
        for seed in frozen['seeds']:
            sampled, sample_info = sample_exact(train, compressed['feature_profile'], compressed['sampler'], seed)
            fit_idx, score_idx = train_test_split(np.arange(TARGET), test_size=0.2,
                                                 stratify=sampled.label, random_state=seed + 73)
            fit, score = sampled.iloc[fit_idx].reset_index(drop=True), sampled.iloc[score_idx].reset_index(drop=True)
            output = folder / 'compressed' / ('seed-' + str(seed))
            output.mkdir(parents=True)
            write_json(output / 'sample.json', sample_info)
            save_frame(fit[['id', 'label']], output / 'fitting_ids.parquet')
            save_frame(score[['id', 'label']], output / 'scoring_ids.parquet')
            train_idx, calibration_idx = train_test_split(np.arange(len(fit)), test_size=0.15,
                                                         stratify=fit.label, random_state=seed + 11)
            model = fit_model(fit.iloc[train_idx], compressed['feature_profile'], compressed['parameters'], seed)
            calibration = fit.iloc[calibration_idx]
            calibration_probability = predict(model, calibration)
            cutoff, calibration_scores = best_cutoff(calibration.label.to_numpy(dtype=int), calibration_probability)
            save_frame(pd.DataFrame(dict(id=calibration.id.to_numpy(), label=calibration.label.to_numpy(),
                                        customer_probability=calibration_probability)), output / 'calibration_predictions.parquet')
            write_json(output / 'cutoff_selection.json', dict(cutoff=cutoff, calibration_scores=calibration_scores))
            for policy, params, threshold in [('reference_fixed', GRID[0], 0.5),
                                              ('training_selected', compressed['parameters'], cutoff)]:
                record = model_run(fit, score, compressed['feature_profile'], params, seed, threshold,
                                   output / policy, 'compressed_training_dataset_holdout')
                record.update(policy=policy, sampler=compressed['sampler'], compressed_total_rows=TARGET,
                              frozen_settings_sha256=frozen_digest)
                compressed_records.append(record)
                write_json(folder / 'compressed_results.json', compressed_records)
                print('COMPRESSED', seed, policy, 'F0.5=', round(record['customer_f0_5'], 6), flush=True)
    assert sha256(args.frozen_settings) == frozen_digest
    write_json(folder / 'complete.json', dict(completed=True, seeds=frozen['seeds'], fixed_test_rows=len(test),
                                            results=len(all_records), frozen_settings_sha256=frozen_digest))
    print('FINAL COMPLETE:', folder, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--locality-metadata', type=Path, required=True)
    parser.add_argument('--mode', choices=['develop', 'final'], default='develop')
    parser.add_argument('--validation-folds', type=Path)
    parser.add_argument('--frozen-settings', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--profiles', nargs='+', choices=['paper_visible', 'paper_without_zip', 'proposal_full'],
                        default=['paper_visible', 'paper_without_zip', 'proposal_full'])
    parser.add_argument('--samplers', nargs='+', choices=['stratified', 'similarity'], default=['stratified', 'similarity'])
    args = parser.parse_args()
    if args.mode == 'develop' and not args.validation_folds:
        args.validation_folds = args.data_dir / 'validation_folds.parquet'
        if not args.validation_folds.exists():
            parser.error('Development requires saved Spark --validation-folds or data/validation_folds.parquet')
    locality = read_frame(args.locality_metadata)
    assert locality.id.is_unique and set(locality.local_zip.unique()) <= {'local', 'outside', 'missing'}
    train = prepare(read_frame(args.data_dir / 'enriched/train.parquet'), locality)
    test_ids = set(read_frame(args.data_dir / 'enriched/test.parquet', ['id']).id)
    assert set(train.id).isdisjoint(test_ids)
    if args.mode == 'develop':
        develop(args, train, test_ids, locality)
    else:
        final(args, train, locality)


if __name__ == '__main__':
    main()
