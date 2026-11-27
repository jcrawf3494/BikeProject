"""Fixed-parameter XGBoost controls sharing Spark inputs, cohorts and calibration IDs."""
import argparse
import atexit
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid

for thread_setting in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[thread_setting]='2'

WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(WORK/'baseline-runtime'),str(WORK/'vendor')]
ROOT=WORK.parent

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)
def win(path):return ('\\\\?\\'+str(path.resolve())) if os.name=='nt' else str(path)

def libraries():
    global np,pd,pa,pq,joblib,ColumnTransformer,Pipeline,SimpleImputer,FunctionTransformer,MinMaxScaler,OneHotEncoder,XGBClassifier
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer,MinMaxScaler,OneHotEncoder
    from xgboost import XGBClassifier

def read_frame(path,columns=None):
    files=sorted(path.glob('*.parquet')) if path.is_dir() else [path]
    assert files
    return pa.concat_tables([pq.ParquetFile(win(f)).read(columns=columns) for f in files]).to_pandas()

def preprocessing(settings):
    numeric=[c for c in settings['numeric_features'] if c!='duration_sec']
    duration=Pipeline([('log',FunctionTransformer(np.log1p,feature_names_out='one-to-one')),
                       ('median',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',MinMaxScaler())])
    continuous=Pipeline([('median',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',MinMaxScaler())])
    categorical=Pipeline([('missing',FunctionTransformer(pd.DataFrame.fillna,kw_args={'value':'__MISSING__'},feature_names_out='one-to-one')),
                          ('strings',FunctionTransformer(np.asarray,kw_args={'dtype':str},feature_names_out='one-to-one')),
                          ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32))])
    return ColumnTransformer([('duration',duration,['duration_sec']),('numeric',continuous,numeric),
                              ('categorical',categorical,settings['categorical_features'])],sparse_threshold=1.0)

def weights(y):
    counts=np.bincount(np.asarray(y,dtype=np.int32),minlength=2)
    assert (counts>0).all()
    values=len(y)/(2.0*counts)
    return values[np.asarray(y,dtype=np.int32)],{'Subscriber':float(values[0]),'Customer':float(values[1])}

def score(y,probability,threshold):
    truth=np.asarray(y,dtype=np.int32);probability=np.asarray(probability)
    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
    pred=probability>threshold
    tp=int(np.sum(pred & (truth==1)));fp=int(np.sum(pred & (truth==0)))
    fn=int(np.sum(~pred & (truth==1)));tn=int(np.sum(~pred & (truth==0)))
    den=1.25*tp+fp+.25*fn
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'customer_f0_5':1.25*tp/den if den else 0.0,
            'customer_precision':tp/(tp+fp) if tp+fp else 0.0,'customer_recall':tp/(tp+fn) if tp+fn else 0.0,
            'accuracy':(tp+tn)/len(truth),'evaluation_rows':len(truth)}

def cutoff(y,probability):
    y=np.asarray(y,dtype=np.int32);p=np.asarray(probability,dtype=np.float64)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    order=np.argsort(-p,kind='stable');ordered=p[order];truth=y[order]
    ends=np.r_[np.flatnonzero(ordered[:-1]!=ordered[1:]),len(ordered)-1]
    tp=np.cumsum(truth)[ends];fp=ends+1-tp;fn=truth.sum()-tp
    values=1.25*tp/(1.25*tp+fp+.25*fn)
    best=int(np.flatnonzero(values==values.max())[0]);end=int(ends[best]);observed=float(ordered[end])
    lower=float(ordered[end+1]) if end+1<len(ordered) else 0.0
    threshold=(observed+lower)/2.0
    if not 0<threshold<1:threshold=.5
    chosen=score(y,p,threshold);default=score(y,p,.5)
    assert chosen['customer_f0_5']+1e-12>=default['customer_f0_5']
    return threshold,{'threshold':threshold,'test_used':False,'rows':len(y),
                      'method':'Exact training-calibration F0.5 curve, highest observed-score tie break and midpoint boundary',
                      'default_threshold_metrics':default,'selected_threshold_metrics':chosen}

def model(settings,parameters,seed):
    classifier=XGBClassifier(**parameters,objective='binary:logistic',tree_method='hist',max_bin=256,
                             subsample=1.0,colsample_bytree=1.0,scale_pos_weight=1.0,
                             random_state=seed,n_jobs=2,device='cpu',eval_metric='logloss',verbosity=0)
    return Pipeline([('preprocessing',preprocessing(settings)),('classifier',classifier)])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,default=ROOT/'outputs/BikeShareModelTraining')
    parser.add_argument('--metadata-dir',type=Path)
    parser.add_argument('--primary-state',type=Path)
    parser.add_argument('--sample-state',type=Path)
    parser.add_argument('--full-data-dir',type=Path)
    parser.add_argument('--sample-data-dir',type=Path)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--check-preprocessing',action='store_true')
    args=parser.parse_args();package=args.package.resolve()
    batch=args.metadata_dir or package/'results/matched-xgb-state'
    primary=args.primary_state or package/'validation/paper_flags_full_20261108'
    samples=args.sample_state or package/'validation/paper_input_sample_controls_20261108'
    full_data=args.full_data_dir or package/'data/paper_flags_full'
    sample_data=args.sample_data_dir or package/'data/pfc'
    settings_path=primary/'frozen_selected_settings.json'
    if not settings_path.exists():settings_path=package/'validation/P1_P2_P3_with_paper_inputs.json'
    settings=load(settings_path);features=settings['numeric_features']+settings['categorical_features']
    assert settings['duration_transform']=='log' and settings.get('duration_cap_seconds',0)==0
    assert len(features)==len(set(features))==39
    assert not set(features)&{'id','label','subscription_type','zip_code','local_zip','trip_date','start_ts'}
    original=package/'validation/reconstruction_10354_frozen_settings.json'
    parameters=load(original)['parameters']
    protocol={'frozen_utc':now(),'model':'XGBoost_matched_inputs_weighted',
              'cohorts':['full','stratified','similarity'],'seeds':[42,43,44],'expected_model_records':9,
              'training_rows':{'full':536406,'stratified':10354,'similarity':10354},'test_rows':133553,
              'feature_count':39,'numeric_features':settings['numeric_features'],'categorical_features':settings['categorical_features'],
              'settings_sha256':sha(settings_path),'source_baseline_settings_sha256':sha(original),'parameters':parameters,
              'parameter_origin':'Existing stronger stratified-reconstruction training-CV selection; transferred unchanged across all39-input cohorts, not newly tuned',
              'class_weighting':'N/(2*N_class), independently recomputed for calibration fit and final fit',
              'calibration_roles':'Exactly the corresponding Spark P3 saved training-role IDs; no extra calibration trips outside the supplied cohort',
              'preprocessing':'Log1p duration, training-only median numeric imputation, min-max scaling, constant categorical missing values and one-hot encoding',
              'implementation':{'tree_method':'hist','max_bin':256,'subsample':1.0,'colsample_bytree':1.0,'n_jobs':2,'device':'cpu','thread_environment_limit':2},
              'source_sha256':sha(Path(__file__)),'dependencies':{name:importlib.metadata.version(name) for name in ['xgboost','scikit-learn','numpy','pyarrow']},
              'scope':'NEW weighted matched-input control; original paper-style XGBoost reconstruction remains a separate reference',
              'test_use':'Repeated fixed benchmark; no test-based hyperparameter or cutoff selection'}
    batch.mkdir(parents=True,exist_ok=True)
    if args.check_preprocessing:
        libraries()
        frame=pd.DataFrame({c:[0.0,1.0,2.0] for c in settings['numeric_features']})
        frame['duration_sec']=[300.,600.,1200.]
        for c in settings['categorical_features']:
            frame[c]=[1,2,3] if c in {'start_hour','day_of_week','month','start_cluster','end_cluster'} else ['a',None,'b']
        frame['rider_locality']=['local','missing','outside']
        prep=preprocessing(settings);matrix=prep.fit_transform(frame)
        check=batch/'preprocessing_check.joblib';joblib.dump(prep,check)
        assert np.array_equal(matrix.toarray(),joblib.load(check).transform(frame).toarray())
        print(json.dumps({'preprocessing_check_passed':True,'rows':3,'input_columns':39,'transformed_columns':matrix.shape[1],'classifier_trained':False}));return
    fingerprints={}
    data_groups=[('full',full_data)] + [(f'{kind}/seed-{seed}',sample_data/kind/f'seed-{seed}')
                                      for kind in ('similarity','stratified') for seed in (42,43,44)]
    for key,folder in data_groups:
        parts=sorted((folder/'enriched/train.parquet').glob('*.parquet'));assert parts
        for part in parts:fingerprints[f'{key}/train/{part.name}']=sha(part)
    for part in sorted((full_data/'enriched/test.parquet').glob('*.parquet')):
        fingerprints[f'common_test/{part.name}']=sha(part)
    protocol['input_parquet_sha256']=fingerprints
    path=batch/'protocol.json'
    if path.exists():
        previous=load(path);protocol['frozen_utc']=previous['frozen_utc'];assert protocol==previous,'Frozen control protocol changed'
    else:write(path,protocol)
    if args.freeze_only:print(json.dumps({'frozen':True,'matched_xgboost_cases':9,'parameters':parameters}));return
    if os.name=='nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        atexit.register(ctypes.windll.kernel32.SetThreadExecutionState,0x80000000)
    write(batch/'queue_status.json',{'phase':'waiting_for_spark_controls','pid':os.getpid(),'started_utc':now()})
    print('WAITING_FOR_SPARK_CONTROLS',flush=True)
    while not (primary/'batch_completed.json').exists() or not (samples/'batch_completed.json').exists():
        for folder in (primary,samples):
            for path in folder.glob('*P*_seed_*.json'):
                try:state=load(path)
                except (OSError,json.JSONDecodeError):continue
                if isinstance(state,dict) and state.get('status')=='failed':raise RuntimeError('A Spark prerequisite failed; recover it before these controls')
        time.sleep(10)
    libraries()
    from sklearn.metrics import precision_recall_curve,auc
    execution=batch/'execution.json'
    if execution.exists():run=package/load(execution)['run']
    else:
        run=package/'results'/('xgb-matched39-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
        run.mkdir(parents=True,exist_ok=True);write(execution,{'run':run.relative_to(package).as_posix()})
        write(run/'manifest.json',protocol)
    test=read_frame(full_data/'enriched/test.parquet',['id','label']+features)
    assert len(test)==133553 and test.id.is_unique and int(test.label.sum())==20723
    rows=[]
    for cohort in protocol['cohorts']:
        for seed in protocol['seeds']:
            key=f'{cohort}_seed_{seed}';status_path=batch/(key+'.json')
            old=load(status_path) if status_path.exists() else {}
            if old.get('status')=='completed':rows.append(load(package/old['metrics']));continue
            data=full_data if cohort=='full' else sample_data/cohort/f'seed-{seed}'
            prefix='full' if cohort=='full' else f'{cohort}/seed-{seed}'
            for part in (data/'enriched/train.parquet').glob('*.parquet'):
                assert sha(part)==protocol['input_parquet_sha256'][f'{prefix}/train/{part.name}']
            for part in (full_data/'enriched/test.parquet').glob('*.parquet'):
                assert sha(part)==protocol['input_parquet_sha256'][f'common_test/{part.name}']
            train=read_frame(data/'enriched/train.parquet',['id','label']+features)
            assert len(train)==protocol['training_rows'][cohort] and train.id.is_unique
            assert set(train.id).isdisjoint(test.id) and set(train.label.unique())=={0.,1.}
            spark_state=load((primary/f'P3_seed_{seed}.json') if cohort=='full' else samples/f'{cohort}_P3_seed_{seed}.json')
            assert spark_state['status']=='completed'
            roles=read_frame(package/spark_state['run']/f'training_roles/seed-{seed}/fraction-10000')
            assert roles.id.is_unique and set(roles.id)==set(train.id)
            indexed=train.set_index('id',drop=False)
            selected=indexed.loc[roles.loc[roles.training_role=='model_selection','id']]
            calibration=indexed.loc[roles.loc[roles.training_role=='calibration','id']]
            assert len(selected)+len(calibration)==len(train) and set(selected.id).isdisjoint(calibration.id)
            assert set(selected.label.unique())==set(calibration.label.unique())=={0.,1.}
            attempt_number=len(old.get('attempts',[]))+1
            out=run/cohort/f'seed-{seed}'/f'attempt-{attempt_number}';out.mkdir(parents=True,exist_ok=True)
            attempt={'started_utc':now(),'status':'running'}
            state={'cohort':cohort,'seed':seed,'status':'running','attempts':old.get('attempts',[])+[attempt]}
            write(status_path,state);print('START '+key,flush=True)
            try:
                calibration_model=model(settings,parameters,seed)
                selected_weights,selected_map=weights(selected.label)
                started=time.perf_counter();cpu=time.process_time()
                calibration_model.fit(selected[features],selected.label.astype(np.int32),classifier__sample_weight=selected_weights)
                calibration_probability=calibration_model.predict_proba(calibration[features])[:,1].astype(float)
                threshold,policy=cutoff(calibration.label,calibration_probability)
                calibration_seconds=time.perf_counter()-started;calibration_cpu=time.process_time()-cpu
                pq.write_table(pa.table({'id':calibration.id.to_numpy(),'label':calibration.label.to_numpy(),
                                         'customer_probability':calibration_probability}),win(out/'calibration_predictions.parquet'))
                write(out/'threshold_selection.json',policy)
                del calibration_model;gc.collect()
                final=model(settings,parameters,seed);final_weights,final_map=weights(train.label)
                started=time.perf_counter();cpu=time.process_time()
                final.fit(train[features],train.label.astype(np.int32),classifier__sample_weight=final_weights)
                fit_seconds=time.perf_counter()-started;fit_cpu=time.process_time()-cpu
                probability=final.predict_proba(test[features])[:,1].astype(float)
                metrics=score(test.label,probability,threshold)
                precision,recall,_=precision_recall_curve(test.label,probability)
                metrics['area_under_pr']=float(auc(recall,precision))
                model_path=out/'model.joblib';joblib.dump(final,win(model_path))
                probe=test[features].head(25)
                assert np.array_equal(final.predict_proba(probe),joblib.load(win(model_path)).predict_proba(probe))
                pq.write_table(pa.table({'id':test.id.to_numpy(),'label':test.label.to_numpy(),
                                         'prediction':(probability>threshold).astype(float),'customer_probability':probability}),
                               win(out/'predictions.parquet'))
                pq.write_table(pa.Table.from_pandas(roles,preserve_index=False),win(out/'training_roles.parquet'))
                pq.write_table(pa.table({'id':train.id.to_numpy(),'label':train.label.to_numpy()}),win(out/'fitting_ids.parquet'))
                affected=(calibration_seconds>120 and calibration_cpu/calibration_seconds<.1) or (fit_seconds>120 and fit_cpu/fit_seconds<.1)
                metrics.update(model='XGBoost_matched_inputs_weighted',seed=seed,cohort=cohort,
                               training_rows=len(train),model_selection_rows=len(selected),calibration_rows=len(calibration),
                               customer_training_rows=int(train.label.sum()),evaluation_scope='fixed_test',feature_count=39,
                               decision_threshold=threshold,final_fit_seconds=fit_seconds,threshold_selection_seconds=calibration_seconds,
                               final_fit_cpu_seconds=fit_cpu,threshold_selection_cpu_seconds=calibration_cpu,
                               runtime_timing_affected=bool(affected),class_weighting='N/(2*N_class)',
                               calibration_fit_weights=selected_map,final_fit_weights=final_map,
                               parameters=parameters,model_reload_verified=True,
                               input_numeric_features=settings['numeric_features'],input_categorical_features=settings['categorical_features'],
                               pr_area_definition='Exact sklearn precision_recall_curve trapezoidal area; Spark uses its default binned computation',
                               spark_calibration_source_run=Path(spark_state['run']).name,
                               raw_artifact_directory=out.relative_to(package).as_posix())
                write(out/'metrics.json',metrics);rows.append(metrics)
                attempt.update(status='completed',ended_utc=now());state.update(status='completed',metrics=(out/'metrics.json').relative_to(package).as_posix())
                write(status_path,state);print(f"COMPLETE {key} F0.5={metrics['customer_f0_5']}",flush=True)
            except Exception:
                attempt.update(status='failed',ended_utc=now());state['status']='failed';write(status_path,state);raise
            finally:
                del train,indexed,selected,calibration;gc.collect()
    assert len(rows)==9
    write(run/'results.json',rows);write(batch/'all_results.json',rows)
    write(batch/'batch_completed.json',{'completed_utc':now(),'model_records':9,'run':run.relative_to(package).as_posix()})
    write(batch/'queue_status.json',{'phase':'completed','ended_utc':now()});print('MATCHED_FEATURE_XGBOOST_COMPLETE',flush=True)

if __name__=='__main__':main()
