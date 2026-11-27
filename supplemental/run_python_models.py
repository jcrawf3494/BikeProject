"""Full-data EBM and CatBoost experiments with training-only selection/calibration."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import uuid

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve, auc

HERE=Path(__file__).resolve().parent
PRESETS={
    'P6':[
        dict(interactions=0,max_rounds=500,learning_rate=.03,max_bins=128,min_samples_leaf=20),
        dict(interactions=8,max_rounds=500,learning_rate=.03,max_bins=128,min_samples_leaf=20)],
    'P7':[
        dict(iterations=400,depth=6,learning_rate=.05,l2_leaf_reg=3.0),
        dict(iterations=400,depth=8,learning_rate=.05,l2_leaf_reg=3.0)]}
FAMILIES={'P6':'explainable_boosting_machine_additive_model','P7':'catboost_gradient_boosted_trees'}
BLOCKED={'id','label','subscription_type','zip_code','local_zip','trip_date','start_ts','class_weight'}


def load(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def win(path): return '\\\\?\\'+str(path.resolve()) if os.name=='nt' else str(path)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)
def read(path,columns=None):
    parts=sorted(path.glob('*.parquet')) if path.is_dir() else [path]
    assert parts,'Missing Parquet input: '+str(path)
    return pa.concat_tables([pq.ParquetFile(win(f)).read(columns=columns) for f in parts]).to_pandas()
def save_frame(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),win(path))


def weights(y):
    counts=np.bincount(np.asarray(y,dtype=int),minlength=2)
    assert (counts>0).all()
    mapping=len(y)/(2*counts)
    return mapping[np.asarray(y,dtype=int)],dict(Subscriber=float(mapping[0]),Customer=float(mapping[1]))


def score(y,probability,threshold):
    y=np.asarray(y,dtype=int);p=np.asarray(probability,dtype=float)
    assert len(y)==len(p) and len(y)>0
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    predicted=p>threshold
    tp=int(np.sum(predicted&(y==1)));fp=int(np.sum(predicted&(y==0)))
    fn=int(np.sum(~predicted&(y==1)));tn=int(np.sum(~predicted&(y==0)))
    denominator=1.25*tp+fp+.25*fn
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,evaluation_rows=len(y),
                customer_f0_5=1.25*tp/denominator if denominator else 0.,
                customer_precision=tp/(tp+fp) if tp+fp else 0.,
                customer_recall=tp/(tp+fn) if tp+fn else 0.,accuracy=(tp+tn)/len(y))


def cutoff(y,probability):
    y=np.asarray(y,dtype=int);p=np.asarray(probability,dtype=float)
    assert len(y)>0 and set(y)=={0,1} and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    order=np.argsort(-p,kind='stable');ordered=p[order];truth=y[order]
    ends=np.r_[np.flatnonzero(ordered[:-1]!=ordered[1:]),len(ordered)-1]
    tp=np.cumsum(truth)[ends];fp=ends+1-tp;fn=truth.sum()-tp
    values=1.25*tp/(1.25*tp+fp+.25*fn)
    selected=int(np.flatnonzero(values==values.max())[0]);end=int(ends[selected])
    upper=ordered[end]
    threshold=float((upper+ordered[end+1])/2) if end+1<len(ordered) else float(np.nextafter(upper,-np.inf))
    if threshold>=upper:threshold=float(np.nextafter(upper,-np.inf))
    measured=score(y,p,threshold)
    assert abs(measured['customer_f0_5']-float(values.max()))<1e-12
    return threshold,dict(threshold=threshold,rows=len(y),test_used=False,
                          method='Exact calibration F0.5; highest observed-score tie break; probability > cutoff',
                          selected_threshold_metrics=measured,default_threshold_metrics=score(y,p,.5))


def transform(frame,settings,medians=None):
    numeric=settings['numeric_features'];categorical=settings['categorical_features']
    x=frame[numeric+categorical].copy()
    x[numeric]=x[numeric].astype(float)
    assert not (x.duration_sec.dropna()<0).any()
    x['duration_sec']=np.log1p(x.duration_sec)
    assert not np.isinf(x[numeric].to_numpy()).any()
    if medians is None:
        medians={name:float(x[name].median()) for name in numeric}
        assert all(np.isfinite(value) for value in medians.values())
    x[numeric]=x[numeric].fillna(medians)
    for name in categorical:x[name]=x[name].fillna('__MISSING__').astype(str)
    assert not x.isna().any().any()
    return x,medians


def classifier(model,parameters,settings,seed,smoke=False):
    parameters=dict(parameters)
    if model=='P6':
        from interpret.glassbox import ExplainableBoostingClassifier
        if smoke:parameters.update(max_rounds=20,interactions=2,max_bins=32,min_samples_leaf=4)
        # No internal early-stopping exclusion: every final training row enters fitting.
        return ExplainableBoostingClassifier(**parameters,feature_types=['continuous']*len(settings['numeric_features'])+
                                              ['nominal']*len(settings['categorical_features']),
                                              outer_bags=1,validation_size=0,early_stopping_rounds=0,
                                              max_interaction_bins=16,n_jobs=2,random_state=seed)
    from catboost import CatBoostClassifier
    if smoke:parameters.update(iterations=25,depth=3)
    return CatBoostClassifier(**parameters,cat_features=settings['categorical_features'],loss_function='Logloss',
                             random_seed=seed,thread_count=2,task_type='CPU',verbose=False,allow_writing_files=False)


def fit(model,parameters,settings,seed,frame,smoke=False):
    x,medians=transform(frame,settings)
    sample_weights,weight_map=weights(frame.label)
    estimator=classifier(model,parameters,settings,seed,smoke)
    estimator.fit(x,frame.label.astype(int),sample_weight=sample_weights)
    return estimator,medians,weight_map


def predict(estimator,medians,frame,settings):
    x,_=transform(frame,settings,medians)
    assert np.array_equal(estimator.classes_,np.array([0,1]))
    return estimator.predict_proba(x)[:,1].astype(float)


def validate_roles(roles,train):
    assert roles.id.is_unique and set(roles.id)==set(train.id)
    assert set(roles.training_role)=={'model_selection','calibration'}
    assert set(roles.loc[roles.training_role=='model_selection','cv_fold'])=={0,1,2}
    assert roles.loc[roles.training_role=='calibration','cv_fold'].isna().all()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['smoke','train','tune'],default='train')
    parser.add_argument('--models',default='P6,P7')
    parser.add_argument('--seeds',default='42,43,44')
    parser.add_argument('--data-dir',type=Path,default=HERE.parent/'data/paper_flags_full')
    parser.add_argument('--roles-dir',type=Path,default=HERE.parent/'data/supplemental_roles')
    parser.add_argument('--features-file',type=Path,default=HERE.parent/'validation/P1_P2_P3_with_paper_inputs.json')
    parser.add_argument('--settings-file',type=Path)
    parser.add_argument('--technical-check',action='store_true',help='Small synthetic CV verification only; its settings cannot be imported for scientific runs')
    parser.add_argument('--output-dir',type=Path,default=HERE.parent/'results')
    args=parser.parse_args()
    assert not args.technical_check or args.mode=='tune','Technical CV checks use tune mode only.'
    settings=load(args.features_file);features=settings['numeric_features']+settings['categorical_features']
    assert len(features)==len(set(features))==39 and not set(features)&BLOCKED
    assert settings['duration_transform']=='log' and settings['duration_cap_seconds']==0
    models=args.models.split(',');seeds=[int(seed) for seed in args.seeds.split(',')]
    assert models and len(set(models))==len(models) and set(models)<=PRESETS.keys()
    assert seeds and len(set(seeds))==len(seeds)
    assert args.mode!='tune' or len(seeds)==1,'Tune one declared seed, freeze parameters, then evaluate three seeds.'
    imported=load(args.settings_file) if args.settings_file else None
    if imported:
        assert args.mode=='train' and imported['selected_by']=='training_cross_validation' and not imported.get('technical_verification',False)
        assert imported['numeric_features']==settings['numeric_features'] and imported['categorical_features']==settings['categorical_features']
    active=args.data_dir/'enriched' if (args.data_dir/'enriched/train.parquet').exists() else args.data_dir
    train=read(active/'train.parquet',['id','label']+features)
    # Tune and smoke never score the fixed project test. Smoke uses a synthetic training holdout.
    test=read(active/'test.parquet',['id','label']+features) if args.mode=='train' else None
    assert train.id.is_unique and set(train.label)=={0.,1.}
    if args.mode!='smoke' and not args.technical_check:assert len(train)==536406 and int(train.label.sum())==82490
    if test is not None:
        assert test.id.is_unique and len(test)==133553 and int(test.label.sum())==20723
        assert set(test.id).isdisjoint(train.id)
    run=args.output_dir/(args.mode+'-additional-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True,exist_ok=False)
    parts=list((active/'train.parquet').glob('*.parquet')) if (active/'train.parquet').is_dir() else [active/'train.parquet']
    if test is not None:parts+=list((active/'test.parquet').glob('*.parquet')) if (active/'test.parquet').is_dir() else [active/'test.parquet']
    manifest=dict(started_utc=now(),mode=args.mode,models=models,seeds=seeds,train_rows=len(train),
                  test_rows=len(test) if test is not None else None,numeric_features=settings['numeric_features'],
                  categorical_features=settings['categorical_features'],source_sha256=sha(Path(__file__)),
                  feature_settings_sha256=sha(args.features_file),input_hashes={p.parent.name+'/'+p.name:sha(p) for p in parts},
                  imported_settings_sha256=sha(args.settings_file) if args.settings_file else None,
                  role_hashes={str(seed):sha(args.roles_dir/f'seed-{seed}.parquet') for seed in seeds},
                  candidate_grid=PRESETS,parameters_selected_using_test=False,
                  selection='training_cross_validation' if args.mode=='tune' else 'imported_training_cross_validation' if imported else 'predeclared_preset_not_tuned',
                  full_data_final_refit=args.mode=='train',technical_verification=args.mode=='smoke' or args.technical_check,
                  class_weighting='N/(2*N_class), recomputed within every fit',threads=2,
                  ebm_final_fit='All supplied fitting rows; validation_size=0, outer_bags=1, fixed round budget',
                  dependencies={name:importlib.metadata.version(name) for name in ['catboost','interpret-core','numpy','pandas','scikit-learn','pyarrow','joblib']})
    write(run/'manifest.json',manifest)
    rows=[];selected_settings=[]
    for seed in seeds:
        roles=read(args.roles_dir/f'seed-{seed}.parquet');validate_roles(roles,train)
        indexed=train.set_index('id',drop=False)
        selection=indexed.loc[roles.loc[roles.training_role=='model_selection','id']]
        calibration=indexed.loc[roles.loc[roles.training_role=='calibration','id']]
        cv_by_id=roles.set_index('id').cv_fold
        scoring=test
        if args.mode=='smoke':
            # Fold 0 is an outer synthetic holdout, excluded from all calibration fitting.
            scoring=selection.loc[cv_by_id.loc[selection.id].to_numpy()==0]
            selection=selection.loc[cv_by_id.loc[selection.id].to_numpy()!=0]
            fitting=pd.concat([selection,calibration],ignore_index=True)
        else:fitting=train
        for model in models:
            directory=run/model/f'seed-{seed}';directory.mkdir(parents=True,exist_ok=False)
            state=dict(status='running',model=model,seed=seed,started_utc=now());write(directory/'state.json',state)
            print(f'START {model} seed={seed} mode={args.mode}',flush=True)
            try:
                if args.mode=='tune':
                    candidates=[]
                    for candidate,parameters in enumerate(PRESETS[model]):
                        fold_scores=[]
                        for fold in range(3):
                            fitting_fold=selection.loc[cv_by_id.loc[selection.id].to_numpy()!=fold]
                            scoring_fold=selection.loc[cv_by_id.loc[selection.id].to_numpy()==fold]
                            estimator,medians,_=fit(model,parameters,settings,seed,fitting_fold,args.technical_check)
                            probability=predict(estimator,medians,scoring_fold,settings)
                            _,policy=cutoff(scoring_fold.label,probability)
                            fold_scores.append(policy['selected_threshold_metrics']['customer_f0_5'])
                            del estimator;gc.collect()
                        candidates.append(dict(candidate=candidate,parameters=parameters,fold_f0_5=fold_scores,mean_f0_5=float(np.mean(fold_scores))))
                    best=max(range(len(candidates)),key=lambda i:candidates[i]['mean_f0_5'])
                    for i,candidate in enumerate(candidates):candidate['selected']=i==best
                    write(directory/'cross_validation.json',candidates)
                    selected_settings.append(dict(model=model,parameters=candidates[best]['parameters']))
                    state.update(status='completed',ended_utc=now());write(directory/'state.json',state)
                    continue
                parameters=next(m['parameters'] for m in imported['models'] if m['model']==model) if imported else PRESETS[model][0]
                if model=='P6' and imported is None:parameters=PRESETS[model][1]
                started=time.perf_counter()
                estimator,medians,calibration_weights=fit(model,parameters,settings,seed,selection,args.mode=='smoke')
                probability=predict(estimator,medians,calibration,settings)
                threshold,policy=cutoff(calibration.label,probability)
                calibration_seconds=time.perf_counter()-started
                save_frame(directory/'calibration_predictions.parquet',pd.DataFrame(dict(id=calibration.id.to_numpy(),label=calibration.label.to_numpy(),customer_probability=probability)))
                write(directory/'threshold_selection.json',policy)
                del estimator;gc.collect()
                started=time.perf_counter()
                estimator,medians,final_weights=fit(model,parameters,settings,seed,fitting,args.mode=='smoke')
                fit_seconds=time.perf_counter()-started
                probability=predict(estimator,medians,scoring,settings)
                measured=score(scoring.label,probability,threshold)
                precision,recall,_=precision_recall_curve(scoring.label,probability)
                payload=dict(model=estimator,medians=medians,numeric_features=settings['numeric_features'],categorical_features=settings['categorical_features'],threshold=threshold,duration_transform='log1p')
                joblib.dump(payload,win(directory/'model.joblib'))
                restored=joblib.load(win(directory/'model.joblib'))
                assert np.array_equal(predict(estimator,medians,scoring.head(25),settings),predict(restored['model'],restored['medians'],scoring.head(25),settings))
                save_frame(directory/'predictions.parquet',pd.DataFrame(dict(id=scoring.id.to_numpy(),label=scoring.label.to_numpy(),customer_probability=probability,prediction=(probability>threshold).astype(float))))
                save_frame(directory/'fitting_ids.parquet',fitting[['id','label']])
                save_frame(directory/'training_roles.parquet',roles)
                measured.update(model=model,model_family=FAMILIES[model],seed=seed,training_rows=len(fitting),
                                feature_count=39,model_selection_rows=len(selection),calibration_rows=len(calibration),
                                evaluation_scope='synthetic_training_holdout' if args.mode=='smoke' else 'fixed_test',
                                technical_verification=args.mode=='smoke',parameter_selection=manifest['selection'],parameters=parameters,
                                effective_parameters={k:v for k,v in estimator.get_params().items() if k in set(parameters)|{'outer_bags','validation_size','early_stopping_rounds','n_jobs','random_state','thread_count','random_seed','iterations','depth','max_rounds','interactions'}},
                                decision_threshold=threshold,area_under_pr=float(auc(recall,precision)),
                                pr_area_definition='Exact sklearn precision-recall curve trapezoidal area',
                                final_fit_seconds=fit_seconds,threshold_selection_seconds=calibration_seconds,
                                class_weighting='N/(2*N_class)',calibration_fit_weights=calibration_weights,final_fit_weights=final_weights,
                                model_reload_verified=True,test_used_for_selection=False)
                write(directory/'metrics.json',measured);rows.append(measured)
                state.update(status='completed',ended_utc=now());write(directory/'state.json',state)
                print(f"COMPLETE {model} seed={seed} F0.5={measured['customer_f0_5']:.6f}",flush=True)
            except Exception:
                state.update(status='failed',ended_utc=now());write(directory/'state.json',state);raise
    if args.mode=='tune':
        write(run/'selected_settings.json',dict(selected_by='training_cross_validation',numeric_features=settings['numeric_features'],categorical_features=settings['categorical_features'],models=selected_settings,selection_seed=seeds[0],source_run=run.name,technical_verification=args.technical_check))
    write(run/'results.json',rows)
    write(run/'COMPLETED.json',dict(completed_utc=now(),model_outcomes=len(rows),mode=args.mode))
    print('OUTPUT '+str(run),flush=True)


if __name__=='__main__':main()
