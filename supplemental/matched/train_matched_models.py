"""Resumable, training-only tuning and cross-fitted stacking on the full split."""
import argparse
import atexit
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run_python_models as common
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, auc

BASES=['P1','P6','P7','P8','P9']
STACK_BASES=['P1','P6','P7']
META_CS=[0.001,0.1,10.0]
GRIDS={
 'P1':[dict(maxDepth=7,maxIter=240,stepSize=.03,minInstancesPerNode=40),
       dict(maxDepth=5,maxIter=160,stepSize=.05,minInstancesPerNode=20),
       dict(maxDepth=7,maxIter=160,stepSize=.05,minInstancesPerNode=40)],
 'P6':[dict(interactions=8,max_rounds=500,learning_rate=.03,max_bins=128,min_samples_leaf=20),
       dict(interactions=16,max_rounds=800,learning_rate=.03,max_bins=128,min_samples_leaf=20),
       dict(interactions=8,max_rounds=1000,learning_rate=.02,max_bins=256,min_samples_leaf=40)],
 'P7':[dict(iterations=400,depth=6,learning_rate=.05,l2_leaf_reg=3.,bootstrap_type='MVS',subsample=.8),
       dict(iterations=800,depth=6,learning_rate=.03,l2_leaf_reg=3.,bootstrap_type='No'),
       dict(iterations=800,depth=8,learning_rate=.03,l2_leaf_reg=5.,bootstrap_type='No')],
 'P8':[dict(n_estimators=400,num_leaves=31,learning_rate=.05,min_child_samples=20,reg_lambda=1.,max_bin=127),
       dict(n_estimators=800,num_leaves=31,learning_rate=.03,min_child_samples=40,reg_lambda=5.,max_bin=255),
       dict(n_estimators=600,num_leaves=63,learning_rate=.03,min_child_samples=40,reg_lambda=5.,max_bin=255)],
 'P9':[dict(n_estimators=400,max_depth=3,learning_rate=.05,min_child_weight=1,reg_lambda=1.),
       dict(n_estimators=600,max_depth=5,learning_rate=.03,min_child_weight=5,reg_lambda=5.),
       dict(n_estimators=600,max_depth=7,learning_rate=.03,min_child_weight=10,reg_lambda=5.)]}

def now():return datetime.now(timezone.utc).isoformat()
def load(path):return common.load(path)
def write(path,value):return common.write(path,value)
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def save(path,frame):return common.save_frame(path,frame)
def read(path,columns=None):return common.read(path,columns)


def make_roles(train,source):
    """B: base search/OOF; H: separate stack selection; C: cutoff calibration."""
    common.validate_roles(source,train)
    selected=source.loc[source.training_role=='model_selection','id'].to_numpy()
    indexed=train.set_index('id')
    base_ids,held_ids=train_test_split(selected,test_size=.20,random_state=20261126,
                                     stratify=indexed.loc[selected,'label'])
    roles=pd.DataFrame({'id':train.id.to_numpy(),'phase':'calibration','cv_fold':np.nan})
    roles.loc[roles.id.isin(base_ids),'phase']='cv'
    roles.loc[roles.id.isin(held_ids),'phase']='stack_validation'
    y=indexed.loc[base_ids,'label'].to_numpy()
    splitter=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    for fold,(_,validation) in enumerate(splitter.split(base_ids,y)):
        roles.loc[roles.id.isin(base_ids[validation]),'cv_fold']=fold
    validate_phases(roles,train)
    return roles


def validate_phases(roles,train):
    assert roles.id.is_unique and set(roles.id)==set(train.id)
    assert set(roles.phase)=={'cv','stack_validation','calibration'}
    assert set(roles.loc[roles.phase=='cv','cv_fold'])=={0,1,2}
    assert roles.loc[roles.phase!='cv','cv_fold'].isna().all()
    for fold in (0,1,2):
        fitted=set(roles.loc[(roles.phase=='cv')&(roles.cv_fold!=fold),'id'])
        evaluated=set(roles.loc[(roles.phase=='cv')&(roles.cv_fold==fold),'id'])
        reserved=set(roles.loc[roles.phase!='cv','id'])
        assert fitted.isdisjoint(evaluated|reserved) and evaluated.isdisjoint(reserved)


def meta_matrix(frame):
    assert np.isfinite(frame.to_numpy()).all()
    p=np.clip(frame.to_numpy(dtype=float),1e-6,1-1e-6)
    return np.log(p)-np.log1p(-p)


def fit_meta(frame,labels,c):
    estimator=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(C=c,max_iter=300,tol=1e-8,solver='lbfgs'))])
    weights,_=common.weights(labels)
    estimator.fit(meta_matrix(frame),np.asarray(labels,dtype=int),classifier__sample_weight=weights)
    return estimator


def fit_python(model,parameters,frame,settings,seed,threads):
    x,medians=common.transform(frame,settings)
    payload=dict(model_id=model,medians=medians,settings=settings)
    values,weight_map=common.weights(frame.label)
    if model=='P6':
        from interpret.glassbox import ExplainableBoostingClassifier
        estimator=ExplainableBoostingClassifier(**parameters,
            feature_types=['continuous']*len(settings['numeric_features'])+['nominal']*len(settings['categorical_features']),
            validation_size=0,outer_bags=1,early_stopping_rounds=0,max_interaction_bins=16,n_jobs=1,random_state=seed)
    elif model=='P7':
        from catboost import CatBoostClassifier
        estimator=CatBoostClassifier(**parameters,cat_features=settings['categorical_features'],loss_function='Logloss',
            random_seed=seed,thread_count=threads,task_type='CPU',verbose=False,allow_writing_files=False)
    elif model=='P8':
        from lightgbm import LGBMClassifier
        categories={column:sorted(x[column].unique().tolist()) for column in settings['categorical_features']}
        for column,vocabulary in categories.items():x[column]=pd.Categorical(x[column],categories=vocabulary)
        payload['categories']=categories
        estimator=LGBMClassifier(**parameters,objective='binary',n_jobs=threads,random_state=seed,
            subsample=1.,subsample_freq=0,colsample_bytree=1.,verbosity=-1,deterministic=True,force_col_wise=True)
    elif model=='P9':
        from xgboost import XGBClassifier
        encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32)
        scaler=MinMaxScaler()
        numeric=scaler.fit_transform(x[settings['numeric_features']].to_numpy(dtype=np.float32))
        categorical=encoder.fit_transform(x[settings['categorical_features']])
        x=sparse.hstack([sparse.csr_matrix(numeric),categorical],format='csr')
        payload.update(encoder=encoder,scaler=scaler)
        estimator=XGBClassifier(**parameters,objective='binary:logistic',tree_method='hist',max_bin=256,
            subsample=1.,colsample_bytree=1.,scale_pos_weight=1.,random_state=seed,n_jobs=threads,device='cpu',verbosity=0)
    else:raise ValueError(model)
    started=time.perf_counter();estimator.fit(x,frame.label.astype(int),sample_weight=values)
    payload.update(model=estimator,class_weights=weight_map,fit_seconds=time.perf_counter()-started)
    return payload


def predict_python(payload,frame):
    settings=payload['settings'];x,_=common.transform(frame,settings,payload['medians'])
    if payload['model_id']=='P8':
        for column,vocabulary in payload['categories'].items():x[column]=pd.Categorical(x[column],categories=vocabulary)
    elif payload['model_id']=='P9':
        numeric=payload['scaler'].transform(x[settings['numeric_features']].to_numpy(dtype=np.float32))
        categorical=payload['encoder'].transform(x[settings['categorical_features']])
        x=sparse.hstack([sparse.csr_matrix(numeric),categorical],format='csr')
    assert np.array_equal(payload['model'].classes_,[0,1])
    return payload['model'].predict_proba(x)[:,1].astype(float)


def prediction_frame(frame,probability,threshold=.5):
    return pd.DataFrame(dict(id=frame.id.to_numpy(),label=frame.label.to_numpy(),
        customer_probability=probability,prediction=(np.asarray(probability)>threshold).astype(float)))


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    handle=path.open('a+b');handle.seek(0)
    if path.stat().st_size==0:handle.write(b'0');handle.flush()
    handle.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:yield
    finally:handle.close()


def technical_grids():
    grids=json.loads(json.dumps(GRIDS))
    for i in range(3):
        grids['P1'][i].update(maxDepth=2+i//2,maxIter=3+i*2,minInstancesPerNode=2)
        grids['P6'][i].update(interactions=i,max_rounds=15+i*5,max_bins=32,min_samples_leaf=4)
        grids['P7'][i].update(iterations=15+i*5,depth=2+i//2)
        grids['P8'][i].update(n_estimators=15+i*5,num_leaves=7+i*4,min_child_samples=4)
        grids['P9'][i].update(n_estimators=15+i*5,max_depth=2+i//2)
    return grids


class Study:
    def __init__(self,args):
        self.args=args;self.root=args.output_dir.resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.settings=load(args.features_file);self.features=self.settings['numeric_features']+self.settings['categorical_features']
        assert len(self.features)==len(set(self.features))==39 and not set(self.features)&common.BLOCKED
        self.active=(args.data_dir/'enriched' if (args.data_dir/'enriched/train.parquet').exists() else args.data_dir).resolve()
        self.train=read(self.active/'train.parquet',['id','label']+self.features)
        assert self.train.id.is_unique and set(self.train.label)=={0.,1.}
        if not args.technical_check:assert len(self.train)==536406 and int(self.train.label.sum())==82490
        self.index=self.train.set_index('id',drop=False)
        self.grids=technical_grids() if args.technical_check else GRIDS
        self.seeds=[42] if args.technical_check else [42,43,44]
        roles_path=self.root/'roles.parquet'
        if roles_path.exists():self.roles=read(roles_path);validate_phases(self.roles,self.train)
        else:
            self.roles=make_roles(self.train,read(args.roles_file));save(roles_path,self.roles)
        self.B=self.index.loc[self.roles.loc[self.roles.phase=='cv','id']]
        self.H=self.index.loc[self.roles.loc[self.roles.phase=='stack_validation','id']]
        self.C=self.index.loc[self.roles.loc[self.roles.phase=='calibration','id']]
        self.S=self.index.loc[self.roles.loc[self.roles.phase!='calibration','id']]
        self.fold_by_id=self.roles.set_index('id').cv_fold
        files={}
        for split in ('train','test'):
            folder=self.active/(split+'.parquet')
            for file in sorted(folder.glob('*.parquet')) if folder.is_dir() else [folder]:files[split+'/'+file.name]=sha(file)
        self.source_files=[Path(__file__),HERE/'TrainMatchedGBT.scala',HERE.parent/'run_python_models.py']
        protocol=dict(version=1,technical_verification=args.technical_check,feature_count=39,
            numeric_features=self.settings['numeric_features'],categorical_features=self.settings['categorical_features'],
            training_rows=len(self.train),cv_rows=len(self.B),stack_validation_rows=len(self.H),calibration_rows=len(self.C),
            input_hashes=files,roles_sha256=sha(roles_path),grids=self.grids,meta_cs=META_CS,
            source_calibration_roles_sha256=sha(args.roles_file),feature_settings_sha256=sha(args.features_file),
            seeds=self.seeds,base_models=BASES,stack_bases=STACK_BASES,
            source_sha256={p.name:sha(p) for p in self.source_files},
            dependencies={n:importlib.metadata.version(n) for n in ['numpy','pandas','pyarrow','scikit-learn','interpret-core','catboost','lightgbm','xgboost','joblib']},
            threads=args.threads,spark_master=f'local[{args.threads}]',spark_heap='4g',ebm_jobs=1,
            selection_policy='Each base: 3 fixed configurations x3 CV folds on B; stack C on disjoint H; cutoff calibration only on C',
            stacking_policy='OOF_B plus base-B predictions for H; combiner selection on H; refit combiner on S, calibrate on C, then refit combiner on all training rows with out-of-fit features',
            seed_policy='Fixed seed42 calibration split for every model/seed. Seeds42/43/44 vary estimator initialization only; shared seed42 OOF_B/H construction transferred unchanged.',
            final_fit_policy='Each base and final stack combiner fit all original training rows; no external training downsampling',
            test_use='Previously examined benchmark. No test features opened until FROZEN_SETTINGS.json exists; no test-based parameter/cutoff/ensemble selection')
        existing=self.root/'protocol.json'
        if existing.exists():assert load(existing)==protocol,'Frozen study inputs/code changed; use a new output directory.'
        else:write(existing,protocol)
        self.protocol=protocol

    def integrity(self):
        assert {p.name:sha(p) for p in self.source_files}==self.protocol['source_sha256'],'Study source changed during execution.'

    def input_integrity(self):
        for relative,fingerprint in self.protocol['input_hashes'].items():
            split,name=relative.split('/',1);folder=self.active/(split+'.parquet')
            assert sha(folder/name if folder.is_dir() else folder)==fingerprint,'Input changed after study freeze.'

    def event(self,phase,**details):
        value=dict(updated_utc=now(),phase=phase,**details)
        write(self.root/'STATUS.json',value)
        print(json.dumps(value),flush=True)

    def spark(self,stage,parameters,seed=42,threshold=.5):
        self.integrity()
        self.input_integrity()
        out=self.root/'P1'/('cv' if stage=='cv' else f'{stage}-{seed}')
        candidates=self.grids['P1'] if stage=='cv' else [parameters]
        final_marker=out/f'candidate-{len(candidates)-1}'/('fold-2' if stage=='cv' else 'fold--1')/'DONE.json'
        if not final_marker.exists():
            config=dict(data_dir=str(self.active),roles_file=str((self.root/'roles.parquet').resolve()),
                output_dir=str(out.resolve()),stage=stage,seed=seed,candidates=candidates,
                threshold=threshold,technical_verification=self.args.technical_check,
                settings_frozen=(self.root/'FROZEN_SETTINGS.json').exists(),
                numeric_features=self.settings['numeric_features'],categorical_features=self.settings['categorical_features'])
            config_path=self.root/'jobs'/f'P1-{stage}-{seed}.json';write(config_path,config)
            log=self.root/'logs'/f'P1-{stage}-{seed}-{time.time_ns()}.log';log.parent.mkdir(parents=True,exist_ok=True)
            command=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(HERE.parent.parent/'run-matched-gbt.ps1'),
                '-ConfigFile',str(config_path.resolve()),'-SparkHome',str(self.args.spark_home.resolve()),
                '-JavaHome',str(self.args.java_home.resolve()),'-HadoopHome',str(self.args.hadoop_home.resolve()),
                '-Master',f'local[{self.args.threads}]']
            env=dict(os.environ,SPARK_LOCAL_IP='127.0.0.1')
            self.args.temp_dir.mkdir(parents=True,exist_ok=True);env['TEMP']=env['TMP']=str(self.args.temp_dir.resolve())
            self.event('spark_'+stage,model='P1',seed=seed)
            with log.open('w',encoding='utf-8') as handle:
                process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                last_line=''
                while process.poll() is None:
                    time.sleep(5)
                    try:
                        lines=log.read_text(encoding='utf-8',errors='replace').splitlines()
                        progress=[line for line in lines if line.startswith(('START P1','COMPLETE P1'))]
                        if progress and progress[-1]!=last_line:
                            last_line=progress[-1];print(last_line,flush=True)
                        write(self.root/'HEARTBEAT.json',dict(updated_utc=now(),phase='spark_'+stage,last_progress=last_line))
                    except OSError:pass
                assert process.returncode==0,f'Spark failed; inspect {log.name}'
        if stage=='cv':
            return [[load(out/f'candidate-{i}'/f'fold-{fold}'/'DONE.json') for fold in range(3)] for i in range(3)]
        return load(out/'candidate-0/fold--1/DONE.json')

    def py_case(self,model,parameters,fitting,scoring,directory,seed=42,save_model=False,threshold=.5):
        self.integrity();marker=directory/'DONE.json'
        if marker.exists():return load(marker)
        self.event('python_fit',model=model,seed=seed,case=directory.relative_to(self.root).as_posix(),fitting_rows=len(fitting),scoring_rows=len(scoring))
        payload=fit_python(model,parameters,fitting,self.settings,seed,self.args.threads)
        probability=predict_python(payload,scoring)
        save(directory/'fitting_ids.parquet',fitting[['id','label']])
        save(directory/'predictions.parquet',prediction_frame(scoring,probability,threshold))
        reload_verified=False
        if save_model:
            joblib.dump(payload,common.win(directory/'model.joblib'))
            restored=joblib.load(common.win(directory/'model.joblib'))
            assert np.array_equal(predict_python(payload,scoring.head(25)),predict_python(restored,scoring.head(25)))
            reload_verified=True
        actual=payload['model'].get_all_params() if model=='P7' else payload['model'].get_params()
        actual={key:actual.get(key) for key in set(parameters)|{'bootstrap_type','subsample','validation_size','outer_bags','n_jobs','thread_count'}}
        record=dict(model=model,seed=seed,training_rows=len(fitting),evaluation_rows=len(scoring),
                    fit_seconds=payload['fit_seconds'],parameters=parameters,effective_parameters=actual,
                    class_weights=payload['class_weights'],predictions=str((directory/'predictions.parquet').resolve()),
                    fitting_ids=str((directory/'fitting_ids.parquet').resolve()),
                    model_reload_verified=reload_verified,threshold=threshold)
        write(marker,record);del payload;gc.collect()
        return record

    def tune(self):
        frozen=self.root/'FROZEN_SETTINGS.json'
        if frozen.exists():return load(frozen)
        selected={};oof={};held={};cv_records=[]
        # The faster Python families run first; the Spark search is checkpointed separately.
        for model in ['P6','P7','P8','P9','P1']:
            self.event('cross_validation',model=model)
            spark_records=self.spark('cv',None) if model=='P1' else None
            candidates=[]
            for candidate,parameters in enumerate(self.grids[model]):
                fold_scores=[];pieces=[]
                for fold in range(3):
                    fitting=self.B.loc[self.fold_by_id.loc[self.B.id].to_numpy()!=fold]
                    scoring=self.B.loc[self.fold_by_id.loc[self.B.id].to_numpy()==fold]
                    if model=='P1':record=spark_records[candidate][fold]
                    else:record=self.py_case(model,parameters,fitting,scoring,self.root/model/'cv'/f'candidate-{candidate}'/f'fold-{fold}')
                    pred=read(Path(record['predictions']))
                    assert len(pred)==len(scoring) and set(pred.id)==set(scoring.id)
                    threshold,policy=common.cutoff(pred.label,pred.customer_probability)
                    fold_scores.append(policy['selected_threshold_metrics']['customer_f0_5']);pieces.append(pred)
                candidates.append(dict(candidate=candidate,parameters=parameters,fold_f0_5=fold_scores,mean_f0_5=float(np.mean(fold_scores))))
                save(self.root/model/'cv'/f'oof-candidate-{candidate}.parquet',pd.concat(pieces,ignore_index=True))
            best=max(range(3),key=lambda i:candidates[i]['mean_f0_5'])
            for i,candidate in enumerate(candidates):candidate['selected']=i==best
            write(self.root/model/'cross_validation.json',candidates)
            selected[model]=self.grids[model][best];cv_records.append(dict(model=model,candidates=candidates))
            oof[model]=read(self.root/model/'cv'/f'oof-candidate-{best}.parquet').set_index('id').customer_probability
            if model in STACK_BASES:
                record=self.spark('stack_validation',selected[model]) if model=='P1' else self.py_case(model,selected[model],self.B,self.H,self.root/model/'stack-validation')
                pred=read(Path(record['predictions']));assert set(pred.id)==set(self.H.id)
                held[model]=pred.set_index('id').customer_probability
        b_features=pd.DataFrame({m:oof[m].loc[self.B.id].to_numpy() for m in STACK_BASES},index=self.B.id)
        h_features=pd.DataFrame({m:held[m].loc[self.H.id].to_numpy() for m in STACK_BASES},index=self.H.id)
        stack_candidates=[]
        for c in META_CS:
            estimator=fit_meta(b_features,self.B.label,c)
            probability=estimator.predict_proba(meta_matrix(h_features))[:,1]
            _,policy=common.cutoff(self.H.label,probability)
            stack_candidates.append(dict(C=c,selection_f0_5=policy['selected_threshold_metrics']['customer_f0_5']))
        best=max(range(len(META_CS)),key=lambda i:stack_candidates[i]['selection_f0_5'])
        for i,candidate in enumerate(stack_candidates):candidate['selected']=i==best
        write(self.root/'P10/selection.json',stack_candidates)
        meta_features=pd.concat([b_features,h_features]).loc[self.S.id]
        assert meta_features.index.is_unique and set(meta_features.index)==set(self.S.id)
        save(self.root/'P10/meta_training_features.parquet',meta_features.reset_index())
        frozen_value=dict(frozen_utc=now(),selected_by='training_only_shared_cv_and_disjoint_stack_validation',
            technical_verification=self.args.technical_check,models=selected,stack_C=META_CS[best],stack_bases=STACK_BASES,
            protocol_sha256=sha(self.root/'protocol.json'),cv_records=cv_records,
            no_test_features_scored=True,all_base_selections_exclude_stack_validation_and_calibration=True)
        write(frozen,frozen_value)
        self.event('settings_frozen',models=BASES+['P10'])
        return frozen_value

    def evaluate(self,frozen):
        assert (self.root/'FROZEN_SETTINGS.json').exists()
        assert frozen['protocol_sha256']==sha(self.root/'protocol.json')
        self.integrity()
        self.input_integrity()
        test=read(self.active/'test.parquet',['id','label']+self.features)
        assert test.id.is_unique and set(test.id).isdisjoint(self.train.id) and set(test.label)=={0.,1.}
        if not self.args.technical_check:assert len(test)==133553 and int(test.label.sum())==20723
        meta_data=read(self.root/'P10/meta_training_features.parquet').set_index('id').loc[self.S.id,STACK_BASES]
        meta_cal=fit_meta(meta_data,self.S.label,frozen['stack_C'])
        rows=[]
        for seed in self.seeds:
            cal_features={};test_features={}
            for model in BASES:
                parameters=frozen['models'][model]
                cal_record=self.spark('calibration',parameters,seed) if model=='P1' else self.py_case(model,parameters,self.S,self.C,self.root/model/f'calibration-{seed}',seed)
                cal=read(Path(cal_record['predictions'])).set_index('id').loc[self.C.id]
                threshold,policy=common.cutoff(cal.label,cal.customer_probability)
                write(self.root/model/f'seed-{seed}/threshold_selection.json',policy)
                record=self.spark('final',parameters,seed,threshold) if model=='P1' else self.py_case(model,parameters,self.train,test,self.root/model/f'final-{seed}',seed,True,threshold)
                pred=read(Path(record['predictions'])).set_index('id').loc[test.id]
                measured=common.score(pred.label,pred.customer_probability,threshold)
                precision,recall,_=precision_recall_curve(pred.label,pred.customer_probability)
                measured.update(model=model,seed=seed,training_rows=len(self.train),feature_count=39,
                    calibration_rows=len(self.C),decision_threshold=threshold,parameters=parameters,
                    final_fit_seconds=record['fit_seconds'],calibration_fit_seconds=cal_record['fit_seconds'],
                    area_under_pr=float(auc(recall,precision)),model_reload_verified=record['model_reload_verified'],
                    evaluation_scope='synthetic_holdout' if self.args.technical_check else 'fixed_test',
                    technical_verification=self.args.technical_check,class_weighting='N/(2*N_class)',
                    training_pool_downsampled=False,calibration_split_seed=42,initialization_seed=seed,
                    predictions=record['predictions'],calibration_predictions=cal_record['predictions'])
                write(self.root/model/f'seed-{seed}/metrics.json',measured);rows.append(measured)
                if model in STACK_BASES:
                    cal_features[model]=cal.customer_probability.to_numpy()
                    test_features[model]=pred.customer_probability.to_numpy()
                self.event('final_model_completed',model=model,seed=seed,f0_5=measured['customer_f0_5'])
            directory=self.root/'P10'/f'seed-{seed}'
            if (directory/'metrics.json').exists():
                rows.append(load(directory/'metrics.json'))
                continue
            cal_matrix=pd.DataFrame(cal_features,columns=STACK_BASES,index=self.C.id)
            calibration_probability=meta_cal.predict_proba(meta_matrix(cal_matrix))[:,1]
            threshold,policy=common.cutoff(self.C.label,calibration_probability)
            write(directory/'threshold_selection.json',policy)
            final_meta_features=pd.concat([meta_data,cal_matrix]).loc[self.train.id]
            assert final_meta_features.index.is_unique and len(final_meta_features)==len(self.train)
            estimator=fit_meta(final_meta_features,self.train.label,frozen['stack_C'])
            probability=estimator.predict_proba(meta_matrix(pd.DataFrame(test_features,columns=STACK_BASES)))[:,1]
            payload=dict(model=estimator,base_models=STACK_BASES,transformation='clip[1e-6,1-1e-6] then logit',threshold=threshold)
            directory.mkdir(parents=True,exist_ok=True);joblib.dump(payload,common.win(directory/'model.joblib'))
            restored=joblib.load(common.win(directory/'model.joblib'))
            probe=meta_matrix(pd.DataFrame(test_features,columns=STACK_BASES).head(25))
            assert np.array_equal(estimator.predict_proba(probe),restored['model'].predict_proba(probe))
            save(directory/'predictions.parquet',prediction_frame(test,probability,threshold))
            save(directory/'calibration_predictions.parquet',prediction_frame(self.C,calibration_probability))
            save(directory/'final_meta_training_features.parquet',final_meta_features.reset_index())
            measured=common.score(test.label,probability,threshold)
            measured.update(model='P10',seed=seed,training_rows=len(self.train),feature_count=3,base_input_feature_count=39,
                calibration_rows=len(self.C),decision_threshold=threshold,parameters=dict(C=frozen['stack_C']),
                evaluation_scope='synthetic_holdout' if self.args.technical_check else 'fixed_test',
                technical_verification=self.args.technical_check,model_reload_verified=True,
                class_weighting='N/(2*N_class)',training_pool_downsampled=False,calibration_split_seed=42,
                shared_seed42_oof_construction=True,predictions=str((directory/'predictions.parquet').resolve()),
                calibration_predictions=str((directory/'calibration_predictions.parquet').resolve()))
            write(directory/'metrics.json',measured);rows.append(measured)
            self.event('final_model_completed',model='P10',seed=seed,f0_5=measured['customer_f0_5'])
        write(self.root/'results.json',rows)
        write(self.root/'COMPLETED.json',dict(completed_utc=now(),model_outcomes=len(rows),independent_audit_pending=True))
        self.event('training_completed',model_outcomes=len(rows))
        return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','tune','evaluate'],default='all')
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--roles-file',type=Path,required=True)
    parser.add_argument('--features-file',type=Path,default=HERE.parent.parent/'validation/P1_P2_P3_with_paper_inputs.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--spark-home',type=Path,required=True)
    parser.add_argument('--java-home',type=Path,required=True)
    parser.add_argument('--hadoop-home',type=Path,required=True)
    parser.add_argument('--temp-dir',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--technical-check',action='store_true')
    args=parser.parse_args();assert 1<=args.threads<=8
    with exclusive_lock(args.output_dir/'.run.lock'):
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            atexit.register(ctypes.windll.kernel32.SetThreadExecutionState,0x80000000)
        study=Study(args)
        if args.stage=='evaluate':frozen=load(study.root/'FROZEN_SETTINGS.json')
        else:frozen=study.tune()
        if args.stage!='tune':study.evaluate(frozen)


if __name__=='__main__':main()
