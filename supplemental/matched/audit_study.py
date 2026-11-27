"""Independent boundary, CV, weighting, cutoff and cross-fitted stack verification."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import statistics
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train_matched_models as study


def labels(frame):
    result=dict(zip(frame.id,frame.label));assert len(result)==len(frame)
    return result


def independent_metrics(y,p,threshold):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    pred=p>threshold
    tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)))
    fn=int(np.sum(~pred&(y==1)));tn=int(np.sum(~pred&(y==0)))
    denominator=1.25*tp+fp+.25*fn
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,customer_f0_5=1.25*tp/denominator if denominator else 0.,
        customer_precision=tp/(tp+fp) if tp+fp else 0.,customer_recall=tp/(tp+fn) if tp+fn else 0.,
        accuracy=(tp+tn)/len(y))


def optimal_f05(y,p):
    precision,recall,_=precision_recall_curve(y,p)
    denominator=.25*precision+recall
    values=np.divide(1.25*precision*recall,denominator,out=np.zeros_like(denominator),where=denominator>0)
    return float(values.max())


def verify_record(record,fitting,evaluation):
    fitted=study.read(Path(record['fitting_ids']))
    predicted=study.read(Path(record['predictions']))
    assert labels(fitted)==labels(fitting) and labels(predicted)==labels(evaluation)
    assert set(fitted.id).isdisjoint(predicted.id)
    assert len(fitted)==record['training_rows'] and len(predicted)==record['evaluation_rows']
    customers=int(fitted.label.sum());subscribers=len(fitted)-customers
    assert abs(record['class_weights']['Customer']-len(fitted)/(2*customers))<1e-12
    assert abs(record['class_weights']['Subscriber']-len(fitted)/(2*subscribers))<1e-12
    return predicted.set_index('id').loc[evaluation.id]


def source_record(root,model,stage,seed=42,candidate=None,fold=None):
    if stage=='cv':
        return study.load(root/model/'cv'/f'candidate-{candidate}'/f'fold-{fold}'/'DONE.json')
    if model=='P1':return study.load(root/model/f'{stage}-{seed}'/'candidate-0/fold--1/DONE.json')
    suffix='stack-validation' if stage=='stack_validation' else f'{stage}-{seed}'
    return study.load(root/model/suffix/'DONE.json')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    args=parser.parse_args();root=args.run
    protocol=study.load(root/'protocol.json');frozen=study.load(root/'FROZEN_SETTINGS.json')
    assert frozen['protocol_sha256']==study.sha(root/'protocol.json')
    for name,fingerprint in protocol['source_sha256'].items():
        path=HERE/name if (HERE/name).exists() else HERE.parent/name
        assert study.sha(path)==fingerprint
    assert study.sha(root/'roles.parquet')==protocol['roles_sha256']
    active=args.data_dir/'enriched' if (args.data_dir/'enriched/train.parquet').exists() else args.data_dir
    for relative,fingerprint in protocol['input_hashes'].items():
        split,name=relative.split('/',1);folder=active/(split+'.parquet')
        assert study.sha(folder/name if folder.is_dir() else folder)==fingerprint
    train=study.read(active/'train.parquet',['id','label']);test=study.read(active/'test.parquet',['id','label'])
    roles=study.read(root/'roles.parquet');study.validate_phases(roles,train)
    index=train.set_index('id',drop=False);fold_by_id=roles.set_index('id').cv_fold
    B=index.loc[roles.loc[roles.phase=='cv','id']]
    H=index.loc[roles.loc[roles.phase=='stack_validation','id']]
    C=index.loc[roles.loc[roles.phase=='calibration','id']]
    S=index.loc[roles.loc[roles.phase!='calibration','id']]
    assert set(train.id).isdisjoint(test.id)
    cv_checks=[];selected_oof={};held={}
    for model in study.BASES:
        candidates=study.load(root/model/'cross_validation.json')
        assert len(candidates)==3
        for candidate,selection in enumerate(candidates):
            values=[];pieces=[]
            for fold in range(3):
                fitting=B.loc[fold_by_id.loc[B.id].to_numpy()!=fold]
                evaluation=B.loc[fold_by_id.loc[B.id].to_numpy()==fold]
                record=source_record(root,model,'cv',candidate=candidate,fold=fold)
                predicted=verify_record(record,fitting,evaluation)
                actual=optimal_f05(predicted.label,predicted.customer_probability)
                assert abs(actual-selection['fold_f0_5'][fold])<1e-10
                values.append(actual);pieces.append(predicted.reset_index())
            assert abs(np.mean(values)-selection['mean_f0_5'])<1e-10
            saved=study.read(root/model/'cv'/f'oof-candidate-{candidate}.parquet').set_index('id')
            combined=pd.concat(pieces).set_index('id')
            assert saved.index.is_unique and set(saved.index)==set(B.id)
            assert np.array_equal(saved.loc[B.id,'customer_probability'],combined.loc[B.id,'customer_probability'])
        chosen=[r for r in candidates if r['selected']];assert len(chosen)==1
        assert chosen[0]['mean_f0_5']==max(r['mean_f0_5'] for r in candidates)
        assert chosen[0]['parameters']==frozen['models'][model]
        cv_checks.append(dict(model=model,candidate_records=3,fold_records=9,all_fitting_ids_and_weights_verified=True))
        if model in study.STACK_BASES:
            selected_oof[model]=study.read(root/model/'cv'/f"oof-candidate-{chosen[0]['candidate']}.parquet").set_index('id').customer_probability
            record=source_record(root,model,'stack_validation')
            held[model]=verify_record(record,B,H).customer_probability
    meta=study.read(root/'P10/meta_training_features.parquet').set_index('id').loc[S.id,study.STACK_BASES]
    assert len(meta)==len(S) and meta.index.is_unique
    for model in study.STACK_BASES:
        assert np.array_equal(meta.loc[B.id,model],selected_oof[model].loc[B.id])
        assert np.array_equal(meta.loc[H.id,model],held[model].loc[H.id])
    choices=study.load(root/'P10/selection.json')
    for row in choices:
        estimator=study.fit_meta(meta.loc[B.id],B.label,row['C'])
        predicted=estimator.predict_proba(study.meta_matrix(meta.loc[H.id]))[:,1]
        assert abs(optimal_f05(H.label,predicted)-row['selection_f0_5'])<1e-10
    chosen=[r for r in choices if r['selected']];assert len(chosen)==1
    assert chosen[0]['C']==frozen['stack_C'] and chosen[0]['selection_f0_5']==max(r['selection_f0_5'] for r in choices)
    meta_cal=study.fit_meta(meta,S.label,frozen['stack_C'])
    outcomes=study.load(root/'results.json');expected={(m,s) for m in study.BASES+['P10'] for s in protocol['seeds']}
    assert {(r['model'],r['seed']) for r in outcomes}==expected and len(outcomes)==len(expected)
    final_checks=[]
    for seed in protocol['seeds']:
        cal_features={};test_features={}
        for model in study.BASES:
            row=next(r for r in outcomes if (r['model'],r['seed'])==(model,seed))
            cal=verify_record(source_record(root,model,'calibration',seed),S,C)
            pred=verify_record(source_record(root,model,'final',seed),train,test)
            actual=independent_metrics(pred.label,pred.customer_probability,row['decision_threshold'])
            for key,value in actual.items():assert abs(row[key]-value)<1e-10
            assert np.array_equal(pred.prediction,pred.customer_probability>row['decision_threshold'])
            assert row['training_rows']==len(train) and row['model_reload_verified'] is True
            policy=study.load(root/model/f'seed-{seed}/threshold_selection.json')
            calibration=independent_metrics(cal.label,cal.customer_probability,row['decision_threshold'])
            assert policy['test_used'] is False and policy['threshold']==row['decision_threshold']
            assert abs(calibration['customer_f0_5']-optimal_f05(cal.label,cal.customer_probability))<1e-10
            if model in study.STACK_BASES:
                cal_features[model]=cal.customer_probability.to_numpy()
                test_features[model]=pred.customer_probability.to_numpy()
            final_checks.append(dict(model=model,seed=seed,full_split_ids_labels_weights_and_cutoffs_verified=True))
        row=next(r for r in outcomes if (r['model'],r['seed'])==('P10',seed))
        directory=root/'P10'/f'seed-{seed}'
        cal_matrix=pd.DataFrame(cal_features,columns=study.STACK_BASES,index=C.id)
        expected_cal=meta_cal.predict_proba(study.meta_matrix(cal_matrix))[:,1]
        saved_cal=study.read(directory/'calibration_predictions.parquet').set_index('id').loc[C.id]
        assert np.allclose(expected_cal,saved_cal.customer_probability,rtol=0,atol=1e-12)
        assert abs(independent_metrics(C.label,expected_cal,row['decision_threshold'])['customer_f0_5']-optimal_f05(C.label,expected_cal))<1e-10
        final_matrix=study.read(directory/'final_meta_training_features.parquet').set_index('id').loc[train.id,study.STACK_BASES]
        assert final_matrix.index.is_unique and len(final_matrix)==len(train)
        assert np.array_equal(final_matrix.loc[S.id],meta.loc[S.id])
        assert np.array_equal(final_matrix.loc[C.id],cal_matrix.loc[C.id])
        estimator=study.fit_meta(final_matrix,train.label,frozen['stack_C'])
        saved=joblib.load(study.common.win(directory/'model.joblib'))
        matrix=study.meta_matrix(pd.DataFrame(test_features,columns=study.STACK_BASES))
        probability=estimator.predict_proba(matrix)[:,1]
        assert np.allclose(probability,saved['model'].predict_proba(matrix)[:,1],rtol=0,atol=1e-12)
        pred=study.read(directory/'predictions.parquet').set_index('id').loc[test.id]
        assert labels(pred.reset_index())==labels(test)
        assert np.allclose(probability,pred.customer_probability,rtol=0,atol=1e-12)
        assert np.array_equal(pred.prediction,probability>row['decision_threshold'])
        for key,value in independent_metrics(test.label,probability,row['decision_threshold']).items():assert abs(row[key]-value)<1e-10
        final_checks.append(dict(model='P10',seed=seed,meta_selection_holdout_excluded_from_base_cv=True,
            all_meta_features_out_of_fit=True,combiner_refit_all_training_rows=True,all_predictions_reproduced=True))
    report=dict(passed=True,audited_utc=datetime.now(timezone.utc).isoformat(),model_records=len(outcomes),
        technical_verification=protocol['technical_verification'],cv_checks=cv_checks,final_checks=final_checks,
        input_and_source_hashes_unchanged=True,stacking_boundaries_and_all_meta_predictions_verified=True,
        settings_frozen_before_test_prediction=True)
    study.write(root/'independent_audit.json',report)
    complete=study.load(root/'COMPLETED.json');complete.update(independent_audit_pending=False,audited_utc=report['audited_utc'])
    study.write(root/'COMPLETED.json',complete)
    summaries=[]
    for model in study.BASES+['P10']:
        selected=[r for r in outcomes if r['model']==model]
        summaries.append(dict(model=model,seeds=protocol['seeds'],mean_f0_5=statistics.mean(r['customer_f0_5'] for r in selected),
            sample_sd=statistics.stdev(r['customer_f0_5'] for r in selected) if len(selected)>1 else None,
            mean_precision=statistics.mean(r['customer_precision'] for r in selected),mean_recall=statistics.mean(r['customer_recall'] for r in selected)))
    study.write(root/'summary.json',summaries)
    print(json.dumps(dict(passed=True,model_records=len(outcomes),base_cv_fold_fits_verified=45,stack_reconstructed=True)))


if __name__=='__main__':main()
