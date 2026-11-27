"""Independent saved-prediction checks for supplemental Spark and Python runs."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import precision_recall_curve

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'tools')]
from audit_improved_runs import audit_run, labels, read_table


def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def verify_predictions(path,record,expected):
    pred=read_table(path)
    assert dict(zip(pred['id'],pred['label']))==expected and len(pred['id'])==len(expected)
    y=np.asarray(pred['label']);p=np.asarray(pred['customer_probability'])
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    predicted=p>record['decision_threshold']
    assert np.array_equal(predicted,np.asarray(pred['prediction']))
    counts=dict(tp=int(np.sum(predicted&(y==1))),fp=int(np.sum(predicted&(y==0))),
                fn=int(np.sum(~predicted&(y==1))),tn=int(np.sum(~predicted&(y==0))))
    assert all(record[key]==value for key,value in counts.items())
    tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
    expected_f05=1.25*tp/(1.25*tp+fp+.25*fn) if tp+fp+fn else 0.
    assert abs(expected_f05-record['customer_f0_5'])<1e-12
    assert abs(record['accuracy']-(counts['tp']+counts['tn'])/len(expected))<1e-12
    return counts


def verify_cutoff(directory,record,training):
    cal=read_table(directory/'calibration_predictions.parquet')
    assert len(cal['id'])==len(set(cal['id']))==record['calibration_rows']
    assert set(cal['id'])<=training.keys()
    assert all(training[i]==y for i,y in zip(cal['id'],cal['label']))
    policy=load(directory/'threshold_selection.json')
    assert policy['test_used'] is False and policy['threshold']==record['decision_threshold']
    y=np.asarray(cal['label']);p=np.asarray(cal['customer_probability'])
    precision,recall,_=precision_recall_curve(y,p)
    den=.25*precision+recall
    scores=np.divide(1.25*precision*recall,den,out=np.zeros_like(den),where=den>0)
    pred=p>record['decision_threshold'];tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)));fn=int(np.sum(~pred&(y==1)))
    selected=1.25*tp/(1.25*tp+fp+.25*fn)
    assert abs(selected-float(scores.max()))<1e-10
    assert abs(selected-policy['selected_threshold_metrics']['customer_f0_5'])<1e-10
    return set(cal['id'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--spark-runs',type=Path,nargs='*',default=[])
    parser.add_argument('--python-runs',type=Path,nargs='*',default=[])
    parser.add_argument('--settings-file',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    active=args.data_dir/'enriched' if (args.data_dir/'enriched/train.parquet').exists() else args.data_dir
    train=labels(active/'train.parquet');test=labels(active/'test.parquet')
    assert train.keys().isdisjoint(test)
    checks=[];outcomes=[]
    for run in args.spark_runs:
        manifest=load(run/'manifest.json')
        settings=load(args.settings_file) if args.settings_file else None
        result=audit_run(run,train,test,settings)
        rows=load(run/'results.json')
        for row in rows:
            directory=run/row['model']/f"seed-{row['seed']}"/'fraction-10000'
            fitting=labels(run/'folds'/f"seed-{row['seed']}"/'fraction-10000')
            verify_cutoff(directory,row,fitting)
            if row['evaluation_scope']=='fixed_test':assert row['training_rows']==536406 and row['evaluation_rows']==133553
            assert row['model'] in ('P4','P5')
            assert ('none:' in row['class_weighting'])==(row['model']=='P4')
            if row['model']=='P5':assert row['classifier_parameters']['weightCol']=='class_weight'
            outcomes.append(dict(row,run_id=run.name))
        result.update(calibration_curve_independently_verified=True,technical_verification=manifest['technical_verification'])
        checks.append(result)
    for run in args.python_runs:
        manifest=load(run/'manifest.json')
        assert hashlib.sha256((HERE/'run_python_models.py').read_bytes()).hexdigest()==manifest['source_sha256']
        assert (run/'COMPLETED.json').exists()
        for row in load(run/'results.json'):
            directory=run/row['model']/f"seed-{row['seed']}"
            fitting=labels(directory/'fitting_ids.parquet')
            assert fitting.keys()<=train.keys() and all(train[i]==y for i,y in fitting.items())
            if row['evaluation_scope']=='fixed_test':
                assert fitting==train and len(train)==536406 and len(test)==133553
                expected=test
            else:
                roles=read_table(directory/'training_roles.parquet')
                holdout={i for i,role,fold in zip(roles['id'],roles['training_role'],roles['cv_fold']) if role=='model_selection' and fold==0}
                expected={i:train[i] for i in holdout}
            assert fitting.keys().isdisjoint(expected)
            verify_predictions(directory/'predictions.parquet',row,expected)
            calibration=verify_cutoff(directory,row,fitting)
            roles=read_table(directory/'training_roles.parquet')
            expected_cal={i for i,role in zip(roles['id'],roles['training_role']) if role=='calibration'}
            assert calibration==expected_cal
            assert row['model_reload_verified'] is True and row['test_used_for_selection'] is False
            assert row['training_rows']==len(fitting) and row['evaluation_rows']==len(expected)
            for name,members in [('final_fit_weights',fitting),('calibration_fit_weights',{i:y for i,y in fitting.items() if i not in calibration})]:
                customers=sum(members.values());subscribers=len(members)-customers
                assert abs(row[name]['Customer']-len(members)/(2*customers))<1e-12
                assert abs(row[name]['Subscriber']-len(members)/(2*subscribers))<1e-12
            if row['model']=='P6':
                assert row['effective_parameters']['validation_size']==0 and row['effective_parameters']['outer_bags']==1
            checks.append(dict(run=run.name,model=row['model'],seed=row['seed'],
                               all_predictions_roles_and_cutoff_verified=True,weights_verified=True,
                               technical_verification=manifest['technical_verification']))
            outcomes.append(dict(row,run_id=run.name))
    assert checks,'Provide at least one saved run.'
    report=dict(passed=True,audited_utc=datetime.now(timezone.utc).isoformat(),model_records=len(outcomes),checks=checks,results=outcomes,
                scope='Saved row-level membership, predictions, calibration optima, weights and persistence attestations; synthetic checks are not project performance results')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,model_records=len(outcomes))))


if __name__=='__main__':main()
