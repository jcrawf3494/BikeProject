"""Independent row-level checks and aggregate reporting for the frozen batch."""
import csv
import argparse
from datetime import date
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

WORK = Path(__file__).resolve().parent
sys.path[:0] = [str(WORK / "baseline-runtime"), str(WORK / "vendor")]
ROOT = WORK.parent
PACKAGE = ROOT / "outputs" / "BikeShareModelTraining"
BATCH = PACKAGE / "validation" / "completion_20261108"

def load(path): return json.loads(path.read_text(encoding="utf-8-sig"))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    global BATCH, PACKAGE
    parser=argparse.ArgumentParser()
    parser.add_argument('--paper-flags',action='store_true')
    parser.add_argument('--package',type=Path)
    args=parser.parse_args()
    if args.package: PACKAGE=args.package.resolve()
    BATCH=PACKAGE/('validation/paper_flags_full_20261108' if args.paper_flags else 'validation/completion_20261108')
    spec = importlib.util.spec_from_file_location("independent_run_audit", WORK / "audit_improved_runs.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    protocol = load(BATCH / "protocol.json")
    settings = load(BATCH / "frozen_selected_settings.json")
    runtime_events = load(BATCH / "runtime_events.json") if (BATCH/"runtime_events.json").exists() else []
    assert sha(BATCH / "frozen_selected_settings.json") == protocol["settings_sha256"]
    for relative, fingerprint in protocol["input_parquet_sha256"].items():
        assert sha(PACKAGE / relative) == fingerprint, "Input changed: " + relative
    for filename, fingerprint in protocol["source_sha256"].items():
        assert sha(PACKAGE / "src" / filename) == fingerprint
    assert sha(PACKAGE / "target" / protocol["compiled_jar"]) == protocol["compiled_jar_sha256"]
    signature = "\n".join(p.name+":"+sha(p).upper() for p in sorted((PACKAGE/"src").glob("*.scala"),key=lambda p:p.name.lower()))
    assert protocol["compiled_jar"] == "bikeshare-models-"+hashlib.sha256(signature.encode()).hexdigest()[:12]+".jar"
    train = audit.labels(PACKAGE / "data" / "train.parquet")
    test = audit.labels(PACKAGE / "data" / "test.parquet")
    assert len(train) == 536406 and len(test) == 133553
    assert sum(train.values()) == 82490 and sum(test.values()) == 20723
    assert train.keys().isdisjoint(test)
    assert audit.labels(PACKAGE / "data" / "enriched" / "train.parquet") == train
    assert audit.labels(PACKAGE / "data" / "enriched" / "test.parquet") == test
    feature_checks=[]
    if args.paper_flags:
        import pyarrow as pa
        import pyarrow.parquet as pq
        preparation=load(BATCH/'feature_preparation_audit.json')
        assert sha(BATCH/'feature_preparation_audit.json')==protocol['feature_preparation_audit_sha256']
        assert sha(WORK/'prepare_full_data_paper_flags.py')==protocol['feature_preparation_source_sha256']
        base_settings_path=PACKAGE/'results/tune-2026-10-30T01-39-38-821598Z-8d5c2d7e/selected_settings.json'
        if not base_settings_path.exists():base_settings_path=PACKAGE/'validation/P1_P2_P3_frozen_settings.json'
        assert sha(base_settings_path)==protocol['base_selected_settings_sha256']==preparation['base_settings_sha256']
        base_settings=load(base_settings_path)
        assert settings['models']==base_settings['models']
        assert settings['numeric_features']==base_settings['numeric_features']+['before_oct_2013']
        assert settings['categorical_features']==base_settings['categorical_features']+['rider_locality']
        assert settings['retuned_with_added_features'] is False
        assert sha(PACKAGE/'data/raw/paper_locality.parquet')==preparation['locality_metadata_sha256']
        locality=audit.read_table(PACKAGE/'data/raw/paper_locality.parquet')
        by_id=dict(zip(locality['id'],locality['local_zip']))
        assert len(by_id)==669959
        for split in ('train','test'):
            for old_path in sorted((PACKAGE/'data/enriched'/(split+'.parquet')).glob('*.parquet')):
                new_path=PACKAGE/'data/paper_flags_full/enriched'/(split+'.parquet')/old_path.name
                assert sha(old_path)==preparation['source_files'][old_path.relative_to(PACKAGE/'data').as_posix()]
                old_table=pq.ParquetFile(old_path).read()
                new_table=pq.ParquetFile(new_path).read()
                assert set(new_table.column_names)-set(old_table.column_names)=={'rider_locality','before_oct_2013'}
                assert new_table.select(old_table.column_names).equals(old_table,check_metadata=False)
                ids=old_table['id'].to_pylist()
                expected_locality=[by_id[i] for i in ids]
                expected_early=[float(date.fromisoformat(s)<date(2013,10,1)) for s in old_table['trip_date'].to_pylist()]
                assert new_table['rider_locality'].to_pylist()==expected_locality
                assert new_table['before_oct_2013'].to_pylist()==expected_early
                feature_checks.append({'split':split,'source_part':old_path.name,'rows':len(ids),
                                       'original_columns_exactly_equal':True,'locality_matches_original_metadata':True,
                                       'early_flag_matches_local_calendar_date':True})
    records, checks, runs = [], [], []
    for model in ("P1", "P2", "P3"):
        for seed in (42, 43, 44):
            status = load(BATCH / f"{model}_seed_{seed}.json")
            assert status["status"] == "completed"
            run = PACKAGE / status["run"]
            checks.append(audit.audit_run(run, train, test, settings))
            manifest = load(run / "manifest.json")
            assert manifest["evaluation"] == "test"
            assert manifest["master"] == "local[2]"
            assert manifest["test_split_recreated"] is False
            assert manifest["test_scores_used_for_selection"] is False
            assert manifest["compiled_artifact"] == protocol["compiled_jar"]
            assert manifest["tree_encoding"] == "native" and manifest["feature_variant"] == "standard"
            assert manifest["numeric_features"] == settings["numeric_features"]
            assert manifest["categorical_features"] == settings["categorical_features"]
            rows = load(run / "results.json")
            flagged_events = [e for e in runtime_events if e["model"]==model and e["seed"]==seed
                              and e.get("attempt",1)==len(status["attempts"]) and e.get("timing_affected")]
            expected_fractions = [1.0]
            assert sorted(r["requested_fraction"] for r in rows) == expected_fractions
            nested = set()
            fractions = []
            for row in sorted(rows, key=lambda r:r["requested_fraction"]):
                assert row["seed"] == seed and row["model"] == model
                fraction = row["requested_fraction"]
                membership = audit.labels(run / "folds" / f"seed-{seed}" / f"fraction-{round(fraction*10000)}")
                assert nested <= membership.keys() <= train.keys()
                assert membership.keys().isdisjoint(test)
                assert len(membership) == row["sample_rows"] == row["training_rows"]
                assert all(train[i] == y for i,y in membership.items())
                if fraction == 1.0: assert membership == train
                nested = set(membership)
                fractions.append({"fraction":fraction,"actual_rows":len(membership),"customers":int(sum(membership.values()))})
                total = row["tp"]+row["fp"]+row["tn"]+row["fn"]
                for field, actual in {
                    "customer_precision":row["tp"]/(row["tp"]+row["fp"]),
                    "customer_recall":row["tp"]/(row["tp"]+row["fn"]),
                    "accuracy":(row["tp"]+row["tn"])/total,
                }.items(): assert abs(actual-row[field]) < 1e-12
                records.append(dict(row, run_id=run.name, data_version=protocol["data_version"],
                                    runtime_timing_affected=bool(flagged_events)))
            runs.append({"model":model,"seed":seed,"run_id":run.name,"mode":status["mode"],
                         "wall_seconds":status["attempts"][-1]["elapsed_seconds"],"fractions":fractions,
                         "all_original_training_rows_retained":True,"runtime_timing_affected":bool(flagged_events)})
    assert len(records) == 9
    paired=[]
    if args.paper_flags:
        for row in records:
            old_status=PACKAGE/'validation/completion_20261108'/f"{row['model']}_seed_{row['seed']}.json"
            if not old_status.exists():old_status=PACKAGE/'validation/proposal_only_completion_20261108'/old_status.name
            if not old_status.exists(): continue
            old_job=load(old_status)
            if old_job.get('status')!='completed':continue
            old_run=PACKAGE/old_job['run']
            old_result=load(old_run/'results.json')[0]
            assert old_result['training_rows']==row['training_rows']==536406
            reference=next(r for r in base_settings['models'] if r['model']==row['model'])
            for parameter,value in reference['parameters'].items():
                assert old_result['classifier_parameters'][parameter]==value
                assert row['classifier_parameters'][parameter]==value
            old_roles=audit.read_table(old_run/'training_roles'/f"seed-{row['seed']}/fraction-10000")
            new_job=load(BATCH/f"{row['model']}_seed_{row['seed']}.json")
            new_run=PACKAGE/new_job['run']
            new_roles=audit.read_table(new_run/'training_roles'/f"seed-{row['seed']}/fraction-10000")
            assert dict(zip(old_roles['id'],old_roles['training_role']))==dict(zip(new_roles['id'],new_roles['training_role']))
            paired.append({'model':row['model'],'seed':row['seed'],'training_rows':536406,'test_rows':133553,
                           'previous_run':old_run.name,'new_run':row['run_id'],
                           'proposal_only_f0_5':old_result['customer_f0_5'],'with_paper_flags_f0_5':row['customer_f0_5'],
                           'difference':row['customer_f0_5']-old_result['customer_f0_5'],
                           'same_hyperparameters_and_calibration_membership':True})
    fields = ["customer_f0_5","customer_precision","customer_recall","accuracy","area_under_pr",
              "final_fit_seconds","threshold_selection_seconds","training_rows"]
    summary = []
    for model in ("P1","P2","P3"):
        for fraction in [1.0]:
            rows = [r for r in records if r["model"]==model and r["requested_fraction"]==fraction]
            assert sorted(r["seed"] for r in rows)==[42,43,44]
            uninterrupted = [r for r in rows if not r["runtime_timing_affected"]]
            summary.append({"model":model,"fraction":fraction,"seeds":[42,43,44],
                            "metrics":{f:{"mean":statistics.mean(r[f] for r in rows),
                                          "sample_sd":statistics.stdev(r[f] for r in rows),
                                          "min":min(r[f] for r in rows),"max":max(r[f] for r in rows)} for f in fields},
                            "timing_without_flagged_jobs":{
                                "jobs":len(uninterrupted),
                                "mean_final_fit_seconds":statistics.mean(r["final_fit_seconds"] for r in uninterrupted) if uninterrupted else None,
                                "mean_threshold_selection_seconds":statistics.mean(r["threshold_selection_seconds"] for r in uninterrupted) if uninterrupted else None}})
    report={"audited_utc":datetime.now(timezone.utc).isoformat(),"passed":True,"expected_model_records":9,
            "train_rows":len(train),"test_rows":len(test),"data_version":protocol["data_version"],
            "input_and_source_hashes_unchanged":True,"all_prediction_ids_labels_cutoffs_confusions_checked":True,
            "selected_parameters_imported_without_retuning":True,"runs":runs,"run_audits":checks,"summary":summary,
            "runtime_events":runtime_events,
            "feature_derivation_checks":feature_checks,"paired_feature_comparisons":paired,
            "limits":["Fixed test has been examined previously; benchmark evaluation rather than a fresh confirmation set.",
                      "Calibration optimum is checked within Spark; this independent audit verifies saved cutoff and final predictions.",
                      "Local wall-clock timings depend on system load and paging; no cluster speedup claim.",
                      "Recorded interruptions affect timing summaries only; predictive scores retain all three seeds. Raw elapsed times remain available.",
                      "Historical V1 inputs remain distinct from updated V2 weather."]}
    (BATCH/"independent_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    (BATCH/"all_results.json").write_text(json.dumps(records,indent=2)+"\n")
    (BATCH/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (BATCH/"paired_feature_comparisons.json").write_text(json.dumps(paired,indent=2)+"\n")
    columns=["model","seed","requested_fraction","training_rows","evaluation_rows","customer_f0_5",
             "customer_precision","customer_recall","accuracy","area_under_pr","decision_threshold",
             "final_fit_seconds","threshold_selection_seconds","runtime_timing_affected","run_id"]
    with (BATCH/"all_results.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore");writer.writeheader();writer.writerows(records)
    text=["# Completed full-data Spark model evaluation","",
          "The frozen expanded settings were evaluated on the historical audited feature version. "
          "All final models use the same 133,553 test trips and seeds 42, 43, and 44. "
          "V2 weather was not substituted into these experiments.","",
          "## Final full-training results","",
          "| Model | Mean Customer F0.5 | Sample SD | Precision | Recall | Accuracy |",
          "|---|---:|---:|---:|---:|---:|"]
    if args.paper_flags:
        text[0]='# Full-data Spark evaluation with locality and the early-period flag'
        text[2]='All final fits use 536,406 training trips and the same 133,553 test trips, with seeds 42, 43, and 44. Two columns were added: rider locality and whether the verified local trip date precedes October 1, 2013. Every original column value and split membership is unchanged. V2 weather was not substituted.'
        text[2]+=' Hyperparameters come from the earlier proposal-only training CV and are held fixed for this feature comparison. They have not been retuned under the expanded feature list. Locality uses the same declared assumption as the XGBoost reconstruction, not an exact recovered author definition.'
    for item in summary:
        if item["fraction"]!=1.0: continue
        m=item["metrics"]
        text.append(f"| {item['model']} | {m['customer_f0_5']['mean']:.6f} | {m['customer_f0_5']['sample_sd']:.6f} | {m['customer_precision']['mean']:.4%} | {m['customer_recall']['mean']:.4%} | {m['accuracy']['mean']:.4%} |")
    if paired:
        text += ['','## Paired feature comparison','','The rows below compare the same hyperparameters, training/test IDs, seed, and calibration memberships. Each run reselects its cutoff using the same training-only procedure. Both added variables change together, so the difference does not isolate locality from the date flag.','',
                 '| Model | Seed | Original inputs F0.5 | With both flags F0.5 | Difference |','|---|---:|---:|---:|---:|']
        for r in paired:text.append(f"| {r['model']} | {r['seed']} | {r['proposal_only_f0_5']:.6f} | {r['with_paper_flags_f0_5']:.6f} | {r['difference']:+.6f} |")
        text += ['','A matching final-test result for the newest proposal-only P1 settings was not completed before the feature revision. The historical P1 mean 0.717260 used another configuration and is a reference rather than a controlled feature ablation.']
    text += ["","## Observed local training cost","",
             "Raw elapsed times are retained in all_results.csv. The table below excludes jobs with "
             "recorded interruptions from timing means only. Every predictive mean above still uses "
             "all three seeds. Absence of a recorded interruption does not establish controlled system load.","",
             "| Model | Jobs without a flagged interruption | Mean final fit seconds | Mean calibration seconds |",
             "|---|---:|---:|---:|"]
    for item in summary:
        if item["fraction"]!=1.0: continue
        t=item["timing_without_flagged_jobs"]
        fit=f"{t['mean_final_fit_seconds']:.1f}" if t['jobs'] else 'unavailable'
        cal=f"{t['mean_threshold_selection_seconds']:.1f}" if t['jobs'] else 'unavailable'
        text.append(f"| {item['model']} | {t['jobs']} | {fit} | {cal} |")
    text += ["","## Full-training-only plan revision","",
             "The user canceled the proposed 10%, 25%, and 50% Spark volume runs. "
             "All nine final fits use every one of the 536,406 original training trips. "
             "The earlier proposed volume protocol is archived as a superseded plan, not a completed experiment. "
             "Training-only cutoff calibration remains part of the procedure before each full-data refit."]
    text += ["","## Interpretation limits","",
             "The earlier P1 configuration scored mean test F0.5 0.717260. Its historical result remains separate. "
             "The calibrated 10,354-row similarity reconstruction scored 0.802624 on the same project test, "
             "but included assumed locality and early-period inputs excluded from P1-P3. "
             "The tuned stratified reconstruction scored 0.879851 with a different sampler. "
             "The 0.815863 compressed reference used a different holdout population. "
             "These comparisons do not isolate a causal training-size or algorithm effect.","",
             "All settings were selected in earlier training-only CV and frozen before this batch. "
             "The test set had been examined in earlier work. These are additional benchmark evaluations. "
             "Timing measures local execution and does not establish a distributed speedup.","",
             "The independent audit checks all nine model records against saved row-level predictions, "
             "training/calibration roles, full training memberships, selected settings, source hashes, and preserved input files.",""]
    if args.paper_flags:
        text=[line.replace("but included assumed locality and early-period inputs excluded from P1-P3.",
                           "and includes assumed locality and early-period variables, which are now also supplied to these expanded Spark runs. Remaining predictor, sampling, and classifier differences still apply.") for line in text]
    (BATCH/"RESULTS.md").write_text("\n".join(text),encoding="utf-8")
    print(json.dumps({"passed":True,"records":len(records),"summary":summary},indent=2))

if __name__ == "__main__": main()
