"""Reuse saved reconstructed-baseline IDs with the full experiment's inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--full-data-dir',type=Path,required=True)
    p.add_argument('--baseline-run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--validation-dir',type=Path,required=True)
    p.add_argument('--settings',type=Path,required=True)
    a=p.parse_args();a.validation_dir.mkdir(parents=True,exist_ok=True)
    test_parts=sorted((a.full_data_dir/'enriched/test.parquet').glob('*.parquet'))
    assert sum(pq.ParquetFile(f).metadata.num_rows for f in test_parts)==133553
    report={'prepared_utc':datetime.now(timezone.utc).isoformat(),
            'source_baseline_run':a.baseline_run.name,'settings_sha256':sha(a.settings),
            'input_feature_count':39,'original_training_pool_rows':536406,'test_rows':133553,
            'cohort_policy':'Saved reconstruction IDs reused exactly; no resampling on the expanded features',
            'class_mix_note':'Similarity reconstruction changes class proportions; stratified control preserves the natural mix.',
            'primary_full_data_experiments_unchanged':True,'cases':[]}
    for sampler in ('similarity','stratified'):
        for seed in (42,43,44):
            saved=a.baseline_run/('paper-similarity' if sampler=='similarity' else '')/f'seed-{seed}'/'sampled_ids.parquet'
            reference=pq.ParquetFile(saved).read()
            assert len(reference)==10354 and pc.count_distinct(reference['id']).as_py()==10354
            expected=dict(zip(reference['id'].to_pylist(),reference['label'].to_pylist()))
            selected=[];input_hashes={}
            for full in sorted((a.full_data_dir/'enriched/train.parquet').glob('*.parquet')):
                table=pq.ParquetFile(full).read()
                mask=pc.is_in(table['id'],value_set=reference['id'])
                selected.append(table.filter(mask))
                input_hashes[full.relative_to(a.full_data_dir).as_posix()]=sha(full)
            train=pa.concat_tables(selected).replace_schema_metadata(None)
            assert len(train)==10354
            actual=dict(zip(train['id'].to_pylist(),train['label'].to_pylist()))
            assert actual==expected and len(actual)==10354
            cohort=a.output_dir/sampler/f'seed-{seed}'
            training=cohort/'enriched/train.parquet/part-00000.parquet'
            training.parent.mkdir(parents=True,exist_ok=True)
            if not training.exists():pq.write_table(train,training,compression='snappy',use_deprecated_int96_timestamps=True)
            assert pq.ParquetFile(training).read().equals(train,check_metadata=False)
            copied=[]
            for i,source in enumerate(test_parts):
                target=cohort/f'enriched/test.parquet/part-{i:05d}.parquet'
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copy2(source,target)
                assert sha(target)==sha(source)
                copied.append({'source':source.name,'copy':target.relative_to(cohort).as_posix(),'sha256':sha(target)})
            report['cases'].append({'sampler':sampler,'seed':seed,'training_rows':10354,
                                    'customer_rows':int(pc.sum(train['label']).as_py()),
                                    'customer_rate':pc.sum(train['label']).as_py()/10354,
                                    'saved_ids_sha256':sha(saved),
                                    'saved_ids_path':saved.relative_to(a.baseline_run).as_posix(),
                                    'cohort_relative_path':cohort.relative_to(a.output_dir).as_posix(),
                                    'training_parquet_sha256':sha(training),
                                    'full_input_hashes':input_hashes,'test_copies':copied,
                                    'same_original_rows_labels_and_39_inputs':True})
    target=a.validation_dir/'cohort_preparation_audit.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cohorts':len(report['cases']),'rows_each':10354,'same_test_rows':133553,
                      'same_39_features':True,'primary_full_training_rows':536406},indent=2))

if __name__=='__main__':main()
