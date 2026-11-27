"""Append two declared paper inputs without filtering or resplitting trips."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--locality-metadata',type=Path,required=True)
    parser.add_argument('--validation-dir',type=Path,required=True)
    parser.add_argument('--base-settings',type=Path,required=True)
    args=parser.parse_args()
    args.validation_dir.mkdir(parents=True,exist_ok=True)
    locality=pq.ParquetFile(args.locality_metadata).read()
    assert set(locality.column_names)=={'id','local_zip'}
    assert len(locality)==669959 and pc.count_distinct(locality['id']).as_py()==len(locality)
    assert locality['id'].null_count==0 and locality['local_zip'].null_count==0
    assert set(pc.unique(locality['local_zip']).to_pylist()) <= {'local','outside','missing'}
    locality_ids=pc.cast(locality['id'],pa.int64())
    locality_values=pc.cast(locality['local_zip'],pa.string())
    report={'prepared_utc':datetime.now(timezone.utc).isoformat(),
            'data_version':'audited historical enriched V1 plus two declared paper inputs',
            'locality_metadata_sha256':sha(args.locality_metadata),
            'added_features':{
                'rider_locality':{'source':'previously audited ID-to-locality metadata; same values as reconstruction local_zip',
                                  'definition':'Valid five-digit US rider postal code (optional ZIP+4), prefix 94 or 95 is local; outside and missing remain separate categories',
                                  'status':'Declared assumption; original paper definition unavailable; no weather/station ZIP substitution'},
                'before_oct_2013':{'definition':'Verified local trip_date earlier than 2013-10-01, encoded 1.0; otherwise 0.0',
                                   'source':'trip_date, not the legacy stored raw timestamp'}},
            'raw_postal_codes_written':False,'rows_filtered':0,'split_recreated':False,
            'source_files':{},'output_files':{},'splits':{}}
    for split,expected,positive in [('train',536406,82490),('test',133553,20723)]:
        source=args.data_dir/'enriched'/(split+'.parquet')
        destination=args.output_dir/'enriched'/(split+'.parquet')
        destination.mkdir(parents=True,exist_ok=True)
        total=customers=0;groups=Counter()
        for part in sorted(source.glob('*.parquet')):
            base=pq.ParquetFile(part).read()
            assert not {'rider_locality','before_oct_2013'} & set(base.column_names)
            indices=pc.index_in(pc.cast(base['id'],pa.int64()),value_set=locality_ids)
            assert indices.null_count==0, 'Every trip needs exactly one original locality category'
            flags=pc.take(locality_values,indices)
            local_dates=pc.cast(base['trip_date'],pa.date32())
            assert local_dates.null_count==0
            early=pc.cast(pc.less(local_dates,pa.scalar(date(2013,10,1),type=pa.date32())),pa.float64())
            augmented=base.append_column('rider_locality',flags).append_column('before_oct_2013',early)
            # Old Spark schema metadata describes only the original 49 columns.
            # Read the physical schema instead; INT96 retains Spark 3.5 timestamp compatibility.
            augmented=augmented.replace_schema_metadata(None)
            output=destination/part.name
            if not output.exists():
                pq.write_table(augmented,output,compression='snappy',use_deprecated_int96_timestamps=True)
            check=pq.ParquetFile(output).read()
            assert check.equals(augmented,check_metadata=False)
            assert check.select(base.column_names).equals(base,check_metadata=False)
            assert len(check)==len(base) and flags.null_count==0
            total+=len(base);customers+=int(pc.sum(base['label']).as_py())
            grouped=augmented.select(['id','label','rider_locality','before_oct_2013']).group_by(
                ['label','rider_locality','before_oct_2013']).aggregate([('id','count')])
            for row in grouped.to_pylist():
                groups[(int(row['label']),row['rider_locality'],int(row['before_oct_2013']))]+=row['id_count']
            report['source_files'][part.relative_to(args.data_dir).as_posix()]=sha(part)
            report['output_files'][output.relative_to(args.output_dir).as_posix()]=sha(output)
        assert total==expected and customers==positive
        report['splits'][split]={'rows':total,'customers':customers,
                                 'original_columns_and_row_order_unchanged':True,
                                 'added_feature_counts':[{'label':y,'rider_locality':loc,'before_oct_2013':early,'rows':n}
                                                       for (y,loc,early),n in sorted(groups.items())]}
    original=json.loads(args.base_settings.read_text())
    settings=json.loads(args.base_settings.read_text())
    settings['numeric_features']=original['numeric_features']+['before_oct_2013']
    settings['categorical_features']=original['categorical_features']+['rider_locality']
    settings['parameter_selection_scope']='Hyperparameter values selected by training-only CV on the original proposal inputs, then held fixed for this two-feature ablation'
    settings['retuned_with_added_features']=False
    settings['base_selected_settings_sha256']=sha(args.base_settings)
    settings['experiment_role']='Full-data feature ablation; not a claim of newly tuned hyperparameters'
    settings['added_numeric_features']=['before_oct_2013']
    settings['added_categorical_features']=['rider_locality']
    settings_path=args.validation_dir/'frozen_paper_input_settings.json'
    encoded=json.dumps(settings,indent=2)+'\n'
    if settings_path.exists():assert settings_path.read_text()==encoded
    else:settings_path.write_text(encoded)
    report['base_settings_sha256']=sha(args.base_settings)
    report['frozen_settings_sha256']=sha(settings_path)
    report['all_original_rows_and_columns_preserved']=True
    target=args.validation_dir/'feature_preparation_audit.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'train_rows':536406,'test_rows':133553,'features_added':2,
                      'rows_filtered':0,'original_columns_unchanged':True,
                      'parameter_values_held_fixed':True,'audit':target.name},indent=2))

if __name__=='__main__': main()
