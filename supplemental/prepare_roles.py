"""Copy the exact saved P3 calibration IDs and CV folds for Python controls."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import pyarrow as pa
import pyarrow.parquet as pq


def read(path):
    parts=sorted(path.glob('*.parquet')) if path.is_dir() else [path]
    return pa.concat_tables([pq.ParquetFile('\\\\?\\'+str(p.resolve()) if os.name=='nt' else str(p)).read() for p in parts])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-package',type=Path,required=True)
    parser.add_argument('--source-state-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    state=args.source_state_dir or args.source_package/'validation/paper_flags_full_20261108'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    audit=[]
    for seed in (42,43,44):
        job=json.loads((state/f'P3_seed_{seed}.json').read_text(encoding='utf-8-sig'))
        assert job['status']=='completed'
        run=args.source_package/job['run'];suffix=f'seed-{seed}/fraction-10000'
        roles=read(run/'training_roles'/suffix).to_pandas()
        folds=read(run/'model_selection_folds'/suffix).to_pandas()
        assert len(roles)==536406 and roles.id.is_unique and folds.id.is_unique
        assert set(roles.training_role)=={'model_selection','calibration'}
        assert set(folds.cv_fold)=={0,1,2}
        assert set(folds.id)==set(roles.loc[roles.training_role=='model_selection','id'])
        joined=roles.merge(folds[['id','cv_fold']],on='id',how='left',validate='one_to_one')
        assert joined.loc[joined.training_role=='calibration','cv_fold'].isna().all()
        output=args.output_dir/f'seed-{seed}.parquet'
        schema=pa.schema([('id',pa.int64()),('training_role',pa.string()),('cv_fold',pa.int32())])
        table=pa.Table.from_pandas(joined,preserve_index=False).cast(schema)
        # Older pandas/Arrow releases encode identical strings and nullable folds
        # with different physical types. Compare canonical values, preserving bytes.
        if output.exists():assert pq.ParquetFile(output).read().cast(schema).equals(table,check_metadata=False)
        else:pq.write_table(table,output)
        audit.append(dict(seed=seed,rows=len(joined),calibration_rows=int((joined.training_role=='calibration').sum()),
                          source_run=run.name,sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
    (args.output_dir/'preparation_audit.json').write_text(json.dumps(dict(cases=audit,exact_saved_roles_preserved=True),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(prepared_seeds=3,rows_each=536406)))


if __name__=='__main__':main()
