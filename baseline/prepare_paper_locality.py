"""Derive the explicitly assumed locality flag for the paper reconstruction.

No raw rider postal codes are written to the output. The proposed Spark models
continue to exclude rider-locality features.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trip-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Locality output already exists; preserve its source version'
    with zipfile.ZipFile(args.trip_archive) as archive:
        candidates = [n for n in archive.namelist() if Path(n).name.lower() == 'trip.csv']
        assert len(candidates) == 1, 'Expected one original trip.csv in archive'
        with archive.open(candidates[0]) as stream:
            raw = pd.read_csv(stream, usecols=['id', 'zip_code'], dtype={'id': 'int64', 'zip_code': 'string'}, keep_default_na=False)
    assert raw.id.is_unique and raw.id.notna().all() and len(raw) == 669959
    postal = raw.zip_code.str.strip()
    valid_us = postal.str.fullmatch(r'\d{5}(?:-\d{4})?')
    missing = postal.str.lower().isin(['', 'nan', 'null', 'none', '0'])
    local = valid_us & postal.str[:2].isin(['94', '95'])
    flag = np.where(missing, 'missing', np.where(local, 'local', 'outside'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': raw.id, 'local_zip': flag}).to_parquet(args.output, index=False)
    report = dict(rows=len(raw), unique_ids=int(raw.id.nunique()),
                  locality_counts=pd.Series(flag).value_counts().to_dict(),
                  definition='Valid five-digit US postal code (optional ZIP+4) with prefix 94 or 95; missing separate',
                  definition_status='Declared assumption; exact original local_zip definition not supplied',
                  source_archive=args.trip_archive.name,
                  archive_sha256=hashlib.sha256(args.trip_archive.read_bytes()).hexdigest(),
                  raw_postal_codes_written=False)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
