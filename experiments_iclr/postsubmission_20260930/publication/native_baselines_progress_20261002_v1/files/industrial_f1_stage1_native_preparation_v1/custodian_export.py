"""Custodian-only export from explicit, already-admitted local native files.

Never run this inside the worker's filesystem. Test parquet is read by key
projection only; no test target values are materialized or copied.
"""
import argparse
from pathlib import Path

from common import COUNTS, TABLES, empty_output, row_hash, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-db', type=Path, required=True)
    p.add_argument('--train-table', type=Path, required=True)
    p.add_argument('--val-table', type=Path, required=True)
    p.add_argument('--test-table', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from relbench.base import Database, Dataset, Table
    out = empty_output(a.output_dir)
    db = Database({name: Table.load(a.source_db / f'{name}.parquet') for name in TABLES})
    db = db.upto(pd.Timestamp('2010-01-01'))
    # Same native validation/correction as Dataset.get_db, without invoking it.
    Dataset().validate_and_correct_db(db)
    database = {}
    for name, table in db.table_dict.items():
        file = f'database/{name}.parquet'
        table.save(out / file)
        database[name] = {'file': file, 'sha256': sha256(out / file), 'rows': len(table),
                          'time_col': table.time_col, 'pkey_col': table.pkey_col,
                          'fkey_col_to_pkey_table': table.fkey_col_to_pkey_table}
    rows = {}
    for split, source in [('train', a.train_table), ('val', a.val_table), ('test_keys', a.test_table)]:
        cols = ['driverId', 'date'] + ([] if split == 'test_keys' else ['position'])
        # Projection also strips original pandas metadata containing target names.
        df = pq.read_table(source, columns=cols).to_pandas()[cols].copy()
        df['date'] = pd.to_datetime(df.date).astype('datetime64[ns]')
        if len(df) != COUNTS[split] or df[cols].isna().any().any():
            raise ValueError(f'Unexpected native split/count/nulls: {split}')
        if df.duplicated(['driverId', 'date']).any():
            raise ValueError(f'Duplicate forecast keys: {split}')
        if split != 'test_keys' and not np.isfinite(df.position.to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite target: {split}')
        df.insert(0, 'row_id', np.arange(len(df)))
        df.insert(1, 'row_hash', [row_hash(split, r.row_id, r.driverId, r.date)
                                  for r in df.itertuples()])
        table = Table(df, {'driverId': 'drivers'}, time_col='date')
        file = f'rows/{split}.parquet'
        table.save(out / file)
        rows[split] = {'file': file, 'sha256': sha256(out / file), 'rows': len(table),
                       'columns': list(df.columns)}
    write_json(out / 'PUBLIC_INPUTS.json', {
        'schema': 'f1-stage1-public-v1', 'task': 'rel-f1/driver-position',
        'native_commit': '9aa346267c2e1c560bd92da07d6f4ad1ca2f0639',
        'public_database_cutoff_inclusive': '2010-01-01',
        'feature_fit_cutoff_inclusive': '2005-01-01',
        'database': database, 'rows': rows,
        'test_targets_present': False, 'row_id_definition': 'original native split row order',
        'untimestamped_metadata': 'native benchmark metadata retained'})
    # Re-read only the exported public files using the worker admission checks.
    from common import read_public
    read_public(out)
    print(f'Public export written to {out}')


if __name__ == '__main__':
    main()
