"""Small local IO/source contracts. Importing this module needs only stdlib."""
from pathlib import Path
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys

PACKET = Path(__file__).resolve().parent
TABLES = ('circuits', 'constructors', 'constructor_results',
          'constructor_standings', 'drivers', 'qualifying', 'races',
          'results', 'standings')
COUNTS = {'train': 7453, 'val': 499, 'test_keys': 760}
PKEYS = {'circuits': 'circuitId', 'constructors': 'constructorId',
         'constructor_results': 'constructorResultsId',
         'constructor_standings': 'constructorStandingsId', 'drivers': 'driverId',
         'qualifying': 'qualifyId', 'races': 'raceId', 'results': 'resultId',
         'standings': 'driverStandingsId'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def relative_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Input file must exist under input root: {name}')
    return path


def empty_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f'Output directory must be empty: {path}')
    return path


def read_public(root):
    """Explicit file list, no dataset/task registry or download/cache calls."""
    from relbench.base import Database, Table
    import numpy as np
    import pandas as pd
    root = Path(root).resolve()
    manifest = json.loads(relative_file(root, 'PUBLIC_INPUTS.json').read_text())
    if manifest['schema'] != 'f1-stage1-public-v1':
        raise ValueError('Unknown public input schema')
    if set(manifest['database']) != set(TABLES):
        raise ValueError('Expected exactly nine F1 database tables')
    for item in list(manifest['database'].values()) + list(manifest['rows'].values()):
        if sha256(relative_file(root, item['file'])) != item['sha256']:
            raise ValueError(f'Public input hash mismatch: {item["file"]}')
    db = Database({name: Table.load(relative_file(root, item['file']))
                   for name, item in manifest['database'].items()})
    for name, table in db.table_dict.items():
        time_col = None if name in ['circuits', 'constructors', 'drivers'] else 'date'
        if table.time_col != time_col or table.pkey_col != PKEYS[name]:
            raise ValueError(f'Unexpected native table metadata: {name}')
        if table.pkey_col and not np.array_equal(table.df[table.pkey_col], np.arange(len(table))):
            raise ValueError(f'Nonconsecutive stable database IDs: {name}')
        if table.time_col and (table.df[table.time_col].isna().any() or
                               (table.df[table.time_col] > pd.Timestamp('2010-01-01')).any()):
            raise ValueError(f'Invalid public timestamp: {name}')
        for col, target in table.fkey_col_to_pkey_table.items():
            keys = table.df[col].dropna()
            if ((keys < 0) | (keys >= len(db.table_dict[target])) | (keys != keys.astype(int))).any():
                raise ValueError(f'Dangling public FK: {name}.{col}')
    rows = {}
    for split, item in manifest['rows'].items():
        table = Table.load(relative_file(root, item['file']))
        expected = {'row_id', 'row_hash', 'driverId', 'date'}
        if split != 'test_keys':
            expected.add('position')
        if set(table.df.columns) != expected or len(table) != COUNTS[split]:
            raise ValueError(f'Unexpected {split} columns/count')
        if not np.array_equal(table.df.row_id, np.arange(len(table))):
            raise ValueError(f'Invalid forecast order: {split}')
        if table.df.duplicated(['driverId', 'date']).any():
            raise ValueError(f'Duplicate forecast keys: {split}')
        for r in table.df.itertuples():
            if r.row_hash != row_hash(split, r.row_id, r.driverId, r.date):
                raise ValueError(f'Forecast row hash mismatch: {split}/{r.row_id}')
        if split != 'test_keys' and not np.isfinite(table.df.position.to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite labels: {split}')
        if ((table.df.driverId < 0) | (table.df.driverId >= len(db.table_dict['drivers']))).any():
            raise ValueError(f'Invalid driver IDs: {split}')
        dates = table.df.date
        if dates.isna().any() or (split == 'val' and (dates < pd.Timestamp('2005-01-01')).any()):
            raise ValueError(f'Invalid forecast dates: {split}')
        if split == 'train' and (dates >= pd.Timestamp('2005-01-01')).any():
            raise ValueError('Training forecast date reaches validation cutoff')
        if split == 'val' and (dates >= pd.Timestamp('2010-01-01')).any():
            raise ValueError('Validation forecast date reaches test cutoff')
        if split == 'test_keys' and (dates < pd.Timestamp('2010-01-01')).any():
            raise ValueError('Test forecast date precedes test cutoff')
        rows[split] = table
    if set(rows) != set(COUNTS):
        raise ValueError('Expected train, val, and masked test_keys')
    return manifest, db, rows


def row_hash(split, row_id, driver_id, date):
    payload = [split, int(row_id), int(driver_id), date.isoformat()]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def metrics(y, pred):
    import numpy as np
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if y.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError('Predictions must be finite and cover every row')
    residual = y - pred
    ss = float(np.square(y - y.mean()).sum())
    return {'mae': float(np.abs(residual).mean()),
            'rmse': float(np.sqrt(np.square(residual).mean())),
            'r2': 1.0 - float(np.square(residual).sum()) / ss if ss else None}


def write_predictions(path, table, pred):
    df = table.df[['row_id', 'row_hash', 'driverId', 'date']].copy()
    if len(pred) != len(df):
        raise ValueError('Prediction count mismatch')
    df['prediction'] = pred
    df.to_csv(path, index=False)


def verify_runtime():
    """Fail on source drift; this is a future runtime gate, not qualification."""
    profile = json.loads((PACKET / 'PROFILE.json').read_text())
    for receipt in json.loads((PACKET / 'evidence/source_receipts.json').read_text()):
        if sha256(PACKET / receipt['file']) != receipt['sha256']:
            raise RuntimeError(f'Saved source integrity mismatch: {receipt["file"]}')
    versions = {name: importlib.metadata.version(name) for name in profile['runtime_versions']}
    for name, wanted in profile['runtime_versions'].items():
        if versions[name].split('+')[0] != wanted:
            raise RuntimeError(f'{name}: expected {wanted}, got {versions[name]}')
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Candidate runtime requires Python 3.11')
    checked = {}
    for module_name, source in profile['source_checks'].items():
        module = importlib.import_module(module_name)
        actual = sha256(module.__file__)
        if actual != sha256(PACKET / source):
            raise RuntimeError(f'Installed source differs from pinned snapshot: {module_name}')
        checked[module_name] = actual
    all_versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    return {'python': sys.version, 'platform': platform.platform(),
            'versions': versions, 'all_distribution_versions': all_versions,
            'verified_source_sha256': checked}
