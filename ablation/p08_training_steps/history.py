"""Explicit ingestion of historical CSV files with repeated embedded headers."""

import csv
import hashlib
from pathlib import Path

import pandas as pd


def read_results_csv(path):
    """Retain every data row before/after repeated headers with explicit provenance."""
    path = Path(path)
    with path.open(newline='') as stream:
        rows = list(csv.reader(stream))
    headers = [row for row in rows if row and row[0] == 'dataset']
    if not headers:
        raise ValueError(f'No dataset header in {path}')
    header = headers[0]
    if any(other != header for other in headers):
        raise ValueError(f'Conflicting embedded CSV headers in {path}')
    records = []
    for line, row in enumerate(rows, start=1):
        if not row or row[0] == 'dataset':
            continue
        if len(row) != len(header):
            raise ValueError(f'Malformed CSV row {path}:{line}')
        record = dict(zip(header, row))
        record['source_file'] = str(path)
        record['source_line'] = line
        records.append(record)
    frame = pd.DataFrame(records)
    for key in ('num_layers', 'split', 'best_step', 'val_metric', 'test_metric',
                'hidden_dim', 'lr'):
        if key in frame:
            frame[key] = pd.to_numeric(frame[key], errors='raise')
    frame['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return frame
