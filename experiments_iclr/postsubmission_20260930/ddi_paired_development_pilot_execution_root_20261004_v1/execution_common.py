"""Stdlib source custody and metadata writes for the fixed pilot queue."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def update(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    save(temporary, value)
    os.replace(temporary, path)


def pin(row):
    path = PHASE / row['path']
    assert path.resolve().is_relative_to(PHASE) and path.is_file() and not path.is_symlink()
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
    return path


def source_inventory():
    seal = json.loads((ROOT / 'ROOT_SOURCE_SEAL.json').read_text())
    assert sha(ROOT / 'ROOT_SOURCE_MANIFEST.json') == seal['manifest_sha256']
    manifest = json.loads((ROOT / 'ROOT_SOURCE_MANIFEST.json').read_text())
    rows = list(json.loads((ROOT / 'STAGE_CLOSURE.json').read_text())['external_files'])
    for row in manifest['payloads']:
        rows.append(dict(row,path=str((ROOT / row['path']).relative_to(PHASE))))
    for name in ('ROOT_SOURCE_MANIFEST.json','ROOT_SOURCE_SEAL.json'):
        path = ROOT / name
        rows.append(dict(path=str(path.relative_to(PHASE)),bytes=path.stat().st_size,sha256=sha(path)))
    rows = sorted({row['path']:row for row in rows}.values(),key=lambda row:row['path'])
    for row in rows:
        pin(row)
    return rows
