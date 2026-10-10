"""Stdlib artifact custody; exact existing signature fingerprint."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
SEEDS=(9101,9203,9307)
CONDITIONS=('shared4_own','shared4_core','independent4_native')
LIMITS=dict(external_active_seconds=300,external_cleanup_seconds=10,
    GPU_bytes=2*1024**3,RSS_bytes=2*1024**3,output_bytes=2*1024**2,
    log_bytes=1024**2,automatic_retry=False)
SERVER_PHASE='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def inside(path,existing=True):
    value=Path(path)
    if not value.is_absolute():value=PHASE/value
    value=value.resolve(strict=existing)
    if value==PHASE or not value.is_relative_to(PHASE):raise ValueError('Only exact project phase artifacts')
    return value


def bind(row):
    path=inside(row['path'])
    if not path.is_file() or sha(path)!=row['sha256']:raise ValueError('Bound artifact changed')
    if row.get('bytes') is not None and path.stat().st_size!=row['bytes']:raise ValueError('Bound artifact byte count changed')
    return path


def fingerprint(array):
    header=json.dumps(dict(dtype=array.dtype.str,shape=list(array.shape)),sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(header+b'\n'+array.tobytes(order='C')).hexdigest()


def sources(digest=None):
    if digest is not None:
        if sha(HERE/'SOURCE_MANIFEST.json')!=digest:raise ValueError('Exact sealed diagnostic source')
        for row in json.loads((HERE/'SOURCE_MANIFEST.json').read_text())['files']:
            path=(HERE/row['path']).resolve(strict=True)
            if not path.is_relative_to(HERE) or sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Diagnostic payload changed')
    binding=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in binding['files']:bind(row)
    manifest=bind(binding['stage1_manifest'])
    for row in json.loads(manifest.read_text())['files']:
        path=(manifest.parent/row['path']).resolve(strict=True)
        if not path.is_relative_to(manifest.parent) or sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Original sealed Stage1 source changed')
    return binding
