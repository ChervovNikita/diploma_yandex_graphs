"""Disabled direct12 source: standard-library custody and release gates."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
SEEDS = (6101, 6203, 6307)
CONDITIONS = ('BE_factor_refit', 'BE_full_refit', 'ordinary_full_refit', 'single_full_refit')
ARM = dict(BE_factor_refit='be_unit_contrastive', BE_full_refit='be_unit_contrastive',
           ordinary_full_refit='independent4', single_full_refit='single')
SUITE = 'learnable_internal_be_contrastive_multitask_suite_20261007_v4'
SUITE_SHA = '76de82781e7fd496a5a3382b3a71a5781ea023dfe3777469cc005a2b19afbfce'
READER_SHA = '476a59b56bd3377c6c93d0048a979f2fe6f08795e77b68e1452ebc874ec96042'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(PHASE)), sha256=sha(path), bytes=path.stat().st_size)


def bound(row):
    relative = Path(row['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Phase-relative binding required')
    path = (PHASE / relative).resolve(strict=True)
    require(path.is_relative_to(PHASE) and path.is_file() and sha(path) == row['sha256'], 'Bound bytes changed')
    if 'bytes' in row:
        require(path.stat().st_size == row['bytes'], 'Bound size changed')
    return path


def seal(root, expected):
    root = Path(root)
    require(sha(root / 'MANIFEST.json') == expected, 'Exact source seal required')
    for row in read(root / 'MANIFEST.json')['files']:
        path = (root / row['path']).resolve(strict=True)
        require(path.is_relative_to(root.resolve()) and sha(path) == row['sha256']
                and path.stat().st_size == row['bytes'], 'Source file changed')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def roster():
    return [dict(condition=c, seed=s, bundle=c + '_' + str(s), arm=ARM[c],
                 members=1 if c == 'single_full_refit' else 4) for s in SEEDS for c in CONDITIONS]


def release(path, digest, stage):
    require(sha(path) == digest, 'Exact root release required')
    cfg = read(path)
    require(cfg.get('schema') == 'Wiki24-direct12-release-v1' and cfg.get('stage') == stage,
            'Fixed direct12 stage required')
    for key in ('enabled', 'root_execution_authorized', 'source_review_approved', 'finite_cost_charged', 'external_hard_bound_confirmed'):
        require(cfg.get(key) is True, 'Disabled pending root release: ' + key)
    require(cfg.get('TEST_access') is False and cfg.get('automatic_retry') is False, 'No TEST/retry')
    seal(ROOT, cfg['source_manifest_sha256'])
    pins = read(ROOT / 'SOURCE_BINDINGS.json')
    bound(pins['fixed_plan'])
    bound(pins['frozen_decision'])
    bound(pins['cohort_clarification'])
    approval = read(bound(cfg['source_review']))
    require(approval.get('approved') is True and approval.get('source_manifest_sha256') == cfg['source_manifest_sha256'],
            'Exact independent review required')
    relative = Path(cfg['output_directory'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Phase-relative fresh output required')
    output = (PHASE / relative).resolve()
    require(output.is_relative_to(PHASE) and not output.exists() and output.parent.is_dir(), 'Fresh output only')
    return cfg, output


def load_arrays(np, row):
    path = bound(row)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    require(all(not value.dtype.hasobject for value in arrays.values()), 'Numeric arrays only')
    return arrays


def backend(torch, np):
    import inspect
    require(torch.__version__ == '2.1.2+cu118' and np.__version__ == '1.26.4', 'Pinned existing backend required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU fit/reader must hide CUDA before import')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return dict(torch=torch.__version__, numpy=np.__version__, device='cpu', dtype='float64',
                threads=1, interop_threads=1, deterministic_algorithms=True,
                LBFGS_source_sha256=sha(inspect.getfile(torch.optim.LBFGS)))


def cpu_env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
