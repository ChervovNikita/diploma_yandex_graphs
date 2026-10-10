"""Fixed support and hash checks; stdlib import only, no launcher or job owner."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
BACKBONES = ('SAGE', 'GCN', 'GAT')
SEEDS = (7301, 7403, 7507)
ARMS = ('ordinary_M1', 'ordinary_genuine_I4', 'factorized_allmap_M1',
        'factorized_allmap_genuine_I4', 'shared4_unchanged')
RULES = ('temperature_global', 'temperature_member', 'reliability_self',
         'linear_stacking', 'reliability_graph')
ARCHIVE_KEYS = {'ids', 'y', 'raw_logits', 'probability_mean', 'member_errors', 'pooled_errors'}
ATOL, RTOL = 1e-6, 1e-5


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256') if hasattr(hashlib, 'file_digest') else None
        if digest is not None:
            return digest.hexdigest()
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
        return value.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_progress(path, value):
    """Atomic counters only; root owns polling and process lifecycle."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def root_runtime():
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd() == REPO,
            'Root must enter the literal authorized allocation/repository first')
    require(subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
        == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Wrong allocation GPU')


def new_output(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PHASE) and not path.exists(), 'Fresh in-scope output required')
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_contract(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PHASE) and path.name == 'FROZEN_SUPPORT.json', 'Exact in-scope source support path')
    contract = json.loads(path.read_text())
    require(contract['schema'] == 'common-wrapper-reliability-support-v1'
            and tuple(contract['backbones']) == BACKBONES
            and tuple(contract['seeds']) == SEEDS and tuple(contract['arms']) == ARMS
            and contract['banks'] == 45 and contract['TEST_access'] is False,
            'Exact frozen45-bank support required')
    for row in contract['source_files']:
        require(sha(PHASE / row['path']) == row['sha256'], 'New source digest mismatch')
    return contract


def closed_family(row):
    """Read already closed descriptors only; verify all inputs before any replay."""
    for bound in row['frozen_inputs']:
        require(sha(REPO / bound['path']) == bound['sha256'], bound['path'])
    cfg = json.loads(Path(row['config']).read_text())
    require(cfg == row['configuration'], 'Selected-family config changed')
    root = Path(row['family_root'])
    complete_path = root / 'COMPLETE_FAMILY.json'
    require(sha(complete_path) == row['complete_sha256'], 'Closed family changed')
    end_path = root.parent / 'OWNER_END.json'
    require(sha(end_path) == row['owner_end_sha256'], 'Closed owner receipt changed')
    complete, end = json.loads(complete_path.read_text()), json.loads(end_path.read_text())
    require(complete['complete'] is True and complete['groups'] == 21
            and complete['fit_units'] == 39 and complete['TEST_access'] is False
            and end['scientific_success'] and end['direct_child_wait']
            and end['child_pid_absent'] and end['owned_cuda_pid_absent'], 'Full closure required')
    require(tuple(cfg['seeds']) == SEEDS, 'Paired seed roster changed')
    index = {(r['arm'], r['seed']): r for r in complete['results']}
    for bank in row['banks']:
        archive = Path(bank['archive'])
        require(sha(archive) == bank['archive_sha256'], 'Selected VALID archive changed')
        units = index[bank['arm'], bank['seed']]['fits']
        require([u['selected_state'] for u in units] == bank['checkpoints'],
                'Exact native selected checkpoint paths required')
    return cfg, index
