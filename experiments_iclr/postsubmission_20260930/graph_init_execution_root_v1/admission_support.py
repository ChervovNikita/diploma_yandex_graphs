"""Stdlib metadata custody shared by the root builder and confined entry."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REMOTE_REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
REMOTE_PHASE = REMOTE_REPO/'experiments_iclr/postsubmission_20260930'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
GPU_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
R17_REL = 'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v2'
MODERN_REL = 'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3'
R17_MANIFEST_SHA = '2a4208581e67e44f8941a43b696fd8ee141b2eec6533893862c1f0dfbe8c0345'
R17_SEAL_SHA = 'd21fc53935ef20e7f9278039a9a4a26fea4f3248c3a75157401fb0d95c345203'
MODERN_MANIFEST_SHA = 'e2fd767c4d08e93ca7ec9c9c526440c4cc5b8f4e7d8bb345de90308e9c57302b'
MODERN_SEAL_SHA = '9589bd1a41ec59fc9259d72c833c5e550e6887196eb773b86c75c6d03b67645d'
SOURCE_BINDING_SHA = 'c93535404c115c0cf68e64b1f39411f0fd799ecb61f142077e2c2f14981ed713'
PROTOCOL_SHA = '44765e8358fac6dd9dc211a13a32f260d304bed7b334c9ea4547d95b2d16856a'
PREPARATION_SHA = '376e8b779d8d71cf21785a62b8543ecadc9c09c9db8174dc230b55b8d0dbd253'
ARMS = ('graph', 'common_only', 'random_tangent', 'topology_permuted', 'warm_copy')
CELLS = (('Squirrel', 'polyformer_mono'), ('Photo', 'polynormer_r'))
SEEDS = (17, 29, 43)
TEXT_EXTENSIONS = {'.json', '.py', '.md', '.csv', '.txt', '.sh', '.toml', '.yaml', '.yml',
                   '.diff', '.patch', '.log', '.jsonl'}
DISCLOSURE = ('This extension was designed after earlier negative screening outcomes. '
    'Photo and Squirrel families were previously exposed; this is exploratory and outcome-aware. '
    'independent_of_stage1_outcomes describes source-pack custody only: no reuse, binding or '
    'selection of Stage1 fitted outputs or labels as this study evidence. It does not describe '
    'independence of idea/cohort choice or an unseen confirmatory cohort. R17 v1 is a sealed '
    'source-only predecessor and draft source audit, with no predecessor R17 numerical registry. '
    'Retained native qualification failures remain provenance, not R17 runs.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def confined(path, base=PHASE):
    path = Path(path)
    require(path.is_absolute() and path == Path(str(path)), 'Absolute path required')
    require('..' not in path.parts and path.is_relative_to(base) and path != base,
            'Path must stay inside the research phase')
    cursor = base
    for part in path.relative_to(base).parts:
        cursor /= part
        require(not cursor.is_symlink(), 'Symlink path forbidden: '+str(cursor))
    return path


def mirror(path):
    """Only text metadata are mirrored for local drafting; runtime paths stay exact."""
    path = Path(path)
    if path.is_relative_to(REMOTE_PHASE) and PHASE != REMOTE_PHASE:
        path = PHASE/path.relative_to(REMOTE_PHASE)
    return confined(path)


def text_path(path):
    path = mirror(path)
    require(path.suffix in TEXT_EXTENSIONS, 'Metadata tool cannot open array/model artifact: '+str(path))
    require(path.is_file(), 'Missing metadata file: '+str(path))
    return path


def sha(path):
    return hashlib.sha256(text_path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(text_path(path).read_text())


def remote(path):
    path = Path(path)
    if path.is_relative_to(PHASE):
        return str(REMOTE_PHASE/path.relative_to(PHASE))
    require(path.is_relative_to(REMOTE_PHASE), 'Remote phase path required')
    return str(path)


def descriptor(path):
    return {'path': remote(path), 'sha256': sha(path)}


def bound(record):
    require(isinstance(record, dict) and {'path', 'sha256'} <= set(record) <= {'path', 'sha256', 'bytes'},
            'Exact path/SHA descriptor with optional bytes required')
    require(isinstance(record['sha256'], str) and len(record['sha256']) == 64,
            'SHA256 required')
    path = text_path(record['path'])
    require(sha(path) == record['sha256'], 'Bound metadata changed: '+str(path))
    if 'bytes' in record:
        require(type(record['bytes']) is int and record['bytes'] >= 0 and path.stat().st_size == record['bytes'],
                'Descriptor byte length differs')
    return path


def write(path, value):
    path = confined(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_descriptors():
    r17 = REMOTE_PHASE/R17_REL
    modern = REMOTE_PHASE/MODERN_REL
    return [
        {'path': str(r17/'MANIFEST.json'), 'sha256': R17_MANIFEST_SHA},
        {'path': str(r17/'SEAL.json'), 'sha256': R17_SEAL_SHA},
        {'path': str(modern/'MANIFEST.json'), 'sha256': MODERN_MANIFEST_SHA},
        {'path': str(modern/'SEAL.json'), 'sha256': MODERN_SEAL_SHA}]


def verify_sources():
    """Hash sealed text/source payloads only; do not import scientific modules."""
    records = source_descriptors()
    for manifest_record, seal_record in zip(records[::2], records[1::2]):
        path = bound(manifest_record)
        seal = read(bound(seal_record))
        require(seal['manifest_sha256'] == manifest_record['sha256'], 'Seal/manifest mismatch')
        manifest = read(path)
        names = set()
        for row in manifest['payload']:
            rel = Path(row['path'])
            require(not rel.is_absolute() and '..' not in rel.parts and str(rel) not in names,
                    'Unsafe or repeated sealed payload')
            names.add(str(rel))
            payload = text_path(path.parent/rel)
            require(sha(payload) == row['sha256'] and payload.stat().st_size == row['bytes'],
                    'Sealed text/source payload changed: '+str(payload))
    require(sha(PHASE/R17_REL/'SOURCE_BINDINGS.json') == SOURCE_BINDING_SHA,
            'R17 source binding differs')
    require(sha(PHASE/R17_REL/'PROTOCOL.json') == PROTOCOL_SHA, 'R17 protocol differs')
    return records


def bindings():
    return ({'path': str(REMOTE_PHASE/R17_REL/'SOURCE_BINDINGS.json'), 'sha256': SOURCE_BINDING_SHA},
            {'path': str(REMOTE_PHASE/R17_REL/'PROTOCOL.json'), 'sha256': PROTOCOL_SHA})


def expected_attempts(contexts, anchor):
    rows = []
    for context in contexts:
        digest = object_hash(context)
        cell = context['graph']+'_seed'+str(context['seed'])
        for phase in ('qualify', 'warm', 'initialize', 'fit'):
            for arm in ARMS if phase in ('initialize', 'fit') else (None,):
                out = Path(anchor)/phase/cell
                if arm is not None:
                    out /= arm
                rows.append({'key': object_hash({'context_sha256': digest, 'phase': phase, 'arm': arm}),
                    'context_sha256': digest, 'phase': phase, 'arm': arm, 'output': str(out)})
    rows.sort(key=lambda row: row['key'])
    require(len(rows) == 72 and len({row['key'] for row in rows}) == 72 and
            len({row['output'] for row in rows}) == 72, 'Exactly72 unique attempts required')
    return rows
