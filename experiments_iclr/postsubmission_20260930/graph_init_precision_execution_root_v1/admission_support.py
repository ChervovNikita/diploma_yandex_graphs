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
R17_REL = 'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v3_precision'
MODERN_REL = 'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3'
R17_MANIFEST_SHA = '5e8250d9fbe219593bb2aa074665e30cc5686d6ed227fc549f83809dcf89f830'
R17_SEAL_SHA = '457cf3bd3275cad315e3d57b5036683577929369f03ea346a718b4f8ffad10f2'
MODERN_MANIFEST_SHA = 'e2fd767c4d08e93ca7ec9c9c526440c4cc5b8f4e7d8bb345de90308e9c57302b'
MODERN_SEAL_SHA = '9589bd1a41ec59fc9259d72c833c5e550e6887196eb773b86c75c6d03b67645d'
SOURCE_BINDING_SHA = 'f6d8ff959204ccc793212b0ff0560213f08de7d92b74e162441888058262bc48'
PROTOCOL_SHA = '91f20afb455c234ce113a5e3d3035955bc42b412ff7f389e5f511c7de47de446'
PREPARATION_SHA = '376e8b779d8d71cf21785a62b8543ecadc9c09c9db8174dc230b55b8d0dbd253'
ARMS = ('graph', 'common_only', 'random_tangent', 'topology_permuted', 'warm_copy')
CELLS = (('Squirrel', 'polyformer_mono'), ('Photo', 'polynormer_r'))
SEEDS = (17, 29, 43)
TEXT_EXTENSIONS = {'.json', '.py', '.md', '.csv', '.txt', '.sh', '.toml', '.yaml', '.yml',
                   '.diff', '.patch', '.log', '.jsonl'}
DISCLOSURE = 'This precision qualification amendment was designed after the known Photo17 v2 finite-difference failure and an exactly replayed arithmetic diagnostic. Photo and Squirrel families were previously exposed; this remains exploratory and outcome-aware. independent_of_stage1_outcomes describes source-pack custody only: no reuse, binding or selection of Stage1 fitted outputs or labels as this study evidence. It does not describe independence of idea/cohort choice or an unseen confirmatory cohort. The failed v2 registry, all old attempts and all diagnostic lineage remain preserved. No old checkpoint, phase output, qualification or optimizer history is inherited. All six cold and every actual-warm state require new v3 qualification with retained original FP32 diagnostics and unchanged thresholds.'


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
