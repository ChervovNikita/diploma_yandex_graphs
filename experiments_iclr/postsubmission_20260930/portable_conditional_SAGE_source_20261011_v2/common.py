"""Local receipts and explicit permission; no numerical imports."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
SEEDS = (8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623)
ARMS = ('BASE', 'E4', 'N4', 'equipped_M1', 'equipped_I4', 'published_mean_SAGE_source_port_M1')


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def checked(item):
    require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'}, 'Exact local file receipt required')
    require(receipt(item['path']) == item, 'Local file differs: ' + item['path'])
    return Path(item['path'])


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); require(not path.exists(), 'Fresh output only; retain prior attempts')
    temporary = path.with_suffix(path.suffix + '.tmp')
    require(not temporary.exists(), 'Partial write retained')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def bundle():
    manifest = read(HERE / 'MANIFEST.json')
    for row in manifest['files']:
        require((HERE / row['path']).stat().st_size == row['bytes'] and sha(HERE / row['path']) == row['sha256'],
                'Public source bundle changed: ' + row['path'])
    return receipt(HERE / 'MANIFEST.json')


def runtime():
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch', 'torch-geometric', 'torch-scatter', 'torch-sparse')}
    require(sys.version_info[:2] == (3, 11), 'Pinned Python3.11 recipe required')
    require(versions['numpy'] == '1.26.4' and versions['torch'] == '2.1.2+cu118'
            and versions['torch-geometric'] == '2.7.0', 'Pinned NumPy/Torch/PyG recipe differs; alternate builds unverified')
    modules = {}
    for distribution, relative in (('numpy', 'numpy/__init__.py'), ('torch', 'torch/__init__.py'),
                                   ('torch-geometric', 'torch_geometric/__init__.py'), ('scipy', 'scipy/__init__.py')):
        modules[distribution] = receipt(importlib.metadata.distribution(distribution).locate_file(relative))
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'package_versions': versions, 'package_entry_sources': modules, 'numeric_qualification_claim': False}


def local_validate(cfg, **context):
    require(cfg.get('permission') == 'LOCAL_RUN_PERMITTED', 'Explicit local command permission required')
    source_bundle = bundle()
    for item in cfg.get('file_bindings', {}).values(): checked(item)
    prepared = read(checked(cfg['preparation_receipt']))
    require(prepared.get('complete') is True and prepared.get('dataset') == 'CoauthorPhysics',
            'Actual canonical preparation must close first')
    actual_cfg = canonical(cfg)
    bound = context.get('source_receipt', {}).get('config_sha256')
    require(bound is None or bound == actual_cfg, 'Engine and local canonical config differ')
    return {'status': 'LOCAL_VALIDATED', 'config_sha256': actual_cfg, 'runtime': runtime(),
            'source_bundle': source_bundle, 'preparation': cfg['preparation_receipt'],
            'action': context.get('action'), 'files_checked': len(cfg.get('file_bindings', {})),
            'permission_only_not_author_PASS_or_portability_certificate': True}


def closed_states(work):
    """Hash all90 states from both complete JSON closures; never deserialize."""
    work = Path(work)
    core_path, author_path = work / 'core_acquisition_v1/COMPLETE_FAMILY.json', work / 'author_acquisition_v1/COMPLETE_AUTHOR_CONTEXT.json'
    core, author = read(core_path), read(author_path)
    require(core.get('complete') is True and core.get('full_same_runtime_stage_complete') is True
            and core.get('groups') == 50 and core.get('fit_units') == 80
            and len(core.get('results', [])) == 50
            and {(r['seed'], r['arm']) for r in core['results']} == {(s, a) for s in SEEDS for a in ARMS[:5]}
            and author.get('complete') is True and author.get('banks') == 10 and author.get('fits') == 10
            and len(author.get('records', [])) == 10 and {r['seed'] for r in author['records']} == set(SEEDS),
            'Complete50/80 core and10/10 reference required; no survivor subset')
    states = [f['selected_state_receipt'] for r in core['results'] for f in r['fits']]
    states += [f['selected_state_receipt'] for r in author['records'] for f in r['fits']]
    require(len(states) == 90 and len({i['path'] for i in states}) == 90, 'Exact90 separate selected-state files')
    for item in states: checked(item)
    return core, author, states, {'core': receipt(core_path), 'author': receipt(author_path)}
