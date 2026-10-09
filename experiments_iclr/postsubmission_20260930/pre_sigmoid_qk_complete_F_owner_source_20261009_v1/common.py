"""Stdlib source/release custody for an inactive serial complete36 queue."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
OPERATORS = ('native_tied', 'active_reversible_exp', 'pre_sigmoid_split', 'full_qk')
KINDS = ('single', 'be_init', 'independent4')
CLEANUP = 15


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(data)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(data)
        os.replace(temporary, path)


def bound(root, row):
    path = (root / row['path']).resolve(strict=True)
    require(path.is_relative_to(root) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
            'Changed exact file binding: ' + row['path'])
    return path


def sealed(root, digest):
    require(sha(root / 'MANIFEST.json') == digest and read(root / 'SEAL.json')['manifest_sha256'] == digest,
            'Changed exact source seal')
    for row in read(root / 'MANIFEST.json')['files']:
        bound(root, row)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def sources():
    seal = read(HERE / 'SEAL.json')
    require(seal['source_only'] is True and seal['execution_enabled'] is False
            and seal['publication_enabled'] is False, 'Exact inactive complete36 source')
    sealed(HERE, seal['manifest_sha256'])
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    sealed(PHASE / pins['adapter_directory'], pins['adapter_manifest_sha256'])
    require(sha(PHASE / pins['public_directory'] / 'MANIFEST.json') == pins['public_manifest_sha256'], 'Exact original public manifest')
    for row in read(PHASE / pins['public_directory'] / 'MANIFEST.json')['files']:
        bound(PHASE / pins['public_directory'], row)
    for row in pins['source_files']:
        bound(PHASE, row)
    return pins, seal['manifest_sha256']


def release_config(path, digest, authorized=False):
    require(authorized is True and sha(path) == digest, 'Separate exact enabled root release required')
    cfg, template = read(path), read(HERE / 'ROOT_RELEASE_TEMPLATE_DISABLED.json')
    variable = set(template['root_variable_fields'])
    require(set(cfg) == set(template) and all(cfg[k] == template[k] for k in template if k not in variable)
            and all(cfg[k] is True for k in ('enabled', 'publication_approved', 'source_review_approved',
                                           'later_execution_authorized', 'scientific_execution_authorized')),
            'Inactive until root fixes qualification/source/seeds/resources before launch')
    pins, manifest = sources()
    require(cfg['owner_manifest_sha256'] == manifest and isinstance(cfg['execution_source_commit'], str)
            and len(cfg['execution_source_commit']) == 40
            and all(c in '0123456789abcdef' for c in cfg['execution_source_commit']), 'Exact committed complete36 owner')
    seeds = cfg['seeds']
    require(isinstance(seeds, list) and len(seeds) == len(set(seeds)) == 3
            and all(type(s) is int and 0 <= s < 2**63-1000000 for s in seeds), 'Exactly three prospectively fixed paired seeds')
    for key in ('startup_seconds', 'admission_seconds', 'cell_active_seconds', 'terminal_seconds',
                'family_seconds', 'max_owned_GPU_bytes', 'max_owned_RSS_bytes', 'minimum_fresh_GPU_bytes'):
        require(type(cfg[key]) is int and cfg[key] > 0, 'Positive root-admitted finite limit: ' + key)
    require(cfg['family_seconds'] >= cfg['startup_seconds'] + 36 *
            (cfg['admission_seconds'] + cfg['cell_active_seconds'] + CLEANUP + cfg['terminal_seconds']),
            'Full36 envelope; no smaller substituted comparison')
    runtime = pins['runtime']
    require(HERE.parent == Path(runtime['phase']).resolve(strict=True) and socket.gethostname() == runtime['hostname']
            and Path(path).resolve(strict=True).parent == PHASE / cfg['activation_relative'], 'Exact normal-host activation route')
    adoption = read(bound(PHASE, cfg['qualification_adoption']))
    require(adoption['schema'] == 'pre-sigmoid-qk-full-input-qualification-adoption-v1'
            and adoption['qualified'] is True and adoption['adapter_manifest_sha256'] == pins['adapter_manifest_sha256']
            and adoption['local_global_full_input'] is True and adoption['development_scores_computed'] is False
            and adoption['TEST_access'] is False and adoption['scientific_fit'] is False,
            'Successful root-adopted unscored full-input local/global qualification')
    bound(PHASE, adoption['qualification_result'])  # Exact evidence bytes; no quality/value opening.
    return cfg, pins, PHASE


def source_identity(pins, manifest):
    return dict(owner_manifest_sha256=manifest, adapter_manifest_sha256=pins['adapter_manifest_sha256'],
                original_driver_sha256=pins['original_driver']['sha256'], cell_program_sha256=sha(HERE / 'cell.py'))


def queue(cfg):
    return [dict(seed=s, operator=o, kind=k, key='seed' + str(s) + '/' + k + '/' + o)
            for s in cfg['seeds'] for o in OPERATORS for k in KINDS]
