"""Stdlib custody checks for the prospective public MolHIV readout."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent.resolve()
CONDITIONS = ('single', 'independent4', 'O', 'I', 'P', 'G')
SEEDS = (7101, 7203, 7307)
PUBLIC = 'portable_internal_be_public_interface_20261007_v2'
ALLOCATION = 'public_internal_be_allocation_controls_20261007_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def inside(phase, relative, exists=True):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Phase-relative custody required')
    path = (phase / rel).resolve(strict=exists)
    require(path != phase and path.is_relative_to(phase), 'Path leaves the authorized phase')
    return path


def bound(phase, row):
    require(isinstance(row, dict) and set(row) == {'path', 'sha256', 'bytes'}, 'Exact file binding required')
    path = inside(phase, row['path'])
    require(path.is_file() and type(row['bytes']) is int and path.stat().st_size == row['bytes']
            and sha(path) == row['sha256'], 'File custody/hash mismatch: ' + row['path'])
    return path


def binding(phase, path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path.relative_to(phase)), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify_source(root, digest):
    require(sha(root / 'MANIFEST.json') == digest, 'Sealed source manifest differs: ' + root.name)
    for row in read(root / 'MANIFEST.json')['files']:
        bound(root, row)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def release(path, digest):
    """No payload import/read until all source, whole18 and owner checks pass."""
    require(Path(path).resolve(strict=True).is_relative_to(PHASE), 'Root release must be saved in this phase')
    require(sha(path) == digest, 'Root readout release digest differs')
    cfg = read(path)
    require(cfg.get('schema') == 'internal-be-molhiv-readout-release-v1' and cfg.get('enabled') is True,
            'Readout is disabled; a separate explicit release is required')
    require(isinstance(cfg.get('event_id'), str) and cfg['event_id'], 'Separate reevaluation event ID required')
    verify_source(HERE, cfg['readout_source_manifest_sha256'])
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    for name, digest in pins['source_manifests'].items():
        verify_source(PHASE / name, digest)
    adoption = read(bound(PHASE, pins['adoption']))
    require(adoption['conditions'] == list(CONDITIONS) and adoption['paired_development_seeds'] == list(SEEDS)
            and adoption['constructor'] == 'be_init' and adoption['lambda'] == .5,
            'Exact frozen six-condition/three-seed adoption required')
    closure = read(bound(PHASE, cfg['engineering_closure']))
    require(closure.get('schema') == 'internal-be-molhiv-family-engineering-closure-v1'
            and closure.get('closed') is True and closure.get('task') == 'molhiv'
            and closure.get('adoption') == pins['adoption']
            and closure.get('source_manifests') == pins['source_manifests'], 'Exact public family closure required')
    rows = closure['cells']
    keys = [(row['condition'], row['seed']) for row in rows]
    require(len(keys) == 18 and len(set(keys)) == 18
            and set(keys) == {(c, s) for c in CONDITIONS for s in SEEDS}, 'Account for the entire frozen18 roster')
    family = inside(PHASE, closure['family_root'])
    terminal = read(bound(PHASE, cfg['terminal_evidence']))
    require(terminal.get('schema') == 'internal-be-molhiv-terminal-evidence-v1'
            and terminal.get('engineering_closure') == cfg['engineering_closure']
            and terminal.get('owner_and_children_terminal') is True
            and isinstance(terminal.get('observed_at_UTC'), str), 'Bound actual terminal evidence required')
    handles = terminal['handles']
    for handle in handles.values():
        require(type(handle.get('pid')) is int and handle['pid'] > 0
                and type(handle.get('start_ticks')) is int and handle['start_ticks'] > 0
                and type(handle.get('exit_code')) is int and handle.get('reaped') is True
                and handle.get('children_terminal') is True, 'Actual PID/ticks/exit/reap custody required')
    for row in rows:
        require(row['status'] in ('complete', 'failed', 'unlaunched', 'invalid'), 'Unknown engineering cell status')
        require(inside(PHASE, row['fit_output'], exists=row['status'] != 'unlaunched')
                == family / (row['condition'] + '_' + str(row['seed'])), 'Exact family cell output required')
        if row['status'] == 'unlaunched':
            require(row.get('owner_handle_id') is None, 'Unlaunched row cannot invent an owner')
        else:
            require(row.get('owner_handle_id') in handles, 'Attempted fit lacks actual terminal owner custody')
    work = cfg['authorized_work']
    require(work.get('collection') is True and work.get('rank_analysis') is True and work.get('cpu') is True,
            'Separately authorized CPU collection/analysis work required')
    require(work['device'] in ('cpu', 'cuda:0'), 'Explicit CPU or one visible CUDA device required')
    if work['device'] == 'cuda:0':
        require(work.get('gpu') is True and isinstance(work.get('gpu_uuid'), str)
                and work['gpu_uuid'].startswith('GPU-'), 'Separate physical GPU authorization required')
    output = inside(PHASE, cfg['output'], exists=False)
    require(not output.exists() and not output.is_relative_to(family)
            and not any(output.is_relative_to(PHASE / name) for name in pins['source_manifests'])
            and not output.is_relative_to(PHASE / adoption['frozen_role_projection']['root_relative_directory'])
            and not output.is_relative_to(PHASE / Path(pins['adoption']['path']).parent)
            and not output.is_relative_to(PHASE / Path(pins['handoff']['path']).parent)
            and not output.is_relative_to(HERE), 'Fresh readout output outside fit/source directories required')
    return cfg, closure, pins, adoption, output


def paired(values):
    """Three paired optimizer seeds, never molecules/pairs as model replicates."""
    if len(values) != 3 or any(x is None or not math.isfinite(x) for x in values):
        return {'available': False, 'reason': 'All three paired seeds are required', 'seed_values': values}
    mean = sum(values) / 3
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / 2)
    radius = 4.302652729911275 * sd / math.sqrt(3)  # df2 Student t .975.
    return {'available': True, 'n': 3, 'seed_values': values, 'mean': mean, 'sample_SD': sd,
            'exploratory_paired_t95_df2': [mean - radius, mean + radius],
            'unit': 'paired optimizer seed on the fixed scaffold development split'}
