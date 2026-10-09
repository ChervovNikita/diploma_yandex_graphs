"""Inactive source/release checks and reuse of birth-bound stdlib custody helpers."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ASSIGNMENTS = {'allocation_a100': 6101, 'gpu77_a998': 6203, 'gpu77_8ced': 6307}
OPERATORS = ('native_tied', 'active_reversible_exp', 'pre_sigmoid_split', 'full_qk')
KINDS = ('single', 'be_init', 'independent4')


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream: stream.write(content)
    else:
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(content)
        os.replace(temp, path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def inside(relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), 'Relative normal phase path')
    path = (PHASE / relative).resolve()
    require(path.is_relative_to(PHASE) and path != PHASE, 'Confined normal repository path')
    return path


def bound(row):
    path = inside(row['path']).resolve(strict=True)
    require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed bound file: ' + row['path'])
    return path


def sealed(root, digest):
    require(sha(root / 'MANIFEST.json') == digest == read(root / 'SEAL.json')['manifest_sha256'], 'Changed source seal')
    for row in read(root / 'MANIFEST.json')['files']:
        path = (root / row['path']).resolve(strict=True)
        require(path.is_relative_to(root) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed sealed payload')


def sources():
    seal = read(HERE / 'SEAL.json')
    require(seal['source_only'] is True and seal['execution_enabled'] is False, 'Inactive source packet')
    sealed(HERE, seal['manifest_sha256'])
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    for key in ('routing', 'qualifier'):
        sealed(PHASE / pins[key]['directory'], pins[key]['manifest_sha256'])
    custody = module(bound(pins['custody_program']), '_lane_original_custody_stdlib')
    routing = module(PHASE / pins['routing']['directory'] / 'route_adapter.py', 'route_adapter')
    return pins, custody, routing


def config(path, digest, authorized=False):
    require(authorized is True and sha(path) == digest, 'Exact separate enabled root release')
    pins, custody, routing = sources()
    cfg = read(path)
    template = read(HERE / 'ROOT_RELEASE_TEMPLATE_DISABLED.json')
    require(set(cfg) == set(template) and cfg['schema'] == template['schema'] and cfg['enabled'] is True
            and cfg['mode'] in ('qualify', 'seed_block') and cfg['route_id'] in ASSIGNMENTS,
            'Explicit host lane and bounded execution mode')
    require(cfg['owner_manifest_sha256'] == read(HERE / 'SEAL.json')['manifest_sha256'], 'Exact lane source')
    route, route_pins, manifest = routing.sources(cfg['route_id'])
    activation, output = inside(cfg['activation_relative']), inside(cfg['output_relative'])
    require(activation.is_dir() and Path(path).resolve(strict=True).parent == activation
            and not output.is_relative_to(HERE), 'Separate root activation/output paths')
    commit = cfg['execution_source_commit']
    require(isinstance(commit, str) and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'Actual source commit')
    subprocess.run(['git', 'cat-file', '-e', commit + '^{commit}'], cwd=route['repository'], check=True, timeout=10)
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=route['repository'], check=True, timeout=10)
    for directory, expected in ((HERE.name, cfg['owner_manifest_sha256']), (pins['routing']['directory'], manifest),
                                (pins['qualifier']['directory'], pins['qualifier']['manifest_sha256'])):
        data = subprocess.check_output(['git', 'show', commit + ':' + pins['repository_phase_relative'] + '/' + directory + '/MANIFEST.json'],
                                       cwd=route['repository'], timeout=10)
        require(hashlib.sha256(data).hexdigest() == expected, 'Sealed source present in execution commit')
    resource = cfg['resources']
    require(set(resource) == set(template['resources']) and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in resource.values())
            and resource['cleanup_seconds'] <= 15, 'Finite root resource and own-group cleanup bounds')
    count = 1 if cfg['mode'] == 'qualify' else 12
    require(resource['owner_seconds'] >= resource['startup_seconds'] + count * sum(resource[k] for k in
            ('admission_seconds', 'active_seconds', 'cleanup_seconds', 'terminal_seconds')), 'Complete bounded lane phase reserves')
    require(Path(route['python']).is_file() and all(Path(p).is_dir() for p in route['PYTHONPATH']), 'Existing declared native runtime')
    cells = []
    if cfg['mode'] == 'seed_block':
        plan_path = bound(cfg['plan'])
        plan = read(plan_path)
        queue = module(PHASE / pins['routing']['directory'] / 'queue_plan.py', '_lane_frozen_queue')
        require(plan['enabled'] is True and plan['execution_source_commit'] == commit
                and plan['route_adapter_manifest_sha256'] == manifest and plan['required_group_endpoints'] == 36
                and plan['lane_owner_manifest_sha256'] == cfg['owner_manifest_sha256']
                and plan['seeds'] == list(ASSIGNMENTS.values()) and plan['route_ids'] == list(ASSIGNMENTS)
                and plan['operators'] == list(OPERATORS) and plan['kinds'] == list(KINDS)
                and plan['GPU_assignment'][cfg['route_id']] == route['GPU_uuid']
                and plan['role_hashes'] == {k: v['sha256'] for k, v in route_pins['roles'].items()}
                and plan['root_resource_limits'][cfg['route_id']] == resource
                and plan['qualification_adoptions'][cfg['route_id']] == cfg['qualification_adoption']
                and output == inside(plan['fresh_output_relative']) / cfg['route_id'], 'One fixed externally hashed full36 plan')
        cells = queue.freeze_queues(plan['seeds'], plan['route_ids'])[cfg['route_id']]
    else:
        require(cfg['plan'] is None and cfg['qualification_adoption'] is None, 'Engineering-only fresh qualification mode')
    return cfg, pins, custody, routing, route, route_pins, cells


def absence(custody, identities, groups):
    require(identities and all(item and set(custody.CUSTODY).issubset(item) for item in identities), 'Actual saved process identities')
    require(all(not custody.same(custody.identity(item['pid']), item) for item in identities)
            and all(not custody.members(item) for item in groups)
            and not ({item['pid'] for item in identities} & {pid for pid, _ in custody.gpu_rows()}),
            'Actual saved parent/child/group/CUDA absence')


def qualification(cfg, pins, custody, route):
    adoption = cfg['qualification_adoption']
    require(adoption['qualified'] is True and adoption['route_id'] == cfg['route_id']
            and adoption['physical_GPU_uuid'] == route['GPU_uuid'], 'Successful qualification for this exact lane')
    result = read(bound(adoption['report']))
    require(result['complete'] is True and result['scientific_fit'] is False and result['completed_updates'] == 24
            and result['completed_member_view_forwards'] == result['completed_member_view_backwards'] == 144
            and result['completed_Adam_steps'] == 48 and result['identity_member_forwards'] == 144
            and result['VALID_metrics'] == result['VALID_forwards'] == 0 and result['TEST_access'] is False
            and len(result['cells']) == 12 and all(c['passed'] is True for c in result['cells'])
            and result['root_release']['qualifier_manifest_sha256'] == pins['qualifier']['manifest_sha256']
            and result['root_release']['qualifier_program_sha256'] == pins['qualifier']['program_sha256']
            and result['root_release']['identity_absolute_tolerance'] == 2e-5
            and result['root_release']['identity_reference'] == 'same_current_bank_native_class_forward'
            and result['root_release']['adapter_manifest_sha256'] == pins['math_manifest_sha256']
            and result['root_release']['role_files'] == read(PHASE / pins['routing']['directory'] / 'SOURCE_BINDINGS.json')['roles']
            and result['runtime']['hostname'] == route['hostname'] and result['runtime']['GPU_uuid'] == route['GPU_uuid']
            and result['runtime']['python'] == route['python'], 'Unchanged same-current-bank V2 engineering qualification')
    absence(custody, adoption['owned_identities'], adoption['owned_groups'])
    return adoption
