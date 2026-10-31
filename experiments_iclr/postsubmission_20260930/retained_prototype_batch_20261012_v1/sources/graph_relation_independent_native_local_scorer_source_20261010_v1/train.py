"""Disabled one-cell native initializer control over the unchanged full trainer."""
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

import integration
from initializer import require

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
POLICIES = ('alphaF', 'relationJ')
SEEDS = (6101, 6203, 6307)
BASE = 'graph_relation_private_credit_source_20261008_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def phase_path(relative):
    value = Path(relative)
    require(not value.is_absolute() and '..' not in value.parts, 'Canonical phase-relative path')
    path = (PHASE / value).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Inside normal authorized phase')
    return path


def bound(row):
    path = phase_path(row['path'])
    require(path.is_file() and sha(path) == row['sha256'], 'Exact bound file')
    if 'bytes' in row:
        require(path.stat().st_size == row['bytes'], 'Exact recorded byte size')
    return path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def source_gate():
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    for row in read(HERE / 'SOURCE_MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve()
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Reviewed control source')
    for row in pins['source_files']:
        bound(row)
    for row in pins['source_manifests']:
        manifest = bound(row)
        for item in read(manifest)['files']:
            path = manifest.parent / item['path']
            require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Original sealed source unchanged')
    return pins


def identity(policy):
    require(policy in POLICIES, 'Only the two fixed policies')
    return 'graph_relation_independent_native_scorer__' + policy


def base_module():
    pins = source_gate()
    root = PHASE / BASE
    for name in ('permissions', 'replay_adapter'):
        path = root / (name + '.py'); prior = sys.modules.get(name)
        require(prior is None or Path(prior.__file__).resolve() == path.resolve(), 'Fresh process or exact original numerical helper')
        if prior is None:
            load(path, name)
    base = load(bound(pins['base_train']), '_independent_scorer_original_full_trainer')
    base.HERE = HERE
    base.POLICIES = POLICIES
    base.identity = identity
    adapted_sha = integration.install(base, bound(pins['base_train']))
    original_descriptor = base.descriptor

    def descriptor(session):
        value = original_descriptor(session)
        value.update(train_program_sha256=sha(__file__), initialization='Independent native local scorer rows only',
            initializer_source_sha256=sha(HERE / 'initializer.py'), integration_source_sha256=sha(HERE / 'integration.py'),
            adapted_make_session_AST_sha256=adapted_sha, native_local_scorer_start=integration.receipt(session),
            initializer_inclusive_seconds=session.independent_native_start['elapsed_seconds'],
            original_base_train_sha256=pins['base_train']['sha256'], original_base_manifest=pins['base_manifest'],
            graph_novelty_claimed=False)
        return value

    base.descriptor = descriptor
    return base


def dependencies():
    return base_module().dependencies()


def make_session(public, constructor, replay, pool, pins, policy, seed, device, polynormer):
    require(policy in POLICIES and seed in SEEDS, 'Fixed policy/seed')
    return base_module().make_session(public, constructor, replay, pool, pins, policy, seed, device, polynormer)


def run_complete(policy, seed, train, valid, output, polynormer, device='cpu', allocator_cap_bytes=None):
    require(policy in POLICIES and seed in SEEDS, 'One prospectively fixed cell')
    return base_module().run_complete(policy, seed, train, valid, output, polynormer, device, allocator_cap_bytes)


def reconstruct_selected(state, device='cpu'):
    """Fresh exact constructor, then original strict selected load; never resume."""
    saved = state['graph_relation_credit']; policy = saved['partition']['policy']; seed = state['run']['seed']
    require(policy in POLICIES and seed in SEEDS and state['run']['arm'] == identity(policy), 'Exact new selected-cell identity')
    base = base_module(); pins, public, driver, constructor, replay, pool = base.dependencies()
    session = base.make_session(public, constructor, replay, pool, pins, policy, seed, device, bound(pins['native']))
    current = base.descriptor(session)
    for key in ('train_program_sha256', 'new_method_source_manifest_sha256', 'partition', 'constructor_AST_sha256',
            'initializer_source_sha256', 'integration_source_sha256', 'adapted_make_session_AST_sha256',
            'native_local_scorer_start', 'original_base_train_sha256', 'original_base_manifest'):
        require(saved[key] == current[key], 'Exact selected source/partition/native-start contract')
    require(type(state['global']) is bool and state['config']['model'] == session.config['model'], 'Original selected mode/model recipe')
    session.model.load_state_dict(state['model'], strict=True)
    session.config = copy.deepcopy(state['config']); session.model.set_global(state['global']); session.model.eval()

    def refusal(*args, **kwargs):
        raise RuntimeError('Serving-only selected reconstruction; no training or optimizer/RNG resume')

    session.train_step = refusal
    return session


def admission(release, qualification=False, release_path=None):
    pins = source_gate(); runtime = pins['runtime']
    flags = ('enabled', 'root_qualification_authorized', 'source_review_approved') if qualification else (
        'enabled', 'root_execution_authorized', 'source_review_approved', 'native_qualification_approved')
    require(all(release.get(key) is True for key in flags), 'Disabled until separate actual root release')
    require(release['source_manifest_sha256'] == sha(HERE / 'SOURCE_MANIFEST.json')
        and socket.gethostname() == runtime['hostname'] and Path.cwd().resolve() == Path(runtime['repository']).resolve()
        and str(Path(sys.executable).absolute()) == runtime['python'] and os.environ.get('PYTHONPATH', '') == runtime['PYTHONPATH']
        and release['runtime_repository'] == runtime['repository'] and release['runtime_python'] == runtime['python']
        and release['physical_gpu_uuid'] in runtime['physical_gpu_inventory']
        and os.environ.get('CUDA_VISIBLE_DEVICES') == release['physical_gpu_uuid'], 'Exact source and normal77 runtime')
    require(all(importlib.metadata.version(name) == runtime[name] for name in (
        'torch', 'numpy', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'ogb')), 'Unchanged original providers')
    require(release['train'] == pins['author_train'] and release['valid'] == pins['author_valid']
        and release['polynormer'] == pins['native'] and release['owned_GPU_allocator_cap_bytes'] == 32 * 1024**3
        and all(release[key] is False for key in ('TEST_access', 'automatic_retry', 'comparative_opening_authorized', 'anchor_reuse_authorized')),
        'Original roles/source/cap, no outcome opening or anchor loading')
    supervision = read(bound(release['external_supervision']))
    entry = HERE / ('qualify.py' if qualification else 'train.py')
    require(supervision['enabled'] is True and supervision['root_owns_finite_launch_and_actual_costs'] is True
        and supervision['existing_run_fit_helper'] == pins['run_fit_helper']
        and supervision['existing_ownership_helper'] == pins['ownership_helper']
        and supervision['entry_program'] == dict(path=str(entry.relative_to(PHASE)), bytes=entry.stat().st_size, sha256=sha(entry))
        and supervision['argv_prefix'] == [runtime['python'], '-B', str(entry)]
        and supervision['release_argument_path'] == str(Path(release_path).resolve())
        and supervision['physical_gpu_uuid'] == release['physical_gpu_uuid']
        and 0 < supervision['active_seconds'] <= (3590 if qualification else 32390)
        and supervision['cleanup_seconds'] == 10 and supervision['hard_seconds'] == supervision['active_seconds'] + 10
        and supervision['owned_GPU_bytes'] == 32 * 1024**3
        and 0 < supervision['owned_RSS_bytes'] <= 32 * 1024**3
        and 0 < supervision['combined_log_bytes'] <= 8 * 1024**2,
        'Reuse existing finite owner helpers; exact admitted entry/route/limits, no new launcher')
    bound(pins['run_fit_helper']); bound(pins['ownership_helper'])
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True); args = parser.parse_args()
    require(sha(args.release) == args.release_sha256, 'Exact saved root cell release')
    release = read(args.release); pins = admission(release, release_path=args.release)
    require(release['schema'] == 'independent-native-local-scorer-cell-release-v1' and release['policy'] in POLICIES
        and release['seed'] in SEEDS and release['epochs'] == 1100 and release['risk_beta'] == .5
        and release['auxiliary'] is False and release['physical_gpu_uuid'] == pins['GPU_per_seed'][str(release['seed'])], 'Fixed six full-recipe cells')
    adoption = read(bound(release['conditional_adoption']))
    require(adoption['root_adopted'] is True and adoption['whole_relation12_closed'] is True
        and adoption['frozen_primary_failed'] is True and adoption['frozen_placement_failed'] is True,
        'Existing conditional native-initialization intervention adopted by root')
    qualified = read(bound(release['native_qualification']))
    require(qualified['complete'] is True and qualified['source_manifest_sha256'] == release['source_manifest_sha256']
        and qualified['policies'] == list(POLICIES) and qualified['real_complete_TRAIN_updates'] == 4
        and qualified['physical_gpu_uuid'] == release['physical_gpu_uuid'], 'Actual four-update same-GPU qualification')
    output = phase_path(release['output'])
    frozen = {HERE, bound(release['conditional_adoption']).parent,
        *(phase_path(Path(row['path']).parts[0]) for row in pins['source_files']),
        *(phase_path(Path(pins[key]['path']).parts[0]) for key in ('author_train', 'author_valid'))}
    require(not output.exists() and output.parent.is_dir()
        and all(not output.is_relative_to(root) for root in frozen), 'Fresh separate cell output')
    import torch
    torch.cuda.set_device(0)
    cap = release['owned_GPU_allocator_cap_bytes']; total = torch.cuda.get_device_properties(0).total_memory
    require(cap <= total, 'Original owned GPU allocator cap')
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    run_complete(release['policy'], release['seed'], bound(release['train']), bound(release['valid']), output,
        bound(release['polynormer']), 'cuda:0', cap)


if __name__ == '__main__':
    main()
