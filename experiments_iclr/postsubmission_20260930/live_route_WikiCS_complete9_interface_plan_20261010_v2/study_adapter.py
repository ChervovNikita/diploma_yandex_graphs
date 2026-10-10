"""Disabled full WikiCS cell adapter over the immutable public trainer.

No numerical import, constructor, fit or launcher runs on import. Root must
bind an enabled cell release, actual qualification/custody, criteria and the
existing finite owner before the original full driver may be called.
"""
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
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
KINDS = ('baseline', 'exchange', 'separable')
SEEDS = (6101, 6203, 6307)
LIVE_MANIFEST_SHA = '302a1dcf2276a7a8091c3429ab01cf7a034db258fa654647f0b15e5de50b4d6a'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def phase_path(relative):
    value = Path(relative)
    require(not value.is_absolute() and '..' not in value.parts, 'Canonical phase-relative path')
    path = (PHASE / value).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Inside the authorized phase')
    return path


def bound(row):
    path = phase_path(row['path'])
    require(path.is_file() and sha(path) == row['sha256'], 'Exact bound source or role')
    require('bytes' not in row or path.stat().st_size == row['bytes'], 'Exact recorded source size')
    return path


def sealed(row):
    manifest = bound(row)
    for item in read(manifest)['files']:
        path = (manifest.parent / item['path']).resolve()
        require(path.is_relative_to(manifest.parent) and path.stat().st_size == item['bytes']
                and sha(path) == item['sha256'], 'Immutable source payload')
    return manifest.parent


def source_gate():
    sealed({'path': str((HERE / 'MANIFEST.json').relative_to(PHASE)), 'sha256': sha(HERE / 'MANIFEST.json')})
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    require(pins['live_manifest']['sha256'] == LIVE_MANIFEST_SHA, 'Immutable V3 scientific identity')
    sealed(pins['live_manifest']); sealed(pins['public_manifest']); sealed(pins['prior_interface_manifest'])
    for row in pins['source_files']:
        bound(row)
    return pins


def identity(kind):
    require(kind in KINDS, 'Only the frozen three live-route arms')
    return 'live_route_native_init_v3__' + kind


def admission(release_path, release_sha256):
    require(sha(release_path) == release_sha256, 'Exact separately frozen root cell release')
    release = read(release_path); pins = source_gate(); plan = read(HERE / 'STUDY_PLAN_DISABLED.json')
    require(all(release.get(key) is True for key in ('enabled', 'root_execution_authorized',
        'source_review_approved', 'native_qualification_approved', 'root_numeric_criteria_frozen',
        'root_data_authority_approved', 'existing_owner_binding_approved')), 'Disabled pending explicit root bindings')
    require(release['schema'] == 'live-route-complete9-cell-release-v1'
        and release['adapter_manifest_sha256'] == sha(HERE / 'MANIFEST.json')
        and release['scientific_source_manifest'] == pins['live_manifest']
        and {'kind': release['kind'], 'seed': release['seed']} in plan['roster']
        and release['epochs'] == 1100 and release['TEST_access'] is False
        and release['automatic_retry'] is False and release['comparative_opening_authorized'] is False
        and release['anchor_reuse_authorized'] is False, 'Exact frozen nine-fit scope')
    runtime = pins['runtime']
    require(socket.gethostname() == runtime['hostname'] and Path.cwd().resolve() == Path(runtime['repository']).resolve()
        and str(Path(sys.executable).absolute()) == runtime['python'] and os.environ.get('PYTHONPATH', '') == runtime['PYTHONPATH']
        and release['physical_gpu_uuid'] in runtime['physical_gpu_inventory']
        and os.environ.get('CUDA_VISIBLE_DEVICES') == release['physical_gpu_uuid']
        and not os.environ.get('PYTHONHOME'), 'Exact normal77 host/runtime/physical GPU; no overlay')
    require(all(importlib.metadata.version(name) == runtime[name] for name in
        ('torch', 'numpy', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'ogb')), 'Original provider versions')
    require(release['train'] == pins['author_train'] and release['valid'] == pins['author_valid']
        and release['polynormer'] == pins['native'], 'Original full TRAIN/VALID/native source roles')
    authority = read(bound(release['data_authority']))
    require(authority['root_authorized_full_TRAIN_VALID'] is True and authority['TEST_access'] is False
        and authority['TRAIN_role_file'] == pins['author_train'] and authority['VALID_role_file'] == pins['author_valid']
        and authority['complete_TRAIN_objects'] == 580 and authority['complete_VALID_objects'] == 5274,
        'Actual separately bound full-role authority; no inference from public array checks')
    criteria = read(bound(release['quality_criteria']))
    require(criteria['roster'] == plan['roster'] and criteria['frozen_before_first_fit'] is True
        and criteria['accuracy_primary'] is True and criteria['whole9_before_comparison'] is True
        and criteria['all_seeds_and_failures_retained'] is True
        and all(criteria['numeric_criteria'].get(key) is not None for key in plan['required_root_numeric_criteria']),
        'Root numeric criteria frozen for accuracy, competence, NLL and harms before fits')
    qualification_root = sealed(release['qualification_source_manifest'])
    qualified = read(bound(release['qualification_report'])); qr = read(bound(release['qualification_release']))
    require(qualified['schema'] == release['qualification_report_schema']
        and release['qualification_report_schema'].startswith('live-route-native-init-qualification-') and qualified['complete'] is True
        and qualified['source_manifest'] == pins['live_manifest'] and qualified['quality_scoring'] is False
        and qualified['VALID_values_access'] is False and qualified['TEST_access'] is False
        and [row['kind'] for row in qualified['rows']] == list(KINDS)
        and all(row['complete'] is True for row in qualified['rows'])
        and qualified['counts']['TRAIN_update_completions'] == qualified['counts']['Adam_completions'] == 6
        and qualified['counts']['backward_completions'] == 12
        and qualified['aggregate_actual_native_work']['joint_forward_completions'] == 42
        and qr['schema'] == release['qualification_release_schema']
        and release['qualification_release_schema'].startswith('live-route-native-init-qualification-release-')
        and qr['enabled'] is qr['root_execution_authorized'] is qr['source_review_approved'] is True
        and qr['qualification_manifest_sha256'] == release['qualification_source_manifest']['sha256']
        and qr['qualifier_program_sha256'] == sha(qualification_root / 'qualify.py')
        and qr['scientific_source_manifest'] == pins['live_manifest']
        and qr['physical_gpu_uuid'] == release['physical_gpu_uuid'], 'Whole actual source-bound qualification on the declared GPU')
    terminal = read(bound(release['qualification_owner_exit'])); physical = read(bound(release['qualification_physical_terminal']))
    require(terminal['exit_code'] == 0 and terminal['terminal_wait_observed'] is True
        and terminal['reason'] is None and terminal['signals_sent'] == []
        and terminal['job_sha256'] == release['qualification_release']['sha256']
        and physical['physical_gpu_uuid'] == release['physical_gpu_uuid']
        and physical['child_PID'] == terminal['raw_identity_observation']['PID']
        and physical['child_start_ticks'] == terminal['raw_identity_observation']['start_ticks']
        and physical['owned_PID_absent'] is True and physical['owned_PID_no_CUDA_rows'] is True,
        'Actual qualification owner wait and physical absence custody')
    supervision = read(bound(release['external_supervision'])); limits = supervision['resource_limits']
    require(supervision['enabled'] is True and supervision['root_owns_finite_launch_and_actual_costs'] is True
        and supervision['existing_run_fit_helper'] == pins['run_fit_helper']
        and supervision['existing_ownership_helper'] == pins['ownership_helper']
        and supervision['physical_gpu_uuid'] == release['physical_gpu_uuid']
        and supervision['entry_program'] == dict(path=str(Path(__file__).resolve().relative_to(PHASE)),
            bytes=Path(__file__).stat().st_size, sha256=sha(__file__))
        and supervision['argv'] == [runtime['python'], '-B', str(Path(__file__).resolve()), '--release',
            str(Path(release_path).resolve()), '--release-sha256', release_sha256]
        and 0 < supervision['active_seconds'] < supervision['hard_seconds']
        and supervision['hard_seconds'] == supervision['active_seconds'] + supervision['cleanup_seconds']
        and supervision['cleanup_seconds'] > 0 and release['owned_GPU_allocator_cap_bytes'] == 64 * 1024**3
        and limits['owned_tree_GPU_memory_cap_bytes'] == 68 * 1024**3
        and limits['minimum_fresh_GPU_free_bytes'] >= 72 * 1024**3
        and 0 < limits['owned_tree_RSS_cap_bytes'] and 0 < limits['own_fit_output_cap_bytes']
        and 0 < limits['combined_child_log_cap_bytes'] <= 8 * 1024**2,
        'Exact existing finite owner binding; full-fit wall/RSS/output bounds supplied by root')
    return release, pins


def module(name, path, package=False):
    prior = sys.modules.get(name)
    require(prior is None or Path(prior.__file__).resolve() == path.resolve(), 'Exact cached module or fresh process')
    if prior is not None:
        return prior
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def dependencies(pins):
    public_root = sealed(pins['public_manifest'])
    public = module('portable', bound(pins['public_portable']))
    sys.path.insert(0, str(public_root))
    prior = sys.modules.get('data_interface')
    require(prior is None or Path(prior.__file__).resolve() == bound(pins['data_interface']), 'Exact original data reader')
    driver = module('_live_route_original_full_driver', bound(pins['public_driver']))
    live = module('_live_route_source_v3', sealed(pins['live_manifest']) / '__init__.py', package=True)
    return public, driver, live


def recipe(public, kind):
    value = copy.deepcopy(public.recipe('wikics')); value['arms'] = [identity(kind)]
    value['contrastive'].update(alignment_weight=0., residual_weight=0.)
    value['live_route_study'] = dict(kind=kind, source_manifest_sha256=LIVE_MANIFEST_SHA,
        adapter_manifest_sha256=sha(HERE / 'MANIFEST.json'), exploratory=True,
        initializer_confirmation=False, novelty_or_superiority_cleared=False)
    return value


def expected_work(kind):
    return dict(joint_forward_attempts=3300, joint_forward_completions=3300,
        stem_completions=13200, global_completions=12000, local_head_completions=1200,
        global_head_completions=12000, block_completions=0 if kind == 'baseline' else 3300,
        local_conv_completions=[13200]*7, backward_attempts=2200, backward_completions=2200)


def run_complete(release_path, release_sha256):
    """One full original1100-epoch cell; no shortened loop, replay or owner."""
    release, pins = admission(release_path, release_sha256)
    kind, seed = release['kind'], release['seed']; output = phase_path(release['output'])
    frozen = (HERE, sealed(pins['live_manifest']), sealed(pins['public_manifest']),
        bound(pins['author_train']).parent, bound(pins['author_valid']).parent,
        *(phase_path(Path(row['path']).parts[0]) for row in pins['source_files']))
    require(not output.exists() and output.parent.is_dir() and all(not output.is_relative_to(root) for root in frozen),
        'Fresh separate fit output; no historical or scientific source overwrite')
    started = time.monotonic(); box = {}; old_path = list(sys.path)
    public, driver, live = dependencies(pins); facade = live.adapted_public(public, kind)
    original = {name: getattr(driver, name) for name in ('Session', 'recipe', 'json_write', 'joint_snapshot')}

    def factory(task, arm, chosen_seed, device, native, ncn_model, ncn_utils):
        require(task == 'wikics' and arm == identity(kind) and chosen_seed == seed
            and ncn_model is None and ncn_utils is None, 'One frozen native full cell')
        import torch
        torch.cuda.set_device(0)
        cap = release['owned_GPU_allocator_cap_bytes']; total = torch.cuda.get_device_properties(0).total_memory
        require(device == 'cuda:0' and cap <= total, 'Declared full-fit CUDA allocator cap')
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        session = facade.Session(task, 'be_unit', seed, device, native)
        require(session.steps == 0 and len(session.optimizers) == 1, 'Fresh sourceV3 pre-Adam installation')
        session.config = recipe(public, kind); box['session'] = session
        return session

    def annotate(value):
        session = box.get('session')
        return dict(value, live_route_study=recipe(public, kind)['live_route_study'],
            root_cell_release_sha256=release_sha256, live_route=live.descriptor(session) if session else None,
            actual_live_work=copy.deepcopy(session.live_work) if session else None,
            actual_last_update_events=list(session.live_last_update_events) if session else None,
            adapter_inclusive_seconds=time.monotonic()-started)

    def snapshot(session, epoch, metric, per, run):
        value = original['joint_snapshot'](session, epoch, metric, per, annotate(run))
        return annotate(value)

    def write(path, value):
        if Path(path).name in ('RUN.json', 'PROGRESS.json', 'COMPLETE.json', 'FAILURE.json'):
            if Path(path).name == 'COMPLETE.json':
                session = box['session']
                require(value['epochs'] == value['steps'] == session.steps == 1100
                    and session.live_work == expected_work(kind), 'Actual full horizon and live work')
            value = annotate(value)
        original['json_write'](path, value)

    driver.Session, driver.recipe, driver.joint_snapshot, driver.json_write = factory, lambda task: (
        recipe(public, kind) if task == 'wikics' else (_ for _ in ()).throw(ValueError('WikiCS only'))), snapshot, write
    previous = sys.argv
    try:
        sys.argv = [str(bound(pins['public_driver'])), '--task', 'wikics', '--arm', identity(kind), '--seed', str(seed),
            '--device', 'cuda:0', '--train', str(bound(release['train'])), '--valid', str(bound(release['valid'])),
            '--output', str(output), '--polynormer', str(bound(release['polynormer']))]
        driver.main()
    finally:
        sys.argv = previous; sys.path[:] = old_path
        for name, value in original.items():
            setattr(driver, name, value)
    return dict(output=str(output), kind=kind, seed=seed, TEST_scoring=False)


def reconstruct_selected(state, device='cpu'):
    """Fresh same-source model, strict selected load and serving only."""
    pins = source_gate(); meta = state['live_route_study']; kind = meta['kind']; seed = state['run']['seed']
    require(kind in KINDS and seed in SEEDS and state['run']['arm'] == identity(kind)
        and state['checkpoint_kind'] == 'strict-first-maximum complete VALID joint snapshot', 'Exact frozen selected cell')
    old_path = list(sys.path)
    try:
        public, _driver, live = dependencies(pins)
        require(meta == recipe(public, kind)['live_route_study'] and state['config'] == recipe(public, kind)
            and type(state['global']) is bool, 'Exact source/recipe and selected local/global mode')
        session = live.adapted_public(public, kind).Session('wikics', 'be_unit', seed, device, bound(pins['native']))
        require(state['live_route'] == live.descriptor(session), 'Exact selected constructor and V3 live source identity')
        session.model.load_state_dict(state['model'], strict=True)
        session.config = copy.deepcopy(state['config']); session.model.set_global(state['global']); session.model.eval()
        def refusal(*args, **kwargs):
            raise RuntimeError('Serving-only selected reconstruction; no training or optimizer/RNG resume')
        session.train_step = session.restore_training_state = session.save_training_state = refusal
        return session
    finally:
        sys.path[:] = old_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); run_complete(args.release, args.release_sha256)
