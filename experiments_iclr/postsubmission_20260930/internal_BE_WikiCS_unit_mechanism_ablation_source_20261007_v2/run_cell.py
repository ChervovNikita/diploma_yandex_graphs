"""Disabled fixed WikiCS unit ablation; VJP recomputation and unchanged full loop."""
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from recompute import MODE, install

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
CONDITIONS = {'plain': (0., 0.), 'alignment_only': (.05, 0.),
              'residual_only': (0., .05), 'combined': (.05, .05)}
SEEDS = (6101, 6203, 6307)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inside(relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Phase-relative file/path required')
    path = (PHASE / rel).resolve(); require(path != PHASE and path.is_relative_to(PHASE), 'Project phase custody required')
    return path


def bound(row):
    path = inside(row['path']); require(path.is_file() and sha(path) == row['sha256'], 'Exact source/input/evidence binding changed')
    return path


def verify(row):
    path = bound(row)
    for item in read(path)['files']:
        file = path.parent / item['path']; require(sha(file) == item['sha256'] and file.stat().st_size == item['bytes'], 'Sealed dependency bytes changed')
    return path.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


class ObjectivesFacade:
    """Original functions; no new loss or extra forward/backward/label opportunity."""
    def __init__(self, original, condition):
        self.original, self.weights = original, CONDITIONS[condition]
        self.own_supervision = original.own_supervision
        self.alignment_source_calls = self.residual_source_calls = 0

    def alignment_loss(self, a, b, labels, task, temperature=.2, identities=None):
        require(task == 'wikics' and temperature == .2, 'Fixed WikiCS supervised alignment')
        if self.weights[0] == 0:
            return a.sum() * 0
        self.alignment_source_calls += 1
        return self.original.alignment_loss(a, b, labels, task, temperature, identities)

    def residual_member_contrast(self, a, b, labels, temperature=.2):
        require(temperature == .2, 'Fixed residual route contrast temperature')
        if self.weights[1] == 0:
            return a.sum() * 0
        self.residual_source_calls += 1
        return self.original.residual_member_contrast(a, b, labels, temperature)


def identity(condition):
    weights = CONDITIONS[condition]
    return dict(condition=condition, control_id='be_unit__mechanism_v2_' + condition, constructor='be_unit',
        alignment_weight=weights[0], residual_weight=weights[1], temperature=.2, max_objects=512,
        own_views=2, members=4, epochs=1100, private_initialization='unit factors', exploratory_attribution=True,
        underlying_session_arm='be_unit' if condition == 'plain' else 'be_unit_contrastive',
        objectives_facade='per-session shallow core copy; inactive original loss returns differentiable zero',
        ablation_adapter_sha256=sha(__file__), recomputation_sha256=sha(HERE / 'recompute.py'),
        execution_mode=MODE, training_member_forwards_per_update=16, shadow_member_forwards_per_update=8,
        replay_member_forwards_per_update=8, member_reverse_collections_per_update=8,
        output_cotangent_collections_per_update=1, original_stochastic_views=2,
        additional_training_member_forwards_per_update=8, optimizer_bank_updates_per_update=1,
        bitwise_author_parity_claimed=False, author_execution_equivalence_claimed=False)


def configured_recipe(public, condition):
    value = copy.deepcopy(public.recipe('wikics')); specification = identity(condition)
    value['arms'] = [specification['control_id']]
    value['metric'] = 'complete_development_union_split0_validation_and_stopping_accuracy'
    value['contrastive'].update(alignment_weight=specification['alignment_weight'], residual_weight=specification['residual_weight'])
    value['mechanism_ablation'] = specification; return value


def make_session(public, condition, seed, device, polynormer):
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    underlying = 'be_unit' if condition == 'plain' else 'be_unit_contrastive'
    session = public.Session('wikics', underlying, seed, device, polynormer)
    require(session.model.members == 4 and not session.model.independent, 'Original shared unit-factor model')
    require(str(session.torch.__version__) == '2.1.2+cu118' and session.np.__version__ == '1.26.4'
        and importlib.metadata.version('torch-geometric') == '2.7.0' and importlib.metadata.version('ogb') == '1.3.6'
        and importlib.metadata.version('torch-scatter') == '2.1.2+pt21cu118'
        and importlib.metadata.version('torch-sparse') == '0.6.18+pt21cu118', 'Existing GPU77 numeric providers and OGB1.3.6')
    session.torch.backends.cuda.matmul.allow_tf32 = False; session.torch.backends.cudnn.allow_tf32 = False
    session.torch.backends.cudnn.benchmark = False
    facade = ObjectivesFacade(session.core['objectives'], condition)
    session.core = dict(session.core, objectives=facade); session.config = configured_recipe(public, condition)
    install(session)
    return session, facade


def run_complete(cfg, pins, public_root):
    condition, seed = cfg['condition'], cfg['seed']; weights = CONDITIONS[condition]
    specification = identity(condition)
    portable = load('_unit_mechanism_public', public_root / 'portable.py')
    old_portable, old_data = sys.modules.get('portable'), sys.modules.get('data_interface')
    sys.modules['portable'] = portable
    try:
        data = load('_unit_mechanism_data', public_root / 'data_interface.py'); sys.modules['data_interface'] = data
        driver = load('_unit_mechanism_full_driver', public_root / 'train.py')
        original_write, original_snapshot = driver.json_write, driver.joint_snapshot; box = {}

        def recipe(task):
            require(task == 'wikics', 'WikiCS only')
            return configured_recipe(portable, condition)

        def factory(task, arm, actual_seed, device, polynormer, ncn_model, ncn_utils):
            require(task == 'wikics' and arm == specification['control_id'] and actual_seed == seed, 'One exact labelled ablation cell')
            session, facade = make_session(portable, condition, seed, device, polynormer)
            box.update(session=session, facade=facade); return session

        def annotate(value):
            facade = box.get('facade'); session = box.get('session')
            return {**value, 'mechanism_ablation': specification, 'method_identity': specification['control_id'],
                'underlying_session_arm': specification['underlying_session_arm'], 'ablation_adapter_sha256': sha(__file__),
                'completed_source_steps': session.steps if session else 0,
                'execution_mode': MODE, 'execution_accounting': dict(session.execution_totals) if session else None,
                'last_replay_diagnostics': session.last_replay_diagnostics if session else None,
                'bitwise_author_parity_claimed': False,
                'active_original_loss_calls': dict(alignment=facade.alignment_source_calls, residual=facade.residual_source_calls) if facade else None}

        def write(path, value):
            if isinstance(value, dict) and Path(path).name in ('RUN.json', 'COMPLETE.json', 'FAILURE.json', 'PROGRESS.json'):
                value = annotate(value)
                if Path(path).name == 'COMPLETE.json':
                    require(value['epochs'] == 1100 and value['steps'] == 1100
                        and value['active_original_loss_calls'] == {'alignment': 1100 if weights[0] else 0, 'residual': 1100 if weights[1] else 0},
                        'Full original horizon and active component calls')
                    require(value['execution_accounting'] == dict(shadow_member_forwards=8800, replay_member_forwards=8800,
                        output_cotangent_collections=1100, member_reverse_collections=8800, optimizer_bank_updates=1100,
                        exact_member_RNG_endpoint_checks=1100), 'All16forwards/8VJPs charged per completed update')
            original_write(path, value)

        def snapshot(session, epoch, metric, per, run):
            value = original_snapshot(session, epoch, metric, per, annotate(run))
            value['mechanism_ablation'] = specification; return value

        driver.Session, driver.recipe, driver.json_write, driver.joint_snapshot = factory, recipe, write, snapshot
        args = ['unit-mechanism-full-loop', '--task', 'wikics', '--arm', specification['control_id'], '--seed', str(seed),
            '--device', 'cuda:0', '--polynormer', str(bound(pins['polynormer'])), '--train', str(bound(cfg['train'])),
            '--valid', str(bound(cfg['development'])), '--output', str(inside(cfg['output']))]
        old_argv = sys.argv
        try:
            sys.argv = args; driver.main()  # Original1100 epochs, two CE views, data, transitions, selectors, failures.
        finally:
            sys.argv = old_argv
    finally:
        for name, value in (('portable', old_portable), ('data_interface', old_data)):
            if value is None: sys.modules.pop(name, None)
            else: sys.modules[name] = value


def admit(release, release_sha256, engineering=False):
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    require(sha(release) == release_sha256, 'Exact separate root release')
    cfg = read(release)
    schema = 'WikiCS-unit-mechanism-qualifier-release-v2' if engineering else 'WikiCS-unit-mechanism-cell-release-v2'
    require(cfg.get('schema') == schema and cfg.get('enabled') is True
        and cfg.get('root_execution_authorized') is True and cfg.get('source_review_approved') is True
        and cfg.get('caller_raw_tensors_match_public_pins') is True and cfg.get('full_two_view_resource_readiness_confirmed') is True
        and cfg.get('external_hard_bound_confirmed') is True and cfg.get('one_owned_fullgraph_process_per_GPU') is True
        and cfg.get('recomputation_execution_review_approved') is True
        and cfg.get('TEST_access') is False and cfg.get('automatic_retry') is False,
        'Disabled pending root scientific, data/runtime/resource and owned-cap admission')
    if not engineering:
        require(cfg.get('fixed12_protocol_adopted') is True and cfg.get('same_runtime_and_GPU_per_seed_confirmed') is True
            and cfg.get('v2_fullgraph_local_global_qualified') is True and cfg['condition'] in CONDITIONS, 'Exact adopted fixed paired protocol and actual V2 qualification')
    require(cfg['seed'] in SEEDS and cfg['source_manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Exact frozen seed/source')
    verify(dict(path=str((HERE / 'MANIFEST.json').relative_to(PHASE)), sha256=cfg['source_manifest_sha256']))
    public_root = verify(pins['public_manifest'])
    verify(pins['preserved_v1_manifest'])
    require(socket.gethostname() == 'peptide' and Path.cwd().resolve() == Path(pins['repository']).resolve(), 'Existing authorized GPU77 host/repository')
    gpu = pins['GPU_per_seed'][str(cfg['seed'])]
    require(cfg['physical_gpu_uuid'] == gpu and os.environ.get('CUDA_VISIBLE_DEVICES') == gpu, 'Same fixed physical GPU for every condition of this seed')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
    require(rows == pins['physical_gpu_inventory'], 'Actual full two-GPU inventory')
    free_mib = int(subprocess.check_output(['nvidia-smi', '--id=' + gpu, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
    require(free_mib * 1048576 >= pins['minimum_fresh_free_GPU_bytes'], 'Shared-host fresh memory for one live member graph plus bounded output/cotangent storage')
    require(pins['minimum_owned_GPU_memory_cap_bytes'] <= cfg['owned_GPU_memory_cap_bytes'] <= pins['maximum_owned_GPU_memory_cap_bytes'], 'Recomputation owned GPU cap and shared-host headroom')
    require(str(Path(sys.executable).absolute()) == pins['python'], 'Existing GPU77 virtualenv executable path')
    require(os.environ.get('PYTHONPATH', '') == '', 'Existing repo-owned runtime without allocation overlay')
    require(cfg['train'] == pins['train'] and cfg['development'] == pins['development'], 'Exact completed safe role files')
    for key in ('runtime_inventory', 'data_conversion_equality', 'resource_readiness', 'external_supervision'):
        bound(cfg[key])
    if not engineering:
        qualification = read(bound(cfg['qualification_receipt']))
        require(qualification.get('schema') == 'WikiCS-unit-mechanism-representative-qualification-v2'
            and qualification.get('complete') is True and qualification.get('execution_mode') == MODE
            and qualification.get('source_manifest_sha256') == cfg['source_manifest_sha256'], 'Actual complete V2 fullgraph qualification bound to this source')
        require(cfg['external_active_seconds'] == 32390 and cfg['external_cleanup_seconds'] == 10 and cfg['external_hard_seconds'] == 32400,
            'Original finite active/cleanup/hard envelope')
    else:
        require(0 < cfg['external_active_seconds'] <= 3600 and cfg['external_cleanup_seconds'] == 10
            and cfg['external_hard_seconds'] == cfg['external_active_seconds'] + 10, 'Finite owned engineering envelope')
    return cfg, pins, public_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    cfg, pins, public_root = admit(args.release, args.release_sha256)
    run_complete(cfg, pins, public_root)


if __name__ == '__main__':
    main()
