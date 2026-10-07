"""Full TRAIN/VALID interface for the sealed private-steering adapter.

Stdlib-only import/CLI parsing. Numerical work occurs only when run_complete
is explicitly invoked. This source adds no scientific adoption or campaign.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

PUBLIC_NAME = 'portable_internal_be_public_interface_20261007_v2'
ADAPTER_NAME = 'public_internal_be_private_steering_adapter_20261007_v1'
PUBLIC_SHA = '190940ca9f8141ac45f739d064cb1ef76aaf1965ba8c91adb544ad8e38fef724'
ADAPTER_SHA = '217a093438eac20f68b80fc2d543bfa8cf7ffd15926044fef8d08ad9f4436baa'
ADAPTER_PROGRAM_SHA = '80d148f51018768c40c88f8534a7afbc426390b0e607a43bc8b5d25d8a4f1153'
MODES = ('alignment_private', 'hidden_private', 'gncl_private')
BASE_ARMS = ('be_unit', 'be_init')
TASKS = ('wikics', 'collab', 'molhiv')
UPDATES = {'wikics': 1, 'collab': 18, 'molhiv': 258}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def control_identity(mode, base_arm, lambda_value):
    require(mode in MODES and base_arm in BASE_ARMS, 'Declared mode and shared BE base arm required')
    if mode == 'gncl_private':
        require(not isinstance(lambda_value, bool) and isinstance(lambda_value, (int, float))
                and math.isfinite(lambda_value) and 0 <= lambda_value <= 1,
                'GNCL requires an explicit fixed finite lambda in [0,1]; no strength is adopted by this source')
        lambda_value = float(lambda_value)
    else:
        require(lambda_value is None, 'Reference modes take no GNCL lambda')
    return {'control_id': base_arm + '__' + mode, 'mode': mode, 'base_constructor_arm': base_arm,
        'lambda': lambda_value, 'lambda_learned': False, 'strength_adopted_by_source': False,
        'public_V2_manifest_sha256': PUBLIC_SHA, 'private_adapter_manifest_sha256': ADAPTER_SHA,
        'private_adapter_program_sha256': ADAPTER_PROGRAM_SHA, 'complete_interface_sha256': sha(__file__),
        'reverse_pass_contract': 'Two autograd.grad collections at the same old parameter state before one original Adam transition',
        'charged_reverse_passes_per_update': 2, 'extra_reverse_passes_over_source_shared_update': 1,
        'full_own_member_views_per_update': 8, 'Adam_transitions_per_update': 1,
        'selector': 'Original public strict-first complete VALID joint selector/local transition; no own-bank conversion',
        'auxiliary_log_semantics': 'private_loss minus own_mean; diagnostic difference, not a shared-block auxiliary gradient',
        'source_preparation_adopts_execution': False, 'prior_interface_runtime_qualification': False,
        'exact_resume_supported': False,
        'TEST_scoring': False, 'automatic_campaign': False}


def _verify(root, expected):
    root = Path(root).resolve(strict=True)
    require(sha(root / 'MANIFEST.json') == expected, 'Exact sealed source manifest required: ' + str(root))
    for row in json.loads((root / 'MANIFEST.json').read_text())['files']:
        rel = Path(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'Source path leaves root')
        path = (root / rel).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file() and sha(path) == row['sha256']
                and path.stat().st_size == row['bytes'], 'Sealed source file changed: ' + row['path'])
    return root


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


@contextmanager
def _aliases(values):
    missing = object(); old = {key: sys.modules.get(key, missing) for key in values}
    try:
        sys.modules.update(values)
        yield
    finally:
        for key, value in old.items():
            if value is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def _sources(public_root, adapter_root):
    parent = Path(__file__).resolve().parent.parent
    public_root = _verify(public_root or parent / PUBLIC_NAME, PUBLIC_SHA)
    adapter_root = _verify(adapter_root or parent / ADAPTER_NAME, ADAPTER_SHA)
    require(sha(adapter_root / 'adapter.py') == ADAPTER_PROGRAM_SHA, 'Exact steering program required')
    public = _module(public_root / 'portable.py', '_steering_complete_public_v2')
    with _aliases({'portable': public}):
        data = _module(public_root / 'data_interface.py', '_steering_complete_public_data')
        with _aliases({'data_interface': data}):
            driver = _module(public_root / 'train.py', '_steering_complete_public_driver')
    # Package registration supports the adapter's unchanged relative imports.
    name = '_steering_complete_sealed_adapter'
    spec = importlib.util.spec_from_file_location(name, adapter_root / '__init__.py',
                                                submodule_search_locations=[str(adapter_root)])
    adapter = importlib.util.module_from_spec(spec)
    with _aliases({name: adapter}):
        spec.loader.exec_module(adapter)
    return public, driver, adapter


class SteeringSession:
    """Delegate all Session state/forward/serving to the original audited object."""
    def __init__(self, session, adapter, identity, config):
        self.session = session
        self.steering = adapter.PrivateSteeringAdapter(session, mode=identity['mode'], lambda_value=identity['lambda'])
        self.identity, self.arm, self.config = identity, identity['control_id'], config
        self.last_train = None

    def __getattr__(self, name):
        return getattr(self.session, name)

    def metadata(self):
        return {**self.identity, 'adapter_runtime_metadata': self.steering.metadata(),
                'completed_source_steps': self.session.steps, 'last_TRAIN_objectives': self.last_train}

    def train_step(self, batch, labels):
        result = self.steering.train_step(batch, labels)
        require(result['charged_reverse_passes'] == 2 and result['Adam_steps'] == 1
                and result['member_forwards'] == 8, 'Sealed two-old-state-pass update contract differs')
        auxiliary = result['private_loss'] - result['own_mean']
        mixture = (result['own_mean'] if result['pool_mean'] is None else
                   (1 - self.identity['lambda']) * result['own_mean'] + self.identity['lambda'] * result['pool_mean'])
        def scalar(value):
            return None if value is None else float(value)
        self.last_train = {'own_mean': scalar(result['own_mean']), 'private_loss': scalar(result['private_loss']),
            'private_supervised_mixture_mean': scalar(mixture), 'auxiliary_private_minus_own': scalar(auxiliary),
            'pool_mean': scalar(result['pool_mean']), 'alignment': scalar(result['alignment']),
            'residual': scalar(result['residual']), 'charged_reverse_passes': 2, 'Adam_steps': 1,
            'member_forwards': 8, 'gradient_assignment': 'shared/boundary mean-own; audited internal factors selected private objective'}
        return {**result, 'auxiliary': auxiliary, 'private_supervised_mixture_mean': mixture}

    def save_training_state(self, *args, **kwargs):
        raise NotImplementedError('Use annotated selected snapshots from run_complete; exact resume is unsupported')

    def restore_training_state(self, *args, **kwargs):
        raise NotImplementedError('No unannotated base-arm restart or exact-resume path')


def run_complete(task, mode, base_arm, train, valid, output, seed=6101, device='cpu',
                 lambda_value=None, polynormer=None, ncn_model=None, ncn_utils=None,
                 public_root=None, adapter_root=None):
    """Explicit one-cell full-horizon call; no training loop copy or campaign.

    Calls must be serial per process because the unchanged driver reads sys.argv.
    A successful source integration is not numerical/runtime qualification.
    """
    require(task in TASKS and type(seed) is int, 'Declared task/integer seed required')
    identity = control_identity(mode, base_arm, lambda_value)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic(); box = {}
    try:
        public, driver, adapter = _sources(public_root, adapter_root)
        original_snapshot, original_write = driver.joint_snapshot, driver.json_write

        def recipe(task):
            config = copy.deepcopy(public.recipe(task))
            config['arms'] = [identity['control_id']]
            config['private_steering_control'] = identity
            return config

        def factory(task, arm, seed, device, polynormer, ncn_model, ncn_utils):
            require(arm == identity['control_id'], 'One visibly labelled steering control per fresh run')
            raw = public.Session(task, base_arm, seed, device, polynormer, ncn_model, ncn_utils)
            session = SteeringSession(raw, adapter, identity, recipe(task))
            box['session'] = session
            return session

        def annotate(value):
            session = box.get('session')
            return {**value, 'private_steering_control': session.metadata() if session else identity,
                    'underlying_constructor_arm': base_arm, 'method_identity': identity['control_id']}

        def snapshot(session, epoch, metric, per, run):
            value = original_snapshot(session, epoch, metric, per, annotate(run))
            value.update(private_steering_control=session.metadata(), method_identity=identity['control_id'])
            return value

        def write(path, value):
            name = Path(path).name; session = box.get('session')
            if isinstance(value, dict) and name in ('RUN.json', 'COMPLETE.json', 'FAILURE.json', 'PROGRESS.json'):
                value = annotate(value)
                if name == 'COMPLETE.json':
                    steps = session.config['training']['epochs'] * UPDATES[task]
                    counts = session.steering.counters
                    require(session.steps == counts['two_view_updates'] == counts['Adam_steps'] == steps
                            and counts['autograd_grad_calls'] == 2 * steps and counts['member_forwards'] == 8 * steps,
                            'Complete full horizon and charged two-pass/eight-forward/one-Adam counts required')
            elif name == 'VALID_TRACE.json' and session is not None:
                # Keep every original selector event; only enrich its last-batch TRAIN diagnostic.
                for row in value:
                    if 'private_steering_control_id' not in row:
                        row['private_steering_control_id'] = identity['control_id']
                        row['TRAIN']['private_objectives'] = dict(session.last_train)
            original_write(path, value)

        driver.Session, driver.recipe = factory, recipe
        driver.joint_snapshot, driver.json_write = snapshot, write
        arguments = ['private-steering-complete', '--task', task, '--arm', identity['control_id'],
            '--seed', str(seed), '--device', str(device), '--train', str(train), '--valid', str(valid), '--output', str(output)]
        for flag, value in (('--polynormer', polynormer), ('--ncn-model', ncn_model), ('--ncn-utils', ncn_utils)):
            if value is not None:
                arguments.extend([flag, str(value)])
        old = sys.argv
        try:
            sys.argv = arguments
            driver.main()  # Original full data/epochs/validation/joint snapshots/local restoration/failure preservation.
        finally:
            sys.argv = old
        return {'output': str(output), 'method_identity': identity['control_id'],
                'private_steering_control': box['session'].metadata(), 'TEST_scoring': False}
    except BaseException as error:
        # Original driver preserves its annotated FAILURE. A pre-driver source
        # setup failure also gets a fresh labelled receipt, never an overwrite.
        if not output.exists():
            output.mkdir(parents=True, exist_ok=False)
            value = {'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                'seconds': time.monotonic() - started, 'automatic_retry': False,
                'method_identity': identity['control_id'], 'private_steering_control': identity,
                'failure_stage': 'source interface setup before original full driver', 'TEST_scoring': False}
            (output / 'FAILURE.json').write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--base-arm', choices=BASE_ARMS, required=True)
    parser.add_argument('--lambda-value', type=float, default=None, help='Required explicit fixed GNCL lambda; forbidden for reference modes')
    parser.add_argument('--seed', type=int, default=6101)
    parser.add_argument('--device', default='cpu')
    for name in ('train', 'valid', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('polynormer', 'ncn-model', 'ncn-utils', 'public-root', 'adapter-root'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    try:
        control_identity(args.mode, args.base_arm, args.lambda_value)
    except ValueError as error:
        parser.error(str(error))
    result = run_complete(**vars(args))
    print(json.dumps({'complete': True, 'output': result['output'], 'method_identity': result['method_identity'], 'TEST_scoring': False}))


if __name__ == '__main__':
    main()
