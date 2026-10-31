"""Complete O/I/P/G callable/CLI; delegate the unchanged full public driver."""
import argparse, copy, hashlib, importlib.util, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTERFACE_NAME = 'public_internal_be_private_steering_complete_interface_20261007_v1'
INTERFACE_SHA = 'aa55302f80629971819469e29b6b8c39636825ea2e01ca6c18f019a85de88b0a'
INTERFACE_PROGRAM_SHA = 'b2ae8b0e7241b2deae11fb90887012b1d1ba29557e39ddc1f069364dbae8acae'
BASE_ARMS = ('be_unit', 'be_init')
TASKS = ('wikics', 'collab', 'molhiv')
UPDATES = {'wikics': 1, 'collab': 18, 'molhiv': 258}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok: raise ValueError(message)

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

method = module(HERE / 'method.py', '_sealed_OIPG_method')  # stdlib-only module.

def sources(public_root=None, adapter_root=None, interface_root=None):
    """Reuse sealed source loader, auditor and exact task pool; no numerical load."""
    root = Path(interface_root or HERE.parent / INTERFACE_NAME).resolve(strict=True)
    require(sha(root / 'train_steering.py') == INTERFACE_PROGRAM_SHA, 'Exact original complete interface required')
    original = module(root / 'train_steering.py', '_OIPG_original_complete_interface')
    original._verify(root, INTERFACE_SHA)
    return original._sources(public_root, adapter_root)

def control_identity(policy, base_arm, lambda_value):
    lambda_value = method.validate(policy, lambda_value)
    require(base_arm in BASE_ARMS, 'Explicit original shared BE constructor arm required')
    return {'control_id': base_arm + '__allocation_' + policy, 'mode': 'allocation_controls', 'policy': policy,
        'base_constructor_arm': base_arm, 'lambda': lambda_value, 'lambda_used_by_supervision': policy != 'O',
        'lambda_learned': False, 'strength_adopted_by_source': False, 'supervision_by_role': method.SUPERVISION[policy],
        'alignment_gradient_permission': 'unchanged .05 A on the same audited internal phi ONLY in O/I/P/G',
        'charged_reverse_calls_per_update': method.REVERSE_CALLS[policy],
        'extra_reverse_calls_over_original_shared_update': method.REVERSE_CALLS[policy] - 1,
        'full_member_view_forwards_per_update': 8, 'Adam_transitions_per_update': 1,
        'reverse_contract': 'All supplied block gradients collected at the same old parameters; no upstream detach; one native Adam',
        'auxiliary_log_semantics': 'exact .05 A scalar; gradient permission phi only; supervised J-own difference logged separately',
        'G_scope': 'standard GNCL supervised scalar on this tied architecture; independently controlled internal-only A; no untied/full-aux reproduction',
        'selector': 'Original full complete-VALID strict-first joint/local selector and WikiCS transition',
        'public_V2_manifest_sha256': '190940ca9f8141ac45f739d064cb1ef76aaf1965ba8c91adb544ad8e38fef724',
        'permission_and_pool_adapter_manifest_sha256': '217a093438eac20f68b80fc2d543bfa8cf7ffd15926044fef8d08ad9f4436baa',
        'original_complete_interface_manifest_sha256': INTERFACE_SHA,
        'allocation_method_sha256': sha(HERE / 'method.py'), 'allocation_interface_sha256': sha(__file__),
        'source_preparation_adopts_execution': False, 'runtime_verified_at_preparation': False, 'exact_resume_supported': False,
        'TEST_scoring': False, 'automatic_campaign': False, 'cost_padding': False}

class PolicySession:
    """Full Session delegation with visibly labelled block policy and actual work."""
    def __init__(self, session, auditor, identity, config):
        self.session = session
        self.control = method.PolicyAdapter(session, auditor, policy=identity['policy'], lambda_value=identity['lambda'])
        self.identity, self.arm, self.config = identity, identity['control_id'], config
        self.last_train = None
        self.work = {'external_forward_calls': 0, 'external_member_forwards': 0, 'complete_VALID_evaluations': 0,
            'complete_VALID_forward_calls': 0, 'complete_VALID_member_forwards': 0, 'complete_VALID_dispatch_seconds': 0.}

    def __getattr__(self, name): return getattr(self.session, name)

    def forward(self, *args, **kwargs):
        self.work['external_forward_calls'] += 1
        result = self.session.forward(*args, **kwargs)
        self.work['external_member_forwards'] += self.session.model.members; return result

    def metadata(self):
        return {**self.identity, 'adapter_runtime_metadata': self.control.metadata(), 'actual_external_work': dict(self.work),
            'completed_source_steps': self.session.steps, 'last_TRAIN_objectives': self.last_train}

    def train_step(self, batch, labels):
        result = self.control.train_step(batch, labels)
        require(result['charged_reverse_passes'] == self.identity['charged_reverse_calls_per_update']
            and result['Adam_steps'] == 1 and result['member_forwards'] == 8, 'Actual policy update work differs')
        def scalar(value): return None if value is None else float(value)
        self.last_train = {name: scalar(result[name]) for name in ('own_mean', 'pool_mean', 'mixture_mean', 'alignment', 'auxiliary', 'internal_loss')}
        self.last_train.update(policy=self.identity['policy'], lambda_value=self.identity['lambda'],
            supervised_mixture_minus_own=None if result['mixture_mean'] is None else scalar(result['mixture_mean'] - result['own_mean']),
            auxiliary_permission='internal phi only', supervision_by_role=self.identity['supervision_by_role'],
            charged_reverse_passes=result['charged_reverse_passes'], member_forwards=8, Adam_steps=1)
        return result  # Original driver gets own_mean and the CORRECT auxiliary=.05A.

    def save_training_state(self, *args, **kwargs):
        raise NotImplementedError('Use policy-labelled original selected snapshots; exact resume unsupported')

    def restore_training_state(self, *args, **kwargs):
        raise NotImplementedError('No unlabelled base-arm restart or exact resume')

def run_complete(task, policy, base_arm, train, valid, output, lambda_value, seed=6101, device='cpu',
                 polynormer=None, ncn_model=None, ncn_utils=None, public_root=None, adapter_root=None, interface_root=None):
    """One explicitly chosen full-horizon cell; original driver, no loop copy/grid."""
    require(task in TASKS and type(seed) is int, 'Declared task and integer seed required')
    identity = control_identity(policy, base_arm, lambda_value); output = Path(output)
    if output.exists(): raise FileExistsError(output)
    started = time.monotonic(); box = {}
    try:
        public, driver, auditor = sources(public_root, adapter_root, interface_root)
        original_snapshot, original_write, original_evaluate = driver.joint_snapshot, driver.json_write, driver.evaluate

        def recipe(task):
            config = copy.deepcopy(public.recipe(task)); config['arms'] = [identity['control_id']]
            config['allocation_control'] = identity; return config

        def factory(task, arm, seed, device, polynormer, ncn_model, ncn_utils):
            require(arm == identity['control_id'], 'One visibly labelled allocation policy per fresh run')
            raw = public.Session(task, base_arm, seed, device, polynormer, ncn_model, ncn_utils)
            session = PolicySession(raw, auditor, identity, recipe(task)); box['session'] = session; return session

        def annotate(value):
            session = box.get('session')
            return {**value, 'allocation_control': session.metadata() if session else identity,
                'underlying_constructor_arm': base_arm, 'method_identity': identity['control_id']}

        def snapshot(session, epoch, metric, per, run):
            value = original_snapshot(session, epoch, metric, per, annotate(run))
            value.update(allocation_control=session.metadata(), method_identity=identity['control_id']); return value

        def evaluate(session, train, valid):
            began = time.monotonic(); calls = session.work['external_forward_calls']; members = session.work['external_member_forwards']
            result = original_evaluate(session, train, valid)
            session.work['complete_VALID_evaluations'] += 1
            session.work['complete_VALID_forward_calls'] += session.work['external_forward_calls'] - calls
            session.work['complete_VALID_member_forwards'] += session.work['external_member_forwards'] - members
            # Dispatch elapsed time; no extra device synchronization/forward.
            session.work['complete_VALID_dispatch_seconds'] += time.monotonic() - began; return result

        def write(path, value):
            name = Path(path).name; session = box.get('session')
            if isinstance(value, dict) and name in ('RUN.json', 'COMPLETE.json', 'FAILURE.json', 'PROGRESS.json'):
                value = annotate(value)
                if name == 'COMPLETE.json':
                    steps = session.config['training']['epochs'] * UPDATES[task]; counts = session.control.counters
                    require(session.steps == counts['two_view_updates'] == counts['Adam_calls'] == counts['Adam_steps'] == steps
                        and counts['Session_forward_calls'] == 2 * steps and counts['member_forwards'] == 8 * steps
                        and counts['autograd_grad_calls'] == method.REVERSE_CALLS[policy] * steps,
                        'Original full horizon and actual policy forward/reverse/Adam work required')
                    require(session.work['complete_VALID_evaluations'] == session.config['training']['epochs'], 'Every original full VALID epoch required')
            elif name == 'VALID_TRACE.json' and session is not None:
                for row in value:
                    if 'allocation_control_id' not in row:
                        row['allocation_control_id'] = identity['control_id']; row['TRAIN']['allocation_objectives'] = dict(session.last_train)
            original_write(path, value)

        driver.Session, driver.recipe = factory, recipe
        driver.joint_snapshot, driver.json_write, driver.evaluate = snapshot, write, evaluate
        arguments = ['allocation-complete', '--task', task, '--arm', identity['control_id'], '--seed', str(seed),
            '--device', str(device), '--train', str(train), '--valid', str(valid), '--output', str(output)]
        for flag, value in (('--polynormer', polynormer), ('--ncn-model', ncn_model), ('--ncn-utils', ncn_utils)):
            if value is not None: arguments.extend([flag, str(value)])
        old = sys.argv
        try:
            sys.argv = arguments
            driver.main()  # Original epochs/data/VALID/selector/WikiCS transition/failure preservation.
        finally: sys.argv = old
        return {'output': str(output), 'method_identity': identity['control_id'], 'allocation_control': box['session'].metadata(), 'TEST_scoring': False}
    except BaseException as error:
        if not output.exists():
            output.mkdir(parents=True, exist_ok=False)
            value = {'complete': False, 'error_type': type(error).__name__, 'error': str(error), 'seconds': time.monotonic() - started,
                'automatic_retry': False, 'method_identity': identity['control_id'], 'allocation_control': identity,
                'failure_stage': 'allocation source setup before original full driver', 'TEST_scoring': False}
            (output / 'FAILURE.json').write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--policy', choices=method.POLICIES, required=True)
    parser.add_argument('--base-arm', choices=BASE_ARMS, required=True)
    parser.add_argument('--lambda-value', type=float, required=True, help='Explicit shared fixed lambda; O records it but uses own supervision')
    parser.add_argument('--seed', type=int, default=6101); parser.add_argument('--device', default='cpu')
    for name in ('train', 'valid', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    for name in ('polynormer', 'ncn-model', 'ncn-utils', 'public-root', 'adapter-root', 'interface-root'): parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    try: control_identity(args.policy, args.base_arm, args.lambda_value)
    except ValueError as error: parser.error(str(error))
    result = run_complete(**vars(args))
    print(json.dumps({'complete': True, 'output': result['output'], 'method_identity': result['method_identity'], 'TEST_scoring': False}))

if __name__ == '__main__': main()
