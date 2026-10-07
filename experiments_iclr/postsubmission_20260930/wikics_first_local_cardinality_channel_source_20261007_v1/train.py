"""Conditional Wiki degree-channel single-cell interface; original full driver."""
import argparse, copy, hashlib, importlib.util, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALLOCATION_NAME = 'public_internal_be_allocation_controls_20261007_v1'
ALLOCATION_SHA = 'e45c7c74a4e47864867275b41818d4b3a8f2ccf169288d18e0580b8deb4b43dd'
ALLOCATION_PROGRAM_SHA = '5fb90f0cc6e40acc0308b0a851258fc0f9904cbb933f021d786e7c6e9649c473'
CONSTRUCTORS = ('be_unit', 'be_init', 'single', 'independent4')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)
def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

method = module(HERE / 'method.py', '_conditional_degree_method')  # stdlib-only.

def sources(public_root=None, adapter_root=None, interface_root=None, allocation_root=None):
    root = Path(allocation_root or HERE.parent / ALLOCATION_NAME).resolve(strict=True)
    require(sha(root / 'MANIFEST.json') == ALLOCATION_SHA and sha(root / 'train.py') == ALLOCATION_PROGRAM_SHA, 'Exact sealed allocation source required')
    for row in json.loads((root / 'MANIFEST.json').read_text())['files']:
        path = (root / row['path']).resolve(strict=True)
        require(path.is_relative_to(root) and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Allocation source changed')
    allocation = module(root / 'train.py', '_degree_original_allocation')
    return allocation.sources(public_root, adapter_root, interface_root)

def identity(constructor_arm, kind, policy, lambda_value):
    require(constructor_arm in CONSTRUCTORS and kind in ('shared', 'centered'), 'Explicit original constructor and exact channel kind required')
    native = constructor_arm in ('single', 'independent4')
    if native:
        require(kind == 'shared' and policy is None and lambda_value is None, 'Native same-access controls retain original own-only objective and independent scalar per body')
        risk = 'original native own supervision; no alignment/J or cross-body sharing'
        reverse = 1
    else:
        lambda_value = method.validate(policy, lambda_value); risk = 'fixed original O/I/P/G allocation; new shared scalar own; centered slots internal objective'
        reverse = method.REVERSE_CALLS[policy]
    name = constructor_arm + '__degree_' + ('native_same_access' if native else kind + '__' + policy)
    return {'method_identity': name, 'conditional_fallback': True, 'task': 'wikics', 'constructor_arm': constructor_arm,
        'channel_kind': kind, 'native_same_access': native, 'allocation_policy': policy, 'lambda': lambda_value,
        'risk_contract': risk, 'reverse_collections_per_update': reverse, 'member_view_forwards_per_update': 2 if constructor_arm == 'single' else 8,
        'native_Adam_calls_per_update': 4 if constructor_arm == 'independent4' else 1,
        'single_old_state_transition_per_native_optimizer': True,
        'native_independent_selection_preserved': constructor_arm == 'independent4',
        'shared_coefficient_risk': 'own ONLY', 'centered_coefficients_risk': 'existing internal policy objective with full centered pullback',
        'literal_native_beta_unchanged': True, 'PNA_prior': '2004.05718v1 Sections2.2-2.3',
        'allocation_manifest_sha256': ALLOCATION_SHA, 'channel_program_sha256': sha(HERE / 'channel.py'),
        'method_program_sha256': sha(HERE / 'method.py'), 'complete_interface_sha256': sha(__file__),
        'source_preparation_adopts_execution': False, 'runtime_verified_at_preparation': False,
        'strength_or_policy_or_seed_roster_adopted': False, 'primitive_novelty_diagnosis_quality_efficiency_guarantee': False,
        'exact_resume_supported': False, 'TEST_scoring': False, 'automatic_campaign': False}

class CardinalitySession:
    def __init__(self, raw, auditor, train, control_identity, config):
        self.session, self.identity, self.config = raw, control_identity, config
        # Keep original arm strings: ordinary independent4 selection tests these exactly.
        self.arm = raw.arm; self.last_train = None
        self.work = {'external_forward_calls': 0, 'external_member_forwards': 0, 'complete_VALID_evaluations': 0,
            'native_completed_updates': 0, 'native_completed_backward_calls': 0, 'native_completed_Adam_calls': 0}
        if raw.model.independent:
            base = method.channel_module.native_inventory(raw)
            self.control = None
            self.channel = method.channel_module.CardinalityChannel(raw, train, kind='shared', base_partition=base)
        else:
            self.control = method.CardinalityAdapter(raw, auditor, train, kind=control_identity['channel_kind'],
                policy=control_identity['allocation_policy'], lambda_value=control_identity['lambda'])
            self.channel = self.control.channel

    def __getattr__(self, name): return getattr(self.session, name)

    def forward(self, *args, **kwargs):
        self.work['external_forward_calls'] += 1; result = self.session.forward(*args, **kwargs)
        self.work['external_member_forwards'] += self.model.members; return result

    def metadata(self):
        return {**self.identity, 'channel': self.channel.metadata(), 'completed_source_steps': self.session.steps,
            'actual_external_and_native_work': dict(self.work), 'last_TRAIN_objectives': self.last_train,
            'allocation_runtime_metadata': None if self.control is None else self.control.metadata()}

    def _cpu_tree(self, value):
        # Original own-body and final evaluation-only bank snapshots also carry
        # channel identity; their actual state/selection semantics stay original.
        if isinstance(value, dict) and 'model' in value and ('epoch' in value or value.get('evaluation_only')):
            value = {**value, 'cardinality_control': self.metadata(), 'method_identity': self.identity['method_identity']}
            if 'run' in value: value['run'] = {**value['run'], 'cardinality_control': self.metadata(), 'method_identity': self.identity['method_identity']}
        return self.session._cpu_tree(value)

    def train_step(self, batch, labels):
        self.channel.check_inventory()
        if self.control is None:
            before = self.session.steps; result = self.session.train_step(batch, labels)  # Unchanged native two-view sum-own backward/Adam bank.
            require(self.session.steps == before + 1, 'Original native update must complete one step')
            self.work['native_completed_updates'] += 1; self.work['native_completed_backward_calls'] += 1
            self.work['native_completed_Adam_calls'] += len(self.session.optimizers)
            self.last_train = {'own_mean': float(result['own_mean']), 'auxiliary': float(result['auxiliary']),
                'native_source_update_unchanged': True, 'degree_coefficients_risk': 'own per independent body'}
        else:
            result = self.control.train_step(batch, labels)
            require(result['charged_reverse_passes'] == self.identity['reverse_collections_per_update'] and result['member_forwards'] == 8
                and result['Adam_steps'] == 1, 'Actual augmented allocation work differs')
            self.last_train = {name: None if result[name] is None else float(result[name]) for name in ('own_mean', 'pool_mean', 'mixture_mean', 'alignment', 'auxiliary', 'internal_loss')}
            self.last_train.update(shared_degree_risk='own', slots_risk='internal objective', alignment_permission='old phi plus centered slots ONLY',
                charged_reverse_passes=result['charged_reverse_passes'])
        return result

    def save_training_state(self, *args, **kwargs): raise NotImplementedError('Use labelled original selected snapshots; exact resume unsupported')
    def restore_training_state(self, *args, **kwargs): raise NotImplementedError('No unlabelled/warm acquisition or exact resume')

def run_complete(constructor_arm, kind, train, valid, output, polynormer, policy=None, lambda_value=None,
                 seed=6101, device='cpu', public_root=None, adapter_root=None, interface_root=None, allocation_root=None):
    """One explicitly authorized full Wiki cell; no policy/scaler/horizon grid."""
    require(type(seed) is int, 'Integer seed required'); control_identity = identity(constructor_arm, kind, policy, lambda_value)
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    began = time.monotonic(); box = {}
    try:
        public, driver, auditor = sources(public_root, adapter_root, interface_root, allocation_root)
        original_load, original_write = driver.load_train_valid, driver.json_write
        original_evaluate, original_snapshot = driver.evaluate, driver.joint_snapshot
        def load(task, train_path, valid_path):
            require(task == 'wikics', 'Wiki only'); result = original_load(task, train_path, valid_path)
            box['train'] = result[0]; return result
        def recipe(task):
            require(task == 'wikics', 'Wiki only'); value = copy.deepcopy(public.recipe(task))
            value['arms'] = [constructor_arm]; value['cardinality_control'] = control_identity; return value
        def factory(task, arm, seed, device, polynormer, ncn_model, ncn_utils):
            require(task == 'wikics' and arm == constructor_arm and ncn_model is None and ncn_utils is None, 'Exact Wiki constructor required')
            raw = public.Session(task, arm, seed, device, polynormer, ncn_model, ncn_utils)
            session = CardinalitySession(raw, auditor, box['train'], control_identity, recipe(task)); box['session'] = session; return session
        def annotate(value):
            session = box.get('session')
            return {**value, 'cardinality_control': session.metadata() if session else control_identity,
                'underlying_constructor_arm': constructor_arm, 'method_identity': control_identity['method_identity']}
        def snapshot(session, epoch, metric, per, run):
            return original_snapshot(session, epoch, metric, per, annotate(run))
        def evaluate(session, train, valid):
            result = original_evaluate(session, train, valid); session.work['complete_VALID_evaluations'] += 1; return result
        def write(path, value):
            name = Path(path).name; session = box.get('session')
            if isinstance(value, dict) and name in ('RUN.json', 'COMPLETE.json', 'FAILURE.json', 'PROGRESS.json', 'OWN_BEST_BANK.json'):
                value = annotate(value)
                if name == 'COMPLETE.json':
                    steps = session.config['training']['epochs']; require(session.steps == steps, 'Original full1100 Wiki steps required')
                    session.channel.check_inventory()
                    if session.control is not None:
                        counts = session.control.counters
                        require(counts['two_view_updates'] == counts['Adam_calls'] == counts['Adam_steps'] == steps
                            and counts['Session_forward_calls'] == 2 * steps and counts['member_forwards'] == 8 * steps
                            and counts['autograd_grad_calls'] == control_identity['reverse_collections_per_update'] * steps, 'Actual complete augmented allocation work required')
                    else:
                        require(session.work['native_completed_updates'] == session.work['native_completed_backward_calls'] == steps
                            and session.work['native_completed_Adam_calls'] == steps * len(session.optimizers), 'Original complete native work required')
                    expected_valid = steps + int(constructor_arm == 'independent4')
                    require(session.work['complete_VALID_evaluations'] == expected_valid, 'All original VALID and ordinary own-bank evaluation required')
                    expected_groups = 2 * steps + expected_valid  # Two TRAIN views + one full graph per original evaluation.
                    require(session.channel.work['edge_binding_checks'] == expected_groups
                        and session.channel.work['first_local_hook_calls'] == expected_groups * session.model.members,
                        'Actual first-local channel calls must cover every original member/view/evaluation')
                    value['degree_parameter_tensors'] = len(session.channel.new_shared) + len(session.channel.new_internal)
                    value['degree_parameter_elements'] = sum(parameter.numel() for _, parameter in session.channel.new_shared + session.channel.new_internal)
            elif name == 'VALID_TRACE.json' and session is not None:
                for row in value:
                    if 'cardinality_method_identity' not in row:
                        row['cardinality_method_identity'] = control_identity['method_identity']; row['TRAIN']['cardinality_objectives'] = dict(session.last_train)
            original_write(path, value)
        driver.load_train_valid, driver.Session, driver.recipe = load, factory, recipe
        driver.json_write, driver.joint_snapshot, driver.evaluate = write, snapshot, evaluate
        # Bare native arm is required for unchanged independent4 own selector/local transition.
        arguments = ['conditional-degree-complete', '--task', 'wikics', '--arm', constructor_arm, '--seed', str(seed),
            '--device', str(device), '--train', str(train), '--valid', str(valid), '--output', str(output), '--polynormer', str(polynormer)]
        old = sys.argv
        try: sys.argv = arguments; driver.main()
        finally: sys.argv = old
        return {'output': str(output), 'method_identity': control_identity['method_identity'], 'cardinality_control': box['session'].metadata(), 'TEST_scoring': False}
    except BaseException as error:
        if not output.exists():
            output.mkdir(parents=True, exist_ok=False)
            (output / 'FAILURE.json').write_text(json.dumps({'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                'seconds': time.monotonic() - began, 'cardinality_control': control_identity, 'method_identity': control_identity['method_identity'],
                'automatic_retry': False, 'failure_stage': 'conditional source setup before original full driver', 'TEST_scoring': False}, indent=2, sort_keys=True) + '\n')
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--constructor-arm', choices=CONSTRUCTORS, required=True)
    parser.add_argument('--kind', choices=('shared', 'centered'), required=True)
    parser.add_argument('--policy', choices=method.POLICIES); parser.add_argument('--lambda-value', type=float)
    parser.add_argument('--seed', type=int, default=6101); parser.add_argument('--device', default='cpu')
    for name in ('train', 'valid', 'output', 'polynormer'): parser.add_argument('--' + name, type=Path, required=True)
    for name in ('public-root', 'adapter-root', 'interface-root', 'allocation-root'): parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    try: identity(args.constructor_arm, args.kind, args.policy, args.lambda_value)
    except ValueError as error: parser.error(str(error))
    result = run_complete(**vars(args)); print(json.dumps({'complete': True, 'method_identity': result['method_identity'], 'output': result['output'], 'TEST_scoring': False}))

if __name__ == '__main__': main()
