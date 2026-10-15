"""Native residual-only/private-BN producer plus frozen structural private risk.

No model/data/library is imported until the native runtime calls transform.
Legacy producers, exact fractional table and same-old-state updater are reused.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
PHASE = E.parent
PARENT = E / 'pilot_20261015_residual_only_private_bn_v1'
TABLE = E / 'structural_class_responsibility_v1'
METHODS = {'residual_private_bn_structural_risk_v2': False,
           'residual_private_bn_permuted_risk_v2': True}
SCHEMA = 'structural-residual-private-BN-session-v2'
SOURCES_SHA = '3d8686ce45f38793614a50dfc33536efa5cedf3d413fee7392d2415eee49e5f6'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(path):
    path = Path(path)
    name = '_structural_residual_' + pin(path)['sha256']
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def source_bindings(root, expected):
    root = Path(root).resolve()
    if not HERE.is_relative_to(root) or not PHASE.is_relative_to(root):
        raise ValueError('Bind the actual repository containing this source')
    manifest = pin(HERE / 'MANIFEST.json')
    if expected != manifest['sha256'] or pin(HERE / 'SOURCES.json')['sha256'] != SOURCES_SHA:
        raise ValueError('Exact frozen composition manifest/source closure required')
    receipts = [manifest, pin(HERE / 'SOURCES.json')]
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = (HERE / row['path']).resolve()
        if not path.is_relative_to(HERE): raise ValueError('Composition source escapes its package')
        actual = pin(path)
        if any(actual[k] != row[k] for k in ('bytes', 'sha256')): raise ValueError('Composition source changed')
        receipts.append(actual)
    for row in json.loads((HERE / 'SOURCES.json').read_text())['files']:
        path = (PHASE / row['path_relative_to_phase']).resolve()
        if not path.is_relative_to(root): raise ValueError('Retained source escaped project')
        actual = pin(path)
        if any(actual[k] != row[k] for k in ('bytes', 'sha256')): raise ValueError('Retained source changed')
        receipts.append(actual)
    return receipts


def descriptor(parent, method, manifest_sha256):
    """In-memory descriptor helper; no scientific job file or fit is created."""
    if method not in METHODS or parent['seed'] not in (101, 202, 303):
        raise ValueError('Only the two declared compositions and parent seeds are defined')
    value = copy.deepcopy(parent)
    value.update(id=parent['id'] + '/' + method, procedure=method,
                 method_settings={'manifest_sha256': manifest_sha256},
                 session_transform={'path': str(HERE / 'extension.py'), 'function': 'transform',
                                    'sha256': pin(HERE / 'extension.py')['sha256']})
    return value


def partition(model, optimizer, *, require_zero_up=False):
    """Exhaustive native shared affine/weights and only residual down/up private."""
    native, private = tuple(model.native.named_parameters()), tuple(model.private_parameters())
    expected = (model.input_residual.down, model.input_residual.up)
    if len(private) != 2 or any(a is not b for a, b in zip(private, expected)) \
            or tuple(private[0].shape) != (4, 8, 300) or tuple(private[1].shape) != (4, 300, 8) \
            or sum(p.numel() for p in private) != 19200:
        raise ValueError('Exactly the original residual down/up coordinates must be private')
    if require_zero_up and private[1].count_nonzero().item() != 0:
        raise ValueError('Native-initialized zero-up residual required; no extra gate')
    optimized = tuple(p for group in optimizer.param_groups for p in group['params'])
    ids = tuple(map(id, optimized))
    known = tuple(p for _, p in native) + private
    if len(ids) != len(set(ids)) or {id(p) for p in known} != set(ids) \
            or {id(p) for p in model.parameters()} != set(ids) \
            or len({id(p) for p in known}) != len(known) or not all(p.requires_grad for p in known):
        raise ValueError('Every unchanged native and residual parameter needs exactly one optimizer owner')
    # The unchanged source updater expects this small BE-era partition API.
    # The residual model/its registered native parameters are not rewritten.
    return types.SimpleNamespace(_source_parameters=native, private_parameters=lambda: private)


def optimizer_contract(optimizer):
    """Fixed group boundaries/order, actual Parameters, every option/default.

    Optimizer.load_state_dict may replace group dictionaries. Their Python
    identity is tracked separately and may be renewed only after a valid load.
    """
    return {'parameter_ids_by_group': tuple(tuple(id(p) for p in group['params']) for group in optimizer.param_groups),
            'options_by_group': tuple({k: copy.deepcopy(v) for k, v in group.items() if k != 'params'}
                                     for group in optimizer.param_groups),
            'defaults': copy.deepcopy(optimizer.defaults)}


def check_optimizer_contract(optimizer, expected, group_objects, *, after_parent_restore=False):
    actual = optimizer_contract(optimizer)
    if actual != expected:
        changed_options = [{'group': i, 'fields': sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) or (k in a) != (k in b))}
                           for i, (a, b) in enumerate(zip(actual['options_by_group'], expected['options_by_group'])) if a != b]
        a, b = actual['defaults'], expected['defaults']
        detail = {'parameter_group_layout_or_live_IDs_changed': actual['parameter_ids_by_group'] != expected['parameter_ids_by_group'],
                  'group_count': [len(actual['parameter_ids_by_group']), len(expected['parameter_ids_by_group'])],
                  'changed_group_options': changed_options,
                  'changed_default_fields': sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) or (k in a) != (k in b))}
        raise ValueError('Original Adam semantic ownership/options/defaults changed: ' + json.dumps(detail, sort_keys=True))
    current = tuple(optimizer.param_groups)
    if not after_parent_restore and (len(current) != len(group_objects) or
            any(group is not old for group, old in zip(current, group_objects))):
        raise ValueError('Adam group dictionaries changed outside the admitted parent restore')
    return current


def weights_from_train(session, *, permuted, seed):
    t, view = session.torch, session.train_view
    if getattr(view, 'role', None) != 'TRAIN' or len(view) != len(session.train_ids):
        raise ValueError('Complete authentic native TRAIN graph view required')
    ids, labels, atoms, positions, missing = [], [], [], [], []
    for index in range(len(view)):
        graph = view[index]; graph_id = int(session.train_ids[index])
        if graph.smoke_graph_id.reshape(-1).tolist() != [graph_id] or graph.y.numel() != 1 \
                or graph.x.ndim != 2 or graph.x.shape[1] != 9 or int(graph.num_nodes) != len(graph.x) or not len(graph.x):
            raise ValueError('Original full nine-field molecular graph/identity/target required')
        target = graph.y.reshape(-1)[0]
        if bool(t.isnan(target)):
            missing.append(index); continue
        if not bool(t.isfinite(target)) or float(target) not in (0., 1.):
            raise ValueError('Only observed TRAIN binary classes define assignments')
        ids.append(graph_id); labels.append(int(target)); atoms.append(len(graph.x)); positions.append(index)
    table = load(TABLE / 'table.py').make_table(ids, labels, atoms, role='TRAIN',
                                              permutation_seed=seed if permuted else None)
    weights = t.ones(4, len(view), dtype=t.float64, device='cpu')
    weights[:, t.tensor(positions, dtype=t.long)] = table.tensor_weights(t)
    return table, weights, missing


def native_weighted_terms(session, routes, batch, positions, weights):
    """FP32 native observed BCE, native mean reduction, fixed FP64 weights."""
    t = session.torch
    losses = t.stack([session._native_loss(route[0], batch) for route in routes])
    F = losses.mean()
    observed = batch.y == batch.y
    if tuple(batch.y.shape) != (len(positions), 1) or not observed.any().item() \
            or any(route[0].shape != batch.y.shape for route in routes):
        raise ValueError('Original single-target observed-label BCE batch shape required')
    # Slice observed targets BEFORE BCE: absent labels never enter a gradient.
    records = t.stack([t.nn.functional.binary_cross_entropy_with_logits(
        route[0].to(t.float32)[observed], batch.y.to(t.float32)[observed], reduction='none') for route in routes])
    selected = weights[:, positions.detach().cpu()][:, observed.reshape(-1).detach().cpu()].to(session.device)
    W = session.rules.weighted_native_own_loss(t, records, selected, native_reduce=lambda row: row.mean(),
                                              native_regularizer=F.new_zeros(()))
    return F, W, losses


def transform(session, job):
    method = job.get('procedure')
    if method not in METHODS or job.get('TEST_access') is not False or job.get('family') != 'ogb':
        raise ValueError('Explicit declared composition and closed TEST required')
    settings = job.get('method_settings', {})
    if set(settings) != {'manifest_sha256'}: raise ValueError('No weight/partition/strength settings are adjustable')
    root = Path(job['recipe']['_repository_root']).resolve()
    receipts = source_bindings(root, settings['manifest_sha256'])
    parent_path = PARENT / ('JOB_seed' + str(job['seed']) + '.json')
    parent = json.loads(parent_path.read_text())
    expected = descriptor(parent, method, settings['manifest_sha256'])
    if job.get('device') == 'cpu': expected['device'] = 'cpu'  # explicit engineering execution override
    if job != expected: raise ValueError('Only the exact unchanged parent recipe/policy/seed with this composition is defined')
    if getattr(session, 'full_responsibilities', None) is not None:
        raise ValueError('Fresh source with no previous assignment required')
    load(PARENT / 'session_extension.py').transform(session, parent)
    model, optimizer = session.model, session.optimizer
    view = partition(model, optimizer, require_zero_up=True)
    if Path(session.rules.__file__).resolve() != PHASE / 'reference_wrapper_redesign_20261013_v1/runtime/training_rules.py':
        raise ValueError('Original source block updater required')
    table, weights, missing = weights_from_train(session, permuted=METHODS[method], seed=job['seed'])
    groups = tuple(optimizer.param_groups)
    fixed_optimizer_contract = optimizer_contract(optimizer)
    binding = {'schema': SCHEMA, 'method_id': method, 'job_sha256': digest(job), 'seed': job['seed'],
        'parent_JOB': pin(parent_path), 'source_receipts': receipts, 'table': table.summary(),
        'assignment': 'native TRAIN atom-count quartiles within observed class; exact retained fractional tie rule',
        'control': 'whole four-route vector permutation within class' if METHODS[method] else 'structural',
        'rule': 'w=.5+2q; positive [.5,2.5]; exact original class mass and mean-one per object',
        'missing_TRAIN_positions': missing, 'missing_weights': 'unused ones; native observed-label mask unchanged',
        'shared_credit': 'unweighted native F, including native BN affine',
        'private_credit': 'weighted native W, only original 19200 residual down/up coordinates',
        'native_reduction': 'native mixed-batch observed-label BCE mean then mean across four routes; no minibatch renormalization',
        'optimizer': 'original Adam, groups/options/parameter ownership unchanged; one transition after both old-state collections',
        'new_learned_parameters': 0, 'weight_tensor_bytes': weights.numel() * weights.element_size(),
        'global_scalar_objective_claim': False, 'prediction_pool': 'unchanged mean raw logits',
        'TEST_accessed': False, 'qualification': 'source only; exact native engineering execution pending'}
    original_state, original_load, original_metadata = session.state, session.load_state, session.metadata
    busy = [False]

    def identity():
        if session.model is not model or session.optimizer is not optimizer or session.full_responsibilities is not None:
            raise ValueError('Composition ownership changed')
        current = partition(model, optimizer)
        if any(a is not b for a, b in zip(current.private_parameters(), view.private_parameters())) \
                or tuple(id(p) for _, p in current._source_parameters) != tuple(id(p) for _, p in view._source_parameters):
            raise ValueError('Native/residual parameter partition changed')
        check_optimizer_contract(optimizer, fixed_optimizer_contract, groups)
        model.assert_source_identity(); model.route_bn_state.assert_native_identity()

    def train_update(self):
        if busy[0]: raise ValueError('Nonnested native update required')
        identity(); busy[0] = True
        try:
            self.model.train()
            batch, positions, skipped = self._next_train_batch()
            self.last_train_positions = positions.clone()
            if not self.torch.equal(batch.smoke_graph_id.reshape(-1).detach().cpu(), self.train_ids[positions].cpu()):
                raise ValueError('Native TRAIN positions and actual graph identities differ')
            self.optimizer.zero_grad()
            routes = [self.model.run_member(member, batch) for member in range(4)]
            F, W, losses = native_weighted_terms(self, routes, batch, positions, weights)
            work = self.rules.step_private_responsibility(self.torch, view, self.optimizer, F=F, F_weighted=W)
            identity(); self.updates += 1
            return {'loss': float(W.detach().cpu()), 'native_unweighted_F': float(F.detach().cpu()),
                'private_weighted_W': float(W.detach().cpu()), 'native_member_losses': losses.detach().cpu().tolist(),
                'loss_semantics': 'W diagnostic, not global objective; native shared F/private residual W',
                'optimizer_updates': 1, 'update': self.updates, 'TRAIN_graph_ids': self.train_ids[positions].tolist(),
                'native_skipped_batches': skipped, 'native_batch_graphs': len(positions), 'native_route_forwards': 4,
                'work': {**work, 'extra_backbone_forwards': 0}, 'table_sha256': table.digest()}
        finally:
            busy[0] = False

    def state(self):
        if busy[0]: raise ValueError('Checkpoint only between native updates')
        identity()
        return {'schema': SCHEMA, 'method': copy.deepcopy(binding), 'table_record': table.record(),
                'weights': weights.clone(), 'parent_state': original_state()}

    def load_state(self, value):
        nonlocal groups
        if busy[0] or value.get('schema') != SCHEMA or value.get('method') != binding \
                or value.get('table_record') != table.record() or value['weights'].dtype != self.torch.float64 \
                or value['weights'].shape != weights.shape or not self.torch.equal(value['weights'].cpu(), weights) \
                or value['parent_state']['source_state']['responsibilities'] is not None:
            raise ValueError('Exact composition, seed, source, TRAIN assignment and permutation required')
        # The unchanged residual producer owns its exact nested state and BN law.
        identity()  # Reject pre-existing tampering before invoking a valid load.
        original_load(copy.deepcopy(value['parent_state']))
        # Accept new group dictionaries only here, never recapture a new semantic
        # baseline from checkpoint state. Live Parameter IDs remain exact.
        groups = check_optimizer_contract(optimizer, fixed_optimizer_contract, groups, after_parent_restore=True)
        identity()

    def metadata(self):
        return {**original_metadata(), 'structural_residual_composition': copy.deepcopy(binding)}

    session.train_update = types.MethodType(train_update, session)
    session.state, session.load_state = types.MethodType(state, session), types.MethodType(load_state, session)
    session.metadata = types.MethodType(metadata, session)
    identity()
    return session
