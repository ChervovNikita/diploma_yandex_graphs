"""Exact PubMed-Cheb residual F/shared, W/private composition; stdlib import."""
import copy
import hashlib
import json
from pathlib import Path
import types
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
PHASE = E.parent
METHODS = {'pubmed_cheb_uniform_residual_v1': 'uniform',
           'pubmed_cheb_structural_residual_v1': 'structured',
           'pubmed_cheb_permuted_residual_v1': 'permuted'}
SCHEMA = 'pubmed-cheb-residual-block-session-v1'
PORTABLE = E / 'portable_private_input_v2'
LEGACY = E / 'research/structural_residual_composition_v2/extension.py'


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(path):
    name = '_pubmed_cheb_' + pin(path)['sha256']
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def source_bindings(root, expected):
    root = Path(root).resolve()
    require(HERE.is_relative_to(root), 'Bind the actual repository containing this composition')
    manifest = pin(HERE / 'MANIFEST.json')
    require(manifest['sha256'] == expected, 'Exact reviewed manifest required')
    receipts = [manifest]
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = (HERE / row['path']).resolve()
        require(path.is_relative_to(HERE), 'Package source escaped')
        actual = pin(path); require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Package source changed')
        receipts.append(actual)
    for row in json.loads((HERE / 'SOURCES.json').read_text())['files']:
        path = (PHASE / row['path_relative_to_phase']).resolve()
        require(path.is_relative_to(root), 'Retained source escaped repository')
        actual = pin(path); require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Retained source changed')
        receipts.append(actual)
    return receipts


def policy():
    return {'track': 'matched_extended', 'minimum_epochs': 2000, 'maximum_epochs': 10000,
            'patience': 100, 'metric': 'accuracy', 'direction': 'max', 'ties': 'first',
            'evaluate_TRAIN_each_epoch': False}


def descriptor(root, seed, variant, manifest_sha256):
    """In-memory descriptor only; not a scientific fit/queue admission."""
    require(type(seed) is int and seed in (101, 202, 303) and variant in METHODS.values(), 'Frozen seed/variant required')
    case = json.loads((HERE / 'BASE_CASE.json').read_text())
    recipe = copy.deepcopy(case['recipe']); recipe['_repository_root'] = str(Path(root).resolve())
    method = next(k for k, v in METHODS.items() if v == variant)
    return {'schema': 'native-full-epoch-fit-v1', 'id': 'pubmed_cheb_residual_composition_v1/seed-' + str(seed) + '/' + method,
        'cohort_id': 'pubmed_cheb_residual_composition_v1/split-0/seed-' + str(seed),
        'family': 'node', 'recipe': recipe, 'callback_module': case['callback_module'],
        'callback_sha256': case['callback_sha256'], 'data_root': case['data_root'],
        'policy': policy(), 'TEST_access': False, 'split': 0, 'device': 'cuda:0', 'torch_threads': 2,
        'procedure': method, 'base_procedure': 'gnnm_boundary_native', 'seed': seed, 'body': None,
        'method_settings': {'manifest_sha256': manifest_sha256},
        'session_transform': {'path': str(HERE / 'extension.py'), 'function': 'transform', 'sha256': pin(__file__)['sha256']}}


def partition(model, optimizer, *, initial=False):
    native = tuple(model.native.named_parameters()); private = tuple(model.private_parameters())
    require(len(private) == 2 and private[0] is model.input_residual.down and private[1] is model.input_residual.up and
        tuple(private[0].shape) == (4,8,128) and tuple(private[1].shape) == (4,128,8) and
        sum(p.numel() for p in private) == 8192 and sum(p.numel() for _, p in native) == 592435,
        'Exact native PubMed-Cheb and input-only residual partition required')
    require(not initial or private[1].count_nonzero().item() == 0, 'Native zero-up start required')
    owned = tuple(p for group in optimizer.param_groups for p in group['params'])
    known = tuple(p for _, p in native) + private
    require(len(owned) == len({id(p) for p in owned}) == len(known) and
        {id(p) for p in owned} == {id(p) for p in known} == {id(p) for p in model.parameters()} and
        all(p.requires_grad for p in known), 'Every native/private coordinate must have one optimizer owner')
    return types.SimpleNamespace(_source_parameters=native, private_parameters=lambda: private)


def train_assignment(session, variant, seed):
    t = session.torch; data = session.data
    ids = session.train_ids.detach().cpu()
    labels = session.targets.detach().cpu()
    require(t.equal(ids, t.nonzero(data.train_mask, as_tuple=False).flatten().cpu()) and
        ids.numel() == 11829 and t.equal(labels, data.y[session.train_ids].detach().cpu()) and
        t.bincount(labels, minlength=3).tolist() == [3943]*3 and data.num_nodes == 19717,
        'Complete original TRAIN IDs/classes required')
    authority = json.loads((HERE / 'BINDING.json').read_text())['provider']
    for value, key in ((ids, 'ids_sha256'), (labels, 'labels_sha256')):
        require(hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() == authority['role_digests']['TRAIN'][key],
            'Original TRAIN role identity changed')
    valid_mask = data.val_mask.detach().cpu()
    require(hashlib.sha256(valid_mask.numpy().tobytes()).hexdigest() == authority['VALID_mask_sha256'], 'Original VALID mask changed')
    module = load(E / 'structural_class_responsibility_v1/table.py')
    edges = data.edge_index.detach().cpu()
    require(tuple(edges.shape) == (2,88648), 'Exact source-preprocessed PubMed edge records required')
    degree = module.native_incoming_counts(data.num_nodes, edges[0].tolist(), edges[1].tolist(), ids.tolist())
    table = module.make_table(ids.tolist(), labels.tolist(), degree, role='TRAIN',
        permutation_seed=seed if variant == 'permuted' else None)
    require(any(c['unique_weight_vectors'] > 1 for c in table.summary()['classes']), 'Entire structural table is uniform; no contrast')
    weights = t.ones(4, len(ids), dtype=t.float64) if variant == 'uniform' else table.tensor_weights(t)
    return table, weights


def terms(session, routes, weights, *, uniform):
    t = session.torch; targets = session.targets
    require(len(routes) == 4 and all(tuple(r.raw_logits.shape) == (19717,3) for r in routes), 'Full native four-route logits required')
    losses = t.stack([t.nn.functional.cross_entropy(r.raw_logits[session.train_ids], targets) for r in routes])
    F = losses.mean()
    if uniform: return F, F, losses
    records = t.stack([t.nn.functional.cross_entropy(r.raw_logits[session.train_ids], targets, reduction='none') for r in routes])
    W = session.rules.weighted_native_own_loss(t, records, weights.to(session.device),
        native_reduce=lambda row: row.mean(), native_regularizer=F.new_zeros(()))
    return F, W, losses


def transform(session, job):
    require(job.get('family') == 'node' and job.get('procedure') in METHODS and job.get('TEST_access') is False,
        'Explicit PubMed-Cheb node method with TEST closed required')
    settings = job.get('method_settings', {})
    require(set(settings) == {'manifest_sha256'}, 'No adjustable residual, weight or objective settings')
    root = Path(job['recipe']['_repository_root']).resolve()
    receipts = source_bindings(root, settings['manifest_sha256'])
    variant = METHODS[job['procedure']]
    expected = descriptor(root, job['seed'], variant, settings['manifest_sha256'])
    if job.get('device') == 'cpu': expected['device'] = 'cpu'  # explicit engineering device only
    require(job == expected, 'Exact frozen source preset/split/controller/seed descriptor required')
    require(session.procedure == 'gnnm_boundary_native' and session.updates == 0 and session.responsibility is None and
        not hasattr(session, '_portable_private_input_attached') and session.recipe['id'] == 'polyformer-pubmed-cheb',
        'Attach only to one fresh unextended native PubMed-Cheb session')
    require(not any(isinstance(m, session.torch.nn.modules.batchnorm._BatchNorm) for m in session.model.modules()),
        'No native BN or synthetic private BN is admitted')
    native = session.core.native
    require(tuple(native.lin1.weight.shape) == (128,500) and tuple(native.lin3.weight.shape) == (3,128) and
        native.K == 7 and native.nlayers == 1 and len(session.data.list_mat) == 7 and
        all(tuple(z.shape) == (19717,500) for z in session.data.list_mat) and
        all(p.dtype == session.torch.float32 for p in native.parameters()),
        'Exact full-feature FP32 PubMed-Cheb shape/token contract required')
    portable = load(PORTABLE / 'extension.py')
    parent = copy.deepcopy(job); parent['procedure'] = 'portable_input_residual_only_v1'
    parent['method_settings'] = {'rank_cap': 8, 'frame_policy': 'disjoint_if_fit_else_separate_orthonormal_frames',
        'manifest_sha256': pin(PORTABLE / 'MANIFEST.json')['sha256']}
    portable.transform(session, parent)
    model, optimizer = session.model, session.optimizer
    view = partition(model, optimizer, initial=True)
    require(set(model._modules) == {'native','input_residual'} and session.core is model and
        Path(session.rules.__file__).resolve() == PHASE / 'reference_wrapper_redesign_20261013_v1/runtime/training_rules.py',
        'Residual-only node wrapper and original updater required')
    table, weights = train_assignment(session, variant, job['seed'])
    helper = load(LEGACY); groups = tuple(optimizer.param_groups); contract = helper.optimizer_contract(optimizer)
    original_state, original_load = session.state, session.load_state
    binding = {'schema': SCHEMA, 'method': job['procedure'], 'job_sha256': digest(job), 'variant': variant,
        'seed': job['seed'], 'source_receipts': receipts, 'table': table.summary(),
        'weights': 'all ones, W=F exactly' if variant == 'uniform' else '.5+2q; native degree quartiles; exact tied masses',
        'table_is_unapplied_structural_counterfactual': variant == 'uniform',
        'applied_weight_range': [1,1] if variant == 'uniform' else [.5,2.5],
        'shared': 'all original native coordinates receive unweighted native F', 'private': 'only8192 residual down/up receive W',
        'reverse_collections': 2, 'optimizer_transitions': 1, 'prediction_pool': 'mean raw logits',
        'BN_parameters_or_buffers_added': 0, 'new_auxiliary_loss': False, 'global_scalar_objective_claim': False,
        'weight_FP64_CPU_bytes': weights.numel()*weights.element_size(), 'TEST_accessed': False}
    busy = False

    def identity():
        require(session.model is model and session.core is model and session.optimizer is optimizer and
            session.responsibility is None, 'Native session ownership changed')
        current = partition(model, optimizer)
        require(tuple(id(p) for _, p in current._source_parameters) == tuple(id(p) for _, p in view._source_parameters) and
            tuple(map(id, current.private_parameters())) == tuple(map(id, view.private_parameters())), 'Partition changed')
        helper.check_optimizer_contract(optimizer, contract, groups); model.assert_source_identity()

    def train_update(self):
        nonlocal busy
        require(not busy, 'Nonnested full-graph update required'); identity(); busy = True
        try:
            self.model.train(); self.optimizer.zero_grad()
            routes = self._routes(True)  # unchanged source activation-checkpoint/dropout orchestration
            F, W, losses = terms(self, routes, weights, uniform=variant == 'uniform')
            work = self.rules.step_private_responsibility(self.torch, view, optimizer, F=F, F_weighted=W)
            identity(); self.updates += 1
            return {'loss': float(F.detach()), 'native_unweighted_F': float(F.detach()), 'private_weighted_W': float(W.detach()),
                'native_member_losses': losses.detach().cpu().tolist(), 'optimizer_updates': 1, 'logical_update': self.updates,
                'train_records': len(self.train_ids), 'graph_nodes': int(self.data.num_nodes), 'whole_native_train': True,
                'native_route_forward_views': 4, 'activation_checkpointing': True,
                'checkpoint_recomputation_work_not_counted': True, 'work': work, 'table_sha256': table.digest(),
                'loss_semantics': 'F diagnostic; shared F/private W block field, no single scalar objective claim'}
        finally: busy = False

    def state(self):
        require(not busy, 'State only at completed-update boundary'); identity()
        return {'schema': SCHEMA, 'method': copy.deepcopy(binding), 'table_record': table.record(),
            'weights': weights.clone(), 'parent_state': original_state()}

    def load_state(self, value):
        nonlocal groups
        require(not busy and value.get('schema') == SCHEMA and value.get('method') == binding and
            value.get('table_record') == table.record() and value['weights'].dtype == self.torch.float64 and
            value['weights'].shape == weights.shape and self.torch.equal(value['weights'].cpu(), weights) and
            value['parent_state']['source_state']['responsibility'] is None, 'Exact method/source/degree/weights required')
        identity(); original_load(copy.deepcopy(value['parent_state']))
        groups = helper.check_optimizer_contract(optimizer, contract, groups, after_parent_restore=True)
        identity()

    session.train_update = types.MethodType(train_update, session)
    session.state, session.load_state = types.MethodType(state, session), types.MethodType(load_state, session)
    session.source_metadata['pubmed_cheb_residual_composition'] = copy.deepcopy(binding)
    identity(); return session
