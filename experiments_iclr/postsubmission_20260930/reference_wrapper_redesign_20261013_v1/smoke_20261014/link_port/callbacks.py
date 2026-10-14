"""Actual source-native Collab construction, TRAIN updates and VALID callbacks.

No numerical imports or operations occur merely by importing this frontend.
The common owner admits data/runtime/device and calls create_session. TEST
files are never opened. No subprocess, install, network, fit loop or GO lives
here. One train_update is one complete source-sized minibatch optimizer map.
"""
import copy
import hashlib
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def _local_module(name):
    path = HERE / (name + '.py')
    spec = importlib.util.spec_from_file_location('_redesign_link_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clone(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone(torch, v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(torch, v) for v in value)
    if isinstance(value, list):
        return [_clone(torch, v) for v in value]
    return copy.deepcopy(value)


def _adapter_receipts():
    manifest_path = HERE / 'MANIFEST.json'
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    files = {row['path']: row for row in manifest['files']}
    receipts = {}
    for name in ('callbacks.py', 'boundaries.py', 'source_loader.py'):
        path = HERE / name
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        frozen = files[name]
        if frozen['bytes'] != len(raw) or frozen['sha256'] != digest:
            raise ValueError('Frozen link smoke adapter changed: ' + name)
        receipts[name] = {'path': str(path), 'bytes': len(raw), 'sha256': digest}
    return receipts, {'path': str(manifest_path), 'bytes': len(manifest_raw),
                      'sha256': hashlib.sha256(manifest_raw).hexdigest(),
                      'adapter_hashes_verified': True}


def _procedure(procedure):
    name = str(procedure).lower().replace('-', '_')
    aliases = {'native': 'native', 'native_m1': 'native', 'ordinary_m1': 'native', 'native_single': 'native',
               'base': 'base', 'gnnm_base': 'base', 'gnnm_boundary_own': 'base',
               'gnnm_boundary_native': 'base',
               'global_half': 'global_half', 'global_half_pool': 'global_half',
               'gnnm_global_half_pool': 'global_half', 'gnnm_global_pool': 'global_half',
               'boundary_global_half_pool': 'global_half',
               'private_half': 'private_half', 'private_half_pool': 'private_half',
               'gnnm_private_half_pool': 'private_half', 'gnnm_private_pool': 'private_half',
               'boundary_private_half_pool': 'private_half'}
    if name not in aliases:
        raise ValueError('Link smoke permits native/base/global-half/private-half only: ' + name)
    return aliases[name]


def _settings(recipe):
    return recipe.get('native_fixed_config', recipe.get('settings', recipe.get('hyperparameters', recipe.get('config', {}))))


def _fixed_config(family):
    if family == 'BUDDY':
        return {'K': 50, 'add_normed_features': True, 'batch_size': 1024,
            'cache_subgraph_features': True, 'eval_batch_size': 1000000,
            'feature_dropout': .05, 'floor_sf': False, 'hidden': 1024,
            'hll_p': 8, 'label_dropout': .1, 'lr': .02, 'max_hash_hops': 2,
            'minhash_num_perm': 128, 'num_negs': 1, 'sign_k': 0,
            'subgraph_feature_batch_size': 11000000, 'use_RA': False,
            'use_zero_one': False, 'weight_decay': 0, 'year': 2007}
    return {'alpha': 1.05, 'batch_size': 65536, 'encoder': 'gcn',
        'encoder_dropout': .1, 'encoder_edge_dropout': .25, 'encoder_layers': 1,
        'encoder_lr': .0082, 'hidden': 64, 'ln': True, 'lnnn': True,
        'maskinput': True, 'predictor': 'cn1' if family == 'NCN' else 'incn1cn1',
        'predictor_dropout': .3, 'predictor_edge_dropout': 0,
        'predictor_lr': .0037, 'proboffset': 6., 'probscale': 2.5, 'pt': .1,
        'residual': True, 'tail_dropout': .05, 'tailact': True,
        'test_batch_size': 131072, 'use_xlin': True, 'weight_decay': 0,
        'xdropout': .25}


def _validate_settings(recipe, family):
    declared = _settings(recipe)
    if not isinstance(declared, dict):
        raise ValueError('native_fixed_config must be a dictionary')
    expected = _fixed_config(family)
    unknown = sorted(set(declared) - set(expected))
    conflicts = {key: {'declared': value, 'source_fixed': expected[key]}
                 for key, value in declared.items() if key in expected and value != expected[key]}
    if unknown or conflicts:
        raise ValueError('Only the exact source-fixed Collab config is supported: ' +
                         json.dumps({'family': family, 'unknown_keys': unknown, 'conflicts': conflicts}, sort_keys=True))


def _family(recipe):
    family = recipe.get('architecture', recipe.get('backbone', recipe.get('family', '')))
    if isinstance(family, dict):
        family = family.get('name', '')
    family = str(family).upper()
    if family not in ('NCN', 'NCNC', 'BUDDY'):
        raise ValueError('Exact native NCN/NCNC/BUDDY Collab family required: ' + family)
    return family


def required_data_candidates(data_root):
    """Closed provider-layout candidates; no recursive dataset discovery."""
    root = Path(data_root)
    phase = HERE.parents[2]
    bases = [root, root / 'ogbl_collab', root / 'ogbl-collab' / 'ogbl_collab',
             root / 'dataset' / 'ogbl_collab', root / 'dataset' / 'ogbl-collab' / 'ogbl_collab',
             phase / 'training_pipeline_v2_20261012/qa/link_admitted_transfer_v1/cache/ogbl_collab']
    return [{'provider': str(base), 'graph': str(base / 'processed/geometric_data_processed.pt'),
             'train': str(base / 'split/time/train.pt'), 'valid': str(base / 'split/time/valid.pt')}
            for base in bases]


def load_collab(torch, recipe, data_root):
    from torch_geometric.data import Data
    declared = recipe.get('data_files')
    if declared is not None:
        paths = {k: Path(declared[k]) for k in ('graph', 'train', 'valid')}
        paths = {k: p if p.is_absolute() else Path(data_root) / p for k, p in paths.items()}
    else:
        candidates = required_data_candidates(data_root)
        chosen = next((row for row in candidates if all(Path(row[k]).is_file() for k in ('graph', 'train', 'valid'))), None)
        if chosen is None:
            raise FileNotFoundError('Exact official Collab graph/TRAIN/VALID files required; TEST is excluded. Candidates=' + json.dumps(candidates))
        paths = {k: Path(chosen[k]) for k in ('graph', 'train', 'valid')}
    # Source-preserving graph deserialization in the admitted runtime. Nothing
    # calls PygLinkPropPredDataset.get_edge_split or opens a TEST split.
    packed = torch.load(paths['graph'], map_location='cpu', weights_only=False)
    raw = packed[0] if isinstance(packed, tuple) else packed
    if isinstance(raw, dict):
        raw = Data.from_dict(raw)
    if not hasattr(raw, 'x') or not hasattr(raw, 'edge_index'):
        raise ValueError('Unsupported official processed Collab graph format')
    graph = Data(x=raw.x.detach().cpu().clone(), edge_index=raw.edge_index.detach().cpu().clone(),
                 num_nodes=int(raw.num_nodes))
    weights = getattr(raw, 'edge_weight', None)
    if weights is not None:
        graph.edge_weight = weights.detach().cpu().clone()
    if graph.num_nodes != 235868 or tuple(graph.x.shape) != (235868, 128):
        raise ValueError('Full native Collab235868x128 required; graph/feature shrink is unsupported')
    train = torch.load(paths['train'], map_location='cpu', weights_only=False)
    valid = torch.load(paths['valid'], map_location='cpu', weights_only=False)
    train = {k: torch.as_tensor(v) for k, v in train.items()}
    valid = {k: torch.as_tensor(v) for k, v in valid.items()}
    for k in ('edge', 'edge_neg'):
        if valid[k].ndim != 2 or valid[k].shape[1] != 2:
            raise ValueError('Complete official VALID pair records required: ' + k)
    if tuple(valid['edge'].shape) != (60084, 2) or tuple(valid['edge_neg'].shape) != (100000, 2):
        raise ValueError('Native complete Collab VALID60084/100000 required')
    if train['edge'].ndim != 2 or train['edge'].shape[1] != 2:
        raise ValueError('Native complete TRAIN edge records required')
    return graph, train, valid, {k: str(p) for k, p in paths.items()}


class SessionBase:
    def __init__(self, torch, recipe, procedure, device):
        self.t, self.recipe, self.procedure, self.device = torch, recipe, _procedure(procedure), device
        self.seed = int(recipe.get('_seed', recipe.get('seed', recipe.get('optimizer_seed', 0))))
        self.members = 1 if self.procedure == 'native' else 4
        self.rules = _local_module('source_loader').training_rules()
        self.boundary_module = _local_module('boundaries')
        self.updates = 0
        self.epoch = 0
        self.core = None
        self.private_rng = None
        self.task_call_scope = None
        self.allow_missing_grad_names = []
        self.adapter_source_receipts, self.adapter_manifest_receipt = _adapter_receipts()

    def _install(self, model, sites, optimizer_constructor):
        self.model = model.to(self.device)
        self.source_parameter_references = dict(self.model.named_parameters())
        if self.members == 4:
            # Sites refer to the original modules whose .to already moved
            # their original Parameters; no native class/Parameter replacement.
            self.core = self.boundary_module.make_boundary_core(self.t, self.model, sites, self.seed)
            self.model.add_module('_canonical_boundary_banks', self.core)
        self.optimizer = optimizer_constructor()
        self.optimizers = [self.optimizer]
        self.private_rng = self.boundary_module.PrivateRNG(self.t, self.seed, 4) if self.members == 4 else None
        self.boundary_bindings = self.core.binding_metadata if self.core else []
        self.native_parameter_identity_preserved = all(
            dict(self.model.named_parameters()).get(name) is parameter
            for name, parameter in self.source_parameter_references.items())
        if not self.native_parameter_identity_preserved:
            raise ValueError('Native parameter identity was changed by boundary construction')

    def _route(self, member, training):
        if self.core is None:
            return nullcontext()
        # Compose route identity and an independent persistent source RNG.
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(self.core.member(member))
        if training:
            stack.enter_context(self.private_rng.member(member))
        return stack

    def _step(self, native_losses, pooled_loss):
        t = self.t
        F = t.stack(native_losses).mean()
        if self.procedure in ('native', 'base'):
            F.backward(); self.optimizer.step()
            counts = {'reverse_collections': 1, 'optimizer_transitions': 1, 'all_gradients_from_old_state': True}
        elif self.procedure == 'global_half':
            counts = self.rules.step_global_half_pool(t, self.core, self.optimizer, F=F, L=pooled_loss)
        else:
            counts = self.rules.step_private_half_pool(t, self.core, self.optimizer, F=F, L=pooled_loss)
        self.updates += 1
        objective = F if self.procedure in ('native', 'base') else .5 * F + .5 * pooled_loss
        return {'loss': float(objective.detach()), 'lossfinite': bool(t.isfinite(objective).item()),
                'own_loss': float(F.detach()), 'pool_loss': float(pooled_loss.detach()),
                'optimizer_updates': 1, **counts}

    def _validate_result(self, positive, negative, work):
        t = self.t
        from ogb.linkproppred import Evaluator
        evaluator = Evaluator(name='ogbl-collab'); evaluator.K = 50
        metric = evaluator.eval({'y_pred_pos': positive, 'y_pred_neg': negative})['hits@50']
        # VALID loss is an engineering diagnostic, not a new selection rule.
        loss = -t.nn.functional.logsigmoid(positive).mean() - t.nn.functional.logsigmoid(-negative).mean()
        return {'outputs': t.cat((positive, negative)), 'role': 'VALID',
                'loss': float(loss), 'diagnostic_loss': float(loss), 'lossfinite': bool(t.isfinite(loss).item()),
                'metrics': {'hits@50': float(metric)}, 'work': work,
                'complete_VALID_positive_records': len(positive), 'complete_VALID_negative_records': len(negative),
                'TEST_opened': False, 'TEST_scored': False}

    def state(self):
        return _clone(self.t, {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'updates': self.updates, 'epoch': self.epoch, 'sampler': self.sampler_state(),
            'private_rng': self.private_rng.streams if self.private_rng else None,
            'active_boundary_route': self.core.route if self.core else None,
            'NCNC_task_call_scope': self.task_call_scope.capture() if self.task_call_scope else None})

    def load_state(self, state):
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.updates, self.epoch = int(state['updates']), int(state['epoch'])
        if self.private_rng is not None:
            self.private_rng.streams = _clone(self.t, state['private_rng'])
            self.core.route = int(state['active_boundary_route'])
        if self.task_call_scope is not None:
            self.task_call_scope.restore(state['NCNC_task_call_scope'])
        self.load_sampler_state(state['sampler'])


class NCNSession(SessionBase):
    def __init__(self, torch, recipe, procedure, data_root, device):
        super().__init__(torch, recipe, procedure, device)
        from torch_sparse import SparseTensor
        from torch_geometric.utils import negative_sampling
        self.SparseTensor, self.negative_sampling = SparseTensor, negative_sampling
        self.utils, source, self.source_proofs = _local_module('source_loader').load_ncn()
        self.graph, self.train, self.valid, self.data_files = load_collab(torch, recipe, data_root)
        self.x = self.graph.x.to(device)
        self.edge_index = self.graph.edge_index.to(device)
        self.adj = SparseTensor.from_edge_index(self.edge_index, sparse_sizes=(235868, 235868)).to_symmetric().coalesce()
        self.positive = self.train['edge'].to(device).t()
        self.batch_size = 65536
        self.test_batch_size = 131072
        self.encoder = source.GCN(128, 64, 64, 1, .1, True, True, -1, 'gcn', False, .25,
                                  xdropout=.25, taildropout=.05, noinputlin=False)
        kwargs = dict(use_xlin=True, tailact=True, twolayerlin=False, beta=1.)
        family = _family(recipe)
        if family == 'NCNC':
            kwargs.update(depth=1, splitsize=-1, scale=2.5, offset=6., trainresdeg=-1,
                          testresdeg=-1, pt=.1, learnablept=False, alpha=1.05)
        pred = source.predictor_dict['cn1' if family == 'NCN' else 'incn1cn1']
        self.predictor = pred(64, 64, 1, 3, .3, 0., True, **kwargs)
        model = torch.nn.Module(); model.add_module('encoder', self.encoder); model.add_module('predictor', self.predictor)
        first = self.encoder.convs[0].lin
        last = self.predictor.lin[-1]
        if first.weight.shape != (64, 128) or last.weight.shape != (1, 64):
            raise ValueError('Declared canonical encoder-input/task-final sites changed')
        sites = [('encoder.convs.0.lin', first, True), ('predictor.lin.' + str(len(self.predictor.lin) - 1), last, False)]
        self.task_call_scope = None
        if family == 'NCNC' and self.members == 4:
            # Original forward/recursion/operator bodies still execute. Every
            # native recursive forward dispatches through multidomainforward;
            # only the outer task's final affine receives output factors.
            self.task_call_scope = self.boundary_module.TaskCallScope()
            self.predictor.multidomainforward = self.task_call_scope.wrap(self.predictor.multidomainforward)
            sites[-1] = sites[-1] + (self.task_call_scope.task_output_active, self.task_call_scope.observe_output)

        def optimizer():
            encoder = list(self.encoder.parameters())
            predictor = list(self.predictor.parameters())
            if self.core:
                encoder += list(self.core.banks['site_0'].parameters())
                predictor += list(self.core.banks['site_1'].parameters())
            return torch.optim.Adam([{'params': encoder, 'lr': .0082}, {'params': predictor, 'lr': .0037}])

        self._install(model, sites, optimizer)
        self.allow_missing_grad_names = [n for n in self.source_parameter_references if n.startswith('predictor.ptlin.')]
        self.permutation = None; self.negative = None; self.cursor = 0
        self.metadata = {'family': family, 'source_config': 'literal_collab_cn1/incn1cn1',
            'declared_native_fixed_config': _settings(recipe),
            'canonical_scope': 'first native GCNConv PyGLinear input and task-only final predictor affine; recursive native NCNC operators unchanged',
            'NCNC_output_scope': 'top-level native task dispatch only; recursive completion final affines remain unwrapped' if self.task_call_scope else None,
            'boundary_bindings': self.boundary_bindings, 'native_parameter_identity_preserved': self.native_parameter_identity_preserved,
            'batch_size': self.batch_size, 'source_training_tail': 'dropped', 'data_files': self.data_files,
            'source_proofs': self.source_proofs, 'TEST_opened': False}
        self.metadata.update(adapter_source_receipts=self.adapter_source_receipts,
                             adapter_manifest=self.adapter_manifest_receipt)
        self.source_metadata = self.metadata

    def _next_batch(self):
        if self.permutation is None or self.cursor + self.batch_size > self.positive.shape[1]:
            # Exact source ordering: sample all negatives first, then the native
            # PermIterator's one shuffled TRAIN permutation; discard tail.
            self.negative = self.negative_sampling(self.edge_index, 235868)
            if self.negative.shape[1] < self.positive.shape[1]:
                raise ValueError('Source negative population shorter than TRAIN records')
            self.permutation = self.utils.PermIterator(self.device, self.positive.shape[1], self.batch_size).idx
            self.cursor = 0; self.epoch += 1
        perm = self.permutation[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return perm

    def train_update(self):
        t = self.t
        routing_before = self.task_call_scope.capture() if self.task_call_scope else None
        self.model.train(); self.optimizer.zero_grad()
        perm = self._next_batch()
        # Original record mask occurs before symmetrization/coalescing.
        mask = t.ones(self.positive.shape[1], dtype=t.bool, device=self.device); mask[perm] = False
        adj = self.SparseTensor.from_edge_index(self.positive[:, mask], sparse_sizes=(235868, 235868)).to_device(self.device, non_blocking=True).to_symmetric()
        positive_edge, negative_edge = self.positive[:, perm], self.negative[:, perm]
        positives, negatives, losses = [], [], []
        for member in range(self.members):
            with self._route(member, True):
                h = self.encoder(self.x, adj)
                pos = self.predictor.multidomainforward(h, adj, positive_edge, cndropprobs=[])
                neg = self.predictor.multidomainforward(h, adj, negative_edge, cndropprobs=[])
                losses.append(-t.nn.functional.logsigmoid(pos).mean() - t.nn.functional.logsigmoid(-neg).mean())
                positives.append(pos); negatives.append(neg)
        pooled_pos = t.stack(positives).mean(0); pooled_neg = t.stack(negatives).mean(0)
        pool = -t.nn.functional.logsigmoid(pooled_pos).mean() - t.nn.functional.logsigmoid(-pooled_neg).mean()
        result = self._step(losses, pool)
        result['work'] = {'encoder_forwards': self.members, 'top_level_positive_predictor_calls': self.members,
            'top_level_negative_predictor_calls': self.members, 'recursive_completion_preserved': _family(self.recipe) == 'NCNC',
            'positive_query_records': len(perm), 'negative_query_records': len(perm),
            'source_record_mask_before_coalesce': True, 'complete_source_sized_minibatch': True}
        if self.task_call_scope:
            result['work']['NCNC_output_routing'] = self.task_call_scope.routing_since(routing_before)
        return result

    def validate(self):
        t = self.t
        routing_before = self.task_call_scope.capture() if self.task_call_scope else None
        self.model.eval(); positives, negatives = [], []
        with t.no_grad():
            for member in range(self.members):
                with self._route(member, False):
                    h = self.encoder(self.x, self.adj)
                    route = []
                    for key in ('edge', 'edge_neg'):
                        pairs = self.valid[key].to(self.device)
                        parts = [self.predictor(h, self.adj, pairs[p].t()).squeeze(-1).cpu()
                                 for p in self.utils.PermIterator(self.device, len(pairs), self.test_batch_size, False)]
                        route.append(t.cat(parts))
                    positives.append(route[0]); negatives.append(route[1])
        work = {'encoder_forwards': self.members, 'all_official_VALID_records': True,
                'recursive_completion_preserved': _family(self.recipe) == 'NCNC'}
        if self.task_call_scope:
            work['NCNC_output_routing'] = self.task_call_scope.routing_since(routing_before)
        return self._validate_result(t.stack(positives).mean(0), t.stack(negatives).mean(0), work)

    def sampler_state(self):
        return {'permutation': self.permutation, 'negative': self.negative, 'cursor': self.cursor}

    def load_sampler_state(self, state):
        self.permutation = state['permutation'].to(self.device) if state['permutation'] is not None else None
        self.negative = state['negative'].to(self.device) if state['negative'] is not None else None
        self.cursor = int(state['cursor'])


class BUDDYSession(SessionBase):
    def __init__(self, torch, recipe, procedure, data_root, device):
        super().__init__(torch, recipe, procedure, device)
        import scipy.sparse as ssp
        from torch.utils.data import DataLoader
        from torch_sparse import coalesce
        loader = _local_module('source_loader')
        cls, propagate, Hashes, filter_year, get_negs, self.native_loss, self.source_proofs = loader.load_buddy(torch)
        graph, train, self.valid, self.data_files = load_collab(torch, recipe, data_root)
        if not {'edge', 'year', 'weight'} <= set(train):
            raise ValueError('Native BUDDYyear2007 requires original TRAIN edge/year/weight')
        graph, split = filter_year(graph, {'train': train}, 2007)
        self.positive = split['train']['edge']
        self.train_negative = get_negs(split, graph.edge_index, 235868, 1, 'ogbl-collab')
        self.links = torch.cat((self.positive, self.train_negative), 0)
        self.labels = torch.cat((torch.ones(len(self.positive), dtype=torch.long), torch.zeros(len(self.train_negative), dtype=torch.long)))
        # Native weighted collab coalescing and source SIGN-k0 propagation.
        edge_index, edge_weight = coalesce(graph.edge_index, graph.edge_weight.view(-1), 235868, 235868)
        graph.edge_index, graph.edge_weight = edge_index, edge_weight
        matrix = ssp.csr_matrix((edge_weight.numpy(), (edge_index[0].numpy(), edge_index[1].numpy())), shape=(235868, 235868))
        self.degrees = torch.tensor(matrix.sum(axis=0, dtype=float), dtype=torch.float).flatten()
        self.features = propagate(SimpleNamespace(), graph, edge_index, edge_weight, 0)
        self.args = SimpleNamespace(model='BUDDY', hidden_channels=1024, use_feature=True, feature_dropout=.05,
            label_dropout=.1, max_hash_hops=2, hll_p=8, minhash_num_perm=128, floor_sf=False, use_zero_one=False,
            add_normed_features=True, use_RA=False, sign_k=0, propagate_embeddings=False, sign_dropout=0.,
            batch_size=1024, train_samples=float('inf'), loss='bce', year=2007)
        self.hashes = Hashes(self.args)
        self.hash_tables, self.cards = self.hashes.build_hash_tables(235868, edge_index)
        model = cls(self.args, 128, node_embedding=None)
        sites = [('label_lin_layer', model.label_lin_layer, True), ('lin_feat', model.lin_feat, True), ('lin', model.lin, False)]
        self._install(model, sites, lambda: torch.optim.Adam(self.model.parameters(), lr=.02, weight_decay=0.))
        self.allow_missing_grad_names = [n for n in self.source_parameter_references if n.startswith('bn_RA.')]
        self.DataLoader = DataLoader
        self.sample_indices = None; self.iterator = None; self.batch_cursor = 0; self.loader_rng_before = None
        self.metadata = {'family': 'BUDDY', 'source_config': 'literal_year2007_H1024_lr.02_drop.05/.1_batch1024',
            'declared_native_fixed_config': _settings(recipe),
            'canonical_scope': 'parallel source structural and feature input affines plus source final task affine',
            'native_BN': 'shared original bn_labels/bn_feats/bn_RA; no clones',
            'boundary_bindings': self.boundary_bindings, 'native_parameter_identity_preserved': self.native_parameter_identity_preserved,
            'data_files': self.data_files, 'source_proofs': self.source_proofs,
            'preprocessing': 'unchanged source SIGN-k0 feature propagation and native full-GRAPH hashes; query hydration is streamed without changing query values',
            'all_nodes': 235868, 'feature_width': 128, 'TRAIN_year_cutoff': 2007, 'TEST_opened': False}
        self.metadata.update(adapter_source_receipts=self.adapter_source_receipts,
                             adapter_manifest=self.adapter_manifest_receipt)
        self.source_metadata = self.metadata

    def _new_epoch(self):
        # Source outer sample permutation, followed by native shuffled
        # DataLoader; no capped TRAIN population, skipped query or fake label.
        self.sample_indices = self.t.randperm(len(self.labels))
        self.loader_rng_before = self.t.get_rng_state().clone()
        self.iterator = iter(self.DataLoader(range(len(self.links)), 1024, shuffle=True))
        self.batch_cursor = 0; self.epoch += 1

    def _next_batch(self):
        if self.iterator is None:
            self._new_epoch()
        try:
            indices = next(self.iterator)
        except StopIteration:
            self._new_epoch(); indices = next(self.iterator)
        self.batch_cursor += 1
        return self.sample_indices[indices]

    def _hydrate(self, links):
        # This original source function includes floor/zero-one policies.
        # On-demand hydration replaces an already-built source cache. Its
        # unshuffled feature DataLoader must not add RNG draws to TRAIN's
        # sampler stream; original feature values/operators remain unchanged.
        with self.t.random.fork_rng(devices=[], enabled=True):
            sf = self.hashes.get_subgraph_features(links, self.hash_tables, self.cards, batch_size=11000000)
        x, degrees = self.features[links], self.degrees[links]
        return sf.to(self.device), x.to(self.device), degrees[:, 0].to(self.device), degrees[:, 1].to(self.device)

    def train_update(self):
        t = self.t
        self.model.train(); ids = self._next_batch(); links = self.links[ids]; labels = self.labels[ids].to(self.device)
        inputs = self._hydrate(links)
        self.optimizer.zero_grad(); logits, losses = [], []
        for member in range(self.members):
            with self._route(member, True):
                output = self.model(*inputs, None, None)
                logits.append(output); losses.append(self.native_loss(output, labels))
        pool = self.native_loss(t.stack(logits).mean(0), labels)
        result = self._step(losses, pool)
        result['work'] = {'native_BUDDY_forwards': self.members, 'query_records': len(ids),
            'native_weighted_graph_and_hashes': True, 'native_shared_BN': True,
            'full_source_minibatch': len(ids) == 1024, 'TEST_scored': False}
        return result

    def validate(self):
        t = self.t
        self.model.eval(); outputs = []
        with t.no_grad():
            for key in ('edge', 'edge_neg'):
                pairs = self.valid[key]
                # Native default eval_batch_size1,000,000; all current VALID
                # records fit in one chunk. This is not a query subset.
                chunks = []
                for start in range(0, len(pairs), 1000000):
                    links = pairs[start:start + 1000000]
                    inputs = self._hydrate(links)
                    routes = []
                    for member in range(self.members):
                        with self._route(member, False):
                            routes.append(self.model(*inputs, None, None).reshape(-1).cpu())
                    chunks.append(t.stack(routes).mean(0))
                outputs.append(t.cat(chunks))
        return self._validate_result(outputs[0], outputs[1], {'native_BUDDY_forwards': 2 * self.members,
            'all_official_VALID_records': True, 'no_TEST_graph_or_queries': True})

    def sampler_state(self):
        return {'sample_indices': self.sample_indices, 'batch_cursor': self.batch_cursor,
                'loader_rng_before': self.loader_rng_before, 'train_negative': self.train_negative}

    def load_sampler_state(self, state):
        t = self.t
        self.train_negative = state['train_negative'].cpu()
        self.links = t.cat((self.positive, self.train_negative), 0)
        self.sample_indices = state['sample_indices'].cpu() if state['sample_indices'] is not None else None
        self.batch_cursor = int(state['batch_cursor'])
        self.loader_rng_before = state['loader_rng_before'].cpu() if state['loader_rng_before'] is not None else None
        self.iterator = None
        if self.sample_indices is not None:
            # Reconstruct the original DataLoader iterator/query position under
            # its saved CPU RNG, then restore caller/global state untouched.
            with t.random.fork_rng(devices=[], enabled=True):
                t.set_rng_state(self.loader_rng_before)
                self.iterator = iter(self.DataLoader(range(len(self.links)), 1024, shuffle=True))
                for _ in range(self.batch_cursor):
                    next(self.iterator)


def create_session(recipe: dict, procedure: str, data_root: Path, device: str):
    if recipe.get('dataset', 'ogbl-collab').lower() not in ('ogbl-collab', 'collab'):
        raise ValueError('Only the actual original ogbl-collab task is admitted')
    family = _family(recipe)
    _validate_settings(recipe, family)
    import torch
    return BUDDYSession(torch, recipe, procedure, data_root, device) if family == 'BUDDY' else NCNSession(torch, recipe, procedure, data_root, device)
