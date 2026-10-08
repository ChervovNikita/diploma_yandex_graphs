"""Disabled one-hop label-only corrector core; no native driver or cell owner.

Import is stdlib only. The sole numerical factory defaults to refusal before
loading torch. Caller-owned full-population feature-only H/base logits are the
interface; no dataset, checkpoint, native capture or selected-stage integration.
"""
import ast
import importlib.util
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys

PUBLIC_NAME = 'portable_internal_be_public_interface_20261007_v2'
PUBLIC_MANIFEST_SHA = '190940ca9f8141ac45f739d064cb1ef76aaf1965ba8c91adb544ad8e38fef724'
FACTORS_SHA = '9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3'
MEMBERS = 4
ATTENTION_WIDTH = 64
LABEL_WIDTH = 64
EXECUTION_MODE = 'query_conditioned_dense_label_value_detached_native_core_v1'
GATE_HELPER_SHA = '46e70016760aefbca612f261d87e418ce55abe1921cee9c3e9ed7700d95de7b8'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _isolated_constructor_rng(torch, initializer_seed):
    """Restore ambient CPU/Python/cached-NumPy RNG; no CUDA random call."""
    python_state = random.getstate()
    numpy = sys.modules.get('numpy')
    numpy_state = numpy.random.get_state() if numpy is not None else None
    isolated = torch.Generator(device='cpu').manual_seed(initializer_seed)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(isolated.get_state())
            yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)


@dataclass(frozen=True)
class CommonQueryMask:
    """Immutable, owner-issued positions/IDs; no label-derived fields."""
    owner_id: int
    draw_id: int
    query_positions: tuple
    query_ids: tuple
    allowed_count: int
    query_count: int
    training_value_scale: float


def _factor_conventions(torch, public_root):
    """Compile the three exact pinned primitives, without importing PyG/core.

    FactorLinear arithmetic, private first-factor reset and member_context are
    unmodified AST copies. Unneeded install_factors/native modules are omitted.
    """
    root = Path(public_root).resolve(strict=True)
    require(sha(root / 'MANIFEST.json') == PUBLIC_MANIFEST_SHA, 'Exact public V2 manifest required')
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    row = next(item for item in manifest['files'] if item['path'] == 'core/factors.py')
    source = root / row['path']
    require(source.stat().st_size == row['bytes'] and sha(source) == row['sha256'] == FACTORS_SHA,
            'Exact pinned factor source required')
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in ('FactorLinear', 'initialize_first_factor', 'member_context')]
    require([node.name for node in selected] ==
            ['FactorLinear', 'initialize_first_factor', 'member_context'], 'Exact primitive inventory')
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    ast_sha = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional,
                     contextmanager=contextmanager, __name__='_label_only_pinned_factor_primitives')
    exec(compile(module, str(source) + ':label-only-primitives', 'exec'), namespace)
    return namespace, ast_sha


def _value_gate_helpers():
    path = Path(__file__).resolve().parent / 'query_value_gate_hook.py'
    require(sha(path) == GATE_HELPER_SHA, 'Exact frozen gate/value-input helper required')
    name = '_query_value_gated_helpers_v1'
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def _core_type(torch, factors):
    gates = _value_gate_helpers()
    nn = torch.nn

    class LabelOnlyCorrectorCore(nn.Module):
        def __init__(self, nodes, features, classes, edge_index, train_ids, train_labels,
                     initializer_seed, mask_seed, factor_ast_sha, erase_value_class_identity=False):
            super().__init__()
            self.nodes, self.features, self.classes = nodes, features, classes
            self.initializer_seed, self.mask_seed = initializer_seed, mask_seed
            self.factor_ast_sha = factor_ast_sha
            self.members = MEMBERS
            # All module creation/reset occurs on CPU inside the factory's
            # isolated CPU RNG context. No external native object is accepted.
            self.label_embedding = nn.Embedding(classes, LABEL_WIDTH, dtype=torch.float32)
            self.query = factors['FactorLinear'](nn.Linear(features, ATTENTION_WIDTH, bias=False,
                                                          dtype=torch.float32), MEMBERS)
            self.key = factors['FactorLinear'](nn.Linear(features, ATTENTION_WIDTH, bias=False,
                                                        dtype=torch.float32), MEMBERS)
            self.value = factors['FactorLinear'](nn.Linear(LABEL_WIDTH, LABEL_WIDTH, bias=False,
                                                          dtype=torch.float32), MEMBERS)
            self.output = factors['FactorLinear'](nn.Linear(LABEL_WIDTH, classes, bias=False,
                                                           dtype=torch.float32), MEMBERS)
            self.query_value_gate = gates.make_query_value_gate(
                torch=torch, later_execution_authorized=True)
            self.erase_value_class_identity = erase_value_class_identity
            self._fixed_value_erasure_policy = erase_value_class_identity
            # Established Rademacher first-factor diversity; only Q/K input
            # factors differ. Every s and the V/O input factors start at one.
            factors['initialize_first_factor'](self.query, initializer_seed + 900001)
            factors['initialize_first_factor'](self.key, initializer_seed + 900003)
            with torch.no_grad():
                self.output.weight.zero_()
            nonself = edge_index[0] != edge_index[1]
            self.register_buffer('edge_source', edge_index[0, nonself].clone())
            self.register_buffer('edge_target', edge_index[1, nonself].clone())
            self.register_buffer('train_ids', train_ids.clone())
            self.register_buffer('train_labels', train_labels.clone())
            self.input_edges = int(edge_index.shape[1])
            self.removed_self_edges = self.input_edges - int(self.edge_source.numel())
            self.allowed_count = int(train_ids.numel())
            self.query_count = self.allowed_count // 2
            self.training_value_scale = (self.allowed_count - 1) / (self.allowed_count - self.query_count)
            self.mask_generator = torch.Generator(device='cpu').manual_seed(mask_seed)
            self._pending_mask = None
            self._failed_update = False
            self.steps = 0
            self.counters = dict(mask_draws=0, common_label_fields_built=0,
                route_label_value_fields_built=0,
                training_step_attempts=0, completed_training_steps=0,
                training_route_forward_attempts=0, training_route_forwards=0,
                route_backward_attempts=0, route_backwards=0, corrector_Adam_attempts=0,
                corrector_Adam_steps=0, TRAIN_query_route_label_presentations=0,
                training_edge_score_records=0, inference_edge_score_records=0,
                serving_calls=0, serving_route_forward_attempts=0, serving_route_forwards=0,
                old_corrector_parameter_checks=0, empty_label_neighborhood_rows_checked=0)

        def _record_role_versions(self):
            self._role_versions = tuple(value._version for value in
                (self.edge_source, self.edge_target, self.train_ids, self.train_labels))

        def _check_roles(self):
            require(tuple(value._version for value in
                (self.edge_source, self.edge_target, self.train_ids, self.train_labels)) == self._role_versions,
                'Fixed graph/TRAIN role tensors changed; fresh independently bound core required')
            require(self.erase_value_class_identity is self._fixed_value_erasure_policy,
                    'Permanent value-identity policy cannot change between train/serve calls')

        def _finite_state(self):
            require(all(bool(torch.isfinite(parameter).all()) for parameter in self.parameters()),
                    'Finite corrector parameters required')
            for state in self.optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        require(bool(torch.isfinite(value).all()), 'Finite corrector Adam state required')

        def draw_common_query_mask(self):
            """Uniform fixed-size Q before the external mask-independent capture."""
            self._check_roles()
            require(not self._failed_update and self._pending_mask is None,
                    'No failed update retry or replacement of a pending common mask')
            positions = torch.randperm(self.allowed_count, generator=self.mask_generator,
                                      device='cpu')[:self.query_count].sort().values
            position_tuple = tuple(int(value) for value in positions.tolist())
            id_tuple = tuple(int(value) for value in self.train_ids[positions.to(self.train_ids.device)].cpu().tolist())
            self.counters['mask_draws'] += 1
            mask = CommonQueryMask(id(self), self.counters['mask_draws'], position_tuple, id_tuple,
                                   self.allowed_count, self.query_count, self.training_value_scale)
            self._pending_mask = mask
            return mask

        def _training_context(self, mask):
            self._check_roles()
            require(isinstance(mask, CommonQueryMask) and mask is self._pending_mask
                    and mask.owner_id == id(self) and mask.allowed_count == self.allowed_count
                    and mask.query_count == self.query_count
                    and mask.training_value_scale == self.training_value_scale,
                    'The one owner-issued common Q is required for all routes')
            positions = torch.tensor(mask.query_positions, dtype=torch.long, device=self.train_ids.device)
            queries = self.train_ids[positions]
            require(tuple(int(value) for value in queries.cpu().tolist()) == mask.query_ids,
                    'Common query IDs must preserve the bound TRAIN order')
            visible = torch.ones(self.allowed_count, dtype=torch.bool, device=self.train_ids.device)
            visible[positions] = False
            visible_positions = visible.nonzero(as_tuple=False).flatten()
            # Targets are read only for CE. Context only indexes visible labels.
            targets = self.train_labels[positions]
            self.counters['common_label_fields_built'] += 1
            return queries, targets, visible_positions, self.training_value_scale

        def _capture(self, H, base_logits):
            require(isinstance(H, torch.Tensor) and isinstance(base_logits, torch.Tensor)
                    and H.shape == (self.nodes, self.features)
                    and base_logits.shape == (self.nodes, self.classes)
                    and H.dtype == base_logits.dtype == torch.float32
                    and H.device == base_logits.device == self.train_ids.device,
                    'Full same-node-order native float32 H[N,F] and base logits[N,C] required')
            require(bool(torch.isfinite(H).all()) and bool(torch.isfinite(base_logits).all()),
                    'Finite external native capture required')
            # This is the gradient boundary, not a claim about external selector,
            # local restoration, RNG/buffers, native loss or native optimizer.
            return H.detach(), base_logits.detach()

        def _query_edges(self, queries):
            # Query row restriction retains EVERY fixed nonself neighbor record
            # for a selected row, irrespective of labels or current common Q.
            lookup = torch.full((self.nodes,), -1, dtype=torch.long, device=queries.device)
            lookup[queries] = torch.arange(len(queries), device=queries.device)
            row = lookup[self.edge_target]
            keep = row >= 0
            return self.edge_source[keep], row[keep]

        def _route_delta(self, H, queries, visible_positions, scale, source, row, member, training):
            with factors['member_context'](self, member):
                # Scores have no label/mask-value input. Denominator includes
                # unlabeled/hidden-label neighbors with their zero message values.
                q = self.query(H[queries])
                k = self.key(H)
                scores = (q[row] * k[source]).sum(-1) / math.sqrt(ATTENTION_WIDTH)
                require(bool(torch.isfinite(scores).all()), 'Finite feature-only edge scores required')
                maximum = H.new_full((len(queries),), -float('inf'))
                maximum.scatter_reduce_(0, row, scores, reduce='amax', include_self=True)
                unnormalized = (scores - maximum[row]).exp()
                denominator = H.new_zeros((len(queries),))
                denominator.index_add_(0, row, unnormalized)
                alpha = unnormalized / denominator[row]
                require(bool(torch.isfinite(alpha).all()), 'Finite all-neighbor attention required')
                visible_ids = self.train_ids[visible_positions]
                values = H.new_zeros((self.nodes, LABEL_WIDTH))
                # The only value input is permitted TRAIN-label embedding.
                # There are no cached full-label fields, raw-H values, biases,
                # additive feature values or additional propagation. The dense
                # message is receiver-gated only after the aggregation.
                permitted = gates.lookup_permitted_label_embedding(
                    self.label_embedding, visible_count=len(visible_positions),
                    visible_labels=(None if self.erase_value_class_identity else
                                    self.train_labels[visible_positions]),
                    erase_class_identity=self.erase_value_class_identity)
                values.index_copy_(0, visible_ids, self.value(permitted) * scale)
                self.counters['route_label_value_fields_built'] += 1
                require(bool(torch.isfinite(values).all()), 'Finite permitted label values required')
                message = H.new_zeros((len(queries), LABEL_WIDTH))
                message.index_add_(0, row, alpha[:, None] * values[source])
                message = self.query_value_gate(message, H[queries])
                delta = self.output(message)
                require(bool(torch.isfinite(delta).all()), 'Finite label-only correction required')
                legal = torch.zeros(self.nodes, dtype=torch.bool, device=H.device)
                legal[visible_ids] = True
                visible_count = torch.zeros(len(queries), dtype=torch.long, device=H.device)
                visible_count.index_add_(0, row, legal[source].long())
                empty = visible_count == 0
                require(torch.equal(delta[empty], torch.zeros_like(delta[empty])),
                        'No visible neighborhood label must give exact zero correction')
                self.counters['empty_label_neighborhood_rows_checked'] += int(empty.sum())
                key = 'training_edge_score_records' if training else 'inference_edge_score_records'
                self.counters[key] += int(source.numel())
                return delta

        def train_corrector_step(self, H, base_logits, common_mask):
            """Mean-four own corrected CE on Q; native updates are caller-owned."""
            require(not self._failed_update, 'Failed core updates are not automatically retried')
            queries, targets, visible_positions, scale = self._training_context(common_mask)
            H, base = self._capture(H, base_logits)
            source, row = self._query_edges(queries)
            self.train()
            self._finite_state()
            self.counters['training_step_attempts'] += 1
            self.optimizer.zero_grad(set_to_none=True)
            parameters = tuple(self.parameters())
            versions = tuple(parameter._version for parameter in parameters)
            own_mean = base.new_zeros(())
            try:
                for member in range(MEMBERS):
                    self.counters['training_route_forward_attempts'] += 1
                    delta = self._route_delta(H, queries, visible_positions, scale,
                                              source, row, member, True)
                    logits = base[queries] + delta
                    self.counters['training_route_forwards'] += 1
                    own = torch.nn.functional.cross_entropy(logits, targets) / MEMBERS
                    require(bool(torch.isfinite(own)), 'Finite complete common-Q own CE required')
                    own_mean = own_mean + own.detach()
                    self.counters['TRAIN_query_route_label_presentations'] += len(queries)
                    self.counters['route_backward_attempts'] += 1
                    own.backward()
                    self.counters['route_backwards'] += 1
                    del delta, logits, own
                require(tuple(parameter._version for parameter in parameters) == versions,
                        'All four corrected CE gradients require the same old corrector parameters')
                self.counters['old_corrector_parameter_checks'] += 1
                active = [parameter for parameter in parameters if parameter.grad is not None]
                require(active and all(bool(torch.isfinite(parameter.grad).all()) for parameter in active),
                        'Finite complete old-state corrector gradient bank required')
                self.counters['corrector_Adam_attempts'] += 1
                self.optimizer.step()
                self.counters['corrector_Adam_steps'] += 1
                self._finite_state()
                self.steps += 1
                self.counters['completed_training_steps'] += 1
                self._pending_mask = None
                return dict(own_corrected_CE_mean=float(own_mean), queries=len(queries),
                            query_ids=common_mask.query_ids, value_scale=scale,
                            corrector_steps=self.steps, native_optimizer_updates_performed_by_core=0,
                            counters=dict(self.counters))
            except BaseException:
                self._failed_update = True
                raise

        def serve(self, H, base_logits, heldout_ids):
            """All permitted A labels, scale1; heldout truth is not an argument."""
            require(not self._failed_update, 'Failed update cannot be promoted through serving')
            self._check_roles()
            self._finite_state()
            H, base = self._capture(H, base_logits)
            require(isinstance(heldout_ids, torch.Tensor) and heldout_ids.dtype == torch.long
                    and heldout_ids.ndim == 1 and len(heldout_ids) > 0
                    and heldout_ids.device == H.device and len(heldout_ids.unique()) == len(heldout_ids)
                    and int(heldout_ids.min()) >= 0 and int(heldout_ids.max()) < self.nodes
                    and not bool(torch.isin(heldout_ids, self.train_ids).any()),
                    'Unique in-domain heldout queries disjoint from permitted TRAIN labels required')
            source, row = self._query_edges(heldout_ids)
            visible_positions = torch.arange(self.allowed_count, device=H.device)
            self.counters['common_label_fields_built'] += 1
            self.eval()
            logits = []
            with torch.no_grad():
                for member in range(MEMBERS):
                    self.counters['serving_route_forward_attempts'] += 1
                    delta = self._route_delta(H, heldout_ids, visible_positions, 1.,
                                              source, row, member, False)
                    logits.append(base[heldout_ids] + delta)
                    self.counters['serving_route_forwards'] += 1
                bank = torch.stack(logits)
                served = bank.softmax(-1).mean(0)
            self.counters['serving_calls'] += 1
            return dict(served_probabilities=served, member_logits=bank,
                        heldout_ids=heldout_ids.clone(), value_scale=1.)

        def forward(self, H, base_logits, heldout_ids):
            return self.serve(H, base_logits, heldout_ids)

        def descriptor(self):
            return dict(execution_mode=EXECUTION_MODE, core_source_sha256=sha(__file__),
                factors_source_sha256=FACTORS_SHA, factor_primitives_AST_sha256=self.factor_ast_sha,
                nodes=self.nodes, feature_width=self.features, classes=self.classes,
                members=MEMBERS, attention_width=ATTENTION_WIDTH, label_width=LABEL_WIDTH,
                allowed_TRAIN_labels=self.allowed_count, common_queries=self.query_count,
                conditional_training_value_scale=self.training_value_scale,
                inference_value_scale=1., input_edges=self.input_edges,
                removed_self_edges=self.removed_self_edges, observed_nonself_edges=int(self.edge_source.numel()),
                duplicate_nonself_edge_records='Preserved with original multiplicity/order',
                initializer_seed=self.initializer_seed, mask_seed=self.mask_seed,
                initializer='Shared native Linear/Embedding reset under isolated CPU RNG; Q/K r Rademacher at seed+900001/+900003; all s,V/O r ones; output shared W zero',
                query_conditioned_values=True, query_gate_shared_across_routes=True,
                query_gate_formula='2*sigmoid(A*stopgrad(H_query)+b)', query_gate_parameters=32832,
                value_identity_erased=self.erase_value_class_identity,
                owned_parameter_count=sum(p.numel() for p in self.parameters()),
                correction_gradients_into_external_H_or_base=False, own_corrected_CE=True,
                pooled_training_objective=False, prediction_or_loss_unbiased=False,
                native_capture_or_trajectory_or_selector_integration_implemented=False,
                complete_cell_ready=False, exact_resume_supported=False, counters=dict(self.counters))

    return LabelOnlyCorrectorCore


def make_core(*, nodes, feature_width, classes, edge_index, train_ids, train_labels,
              initializer_seed, mask_seed, device='cpu', public_root=None,
              later_execution_authorized=False, erase_value_class_identity=False):
    """Honest external-capture core factory; default refuses before torch import."""
    require(later_execution_authorized is True, 'Disabled source prototype; later separate adoption required')
    require(all(type(value) is int for value in
                (nodes, feature_width, classes, initializer_seed, mask_seed))
            and nodes > 1 and feature_width > 0 and classes > 1
            and 0 <= initializer_seed < 2**63 and 0 <= mask_seed < 2**63,
            'Explicit positive shape integers and isolated initializer/mask seeds required')
    require(feature_width == 512 and classes == 10 and type(erase_value_class_identity) is bool,
            'Frozen 512D/10class gate task and permanent explicit value policy required')
    import torch
    requested_device = torch.device(device)
    require(requested_device.type in ('cpu', 'cuda'), 'Declared CPU or CUDA core device required')
    require(isinstance(edge_index, torch.Tensor) and edge_index.dtype == torch.long
            and edge_index.ndim == 2 and edge_index.shape[0] == 2,
            'Caller-bound complete observed graph edge_index[2,E] required')
    require(edge_index.numel() == 0 or (int(edge_index.min()) >= 0 and int(edge_index.max()) < nodes),
            'In-domain graph IDs required')
    require(isinstance(train_ids, torch.Tensor) and isinstance(train_labels, torch.Tensor)
            and train_ids.dtype == train_labels.dtype == torch.long
            and train_ids.ndim == train_labels.ndim == 1
            and len(train_ids) == len(train_labels) >= 2
            and len(train_ids.unique()) == len(train_ids)
            and int(train_ids.min()) >= 0 and int(train_ids.max()) < nodes
            and int(train_labels.min()) >= 0 and int(train_labels.max()) < classes,
            'Only complete permitted TRAIN IDs/labels in the declared class domain are accepted')
    root = public_root or Path(__file__).resolve().parent.parent / PUBLIC_NAME
    primitives, ast_sha = _factor_conventions(torch, root)
    core_type = _core_type(torch, primitives)
    # CPU-only module initialization: private generator supplies the state,
    # fork_rng restores the ambient CPU stream. No global manual_seed call,
    # CUDA random draw, numpy/Python random draw or native stream access exists.
    with _isolated_constructor_rng(torch, initializer_seed):
        core = core_type(nodes, feature_width, classes, edge_index.cpu(), train_ids.cpu(),
                         train_labels.cpu(), initializer_seed, mask_seed, ast_sha, erase_value_class_identity)
        core = core.to(device=requested_device, dtype=torch.float32)
        core.optimizer = torch.optim.Adam(core.parameters(), lr=.001, eps=1e-8, weight_decay=0.)
    core._record_role_versions()
    return core
