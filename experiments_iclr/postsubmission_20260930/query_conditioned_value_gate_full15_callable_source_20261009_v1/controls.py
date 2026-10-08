"""Disabled ordinary label-only correctors; native backbones/selectors are external."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

CORE_DIRECTORY = 'query_conditioned_value_gate_full15_callable_source_20261009_v1'
CORE_SHA = '760d8a676c45239e21ee3b34ad23ace1fac1f38b8901051271306d982a8a55c8'
WIDTH = 64


class OptimizerBank:
    """Snapshot facade; each route still owns its separate ordinary Adam state."""
    def __init__(self, optimizers, mode):
        self.optimizers, self.mode = optimizers, mode

    def state_dict(self):
        return dict(control_mode=self.mode, route_states=[optimizer.state_dict() for optimizer in self.optimizers])

    def load_state_dict(self, state):
        require(state['control_mode'] == self.mode and len(state['route_states']) == len(self.optimizers),
                'Exact complete private optimizer bank required')
        for optimizer, item in zip(self.optimizers, state['route_states']):
            optimizer.load_state_dict(item)


def require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _load_core(core_root):
    root = Path(core_root).resolve(strict=True)
    require(root == Path(__file__).resolve().parent, 'Only this localized successor core is allowed')
    seal = json.loads((root / 'SEAL.json').read_text())
    require(seal['execution_enabled'] is False and seal['runtime_qualified'] is False
            and _sha(root / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact disabled successor seal')
    row = next(r for r in json.loads((root / 'MANIFEST.json').read_text())['files'] if r['path'] == 'core.py')
    require(_sha(root / 'core.py') == row['sha256'] == CORE_SHA
            and (root / 'core.py').stat().st_size == row['bytes'], 'Exact localized gated core')
    name = '_query_value_gated_control_core_v1'
    spec = importlib.util.spec_from_file_location(name, root / 'core.py')
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _control_type(torch, core):
    nn = torch.nn
    gates = core._value_gate_helpers()
    # These inherited mask/role/edge methods are the reviewed operation. The
    # candidate constructor, factor maps, numerical factory and update are unused.
    Base = core._core_type(torch, {})

    class OrdinaryCorrectors(Base):
        def __init__(self, mode, nodes, features, classes, edge_index, train_ids, train_labels,
                     initializer_seeds, mask_seed, context_identity, backbone_ids, selection_policy):
            nn.Module.__init__(self)
            self.nodes, self.features, self.classes = nodes, features, classes
            self.erase_value_class_identity = False
            self._fixed_value_erasure_policy = False
            self.mode = mode; self.attention_heads = len(initializer_seeds)
            self.members = 1 if mode == 'multihead_single4' else self.attention_heads
            self.initializer_seeds, self.mask_seed = initializer_seeds, mask_seed
            self.context_identity = dict(context_identity)
            self.capture_backbone_ids, self.selection_policy = backbone_ids, dict(selection_policy)
            self.backbone_ids = backbone_ids if len(backbone_ids) == self.members else backbone_ids * self.members
            self.backward_loss_scale = 1. / self.members if mode == 'shared_backbone_untied_correctors4' else 1.
            self.routes = nn.ModuleList()
            for seed in initializer_seeds:
                with core._isolated_constructor_rng(torch, seed):
                    maps = dict(label_embedding=nn.Embedding(classes, WIDTH, dtype=torch.float32),
                        query=nn.Linear(features, WIDTH, bias=False, dtype=torch.float32),
                        key=nn.Linear(features, WIDTH, bias=False, dtype=torch.float32),
                        value=nn.Linear(WIDTH, WIDTH, bias=False, dtype=torch.float32))
                    if mode != 'multihead_single4':
                        maps['output'] = nn.Linear(WIDTH, classes, bias=False, dtype=torch.float32)
                        with torch.no_grad():
                            maps['output'].weight.zero_()
                    route = nn.ModuleDict(maps)
                    if mode == 'shared_backbone_untied_correctors4':
                        route['query_value_gate'] = gates.make_query_value_gate(
                            torch=torch, later_execution_authorized=True)
                self.routes.append(route)
            if mode != 'shared_backbone_untied_correctors4':
                self.query_value_gate = gates.make_query_value_gate(
                    torch=torch, later_execution_authorized=True)
            if mode == 'multihead_single4':
                with core._isolated_constructor_rng(torch, initializer_seeds[0] + 700001):
                    self.joint_output = nn.Linear(4 * WIDTH, classes, bias=False, dtype=torch.float32)
                    with torch.no_grad():
                        self.joint_output.weight.zero_()
            nonself = edge_index[0] != edge_index[1]
            self.register_buffer('edge_source', edge_index[0, nonself].clone())
            self.register_buffer('edge_target', edge_index[1, nonself].clone())
            self.register_buffer('train_ids', train_ids.clone())
            self.register_buffer('train_labels', train_labels.clone())
            self.input_edges = int(edge_index.shape[1])
            self.removed_self_edges = self.input_edges - int(self.edge_source.numel())
            self.allowed_count = int(train_ids.numel()); self.query_count = self.allowed_count // 2
            self.training_value_scale = (self.allowed_count - 1) / (self.allowed_count - self.query_count)
            self.mask_generator = torch.Generator(device='cpu').manual_seed(mask_seed)
            self._pending_mask = None; self._failed_update = False
            self.steps = 0
            self._selected = [None] * self.members; self._selected_versions = [None] * self.members
            self.last_native_capture = None
            self.counters = {key: 0 for key in ('mask_draws', 'common_label_fields_built',
                'route_label_value_fields_built', 'training_step_attempts', 'completed_training_steps',
                'training_route_forward_attempts', 'training_route_forwards', 'route_backward_attempts',
                'route_backwards', 'corrector_Adam_attempts', 'corrector_Adam_steps',
                'TRAIN_query_route_label_presentations', 'training_edge_score_records',
                'inference_edge_score_records', 'serving_calls', 'serving_route_forward_attempts',
                'serving_route_forwards', 'old_corrector_parameter_checks',
                'empty_label_neighborhood_rows_checked')}

        def _finite_state(self):
            require(all(bool(torch.isfinite(p).all()) for p in self.parameters()), 'Finite corrector parameters')
            for optimizer in self.optimizers:
                for state in optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            require(bool(torch.isfinite(value).all()), 'Finite private Adam state')

        def _captures(self, captures):
            require(isinstance(captures, (tuple, list)) and len(captures) == len(self.capture_backbone_ids),
                    'One full H/logit capture per externally owned backbone required')
            result = []
            for member, capture in enumerate(captures):
                epoch = capture.get('native_epoch')
                require(capture['backbone_id'] == self.capture_backbone_ids[member]
                    and ((type(epoch) is int and epoch >= 0) or (epoch is None and self.mode != 'ordinary_independent4')),
                    'Capture order/identity/epoch must match the independent backbone contract')
                result.append(Base._capture(self, capture['H'], capture['base_logits']))
            return result if len(result) == self.members else result * self.members

        def _shared_capture(self, H, base, native_capture=None):
            require(self.mode != 'ordinary_independent4', 'Single-backbone protocol cannot supply an independent GNN4')
            # Current-view calls carry no claimed epoch or selected-state proof.
            capture = dict(H=H, base_logits=base, backbone_id=self.capture_backbone_ids[0])
            if native_capture is not None:
                require(isinstance(native_capture, dict) and native_capture['backbone_id'] == self.capture_backbone_ids[0]
                    and type(native_capture['source_parameter_epoch']) is int and native_capture['source_parameter_epoch'] >= 0
                    and type(native_capture['native_logical_steps_before_update']) is int
                    and native_capture['native_logical_steps_before_update'] >= 0
                    and type(native_capture['global_mode']) is bool, 'Explicit actual capture-state/logical-step metadata')
                self.last_native_capture = dict(native_capture)
                capture['native_epoch'] = native_capture['source_parameter_epoch']
            else:
                self.last_native_capture = None
            return [capture]

        def _route_message(self, H, queries, visible_positions, scale, source, row, member, training):
            route = self.routes[member]
            q, k = route['query'](H[queries]), route['key'](H)
            scores = (q[row] * k[source]).sum(-1) / math.sqrt(WIDTH)
            require(bool(torch.isfinite(scores).all()), 'Finite feature-only scores')
            maximum = H.new_full((len(queries),), -float('inf'))
            maximum.scatter_reduce_(0, row, scores, reduce='amax', include_self=True)
            unnormalized = (scores - maximum[row]).exp()
            denominator = H.new_zeros((len(queries),)); denominator.index_add_(0, row, unnormalized)
            alpha = unnormalized / denominator[row]
            require(bool(torch.isfinite(alpha).all()), 'Finite all-nonself-neighbor attention')
            visible_ids = self.train_ids[visible_positions]
            values = H.new_zeros((self.nodes, WIDTH))
            permitted = route['label_embedding'](self.train_labels[visible_positions])
            values.index_copy_(0, visible_ids, route['value'](permitted) * scale)
            self.counters['route_label_value_fields_built'] += 1
            require(bool(torch.isfinite(values).all()), 'Finite visible-label-only values')
            message = H.new_zeros((len(queries), WIDTH))
            message.index_add_(0, row, alpha[:, None] * values[source])
            gate = (route['query_value_gate'] if self.mode == 'shared_backbone_untied_correctors4'
                    else self.query_value_gate)
            message = gate(message, H[queries])
            require(bool(torch.isfinite(message).all()), 'Finite label-only message')
            legal = torch.zeros(self.nodes, dtype=torch.bool, device=H.device); legal[visible_ids] = True
            count = torch.zeros(len(queries), dtype=torch.long, device=H.device)
            count.index_add_(0, row, legal[source].long()); empty = count == 0
            require(torch.equal(message[empty], torch.zeros_like(message[empty])), 'Exact zero with no visible label')
            self.counters['empty_label_neighborhood_rows_checked'] += int(empty.sum())
            self.counters['training_edge_score_records' if training else 'inference_edge_score_records'] += int(source.numel())
            return message

        def _readout(self, message, output):
            delta = output(message); zero = (message == 0).all(-1)
            require(bool(torch.isfinite(delta).all()) and torch.equal(delta[zero], torch.zeros_like(delta[zero])),
                    'Finite bias-free correction; zero label messages give exact zero')
            return delta

        def _route_delta(self, H, queries, visible_positions, scale, source, row, member, training):
            return self._readout(self._route_message(H, queries, visible_positions, scale, source, row, member, training),
                                 self.routes[member]['output'])

        def _joint_delta(self, H, queries, visible, scale, source, row, training):
            messages = []
            for head in range(self.attention_heads):
                prefix = 'training' if training else 'serving'
                self.counters[prefix + '_route_forward_attempts'] += 1
                messages.append(self._route_message(H, queries, visible, scale, source, row, head, training))
                self.counters[prefix + '_route_forwards'] += 1
            return self._readout(torch.cat(messages, dim=-1), self.joint_output)

        def train_corrector_step(self, captures, common_mask, mask_token=None, *, native_capture=None):
            if mask_token is not None:
                captures, common_mask = self._shared_capture(captures, common_mask, native_capture), mask_token
            else:
                require(native_capture is None, 'General capture dictionaries already carry their provenance')
            require(not self._failed_update, 'No failed-update retry')
            queries, targets, visible, scale = self._training_context(common_mask)
            captured = self._captures(captures); source, row = self._query_edges(queries)
            self.train(); self._finite_state(); self.counters['training_step_attempts'] += 1
            self._selected = [None] * self.members
            parameters = tuple(self.parameters()); versions = tuple(p._version for p in parameters)
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
            total = captured[0][0].new_zeros(())
            try:
                for member, (H, base) in enumerate(captured):
                    if self.mode == 'multihead_single4':
                        logits = base[queries] + self._joint_delta(H, queries, visible, scale, source, row, True)
                    else:
                        self.counters['training_route_forward_attempts'] += 1
                        logits = base[queries] + self._route_delta(H, queries, visible, scale, source, row, member, True)
                        self.counters['training_route_forwards'] += 1
                    # Full ordinary4/single: unscaled own CE. Shared-backbone
                    # private-map mechanism control: CE/4, as in the candidate.
                    own = torch.nn.functional.cross_entropy(logits, targets)
                    require(bool(torch.isfinite(own)), 'Finite own corrected CE')
                    total += own.detach() / self.members
                    self.counters['TRAIN_query_route_label_presentations'] += len(queries)
                    self.counters['route_backward_attempts'] += 1; (own * self.backward_loss_scale).backward()
                    self.counters['route_backwards'] += 1
                    del own, logits
                require(tuple(p._version for p in parameters) == versions, 'All gradients at old corrector state')
                self.counters['old_corrector_parameter_checks'] += 1
                for route in self.routes:
                    active = [p for p in route.parameters() if p.grad is not None]
                    require(active and all(bool(torch.isfinite(p.grad).all()) for p in active), 'Finite private gradients')
                require(all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None),
                        'Finite full-map and joint-readout gradient bank')
                for member, optimizer in enumerate(self.optimizers):
                    self.counters['corrector_Adam_attempts'] += 1; optimizer.step()
                    self.counters['corrector_Adam_steps'] += 1
                self._finite_state(); self.steps += 1; self.counters['completed_training_steps'] += 1
                self._pending_mask = None
                return dict(own_corrected_CE_mean=float(total), backward_loss_scale=self.backward_loss_scale, queries=len(queries),
                    query_ids=common_mask.query_ids, value_scale=scale, counters=dict(self.counters), native_updates=0)
            except BaseException:
                self._failed_update = True
                raise

        def route_snapshot(self, member, *, native_epoch):
            require(self.mode != 'multihead_single4', 'Four-head single snapshots belong to its whole model/Adam, not separate heads')
            require(not self._failed_update and self._pending_mask is None and type(member) is int
                and 0 <= member < self.members and type(native_epoch) is int and native_epoch >= 0,
                'Coherent externally identified native epoch required')
            self._check_roles(); self._finite_state()
            return dict(control_mode=self.mode, backbone_id=self.backbone_ids[member], native_epoch=native_epoch,
                context_identity=dict(self.context_identity), initializer_seed=self.initializer_seeds[member],
                model=copy.deepcopy(self.routes[member].state_dict()),
                optimizer=copy.deepcopy(self.optimizers[member].state_dict()))

        def restore_route(self, member, state, *, purpose, selected_binding=None):
            require(self.mode != 'multihead_single4', 'Restore the complete four-head single and its one Adam coherently')
            require(not self._failed_update and self._pending_mask is None and type(member) is int
                and 0 <= member < self.members and purpose in ('native_own_local_restore', 'final_selected_state'),
                'Separate own-member restoration; no shared candidate-selected epoch')
            require(state['control_mode'] == self.mode and state['backbone_id'] == self.backbone_ids[member]
                and state['context_identity'] == self.context_identity
                and state['initializer_seed'] == self.initializer_seeds[member]
                and type(state['native_epoch']) is int and state['native_epoch'] >= 0,
                'Corrector snapshot must match its own native epoch/backbone/context')
            if purpose == 'final_selected_state':
                require(isinstance(selected_binding, dict) and selected_binding['backbone_id'] == self.backbone_ids[member]
                    and selected_binding['native_epoch'] == state['native_epoch']
                    and selected_binding['selector_policy_sha256'] == self.selection_policy['policy_sha256']
                    and all(_digest(selected_binding[key]['sha256']) for key in ('native_checkpoint', 'corrector_checkpoint')),
                    'Own-member native+corrector selected-state bindings required')
            try:
                self.routes[member].load_state_dict(state['model'], strict=True)
                self.optimizers[member].load_state_dict(state['optimizer'])
                self._finite_state()
                self._selected[member] = copy.deepcopy(selected_binding) if purpose == 'final_selected_state' else None
                self._selected_versions[member] = tuple(p._version for p in self.routes[member].parameters())
            except BaseException:
                self._failed_update = True
                raise
            # Mask generator/counters and external end-local RNG streams stay live.

        def serve(self, captures, heldout_ids, ids=None, *, purpose='current_validation', native_capture=None):
            if ids is not None:
                require(purpose == 'current_validation', 'Explicit capture receipts required for final-state verification')
                captures, heldout_ids = self._shared_capture(captures, heldout_ids, native_capture), ids
            else:
                require(native_capture is None, 'General capture dictionaries already carry their provenance')
            require(not self._failed_update and purpose in ('current_validation', 'selected_final'),
                    'Legal validation or independently selected serving only')
            self._check_roles(); self._finite_state(); captured = self._captures(captures)
            if purpose == 'selected_final':
                require(self.mode != 'multihead_single4', 'Four-head single final custody belongs to the external complete-model driver')
                for member in range(self.members):
                    capture = captures[member] if self.mode == 'ordinary_independent4' else captures[0]
                    selected = self._selected[member]
                    require(selected is not None and capture['native_epoch'] == selected['native_epoch']
                        and capture['native_checkpoint'] == selected['native_checkpoint']
                        and tuple(p._version for p in self.routes[member].parameters()) == self._selected_versions[member],
                        'Serve each separately selected coherent member; no common candidate epoch shortcut')
                if self.mode == 'shared_backbone_untied_correctors4':
                    require(all(item['native_epoch'] == self._selected[0]['native_epoch']
                        and item['native_checkpoint'] == self._selected[0]['native_checkpoint'] for item in self._selected),
                        'Shared-backbone correction bank requires one coherent family-selected native state')
            H = captured[0][0]
            require(isinstance(heldout_ids, torch.Tensor) and heldout_ids.dtype == torch.long
                and heldout_ids.ndim == 1 and len(heldout_ids) > 0 and heldout_ids.device == H.device
                and len(heldout_ids.unique()) == len(heldout_ids) and int(heldout_ids.min()) >= 0
                and int(heldout_ids.max()) < self.nodes and not bool(torch.isin(heldout_ids, self.train_ids).any()),
                'Unique heldout IDs disjoint from permitted TRAIN anchors')
            source, row = self._query_edges(heldout_ids)
            visible = torch.arange(self.allowed_count, device=H.device)
            self.counters['common_label_fields_built'] += 1; self.eval(); logits = []
            with torch.no_grad():
                for member, (H, base) in enumerate(captured):
                    if self.mode == 'multihead_single4':
                        logits.append(base[heldout_ids] + self._joint_delta(H, heldout_ids, visible, 1., source, row, False))
                    else:
                        self.counters['serving_route_forward_attempts'] += 1
                        logits.append(base[heldout_ids] + self._route_delta(H, heldout_ids, visible, 1., source, row, member, False))
                        self.counters['serving_route_forwards'] += 1
                bank = torch.stack(logits); probabilities = bank.softmax(-1).mean(0)
            self.counters['serving_calls'] += 1
            return dict(served_probabilities=probabilities, member_logits=bank, heldout_ids=heldout_ids.clone(), value_scale=1.)

        def forward(self, captures, heldout_ids, ids=None, *, purpose='current_validation', native_capture=None):
            return self.serve(captures, heldout_ids, ids, purpose=purpose, native_capture=native_capture)

        def descriptor(self):
            return dict(control=self.mode, members=self.members, attention_heads=self.attention_heads,
                nodes=self.nodes, feature_width=self.features, classes=self.classes,
                allowed_TRAIN_labels=self.allowed_count, common_queries=self.query_count,
                training_value_scale=self.training_value_scale, inference_value_scale=1.,
                input_edges=self.input_edges, removed_self_edges=self.removed_self_edges,
                observed_nonself_edges=int(self.edge_source.numel()),
                initializer_seeds=self.initializer_seeds, mask_seed=self.mask_seed,
                attention_width=WIDTH, label_width=WIDTH, ordinary_head_parameter_formula='128*F+128*C+4096',
                learned_parameter_formula='4*(128*F+128*C+4096)+64*(F+1)' if self.mode == 'multihead_single4'
                    else str(self.members) + '*(128*F+128*C+4096)' + ('+4*64*(F+1)' if self.mode == 'shared_backbone_untied_correctors4' else '+64*(F+1)'),
                owned_parameter_count=sum(p.numel() for p in self.parameters()),
                shared_corrector_parameters=self.mode == 'multihead_single4', independent_native_backbones_required=self.mode == 'ordinary_independent4',
                backbone_ids=self.backbone_ids, context_identity=dict(self.context_identity),
                selection_policy=dict(self.selection_policy), backward_own_CE_scale=self.backward_loss_scale,
                Adam=dict(lr=.001, eps=1e-8, weight_decay=0.), corrector_optimizer_count=len(self.optimizers),
                query_conditioned_values=True, value_identity_erased=False,
                query_gate_shared_across_routes=self.mode != 'shared_backbone_untied_correctors4',
                query_gate_parameters=(4 if self.mode == 'shared_backbone_untied_correctors4' else 1)*32832,
                correction_gradients_into_H_or_base=False, all_neighbor_denominator=True,
                no_raw_H_values=True, mean_probability_serving=True,
                runtime_qualified=False, baseline_competence_confirmed=False, complete_cell_ready=False,
                last_native_capture=copy.deepcopy(self.last_native_capture),
                joint_readout='linear256_to_C_rank_at_most_min(C,256)' if self.mode == 'multihead_single4' else None,
                source_sha256=_sha(__file__), reviewed_core_sha256=CORE_SHA, counters=dict(self.counters))

    return OrdinaryCorrectors


def make_controls(*, mode, nodes, feature_width, classes, edge_index, train_ids, train_labels,
                  initializer_seeds, mask_seed, context_identity, backbone_ids, selection_policy,
                  device='cpu', core_root=None, later_execution_authorized=False):
    """Default refuses before torch/dependency import; source controls only."""
    require(later_execution_authorized is True, 'Disabled control source; separate later adoption required')
    count = {'one_path': 1, 'shared_backbone_untied_correctors4': 4, 'multihead_single4': 4}.get(mode)
    require(count is not None and all(type(x) is int for x in (nodes, feature_width, classes, mask_seed))
        and nodes > 1 and feature_width > 0 and classes > 1 and 0 <= mask_seed < 2**63,
        'Explicit valid mode/shapes/private mask seed')
    require(isinstance(initializer_seeds, tuple) and len(initializer_seeds) == count
        and len(set(initializer_seeds)) == count and all(type(x) is int and 0 <= x < 2**63 for x in initializer_seeds),
        'Explicit distinct independent corrector initializer seeds')
    backbone_count = count if mode == 'ordinary_independent4' else 1
    require(isinstance(backbone_ids, tuple) and len(backbone_ids) == backbone_count
        and len(set(backbone_ids)) == backbone_count and all(isinstance(x, str) and x for x in backbone_ids),
        'Distinct externally owned full native backbone IDs; one shared B with four heads is not untied4')
    require(isinstance(context_identity, dict) and all(_digest(context_identity.get(key)) for key in
        ('node_order_sha256', 'observed_edges_sha256', 'TRAIN_ids_sha256', 'TRAIN_labels_sha256', 'context_policy_sha256')),
        'Explicit identical candidate/control graph, node order, permitted labels and masking policy custody')
    require(isinstance(selection_policy, dict) and _digest(selection_policy.get('policy_sha256'))
        and selection_policy.get('native_local_restore') == 'each_own_native_local_selector'
        and selection_policy.get('corrector_local_restore') == 'same_own_native_selected_epoch'
        and selection_policy.get('end_local_streams') == 'live_no_rewind'
        and selection_policy.get('final_selector') == ('mean_probability_family' if mode == 'shared_backbone_untied_correctors4'
                                                      else 'each_own_corrected_predictor')
        and selection_policy.get('role') == 'VALID' and isinstance(selection_policy.get('metric_name'), str)
        and bool(selection_policy['metric_name']), 'Explicit per-member independent native+corrector selection policy')
    require(feature_width == 512 and classes == 10, 'Frozen 512D/10class gated task required')
    core = _load_core(core_root or Path(__file__).resolve().parent.parent / CORE_DIRECTORY)
    import torch
    target = torch.device(device)
    require(target.type in ('cpu', 'cuda'), 'Declared CPU/CUDA device')
    require(isinstance(edge_index, torch.Tensor) and edge_index.dtype == torch.long
        and edge_index.ndim == 2 and edge_index.shape[0] == 2
        and (edge_index.numel() == 0 or (int(edge_index.min()) >= 0 and int(edge_index.max()) < nodes)), 'Complete observed graph IDs')
    require(isinstance(train_ids, torch.Tensor) and isinstance(train_labels, torch.Tensor)
        and train_ids.dtype == train_labels.dtype == torch.long and train_ids.ndim == train_labels.ndim == 1
        and len(train_ids) == len(train_labels) >= 2 and len(train_ids.unique()) == len(train_ids)
        and int(train_ids.min()) >= 0 and int(train_ids.max()) < nodes
        and int(train_labels.min()) >= 0 and int(train_labels.max()) < classes, 'Only complete permitted TRAIN IDs/labels')
    obj = _control_type(torch, core)(mode, nodes, feature_width, classes, edge_index.cpu(), train_ids.cpu(), train_labels.cpu(),
        initializer_seeds, mask_seed, context_identity, backbone_ids, selection_policy).to(target, dtype=torch.float32)
    obj.optimizers = [torch.optim.Adam(obj.parameters(), lr=.001, eps=1e-8, weight_decay=0.)] if mode in ('multihead_single4', 'one_path') else [
        torch.optim.Adam(route.parameters(), lr=.001, eps=1e-8, weight_decay=0.) for route in obj.routes]
    obj.optimizer = OptimizerBank(obj.optimizers, mode)
    obj._record_role_versions()
    return obj


def make_shared_controls(**kwargs):
    """Single-backbone driver factory; same kwargs as make_controls, restricted modes."""
    require(kwargs.get('mode') in ('one_path', 'shared_backbone_untied_correctors4', 'multihead_single4'),
            'Single-backbone driver must reject ordinary_independent4')
    return make_controls(**kwargs)
