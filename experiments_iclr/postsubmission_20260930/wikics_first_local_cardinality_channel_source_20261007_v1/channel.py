"""PNA-attributed first-local output hook and explicit augmented permissions."""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NATIVE_SHA = '9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8'

def require(ok, message):
    if not ok: raise ValueError(message)

def native_inventory(session):
    """Exact factor-free native inventory; no old private auditor on augmentation."""
    torch = session.torch; profile = json.loads((HERE / 'NATIVE_INVENTORY.json').read_text())
    require(session.task == 'wikics' and session.arm in ('single', 'independent4') and session.steps == 0, 'Fresh native Wiki control required')
    require(session.core_provenance == profile['core_sha256'] and session.config['model'] == profile['model_config'], 'Exact source/core/config required')
    require(session.native_provenance == {'polynormer_model_sha256': NATIVE_SHA}, 'Exact native source required')
    path = Path(session.forward.__func__.__code__.co_filename)
    require(hashlib.sha256(path.read_bytes()).hexdigest() == profile['portable_py_sha256'], 'Original native Session/streams required')
    model = session.model; count = 1 if session.arm == 'single' else 4
    require(type(model) is session.core['models'].Ensemble and model.independent and model.members == count and len(model.models) == count, 'Ordinary native body bank required')
    require(len(session.optimizers) == count, 'Original per-body optimizer bank required')
    expected = {}
    for member in range(count):
        require(type(model.models[member]) is session.core['models'].WikiBackbone, 'Exact WikiBackbone required')
        for name, shape in profile['body_parameter_shapes'].items(): expected['models.' + str(member) + '.' + name] = shape
    named = list(model.named_parameters(remove_duplicate=False))
    require({name for name, _ in named} == set(expected) and len(named) == len(expected), 'Unknown/missing/aliased native parameter')
    require(len({id(parameter) for _, parameter in named}) == len(named), 'No shared aliases between independently selected bodies')
    for name, parameter in named:
        require(list(parameter.shape) == expected[name] and parameter.requires_grad and parameter.dtype == torch.float32, 'Exact native shape/dtype required: ' + name)
    for member, (body, optimizer) in enumerate(zip(model.models, session.optimizers)):
        require(type(optimizer) is torch.optim.Adam and len(optimizer.param_groups) == 1 and not optimizer.state, 'Fresh original Adam group required')
        optimized = optimizer.param_groups[0]['params']; own = list(body.parameters())
        require(len(optimized) == len(own) and {id(value) for value in optimized} == {id(value) for value in own}, 'Exact independent optimizer coverage required')
    return {'all': named, 'roles': {name: 'native_body_own_' + name.split('.')[1] for name, _ in named}}

class CardinalityChannel:
    """One scaler only: log1p(incoming multiplicity)/positive TRAIN mean."""
    def __init__(self, session, train, *, kind, base_partition):
        torch = session.torch; model = session.model; self.session = session
        self.native = model.independent; self.kind = kind
        require(session.task == 'wikics' and session.steps == 0 and kind in ('shared', 'centered'), 'Fresh Wiki channel required')
        require(not self.native or kind == 'shared', 'Native controls have one independent scalar per body; no cross-body centering')
        require(not any(hasattr(body.body, 'degree_shared') or hasattr(body.body, 'degree_slots') for body in model.models), 'No repeated channel installation')
        edge, ids = train['edge_index'].to(session.device), train['ids'].to(session.device)
        require(edge.dtype == torch.long and edge.shape[0] == 2 and ids.dtype == torch.long and len(ids) == 580, 'Complete original TRAIN topology/IDs required')
        nodes = train['x'].shape[0]
        self.edge = edge.clone(); self.nodes = nodes
        degree = torch.bincount(self.edge[1], minlength=nodes)
        require(int(degree.sum()) == self.edge.shape[1], 'Count every incoming edge record, including loops/duplicates')
        log_degree = torch.log1p(degree.to(dtype=torch.float32)); normalizer = log_degree[ids].mean()
        require(torch.isfinite(normalizer) and normalizer > 0, 'Positive finite TRAIN-only log1p normalizer required; no epsilon/fallback')
        self.scale = log_degree / normalizer
        require(torch.isfinite(self.scale).all(), 'Finite degree scale required')
        self.normalizer = float(normalizer)
        self.work = {'degree_build_passes': 1, 'degree_build_edge_records': int(edge.shape[1]),
            'edge_binding_checks': 0, 'edge_records_checked': 0, 'first_local_hook_calls': 0, 'scaled_output_elements': 0}
        self.new_shared = []; self.new_internal = []; self.handles = []
        self.optimizer_settings = [{key: value for key, value in optimizer.param_groups[0].items() if key != 'params'} for optimizer in session.optimizers]
        original = list(model.named_parameters(remove_duplicate=False))
        require(len(original) == len(base_partition['all']) and all(name == old_name and parameter is old_parameter
            for (name, parameter), (old_name, old_parameter) in zip(original, base_partition['all'])), 'Audited original inventory must precede augmentation')
        self.active_edge = None
        for member, body in enumerate(model.models):
            first = body.body.local_convs[0]
            require(type(first).__name__ == 'GATConv' and type(first).__module__ == 'torch_geometric.nn.conv.gat_conv', 'Original PyG GATConv required')
            require(first.flow == 'source_to_target' and first.add_self_loops is False and first.bias is None
                and first.edge_dim is None and getattr(first, 'res', None) is None and first.concat and first.heads == 1,
                'Effective Tensor message multiset must equal supplied edge_index; no residual/edge features/automatic loops')
            require(not first._forward_hooks and not first._forward_pre_hooks, 'No competing first-local hooks')
            body.body.register_parameter('degree_shared', torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=session.device)))
            alpha = body.body.degree_shared; prefix = 'models.' + str(member) + '.body.'
            self.new_shared.append((prefix + 'degree_shared', alpha))
            new = [alpha]
            if kind == 'centered':
                require(not self.native and model.members == 4, 'Four-member shared bank required for centered slots')
                body.body.register_parameter('degree_slots', torch.nn.Parameter(torch.zeros(4, dtype=torch.float32, device=session.device)))
                self.new_internal.append((prefix + 'degree_slots', body.body.degree_slots)); new.append(body.body.degree_slots)
            optimizer = session.optimizers[member if self.native else 0]
            require(type(optimizer) is torch.optim.Adam and len(optimizer.param_groups) == 1 and not optimizer.state, 'Append only to original fresh native Adam')
            optimizer.param_groups[0]['params'].extend(new)  # Keep original group/hyperparameters; no new optimizer.
            def hook(module, arguments, output, owner=body.body):
                require(len(arguments) == 2 and arguments[1] is self.active_edge and self.active_edge is not None,
                    'Use the bound Session forward; actual first-local edge object required')
                require(isinstance(output, torch.Tensor) and output.shape == (self.nodes, 512), 'Literal first local aggregation output required')
                coefficient = owner.degree_shared
                if hasattr(owner, 'degree_slots'):
                    active = module.lin.member  # Original FactorLinear member_context, not an inferred model index.
                    require(0 <= active < 4, 'Actual four-member context required')
                    slots = owner.degree_slots
                    coefficient = coefficient + slots[active] - slots.mean()  # Live full-vector centered pullback.
                self.work['first_local_hook_calls'] += 1; self.work['scaled_output_elements'] += output.numel()
                return output + coefficient * self.scale[:, None] * output
            self.handles.append(first.register_forward_hook(hook))
        self.roles = dict(base_partition['roles'])
        for name, _ in self.new_shared: self.roles[name] = 'degree_shared_own' if not self.native else 'native_body_own_' + name.split('.')[1]
        for name, _ in self.new_internal: self.roles[name] = 'internal_private'
        self.all = list(model.named_parameters(remove_duplicate=False))
        self.expected_shapes = {name: list(parameter.shape) for name, parameter in original}
        self.expected_shapes.update({name: [] for name, _ in self.new_shared})
        self.expected_shapes.update({name: [4] for name, _ in self.new_internal})
        require(set(self.roles) == set(self.expected_shapes), 'Explicit augmented role/shape inventory required')
        self.original_forward = session.forward
        def checked_forward(batch):
            actual = batch['edge_index']
            require(batch['x'].shape[0] == self.nodes and actual.dtype == torch.long and actual.device == self.edge.device
                and torch.equal(actual, self.edge), 'Actual first-local topology differs from the bound edge multiset/cache')
            self.work['edge_binding_checks'] += 1; self.work['edge_records_checked'] += self.edge.shape[1]
            require(self.active_edge is None, 'No recursive/concurrent channel forward')
            self.active_edge = actual
            try: return self.original_forward(batch)  # Original persistent RNG/member loop unchanged.
            finally: self.active_edge = None
        self.checked_forward = checked_forward; session.forward = checked_forward
        self.check_inventory()

    def check_inventory(self):
        session = self.session; current = list(session.model.named_parameters(remove_duplicate=False))
        require(len(current) == len(self.all) and all(name == old_name and parameter is old_parameter
            for (name, parameter), (old_name, old_parameter) in zip(current, self.all)), 'Augmented parameter identity/catalog changed')
        require(len({id(parameter) for _, parameter in current}) == len(current), 'No augmented aliases')
        for name, parameter in current:
            require(list(parameter.shape) == self.expected_shapes[name] and parameter.requires_grad
                and parameter.dtype == session.torch.float32 and parameter.device == session.device, 'Augmented shape/dtype/device mismatch: ' + name)
        optimized = []
        for member, optimizer in enumerate(session.optimizers):
            require(type(optimizer) is session.torch.optim.Adam and len(optimizer.param_groups) == 1, 'Original native Adam/group required')
            require({key: value for key, value in optimizer.param_groups[0].items() if key != 'params'} == self.optimizer_settings[member], 'Original Adam group settings changed')
            values = optimizer.param_groups[0]['params']; optimized.extend(values)
            if self.native:
                own = list(session.model.models[member].parameters())
                require(len(values) == len(own) and {id(value) for value in values} == {id(value) for value in own}, 'Independent body optimizer coverage changed')
        require(len(optimized) == len(current) and {id(value) for value in optimized} == {id(value) for _, value in current}, 'Exhaustive unique augmented optimizer coverage required')
        require(session.forward is self.checked_forward, 'Own topology-binding wrapper changed')

    def metadata(self):
        return {'prior': 'PNA 2004.05718v1 Sections2.2-2.3 log-degree scaling; adapted residual attention channel',
            'hook': 'ONLY literal body.local_convs[0] output, before unchanged root/ReLU/dropout/beta/LN',
            'kind': self.kind, 'native_independent_per_body_scalars': self.native,
            'normalizer': self.normalizer, 'normalizer_role': 'mean log1p(incoming edge-record degree) over580 TRAIN IDs only',
            'effective_multiset': 'bound source_to_target Tensor edges, preserving supplied loops/duplicates/order',
            'zero_initialization': True, 'zero_forward_recovery': 'algebraic; no runtime parity claimed',
            'centered_slot_pullback': 'full u vector: da_m/du_j = delta_mj - 1/4; no detach or per-member gradient mask',
            'shared_coefficient_risk': 'own ONLY, including when original allocation policy is G',
            'augmented_roles': dict(self.roles), 'augmented_parameter_shapes': dict(self.expected_shapes),
            'new_shared_parameter_tensors': len(self.new_shared), 'new_internal_slot_tensors': len(self.new_internal),
            'new_slot_elements': 4 * len(self.new_internal), 'functional_member_contrasts': 3 * len(self.new_internal),
            'actual_channel_work': dict(self.work), 'old_auditor_called_after_augmentation': False,
            'primitive_novelty_quality_efficiency_or_diagnosis_claim': False, 'runtime_verified_at_preparation': False}
