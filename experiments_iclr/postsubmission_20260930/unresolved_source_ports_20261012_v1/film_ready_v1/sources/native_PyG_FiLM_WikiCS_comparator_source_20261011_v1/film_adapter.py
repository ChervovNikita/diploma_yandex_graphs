"""Explicit native FiLM scaffold adapter; callers supply Torch and FiLMConv.

No numerical imports, constructor, graph, optimizer or fit at module load.
The installed FiLMConv class remains unmodified; hooks count completed calls.
"""

COUNT_KEYS = ('film_conv_calls', 'film_node_dense_maps', 'film_node_dense_rows',
    'film_node_dense_MACs', 'film_relation_builder_rows', 'film_skip_builder_rows',
    'film_message_rows', 'film_edge_modulation_multiplications',
    'film_edge_beta_additions', 'film_edge_ReLU_coordinates',
    'film_skip_modulation_multiplications', 'film_skip_beta_additions',
    'film_skip_ReLU_coordinates', 'film_mean_output_rows',
    'film_factual_self_message_rows')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _add(total, unit):
    for key, value in unit.items():
        total[key] = total.get(key, 0) + value


def install(torch, body, FiLMConv, edge_metadata):
    """Replace two fresh SAGE convs before generic factor installation."""
    _require(body.model_name == 'SAGE' and len(body.residual_modules) == 2,
             'Original depth-two SAGE scaffold required before FiLM replacement')
    _require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in body.parameters()),
             'Fresh float32 CPU scaffold before replacement/factorization')
    for residual in body.residual_modules:
        _require(residual.module.__class__.__name__ == 'SAGEModule'
                 and residual.module.conv.__class__.__name__ == 'SAGEConv',
                 'Fresh original SAGEModule/SAGEConv scaffold only')
        residual.module.conv = FiLMConv(128, 128, num_relations=1, nn=None,
                                       act=torch.nn.ReLU(), aggr='mean')
    return FiLMAdapter(torch, body, FiLMConv, edge_metadata)


class FiLMAdapter:
    def __init__(self, torch, body, FiLMConv, edge_metadata):
        self.torch, self.body, self.width = torch, body, 128
        self.name, self.site, self.continuing_blocks = 'native_PyG_FiLM', 'after_complete_residual_module0', 1
        self.projection_maps = ('lins.0', 'films.0', 'lin_skip', 'film_skip')
        self.edge_metadata = dict(edge_metadata)
        _require(self.edge_metadata['nodes'] == 11701 and self.edge_metadata['edges'] > 0
                 and self.edge_metadata['self_edges'] > 0,
                 'Complete original WikiCS support including stored self records')
        _require(body.input_linear.in_features == 300 and body.input_linear.out_features == 128
                 and body.output_linear.in_features == 128 and body.output_linear.out_features == 10
                 and len(body.residual_modules) == 2 and body.act.__class__.__name__ == 'GELU',
                 'Unchanged native stem, two graph layers and head required')
        _require(all(n.__class__.__name__ == 'LayerNorm' for n in
                    [r.normalization for r in body.residual_modules] + [body.output_normalization]),
                 'Unchanged buffer-free native LayerNorm scaffold')
        self.convs = [r.module.conv for r in body.residual_modules]
        _require(len({id(c.act) for c in self.convs}) == 2,
                 'Fresh separate ReLU activation module in each FiLM convolution')
        self.installations = []
        self._handles = []
        for index, conv in enumerate(self.convs):
            _require(type(conv) is FiLMConv and conv.in_channels == conv.out_channels == 128
                     and conv.num_relations == 1 and conv.aggr == 'mean'
                     and conv.flow == 'source_to_target' and conv.node_dim == -2
                     and conv.decomposed_layers == 1 and not conv.explain
                     and type(conv.act) is torch.nn.ReLU and not conv.act.inplace,
                     'Exact installed one-relation default mean/ReLU native FiLM operator')
            maps = (conv.lins[0], conv.films[0], conv.lin_skip, conv.film_skip)
            _require(len(conv.lins) == len(conv.films) == 1
                     and [tuple(m.weight.shape) for m in maps] == [(128, 128), (256, 128), (128, 128), (256, 128)]
                     and maps[0].bias is None and maps[1].bias.shape == (256,)
                     and maps[2].bias is None and maps[3].bias is None,
                     'Four exact native materialized FiLM affine maps and biases')
            _require(sum(p.numel() for p in conv.parameters()) == 98560,
                     'Actual native FiLM slow convolution capacity differs')
            conv.film_counts_train, conv.film_counts_eval, conv.film_last_unit = {}, {}, None
            self._handles.append(conv.register_forward_pre_hook(self._before))
            self._handles.append(conv.register_forward_hook(self._after))
            self.installations.append({'layer': index, 'operator': 'installed_PyG2.7.0_FiLMConv',
                'one_relation': True, 'aggregation': 'mean', 'activation': 'edgewise_ReLU_and_skip_ReLU',
                'shared_conv_parameters': 98560, 'private_factor_parameters_per_member': 1280,
                'self_policy': 'stored_diagonal_in_ordinary_relation_mean_plus_native_separate_skip; no loop insertion/removal',
                'initialization': 'native_FiLM_constructor_reset_before_factorization; no_zero_or_identity_generator_override'})

    def _before(self, conv, args):
        x, edges = args[:2]
        _require(self.torch.is_tensor(x) and x.shape == (11701, 128) and x.dtype == self.torch.float32
                 and self.torch.is_tensor(edges) and edges.layout == self.torch.strided
                 and edges.shape == (2, self.edge_metadata['edges']) and edges.dtype == self.torch.long
                 and edges.device == x.device and len(args) == 2,
                 'Full factual homogeneous native FiLM input/support; no typed/subsampled override')

    def _after(self, conv, args, output):
        _require(output.shape == (11701, 128) and output.dtype == args[0].dtype,
                 'Complete native FiLM output required')
        n, e, c = 11701, self.edge_metadata['edges'], 128
        unit = dict(film_conv_calls=1, film_node_dense_maps=4, film_node_dense_rows=4*n,
            film_node_dense_MACs=6*n*c*c, film_relation_builder_rows=n, film_skip_builder_rows=n,
            film_message_rows=e, film_edge_modulation_multiplications=e*c,
            film_edge_beta_additions=e*c, film_edge_ReLU_coordinates=e*c,
            film_skip_modulation_multiplications=n*c, film_skip_beta_additions=n*c,
            film_skip_ReLU_coordinates=n*c, film_mean_output_rows=n,
            film_factual_self_message_rows=self.edge_metadata['self_edges'])
        conv.film_last_unit = unit
        _add(conv.film_counts_train if conv.training else conv.film_counts_eval, unit)

    def assert_factorized(self, factors, members):
        _require(members in (1, 4) and not list(self.body.named_buffers()),
                 'One or four native routes, no persistent model buffers')
        for conv in self.convs:
            maps = (conv.lins[0], conv.films[0], conv.lin_skip, conv.film_skip)
            _require(all(isinstance(m, factors.FactorLinear) and m.r.shape[0] == m.s.shape[0] == members for m in maps),
                     'All four native FiLM affine maps must use exact existing FactorLinear')
        _require(sum(isinstance(m, factors.FactorLinear) for m in self.body.modules()) == 14
                 and sum(p.numel() for p in self.body.parameters()) == 336522 + 4406*members,
                 'Actual whole-scaffold FiLM/factor capacity differs')
        return self.parameter_roles(factors)

    def parameter_roles(self, factors):
        private = {id(p) for conv in self.convs for m in conv.modules()
                   if isinstance(m, factors.FactorLinear) for p in (m.r, m.s)}
        parameters = [p for conv in self.convs for p in conv.parameters()]
        result = {'shared_conv_parameters': sum(p.numel() for p in parameters if id(p) not in private),
                  'private_conv_factor_parameters': sum(p.numel() for p in parameters if id(p) in private),
                  'total': sum(p.numel() for p in parameters)}
        _require(result['shared_conv_parameters'] == 197120, 'Two native FiLM slow parameter sets')
        return result

    def operation_counts(self, members, updates):
        train, evaluation, selected = {}, {}, {}
        for conv in self.convs:
            _require(conv.film_last_unit is not None, 'Native FiLM operator was never exercised')
            _add(train, conv.film_counts_train); _add(evaluation, conv.film_counts_eval)
            _add(selected, {k: v*members for k, v in conv.film_last_unit.items()})
        expected = 2*members*updates
        _require(train['film_conv_calls'] == expected and evaluation['film_conv_calls'] == expected+2*members,
                 'Actual TRAIN, VALID selection and selected-restore native FiLM calls differ')
        selector = {k: evaluation[k]-selected[k] for k in evaluation}
        _require(all(v >= 0 for v in selector.values()), 'Selected call accounting exceeds observed evaluation work')
        total = dict(train); _add(total, evaluation)
        return {'TRAIN': train, 'VALID_selector': selector, 'selected_VALID': selected, 'total': total}

    def check(self, batch):
        if 'graph' not in batch or batch['graph'].edge_index is not batch['edge_index']:
            raise ValueError('Supply the same factual graph edge_index in graph and batch')
        if 'edge_attr' in batch or 'edge_weight' in batch:
            raise ValueError('Original Model does not consume these edge fields')

    def native(self, batch):
        captured = []; head = self.body.output_linear
        hook = head.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
        try:
            logits = self.body(batch['graph'], batch['x'])
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError('Unique original output head required')
        return logits.unsqueeze(-1) if logits.ndim == 1 else logits, captured[0]
