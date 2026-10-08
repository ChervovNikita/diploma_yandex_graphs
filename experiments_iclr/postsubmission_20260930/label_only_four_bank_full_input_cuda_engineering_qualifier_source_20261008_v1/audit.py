"""Read-only engineering probes on the actual full-graph corrector calls."""


def require(value, message):
    if not value:
        raise ValueError(message)


def same_tree(torch, actual, expected, location='state'):
    """Exact copied learned/Adam custody, allowing CPU snapshot/device placement."""
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape
                and actual.dtype == expected.dtype
                and torch.equal(actual.detach().cpu(), expected.detach().cpu()),
                'Copied tensor differs: ' + location)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(),
                'Copied mapping inventory differs: ' + location)
        for key in expected:
            same_tree(torch, actual[key], expected[key], location + '.' + str(key))
    elif isinstance(expected, (list, tuple)):
        require(type(actual) is type(expected) and len(actual) == len(expected),
                'Copied sequence differs: ' + location)
        for index, (left, right) in enumerate(zip(actual, expected)):
            same_tree(torch, left, right, location + '.' + str(index))
    else:
        require(type(actual) is type(expected) and actual == expected,
                'Copied scalar differs: ' + location)


class RouteAudit:
    """Inspect real Q/context/edge/readout calls without replacing their values.

    Every graph remains complete. This probe only wraps existing route methods
    and reads the existing Embedding input; it performs no forward or backward.
    """
    def __init__(self, torch, banks, heads, conditions):
        self.torch, self.banks, self.heads, self.conditions = torch, banks, heads, conditions
        self.expected = {}; self.events = []; self.handles = []; self.originals = []
        for arm, bank in banks.items():
            candidate = conditions[arm] == 'label_only4_detached'
            method = '_route_delta' if candidate else '_route_message'
            original = getattr(bank, method)
            self.originals.append((bank, method, original))
            setattr(bank, method, self._wrap(arm, bank, original))
            embeddings = [module for module in bank.modules()
                          if isinstance(module, torch.nn.Embedding)]
            require(len(embeddings) == (1 if candidate else heads[arm]),
                    'Exact label embedding inventory for ' + arm)
            for module in embeddings:
                self.handles.append(module.register_forward_pre_hook(self._embedding(arm, bank)))

    def begin(self, mode, *, masks=None, ids=None):
        require(not self.expected and mode in ('TRAIN', 'SERVE'), 'One diagnostic event')
        torch = self.torch
        reference = None
        for arm, bank in self.banks.items():
            if mode == 'TRAIN':
                mask = masks[arm]
                identity = (mask.query_positions, mask.query_ids, mask.training_value_scale)
                require(reference is None or identity == reference, 'One identical Q across all four banks')
                reference = identity
                positions = torch.tensor(mask.query_positions, dtype=torch.long,
                                         device=bank.train_ids.device)
                queries = bank.train_ids[positions]
                visible = torch.ones(bank.allowed_count, dtype=torch.bool, device=positions.device)
                visible[positions] = False
                visible = visible.nonzero(as_tuple=False).flatten()
                scale = (580 - 1) / (580 - 290)
                require(len(positions) == 290 and len(visible) == 290
                        and tuple(queries.cpu().tolist()) == mask.query_ids,
                        'Complete half-TRAIN Q and complement')
            else:
                queries = ids.to(bank.train_ids.device)
                visible = torch.arange(bank.allowed_count, device=bank.train_ids.device)
                scale = 1.
            require(not bool(torch.isin(queries, bank.train_ids[visible]).any()),
                    'Query labels are excluded from the actual context')
            self.expected[arm] = dict(mode=mode, queries=queries, visible=visible,
                scale=scale, routes=0, embedding_calls=0, zero_rows=None)

    def _embedding(self, arm, bank):
        def inspect(module, arguments):
            expected = self.expected[arm]
            require(len(arguments) == 1 and self.torch.equal(arguments[0],
                    bank.train_labels[expected['visible']]),
                    'Only the exact permitted TRAIN label vector enters Embedding')
            expected['embedding_calls'] += 1
            return None
        return inspect

    def _wrap(self, arm, bank, original):
        def inspect(H, queries, visible, scale, source, row, member, training):
            torch = self.torch; expected = self.expected[arm]
            require(H.shape == (11701, 512) and H.grad_fn is None and not H.requires_grad,
                    'Actual route sees the detached full native representation')
            require(torch.equal(queries, expected['queries'])
                    and torch.equal(visible, expected['visible']) and scale == expected['scale']
                    and training == (expected['mode'] == 'TRAIN'),
                    'Actual route uses the common Q and exact permitted label complement')
            lookup = torch.full((11701,), -1, dtype=torch.long, device=queries.device)
            lookup[queries] = torch.arange(len(queries), device=queries.device)
            actual_rows = lookup[bank.edge_target]; keep = actual_rows >= 0
            require(torch.equal(source, bank.edge_source[keep])
                    and torch.equal(row, actual_rows[keep]),
                    'All original nonself neighbor records, order and multiplicity retained')
            legal = torch.zeros(11701, dtype=torch.bool, device=H.device)
            legal[bank.train_ids[visible]] = True
            count = torch.zeros(len(queries), dtype=torch.long, device=H.device)
            count.index_add_(0, row, legal[source].long()); empty = count == 0
            result = original(H, queries, visible, scale, source, row, member, training)
            require(torch.equal(result[empty], torch.zeros_like(result[empty])),
                    'Actual learned correction/message is exactly zero without a visible anchor')
            zero_rows = int(empty.sum())
            require(zero_rows > 0, 'Full real-graph zero-context check must be nonvacuous')
            require(expected['zero_rows'] in (None, zero_rows), 'Identical route context')
            expected['zero_rows'] = zero_rows; expected['routes'] += 1
            return result
        return inspect

    def finish(self):
        require(self.expected, 'A diagnostic event must have actual calls')
        event = {}
        for arm, expected in self.expected.items():
            require(expected['routes'] == expected['embedding_calls'] == self.heads[arm],
                    'All existing heads audited once for ' + arm)
            event[arm] = dict(mode=expected['mode'], queries=len(expected['queries']),
                route_calls=expected['routes'], permitted_label_embedding_calls=expected['embedding_calls'],
                zero_context_rows=expected['zero_rows'], common_Q_exclusion=True,
                original_all_neighbor_records=True, detached_full_H=True)
        self.events.append(event); self.expected = {}
        return event

    def close(self):
        for handle in self.handles:
            handle.remove()
        for bank, method, original in self.originals:
            setattr(bank, method, original)
        self.handles = []; self.originals = []; self.expected = {}
