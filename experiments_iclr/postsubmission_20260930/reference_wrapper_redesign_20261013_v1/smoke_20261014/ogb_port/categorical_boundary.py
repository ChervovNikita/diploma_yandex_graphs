"""Full AtomEncoder logical input boundary; no categorical-ID mutation."""
from contextlib import contextmanager
import weakref


MEMBERS = 4


def attach_ogb_boundary(torch, native, *, factor_seed):
    encoder, head = native.gnn_node.atom_encoder, native.graph_pred_linear
    if type(head) is not torch.nn.Linear or not hasattr(encoder, "atom_embedding_list"):
        raise TypeError("Exact native AtomEncoder and graph_pred_linear interfaces required")
    tables = tuple(encoder.atom_embedding_list)
    if not tables or any(type(e) is not torch.nn.Embedding for e in tables):
        raise TypeError("Every original categorical atom field must retain its native Embedding")
    width = tables[0].embedding_dim
    if any(e.embedding_dim != width for e in tables) or head.in_features != width:
        raise ValueError("Native mean-pooling atom/head dimensions differ")
    source_parameters, source_buffers = tuple(native.named_parameters()), tuple(native.named_buffers())

    class AtomBank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            options = dict(dtype=tables[0].weight.dtype, device=tables[0].weight.device)
            self.r = torch.nn.ParameterList(torch.nn.Parameter(torch.ones(MEMBERS, e.num_embeddings, **options))
                                            for e in tables)
            self.s = torch.nn.Parameter(torch.ones(MEMBERS, width, **options))
            self.delta_b = torch.nn.Parameter(torch.zeros(MEMBERS, width, **options))
            generator = torch.Generator(device="cpu").manual_seed(factor_seed)
            with torch.no_grad():
                for bank in self.r:
                    signs = torch.randint(0, 2, bank.shape, generator=generator, device="cpu") * 2 - 1
                    bank.copy_(signs.to(device=bank.device, dtype=bank.dtype))

    class HeadBank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            options = dict(dtype=head.weight.dtype, device=head.weight.device)
            self.r = torch.nn.Parameter(torch.ones(MEMBERS, head.in_features, **options))
            self.s = torch.nn.Parameter(torch.ones(MEMBERS, head.out_features, **options))
            self.delta_b = torch.nn.Parameter(torch.zeros(MEMBERS, head.out_features, **options))

    class OGBBoundary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = native
            self.input_bank, self.output_bank = AtomBank(), HeadBank()
            self._source_parameters, self._source_buffers = source_parameters, source_buffers
            self._encoder_ref, self._head_ref = weakref.ref(encoder), weakref.ref(head)
            self._tables = tuple(weakref.ref(e) for e in tables)
            self._member = None
            self._handles = []
            self._captures, self._calls = {}, {}
            try:
                self._handles.append(encoder.register_forward_pre_hook(self._encoder_pre))
                for index, table in enumerate(tables):
                    self._handles.append(table.register_forward_hook(self._field_post(index)))
                self._handles.append(encoder.register_forward_hook(self._encoder_post))
                self._handles.append(head.register_forward_pre_hook(self._head_pre))
                self._handles.append(head.register_forward_hook(self._head_post))
            except BaseException:
                self.close()
                raise
            self.assert_source_identity()

        def _encoder_pre(self, module, inputs):
            if self._member is None:
                return None
            if len(inputs) != 1 or inputs[0].ndim != 2 or inputs[0].shape[1] != len(tables):
                raise ValueError("All original full categorical atom fields are required")
            if inputs[0].dtype not in (torch.int64, torch.int32):
                raise TypeError("Categorical atom IDs must remain integer tensors")
            self._calls["encoder"] += 1
            return None  # original ID tensor is passed to native AtomEncoder unchanged

        def _field_post(self, index):
            def hook(module, inputs, output):
                if self._member is None:
                    return None
                if len(inputs) != 1 or inputs[0].ndim != 1:
                    raise ValueError("Native AtomEncoder must perform its original one-field lookup")
                self._calls["fields"][index] += 1
                return output * self.input_bank.r[index][self._member][inputs[0]].unsqueeze(-1)
            return hook

        def _encoder_post(self, module, inputs, output):
            if self._member is None:
                return None
            # Native sums all field embeddings first. One S and one bias residual
            # act on the logical encoded input; no per-field bias is introduced.
            return output * self.input_bank.s[self._member] + self.input_bank.delta_b[self._member]

        def _head_pre(self, module, inputs):
            if self._member is None:
                return None
            if len(inputs) != 1:
                raise ValueError("Native graph head must receive its original one-tensor input")
            self._calls["head"] += 1
            self._captures["prehead"] = inputs[0]
            return (inputs[0] * self.output_bank.r[self._member],)

        def _head_post(self, module, inputs, output):
            if self._member is None:
                return None
            result = output * self.output_bank.s[self._member] + self.output_bank.delta_b[self._member]
            self._captures["raw_logits"] = result
            return result

        def forward(self, *args, **kwargs):
            return self.native(*args, **kwargs)

        @contextmanager
        def route(self, member):
            if self._member is not None or not 0 <= member < MEMBERS:
                raise ValueError("One valid, nonnested route context required")
            self._member = member
            self._captures = {}
            self._calls = {"encoder": 0, "head": 0, "fields": [0] * len(tables)}
            try:
                yield
            finally:
                self._member = None

        def run_member(self, member, batch):
            with self.route(member):
                prediction = self(batch)
                if self._calls != {"encoder": 1, "head": 1, "fields": [1] * len(tables)}:
                    raise ValueError("Every full atom field and both logical boundaries must execute once")
                return prediction, self._captures["raw_logits"], self._captures["prehead"]

        def private_parameters(self):
            return tuple(self.input_bank.r) + (self.input_bank.s, self.input_bank.delta_b,
                                               self.output_bank.r, self.output_bank.s, self.output_bank.delta_b)

        def assert_source_identity(self):
            now = tuple(self.native.named_parameters())
            buffers = tuple(self.native.named_buffers())
            if len(now) != len(source_parameters) or any(n != old_n or p is not old_p
                    for (n, p), (old_n, old_p) in zip(now, source_parameters)):
                raise ValueError("Original native parameter identity changed")
            if len(buffers) != len(source_buffers) or any(n != old_n or b is not old_b
                    for (n, b), (old_n, old_b) in zip(buffers, source_buffers)):
                raise ValueError("Original native buffer identity changed")

        def close(self):
            if self._member is not None:
                raise RuntimeError("Cannot remove hooks during a route")
            for handle in self._handles:
                handle.remove()
            self._handles = []

        def metadata(self):
            return {"input_site": "gnn_node.atom_encoder", "output_site": "graph_pred_linear",
                    "atom_fields": len(tables), "vocabulary_sizes": [e.num_embeddings for e in tables],
                    "embedding_width": width, "native_parameters_replaced": 0, "categorical_IDs_changed": False,
                    "bond_encoder_factors": False, "virtual_node_factors": False,
                    "native_BN_policy": "same shared original buffers; native forward once per route",
                    "first_R": "one isolated CPU generator, field-order Rademacher tables",
                    "other_R_S": "ones", "delta_b": "zeros", "factor_seed": factor_seed}

    return OGBBoundary()


def attach_native_optimizer_groups(model, optimizer):
    """Preserve all native groups/options; append banks to native weight/bias owners."""
    owners = {id(p): index for index, group in enumerate(optimizer.param_groups) for p in group["params"]}
    native_ids = {id(p) for _, p in model._source_parameters}
    if set(owners) != native_ids:
        raise ValueError("The source optimizer must initially own exactly the original native parameters")
    assignments = []
    for ref, bank in zip(model._tables, model.input_bank.r):
        assignments.append((bank, owners[id(ref().weight)]))
    input_group = owners[id(model._tables[0]().weight)]
    if any(owners[id(ref().weight)] != input_group for ref in model._tables):
        raise ValueError("Common input S/bias requires the native OGB shared atom optimizer group")
    assignments.extend(((model.input_bank.s, input_group), (model.input_bank.delta_b, input_group)))
    head = model._head_ref()
    weight_group = owners[id(head.weight)]
    bias_group = owners[id(head.bias)] if head.bias is not None else weight_group
    assignments.extend(((model.output_bank.r, weight_group), (model.output_bank.s, weight_group),
                        (model.output_bank.delta_b, bias_group)))
    if len({id(p) for p, _ in assignments}) != len(model.private_parameters()):
        raise ValueError("Every boundary bank needs one native group assignment")
    for parameter, index in assignments:
        optimizer.param_groups[index]["params"].append(parameter)
    return [{"private_index": i, "native_group": group} for i, (_, group) in enumerate(assignments)]
