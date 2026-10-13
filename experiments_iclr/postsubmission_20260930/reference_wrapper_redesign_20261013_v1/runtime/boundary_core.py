"""Hooks at two explicit native boundaries. No numerical import at module load.

This implements a portable core, not any qualified paper-model/data adapter.
The caller constructs its original native model and injects its own torch.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import weakref


MEMBERS = 4


@dataclass(frozen=True)
class BoundaryBinding:
    module_path: str
    kind: str
    role: str

    def validate(self):
        if not self.module_path or self.kind not in ("linear", "embedding"):
            raise ValueError("An exact module path and linear/embedding kind are required")
        if self.role not in ("input", "output"):
            raise ValueError("Only the declared input/output boundary roles are supported")


@dataclass(frozen=True)
class RouteOutput:
    prediction: object
    raw_logits: object
    prehead: object


def resolve_exact_module(native_model, path):
    """Resolve only the given path; no traversal or placement inference."""
    current = native_model
    for part in path.split("."):
        if not part or part not in current._modules:
            raise ValueError("Boundary path does not resolve through native registered modules")
        current = current._modules[part]
    return current


def attach_boundary_core(torch, native_model, bindings, *, factor_seed, members=MEMBERS):
    """Keep native objects and add private banks plus removable hooks.

    Linear: native_affine(x * R_m) * S_m + delta_b_m.
    Embedding: native_embedding(IDs) * R_m[IDs][...,None] * S_m + delta_b_m.
    The first boundary R has isolated-CPU-RNG Rademacher initialization;
    every other R/S is one and delta_b is zero. The neutral unit/zero state
    recovers the native operation, but the adopted sign start is not neutral.
    Native initialization, categorical IDs, forward and activation stay intact.
    """
    if members != MEMBERS or isinstance(factor_seed, bool) or not isinstance(factor_seed, int) or factor_seed < 0:
        raise ValueError("Four routes and an explicit nonnegative factor seed are required")
    bindings = tuple(bindings)
    if len(bindings) != 2 or {b.role for b in bindings} != {"input", "output"}:
        raise ValueError("Exactly one input and one output binding are required")
    for binding in bindings:
        binding.validate()
    bindings = tuple(sorted(bindings, key=lambda b: b.role != "input"))
    if len({b.module_path for b in bindings}) != 2:
        raise ValueError("The two boundary module paths must be distinct")
    resolved = tuple(resolve_exact_module(native_model, b.module_path) for b in bindings)
    if len({id(m) for m in resolved}) != 2:
        raise ValueError("Aliased input/output modules need an explicit source-specific adapter")
    for binding, module in zip(bindings, resolved):
        expected = torch.nn.Linear if binding.kind == "linear" else torch.nn.Embedding
        if type(module) is not expected:
            raise TypeError("Only exact native nn.Linear/nn.Embedding leaves are supported")
        if not module.weight.is_floating_point():
            raise TypeError("Native boundary weights must already be floating tensors")
    if bindings[1].kind != "linear":
        raise ValueError("Raw-logit/prehead capture currently requires a Linear output boundary")
    before_parameters = tuple(native_model.named_parameters())
    before_buffers = tuple(native_model.named_buffers())

    class FactorBank(torch.nn.Module):
        def __init__(self, module, binding, first):
            super().__init__()
            rows = module.in_features if binding.kind == "linear" else module.num_embeddings
            columns = module.out_features if binding.kind == "linear" else module.embedding_dim
            options = dict(dtype=module.weight.dtype, device=module.weight.device)
            self.r = torch.nn.Parameter(torch.ones(members, rows, **options))
            self.s = torch.nn.Parameter(torch.ones(members, columns, **options))
            self.delta_b = torch.nn.Parameter(torch.zeros(members, columns, **options))
            self.binding = binding
            self._native_ref = weakref.ref(module)
            if first:
                generator = torch.Generator(device="cpu").manual_seed(factor_seed)
                signs = torch.randint(0, 2, (members, rows), generator=generator, device="cpu") * 2 - 1
                with torch.no_grad():
                    self.r.copy_(signs.to(device=self.r.device, dtype=self.r.dtype))

    class NativeBoundaryCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = native_model
            self.banks = torch.nn.ModuleList(
                FactorBank(module, binding, index == 0)
                for index, (binding, module) in enumerate(zip(bindings, resolved)))
            self._source_parameters = before_parameters
            self._source_buffers = before_buffers
            self._active_member = None
            self._captures = {}
            self._calls = {}
            self._handles = []
            self._factor_seed = factor_seed
            self._removed = False
            try:
                for bank, module in zip(self.banks, resolved):
                    # Append hooks; never clear, replace or reorder existing native hooks.
                    self._handles.append(module.register_forward_pre_hook(self._prehook(bank)))
                    self._handles.append(module.register_forward_hook(self._posthook(bank)))
                self.assert_native_parameter_identity()
            except BaseException:
                self.remove_hooks()
                raise

        def _prehook(self, bank):
            def hook(module, inputs):
                member = self._active_member
                if member is None:
                    return None
                if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
                    raise ValueError("Bound leaves require one positional native tensor input")
                self._calls[bank.binding.role] += 1
                x = inputs[0]
                if bank.binding.role == "output":
                    self._captures["prehead"] = x  # live tensor, before this boundary's R
                if bank.binding.kind == "embedding":
                    if x.dtype not in (torch.int32, torch.int64):
                        raise TypeError("Keep native categorical IDs as integer tensors")
                    return None  # no cast, multiplication, replacement or mutation of IDs
                return (x * bank.r[member],)
            return hook

        def _posthook(self, bank):
            def hook(module, inputs, output):
                member = self._active_member
                if member is None:
                    return None
                if not isinstance(output, torch.Tensor):
                    raise TypeError("Bound native leaves must return a tensor")
                if bank.binding.kind == "embedding":
                    output = output * bank.r[member][inputs[0]].unsqueeze(-1)
                result = output * bank.s[member] + bank.delta_b[member]
                if bank.binding.role == "output":
                    self._captures["raw_logits"] = result
                return result
            return hook

        def forward(self, *args, **kwargs):
            return self.native(*args, **kwargs)

        @contextmanager
        def route(self, member):
            if self._removed or self._active_member is not None:
                raise RuntimeError("Hooks must be attached and route contexts cannot nest")
            if isinstance(member, bool) or not isinstance(member, int) or not 0 <= member < members:
                raise ValueError("Explicit route index in [0,4) required")
            self._active_member = member
            self._captures = {}
            self._calls = {"input": 0, "output": 0}
            try:
                yield self
            finally:
                self._active_member = None

        def run_member(self, member, *args, **kwargs):
            """Run the unchanged native forward; keep its activated prediction."""
            with self.route(member):
                prediction = self(*args, **kwargs)
                if self._calls != {"input": 1, "output": 1}:
                    raise ValueError("Both exact boundary sites must execute once per native forward")
                return RouteOutput(prediction, self._captures["raw_logits"], self._captures["prehead"])

        def assert_native_parameter_identity(self):
            current = tuple(self.native.named_parameters())
            if len(current) != len(self._source_parameters) or any(
                    name != old_name or parameter is not old_parameter
                    for (name, parameter), (old_name, old_parameter) in zip(current, self._source_parameters)):
                raise ValueError("Native parameter objects or registration changed")
            current_buffers = tuple(self.native.named_buffers())
            if len(current_buffers) != len(self._source_buffers) or any(
                    name != old_name or buffer is not old_buffer
                    for (name, buffer), (old_name, old_buffer) in zip(current_buffers, self._source_buffers)):
                raise ValueError("Native buffer objects or registration changed during attachment")

        def private_parameters(self):
            return tuple(parameter for bank in self.banks for parameter in (bank.r, bank.s, bank.delta_b))

        def remove_hooks(self):
            if self._active_member is not None:
                raise RuntimeError("Cannot detach hooks during a native route forward")
            for handle in self._handles:
                handle.remove()
            self._handles = []
            self._removed = True

        def metadata(self):
            return {"schema": "injected-native-boundary-core-v1", "members": members,
                    "native_parameters_retained": True, "native_buffers_retained": True,
                    "categorical_IDs_modified": False, "native_activation_removed": False,
                    "first_R": "isolated CPU torch.Generator Rademacher", "factor_seed": self._factor_seed,
                    "other_R_S": "ones", "delta_b": "zeros", "neutral_state_equivalence_verified": False,
                    "native_initial_function_equal_at_sign_start": False,
                    "recipe_adapters_qualified": False,
                    "sites": [{"path": bank.binding.module_path, "kind": bank.binding.kind,
                               "role": bank.binding.role, "r_shape": list(bank.r.shape),
                               "s_shape": list(bank.s.shape), "delta_b_shape": list(bank.delta_b.shape)}
                              for bank in self.banks]}

    return NativeBoundaryCore()


def extend_native_optimizer_groups(core, optimizer):
    """Add only private banks to existing native boundary owner groups.

    Native parameter objects, existing group dictionaries/options and existing
    optimizer state are retained. The caller owns its source optimizer and
    scheduler. Unsupported optimizer interfaces require a separate native port.
    """
    if optimizer.__class__.__name__ in ("SparseAdam", "LBFGS"):
        raise ValueError("Dense private factors need an explicit source optimizer adapter here")
    native_ids = {id(p) for _, p in core._source_parameters}
    private = core.private_parameters()
    private_ids = {id(p) for p in private}
    owners = {}
    for index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            key = id(parameter)
            if key in owners or key not in native_ids:
                raise ValueError("Expected disjoint original native optimizer parameter groups")
            owners[key] = index
    if private_ids & owners.keys():
        raise ValueError("Private banks are already registered with the optimizer")
    assignments = []
    for bank in core.banks:
        module = bank._native_ref()
        if module is None or id(module.weight) not in owners:
            raise ValueError("Native boundary weight must have its predeclared source optimizer group")
        weight_group = owners[id(module.weight)]
        native_bias = getattr(module, "bias", None)
        if native_bias is not None and id(native_bias) not in owners:
            raise ValueError("Native boundary bias group must remain explicitly owned")
        bias_group = owners[id(native_bias)] if native_bias is not None else weight_group
        assignments.extend(((bank.r, weight_group), (bank.s, weight_group), (bank.delta_b, bias_group)))
    # Validate the entire plan before the first mutation.
    if len({id(p) for p, _ in assignments}) != len(private_ids):
        raise ValueError("Every private bank must be assigned exactly once")
    for parameter, group_index in assignments:
        optimizer.param_groups[group_index]["params"].append(parameter)
    return {"schema": "native-boundary-private-group-assignment-v1",
            "existing_group_options_changed": False, "native_group_members_removed": 0,
            "private_parameters_added": len(assignments),
            "assignments": [{"private_index": index, "native_group_index": group}
                            for index, (_, group) in enumerate(assignments)]}
