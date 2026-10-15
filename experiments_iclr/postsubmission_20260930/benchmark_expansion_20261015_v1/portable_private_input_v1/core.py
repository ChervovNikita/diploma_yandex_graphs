"""Four private nonlinear input residuals; tensor/route mechanics only.

No native model, dataset, server, fit loop or numerical library is imported at
module load. The actual source session supplies torch and native boundaries.
"""
from __future__ import annotations
from contextlib import contextmanager
from types import SimpleNamespace

FRAME_POLICY = "disjoint_if_fit_else_separate_orthonormal_frames"
MEMBERS = 4
RANK_CAP = 8


def build_residual(torch, reference, *, width, seed, rank_cap=RANK_CAP, frame_policy=FRAME_POLICY):
    if type(width) is not int or width < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Explicit positive native boundary width and nonnegative seed required")
    if rank_cap != RANK_CAP or frame_policy != FRAME_POLICY:
        raise ValueError("This finite candidate fixes rank cap8 and its explicit dimension-aware frame policy")
    rank = min(rank_cap, width)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def orthonormal(columns):
        # Retain the existing molecular candidate's float32 QR/sign convention
        # whenever the four-route frame fits, rather than silently changing it.
        frame, _ = torch.linalg.qr(torch.randn(width, columns, generator=generator,
                                               dtype=torch.float32), mode="reduced")
        return frame

    if MEMBERS*rank <= width:
        down = orthonormal(MEMBERS*rank).T.reshape(MEMBERS, rank, width)
        actual_policy = "one_disjoint_orthonormal_frame"
    else:
        down = torch.stack([orthonormal(rank).T for _ in range(MEMBERS)])
        actual_policy = "separate_per_route_orthonormal_frames_with_unavoidable_overlap"

    class InputResidual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down = torch.nn.Parameter(down.to(device=reference.device, dtype=reference.dtype))
            self.up = torch.nn.Parameter(reference.new_zeros(MEMBERS, width, rank))

        def forward(self, encoded, member):
            if type(member) is not int or not 0 <= member < MEMBERS or encoded.shape[-1] != width:
                raise ValueError("The declared route and exact native encoded width are required")
            hidden = torch.nn.functional.gelu(torch.nn.functional.linear(encoded, self.down[member]))
            return encoded+torch.nn.functional.linear(hidden, self.up[member])

        def metadata(self):
            return {"width": width, "rank_cap": rank_cap, "rank": rank, "members": MEMBERS,
                "frame_policy": frame_policy, "actual_frame": actual_policy,
                "four_disjoint_subspaces_fit": MEMBERS*rank <= width,
                "initializer": "private CPU float32 Gaussian QR down frames; library QR sign convention; zero up matrices",
                "factor_seed": seed, "global_RNG_consumed": False,
                "law": "encoded + up[route] @ GELU(down[route] @ encoded)",
                "extra_parameters": MEMBERS*2*width*rank,
                "initial_added_residual": "exact zero", "guaranteed_diversity_or_accuracy_gain": False}

    return InputResidual()


def assign_private_parameters(optimizer, native_anchor, parameters):
    """Use the anchor's existing source group; retain options/group objects."""
    parameters = tuple(parameters)
    owners = [group for group in optimizer.param_groups if any(p is native_anchor for p in group["params"])]
    existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if len(owners) != 1 or any(id(p) in existing for p in parameters):
        raise ValueError("One native anchor owner and previously unowned private parameters are required")
    owners[0]["params"].extend(parameters)
    return next(index for index, group in enumerate(optimizer.param_groups) if group is owners[0])


def remove_private_ownership(optimizer, parameters):
    """Remove an untrained wrapper bank; never alter any native Parameter."""
    ids = {id(p) for p in parameters}
    found = {id(p) for group in optimizer.param_groups for p in group["params"] if id(p) in ids}
    if found != ids or any(parameter in optimizer.state for group in optimizer.param_groups
                            for parameter in group["params"] if id(parameter) in ids):
        raise ValueError("Residual-only conversion requires exactly the untrained wrapper banks")
    for group in optimizer.param_groups:
        group["params"][:] = [p for p in group["params"] if id(p) not in ids]


def make_residual_only_wrapper(torch, native, input_sites, output_site, *, seed, family):
    """Preserve one native body; add only four input residuals and route API.

    Node and OGB sessions need their existing four-route wrapper API. Link
    sessions use make_link_residual_core because native owns their route API.
    """
    if family not in ("node", "ogb") or len(input_sites) != 1:
        raise ValueError("This wrapper supports exactly the declared node/complete-AtomEncoder boundary")
    before = tuple(native.named_parameters())
    before_buffers = tuple(native.named_buffers())
    input_name, input_module, width, anchor = input_sites[0]
    output_name, output_module = output_site

    class ResidualWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = native
            self.input_residual = build_residual(torch, anchor, width=width, seed=seed)
            self._member = None
            self._handles, self._captures = [], {}
            self._calls = {}
            self._handles.append(input_module.register_forward_hook(self._input_post))
            self._handles.append(output_module.register_forward_pre_hook(self._output_pre))
            self._handles.append(output_module.register_forward_hook(self._output_post))

        def _input_post(self, module, inputs, output):
            if self._member is None:
                return None
            self._calls["input"] += 1
            return self.input_residual(output, self._member)

        def _output_pre(self, module, inputs):
            if self._member is not None:
                self._calls["output"] += 1
                self._captures["prehead"] = inputs[0]

        def _output_post(self, module, inputs, output):
            if self._member is not None:
                self._captures["raw_logits"] = output

        def forward(self, *args, **kwargs):
            return self.native(*args, **kwargs)

        @contextmanager
        def route(self, member):
            if self._member is not None or type(member) is not int or not 0 <= member < MEMBERS:
                raise ValueError("One valid nonnested route required")
            self._member, self._captures, self._calls = member, {}, {"input": 0, "output": 0}
            try:
                yield self
            finally:
                self._member = None

        def run_member(self, member, *args, **kwargs):
            with self.route(member):
                prediction = self(*args, **kwargs)
                if self._calls != {"input": 1, "output": 1}:
                    raise ValueError("Both declared native logical boundaries must execute once")
                values = prediction, self._captures["raw_logits"], self._captures["prehead"]
                return values if family == "ogb" else SimpleNamespace(
                    prediction=values[0], raw_logits=values[1], prehead=values[2])

        def private_parameters(self):
            return tuple(self.input_residual.parameters())

        def assert_source_identity(self):
            current = tuple(self.native.named_parameters())
            if len(current) != len(before) or any(name != oldname or p is not old
                    for (name, p), (oldname, old) in zip(current, before)):
                raise ValueError("An original native parameter object/registration changed")
            buffers = tuple(self.native.named_buffers())
            if len(buffers) != len(before_buffers) or any(name != oldname or value is not old
                    for (name, value), (oldname, old) in zip(buffers, before_buffers)):
                raise ValueError("An original native buffer object/registration changed")

        assert_native_parameter_identity = assert_source_identity

        def close(self):
            if self._member is not None:
                raise ValueError("Cannot detach during an active native route")
            for handle in self._handles:
                handle.remove()
            self._handles = []

        remove_hooks = close

        def metadata(self):
            return {"schema": "residual-only-four-route-native-wrapper-v1", "members": MEMBERS,
                "input_site": input_name, "output_observer_site": output_name,
                "input_residual": self.input_residual.metadata(), "BE_factor_parameters": 0,
                "native_backbone_copies": 1, "private_parameter_scope": "input residuals only",
                "source_operator_or_parameter_replacements": 0}

    return ResidualWrapper()


def make_link_residual_core(torch, native_model, input_sites, output_site, *, seed, task_scope):
    """Native link callbacks retain their existing four-route/RNG orchestration."""
    class LinkResidualCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.residuals = torch.nn.ModuleList(build_residual(torch, anchor, width=width,
                seed=seed+104729*index) for index, (_, _, width, anchor) in enumerate(input_sites))
            self.route, self._active, self._handles = 0, False, []
            self.binding_metadata = []
            for index, (path, module, width, anchor) in enumerate(input_sites):
                residual = self.residuals[index]
                def post(source, inputs, output, adapter=residual):
                    if not self._active:
                        raise ValueError("A native link input boundary executed outside its declared route")
                    return adapter(output, self.route)
                self._handles.append(module.register_forward_hook(post))
                self.binding_metadata.append({"site": path, "role": "input_residual",
                    "native_module_and_parameters_preserved": True, "residual": residual.metadata()})
            if task_scope is not None:
                # Preserve the existing native NCNC recursion/task observer
                # contract. This hook observes eligible task scope and returns
                # native output unchanged; it owns no output factors.
                module = output_site[1]
                def observe(source, inputs, output):
                    task_scope.observe_output(self.route, task_scope.task_output_active())
                self._handles.append(module.register_forward_hook(observe))

        @contextmanager
        def member(self, member):
            if self._active or type(member) is not int or not 0 <= member < MEMBERS:
                raise ValueError("One valid nonnested native link route required")
            old = self.route
            self.route, self._active = member, True
            try:
                yield
            finally:
                self.route, self._active = old, False

        def private_parameters(self):
            return tuple(self.residuals.parameters())

        def close(self):
            if self._active:
                raise ValueError("Cannot detach during an active link route")
            for handle in self._handles:
                handle.remove()
            self._handles = []

    return LinkResidualCore()
