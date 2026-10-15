"""Route-owned BatchNorm running state; all native learned parameters stay shared.

No task, data loader or training loop is changed here. The caller must enter the
appropriate route around the unchanged native forward, in TRAIN and in eval.
"""
from contextlib import contextmanager
import weakref

import torch


class _State(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            value = getattr(source, name)
            self.register_buffer(name, value.detach().clone())


class RouteBatchNormState(torch.nn.Module):
    """Register private state without registering/copying the native backbone.

    Native buffer objects remain canonical outside each context. Inside the
    context the original BN module sees one route's registered buffers. Pointer
    binding avoids in-place copies that can invalidate an outstanding autograd
    graph when several route forwards precede a single backward.
    """
    def __init__(self, native, members=4):
        super().__init__()
        if type(members) is not int or members < 2:
            raise ValueError("At least two explicitly declared routes required")
        self.members = members
        self._native_ref = weakref.ref(native)
        self._native_parameters = tuple(native.named_parameters())
        self._native_buffers = tuple(native.named_buffers())
        self._active = None
        self.names, self._refs, self._original, self._configuration = [], [], [], []
        self.banks = torch.nn.ModuleList()
        self.ignored_untracked = []
        for name, module in native.named_modules():
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue
            if isinstance(module, torch.nn.SyncBatchNorm):
                raise TypeError("Distributed synchronized BN needs a separate context law")
            if not module.track_running_stats:
                self.ignored_untracked.append(name)
                continue
            if any(getattr(module, key) is None for key in
                   ("running_mean", "running_var", "num_batches_tracked")):
                raise ValueError("Tracked native BN must have materialized state")
            self.names.append(name)
            self._refs.append(weakref.ref(module))
            self._original.append(tuple(getattr(module, key) for key in
                                        ("running_mean", "running_var", "num_batches_tracked")))
            self._configuration.append((type(module), module.num_features, module.eps,
                                        module.momentum, module.affine, module.track_running_stats))
            self.banks.append(torch.nn.ModuleList(_State(module) for _ in range(members)))
        if not self.names:
            raise ValueError("No tracked native BN; do not add BN to create a candidate")

    def assert_native_identity(self):
        if self._active is not None:
            raise RuntimeError("Identity is canonical only outside route contexts")
        native = self._native_ref()
        if native is None:
            raise RuntimeError("Native backbone no longer exists")
        for current, old, label in ((tuple(native.named_parameters()), self._native_parameters, "parameter"),
                                    (tuple(native.named_buffers()), self._native_buffers, "buffer")):
            if len(current) != len(old) or any(n != o or x is not y
                                                for (n, x), (o, y) in zip(current, old)):
                raise ValueError("Original native " + label + " identity changed")

    @contextmanager
    def route(self, member):
        if self._active is not None or type(member) is not int or not 0 <= member < self.members:
            raise ValueError("One nonnested valid route context required")
        self.assert_native_identity()
        bound = []
        self._active = member
        try:
            for index, reference in enumerate(self._refs):
                module, bank = reference(), self.banks[index][member]
                if module is None:
                    raise RuntimeError("Native BN module no longer exists")
                config = (type(module), module.num_features, module.eps,
                          module.momentum, module.affine, module.track_running_stats)
                if config != self._configuration[index]:
                    raise ValueError("Native BN configuration changed")
                bound.append((module, self._original[index]))
                for key in ("running_mean", "running_var", "num_batches_tracked"):
                    module._buffers[key] = getattr(bank, key)
            yield
        finally:
            for module, original in reversed(bound):
                for key, value in zip(("running_mean", "running_var", "num_batches_tracked"), original):
                    module._buffers[key] = value
            self._active = None
            self.assert_native_identity()

    def metadata(self):
        return {"method": "route_private_bn_running_state_shared_affine_v2",
                "members": self.members, "tracked_modules": list(self.names),
                "untracked_modules_unchanged": list(self.ignored_untracked),
                "native_parameters_copied": 0, "additional_learned_parameters": 0,
                "bank_bytes": sum(x.numel() * x.element_size() for x in self.buffers()),
                "resident_native_buffers": "original objects and initial state outside route",
                "state_access": "TRAIN updates its own route; eval reads it without recalibration",
                "synchronized_or_activation_recomputed_BN": "not qualified"}

    def state_dict(self, *args, **kwargs):
        if self._active is not None:
            raise RuntimeError("Cannot save route state during an active forward context")
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        if self._active is not None:
            raise RuntimeError("Cannot load route state during an active forward context")
        return super().load_state_dict(*args, **kwargs)


def attach_molecular(session):
    """Attach to the exact four-route molecular boundary interface.

    Register all banks in model.state_dict() so the existing session checkpoint
    exports/restores them. Native source-identity guards remain intact. This
    extension is not injected into any frozen running pilot.
    """
    import types
    model = session.model
    if not hasattr(model, "run_member") or not hasattr(model, "native"):
        raise TypeError("An existing native-preserving four-route model is required")
    if hasattr(model, "route_bn_state"):
        raise ValueError("Route state already attached")
    manager = RouteBatchNormState(model.native, members=4)
    model.add_module("route_bn_state", manager)
    # Direct forward would silently read canonical *initial* running state.
    def require_route(module, inputs):
        if manager._active is None:
            raise RuntimeError("Private-BN candidate forward requires run_member route context")

    def require_idle(module, *args):
        if manager._active is not None:
            raise RuntimeError("Cannot save/load model during an active BN route context")

    manager._handles = [model.native.register_forward_pre_hook(require_route),
                        model.register_state_dict_pre_hook(require_idle),
                        model.register_load_state_dict_pre_hook(require_idle),
                        model.native.register_state_dict_pre_hook(require_idle),
                        model.native.register_load_state_dict_pre_hook(require_idle)]
    manager._handles += [ref().register_forward_pre_hook(require_route) for ref in manager._refs]
    original = model.run_member

    def run_member(self, member, batch):
        with manager.route(member):
            result = original(member, batch)
        return result

    model.run_member = types.MethodType(run_member, model)
    model.assert_source_identity()
    return manager.metadata()
