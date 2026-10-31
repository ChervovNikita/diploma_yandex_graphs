"""Inactive context extension of the exact existing six-site SeHGNN factor bank.

Torch and the pinned adapter are loaded only after explicit root source/model
capabilities. Neither native weights, data nor providers are created at import.
"""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import math
from pathlib import Path

from .caps import CLOSED
from .context import FIELD_DIM


NATIVE_MODEL_SHA256 = "0948239c4c4d06dd258cd106fd2fa7b6aaa2a62688fe70dd662413fa1eae532a"
ADAPTER_SHA256 = "7d5f899645361212a64fe70ab0b06fd74cc5cfcf8d2ca361e833b884cadd5e47"
CONTEXT_SITES = ("feature_projection.0", "feature_projection.4")
ORIGINAL_SITES = CONTEXT_SITES + ("semantic_fusion.query", "semantic_fusion.key",
                                  "semantic_fusion.value", "fc_after_concat")
CONDITIONS = ("shared_own_pair4", "shared_local_mul4", "shared_local_add4",
              "shared_global_mul4", "single_local_mul1", "untied_local_mul4")


def load_adapter(path, caps=CLOSED):
    caps.require("source_bound", "model", "runtime")
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(Path(__file__).resolve().parent.parent):
        raise ValueError("Adapter source must remain inside the project research repository")
    if hashlib.sha256(path.read_bytes()).hexdigest() != ADAPTER_SHA256:
        raise ValueError("Exact original grouped six-site adapter bytes required")
    spec = importlib.util.spec_from_file_location("pinned_sehgnn_original_factor_adapter", path)
    module = importlib.util.module_from_spec(spec)
    # Dataclass decorators expect their module to be registered. No provider/model
    # is constructed by this pinned adapter import.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if tuple(module.SITES) != ORIGINAL_SITES:
        raise ValueError("Do not change the original four-route factor sites")
    return module


def _seed(base, member, site):
    token = f"typed-context-H:{base}:{member}:{site}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**63-1)


def _context_type(torch):
    class ContextGrouped(torch.nn.Module):
        def __init__(self, previous, hidden, seed, additive):
            super().__init__()
            self.cin, self.cout, self.num_metapaths = previous.cin, previous.cout, previous.num_metapaths
            self.W, self.bias = previous.W, previous.bias
            self.input_factor, self.output_factor = previous.input_factor, previous.output_factor
            object.__setattr__(self, "_native_forward", previous._native_forward)
            self.additive = additive
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            # Dedicated initialization leaves all native global RNG streams intact.
            bound = math.sqrt(6.0 / (FIELD_DIM + hidden))
            h = torch.empty((self.num_metapaths, FIELD_DIM, hidden), dtype=torch.float32)
            h.uniform_(-bound, bound, generator=generator)
            self.context_H = torch.nn.Parameter(h.to(device=self.W.device, dtype=self.W.dtype))
            self.context_U = torch.nn.Parameter(torch.zeros(
                (self.num_metapaths, hidden, self.cout), device=self.W.device, dtype=self.W.dtype))
            object.__setattr__(self, "_context_field", None)
            self.train(previous.training)

        def forward(self, x):
            field = self._context_field
            if (field is None or tuple(field.shape) != (x.shape[0], FIELD_DIM)
                    or tuple(x.shape[1:]) != (self.num_metapaths, self.cin)):
                raise ValueError("Explicit same-batch full context field required")
            r = self.input_factor.to(dtype=x.dtype)
            native_value = self._native_forward(self, x*r.unsqueeze(0))
            s = self.output_factor.to(dtype=native_value.dtype).unsqueeze(0)
            bias = self.bias.to(dtype=native_value.dtype).unsqueeze(0)
            centered = native_value-bias
            base = native_value+(s-1)*centered  # Original exact bias-outside-factor rule.
            context = field.to(device=x.device, dtype=x.dtype)
            encoded = torch.tanh(torch.einsum("bg,cgh->bch", context, self.context_H.to(dtype=x.dtype)))
            delta = torch.einsum("bch,chk->bck", encoded,
                                 self.context_U.to(dtype=encoded.dtype)).to(dtype=native_value.dtype)
            # U=0 preserves base. dL/dH=0 initially; dL/dU need not be0.
            return base+delta if self.additive else base+delta*centered

    return ContextGrouped


def build_bank(native_module, prototype, adapter_path, condition, generator_seed, caps=CLOSED):
    caps.require("source_bound", "model", "runtime")
    if condition not in CONDITIONS or type(generator_seed) is not int:
        raise ValueError("One fixed declared condition and prospective generator seed required")
    adapter = load_adapter(adapter_path, caps)
    native_path = Path(native_module.__file__).resolve(strict=True)
    if not native_path.is_relative_to(Path(__file__).resolve().parent.parent):
        raise ValueError("Native source must remain inside the project research repository")
    if hashlib.sha256(native_path.read_bytes()).hexdigest() != NATIVE_MODEL_SHA256:
        raise ValueError("Native SeHGNN IMDB source changed")
    torch = native_module.torch
    count = 1 if condition == "single_local_mul1" else 4
    shared = condition != "untied_local_mul4"
    installed = []
    if shared:
        old = adapter.install_member_bank(native_module, prototype,
                                         adapter.AdapterConfig(enabled=True, members=count))
        installed = list(old.members)
    else:
        # Matched native initial values, disjoint complete parameter/buffer objects.
        # These are not fitted teachers. Genuine independent initialization remains
        # a separately qualified native reference, not this control.
        for _ in range(count):
            body = copy.deepcopy(prototype)
            one = adapter.install_member_bank(native_module, body,
                                             adapter.AdapterConfig(enabled=True, members=1))
            installed.append(one.members[0])
        old = one
    contextual = condition != "shared_own_pair4"
    hidden = 128 if count == 1 else 32
    private_names = tuple(old.private_names)
    if contextual:
        wrapper = _context_type(torch)
        for m, member in enumerate(installed):
            for site in CONTEXT_SITES:
                owner, key, previous = adapter._site(member, site)
                owner._modules[key] = wrapper(previous, hidden, _seed(generator_seed, m, site),
                                             condition == "shared_local_add4")
        private_names += tuple(f"{site}.{name}" for site in CONTEXT_SITES
                               for name in ("context_H", "context_U"))

    class ConditionalBank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.members = torch.nn.ModuleList(installed)
            self.channels = old.channels
            self.slow_names, self.private_names = tuple(old.slow_names), private_names
            self.shared, self.condition = shared, condition
            self.contextual = contextual
            self.native_model_sha256 = NATIVE_MODEL_SHA256
            self.native_commit = adapter.PINNED_COMMIT

        def forward_member(self, m, batch, features, labels, context_field, mask=None):
            body = self.members[m]
            if (set(features) != set(body.feat_keys) or set(labels) != set(body.label_feat_keys)
                    or len(features) != 25 or len(labels) != 12 or mask is not None):
                raise ValueError("All25 native feature/all12 label channels and factual unmasked serving required")
            sites = [adapter._site(self.members[m], site)[2] for site in CONTEXT_SITES] if contextual else []
            with self.context_bound(sites, context_field):
                return self.members[m](batch, features, labels, mask=mask)

        @contextmanager
        def context_bound(self, sites, field):
            if any(site._context_field is not None for site in sites):
                raise ValueError("Nested/stale context binding is forbidden")
            try:
                for site in sites:
                    object.__setattr__(site, "_context_field", field)
                yield
            finally:
                for site in sites:
                    object.__setattr__(site, "_context_field", None)

        def slow_parameters(self):
            if not shared:
                return ()
            named = dict(self.members[0].named_parameters())
            return tuple(named[name] for name in self.slow_names)

        def private_parameters(self, m):
            named = dict(self.members[m].named_parameters())
            names = self.private_names if shared else self.slow_names+self.private_names
            return tuple(named[name] for name in names)

        def verify_shared_state_dict(self, state):
            if shared:
                for name in self.slow_names:
                    values = [state[f"members.{m}.{name}"] for m in range(len(self.members))]
                    if any(not torch.equal(value, values[0]) for value in values[1:]):
                        raise ValueError("Shared slow aliases in saved state must have identical values")

        def verify_ownership(self):
            maps = [dict(member.named_parameters(remove_duplicate=False)) for member in self.members]
            expected = set(self.slow_names) | set(self.private_names)
            if any(set(row) != expected for row in maps):
                raise ValueError("Native or private parameter coverage changed")
            shared_ids = {id(p) for p in self.slow_parameters()}
            storage = lambda p: (str(p.device), int(p.untyped_storage().data_ptr()))
            shared_storage = {storage(p) for p in self.slow_parameters()}
            if len(shared_storage) != len(shared_ids):
                raise ValueError("Distinct shared parameters alias storage")
            seen = set()
            seen_storage = set()
            for m, row in enumerate(maps):
                if shared and any(row[name] is not maps[0][name] for name in self.slow_names):
                    raise ValueError("Native weights must be genuinely shared")
                private = {id(p) for p in self.private_parameters(m)}
                private_storage = {storage(p) for p in self.private_parameters(m)}
                if len(private_storage) != len(private) or private & (shared_ids | seen):
                    raise ValueError("Private/untied parameter ownership overlaps")
                if private_storage & (shared_storage | seen_storage):
                    raise ValueError("Private/untied parameter storage overlaps")
                seen |= private
                seen_storage |= private_storage
            # Native running buffers must remain member-owned in either condition.
            buffers = set()
            buffer_storage = set()
            for member in self.members:
                current = {id(p) for p in member.buffers()}
                current_storage = {storage(p) for p in member.buffers() if p.numel()}
                if current & buffers or current_storage & buffer_storage:
                    raise ValueError("Member native buffers unexpectedly shared")
                buffers |= current
                buffer_storage |= current_storage
            return {"members": count, "shared_native_weights": shared,
                    "native_factor_sites": ORIGINAL_SITES, "new_context_sites": CONTEXT_SITES if contextual else (),
                    "generator_width": hidden if contextual else 0,
                    "zero_U_suppresses_H_gradient_at_start": contextual,
                    "context_binding_checkpointed": False}

    bank = ConditionalBank()
    bank.verify_ownership()
    return bank
