"""New attributed all-affine-map BE composition, prepared and unexecuted.

Preserves pinned native class forwards with explicit complete member trajectories.
This is a sequential source reference, not a qualified packed backend. Private
factors exist before Stage A/warm acquisition; Stage B creates no parameters.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
from types import SimpleNamespace
import math
from typing import Any
from core.transaction import Custody, ExternalStateHook, MOMENT_POLICY, ZERO_POLICY


@dataclass(frozen=True)
class NativeSpec:
    dataset: str
    num_features: int
    num_classes: int
    hidden: int
    d_ffn: int
    K: int
    nlayer: int
    n_head: int
    q: float
    multi: float
    dropout: float
    dprate: float
    base: str
    recipe_receipt: str

    def __post_init__(self):
        if not self.dataset or not self.recipe_receipt or self.base != "mono":
            raise ValueError("Explicit dataset/recipe and qualified mono scope required; no default recipe")
        for name in ("num_features", "num_classes", "hidden", "d_ffn", "nlayer", "n_head"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Positive integer {name} required")
        if self.num_classes < 2 or type(self.K) is not int or self.K < 0 or self.hidden % self.n_head:
            raise ValueError("Class count, polynomial order or head divisibility invalid")
        if (not math.isfinite(self.q) or not math.isfinite(self.multi) or
                int(self.hidden * self.multi) <= 0 or not all(0 <= v <= 1 for v in (self.dropout, self.dprate))):
            raise ValueError("Native MLP width or dropout invalid")


def site_inventory(spec: NativeSpec):
    """All native affine sites, with explicit continuation permissions."""
    result = [("lin1", "stem", spec.num_features, spec.hidden),
              ("lin2", "readout", spec.hidden, spec.hidden),
              ("lin3", "classifier", spec.hidden, spec.num_classes)]
    t = int(spec.hidden * spec.multi)
    for layer in range(spec.nlayer):
        for k in range(spec.K + 1):
            result += [(f"attn.{layer}.attnmodule.token_wise_network.{k}.0", "intermediate", spec.hidden, t),
                       (f"attn.{layer}.attnmodule.token_wise_network.{k}.2", "intermediate", t, spec.hidden)]
        result += [(f"attn.{layer}.attnmodule.W_Q", "intermediate", spec.hidden, spec.hidden),
                   (f"attn.{layer}.attnmodule.W_K", "intermediate", spec.hidden, spec.hidden),
                   (f"attn.{layer}.ffnmodule.ffn_net.lin1", "intermediate", spec.hidden, spec.d_ffn),
                   (f"attn.{layer}.ffnmodule.ffn_net.lin2", "intermediate", spec.d_ffn, spec.hidden)]
    return tuple(result)


def parameter_algebra(spec: NativeSpec, members: int = 4):
    """Complete source algebra; actual numel comparison is unexecuted."""
    h, t, d, c, f, k = spec.hidden, int(spec.hidden * spec.multi), spec.d_ffn, spec.num_classes, spec.num_features, spec.K + 1
    native = f * h + h * h + h * c + 2 * h + c
    native += spec.nlayer * (k * (2 * h * t + t + h) + 2 * h * h + 2 * h * d + d + 5 * h + spec.n_head * k)
    sites = site_inventory(spec)
    factors = members * sum(i + o for _, _, i, o in sites)
    private_b = members * sum(i + o for _, role, i, o in sites if role == "intermediate")
    return {"native_common_parameters": native, "added_R_S_parameters": factors,
            "complete_parameters": native + factors, "stage_b_private_parameters": private_b,
            "affine_sites": len(sites), "members": members, "actual_module_count_verified": False}


def build_all_layer_polyformer(spec: NativeSpec, *, seed: int, members: int = 4,
                              composition_receipt: str):
    """Numerical construction deferred until separate permission.

    y_m = shared_affine(x * R_m) * S_m. No new private bias is added. This uses
    the local MemberFactorLinear algebra with B fixed to zero, and therefore
    openly scales the shared affine bias by S. Edward2's separately private
    ensemble-bias convention is different; this is a new attributed composition,
    not an exact upstream TensorFlow implementation. All-one R/S preserves native.
    """
    import torch
    from torch import nn
    from .native_polyformer_outer import PolyFormer
    if type(seed) is not int or type(members) is not int or members < 1 or not composition_receipt:
        raise ValueError("Explicit construction seed, members and composition receipt required")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Pinned native mono construction requires explicit float32 default dtype")
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        native = PolyFormer(None, SimpleNamespace(**asdict(spec)))

    class MemberAffine(nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.shared = shared
            self.R = nn.Parameter(shared.weight.new_ones((members, shared.in_features)))
            self.S = nn.Parameter(shared.weight.new_ones((members, shared.out_features)))
            self.active_member = None
            self.members = members

        def forward(self, x):
            m = self.active_member
            if type(m) is not int or not 0 <= m < self.members:
                raise RuntimeError("Complete member trajectory must select every factor")
            return self.shared(x * self.R[m]) * self.S[m]

    sites = site_inventory(spec)
    original_linears = {name: module for name, module in native.named_modules() if type(module) is nn.Linear}
    if set(original_linears) != {name for name, _, _, _ in sites}:
        raise RuntimeError("Pinned native affine site inventory changed")
    for name, _, inputs, outputs in sites:
        linear = original_linears[name]
        if (linear.in_features, linear.out_features) != (inputs, outputs):
            raise RuntimeError("Pinned native map shape changed")
        owner_path, _, leaf = name.rpartition(".")
        owner = native.get_submodule(owner_path) if owner_path else native
        setattr(owner, leaf, MemberAffine(linear))
    # Native PolyAttn.bias is a plain tensor and absent from state_dict. Register
    # the same values before any warm bank so device/dtype/custody are explicit.
    for block in native.attn:
        attention = block.attnmodule
        bias = attention.bias
        delattr(attention, "bias")
        attention.register_buffer("bias", bias, persistent=True)

    class AllLayerPolyFormer(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = native
            self.spec = spec
            self.members = members
            self.site_names = tuple(name for name, _, _, _ in sites)
            self.intermediate_sites = tuple(name for name, role, _, _ in sites if role == "intermediate")
            self.composition_receipt = composition_receipt
            self.stage = "A_unfitted"
            self.warm_bank_receipt = ""
            self._in_forward = False

        def forward_member(self, tokens, member):
            if self._in_forward or type(member) is not int or not 0 <= member < self.members:
                raise RuntimeError("Sequential nonreentrant member forward required")
            if (tokens.ndim != 3 or tokens.shape[1] != self.spec.K + 1 or
                    tokens.shape[2] != self.spec.num_features):
                raise ValueError("Complete [nodes,K+1,original_features] tokens required")
            factors = [self.core.get_submodule(name) for name in self.site_names]
            previous = [factor.active_member for factor in factors]
            self._in_forward = True
            try:
                for factor in factors:
                    factor.active_member = member
                # No member pooling, detached frozen path, node slicing or cache
                # mutation occurs inside the complete attention/FFN trajectory.
                return self.core(SimpleNamespace(list_mat=tokens.unbind(1)))
            finally:
                for factor, value in zip(factors, previous):
                    factor.active_member = value
                self._in_forward = False

        def forward(self, tokens):
            return torch.stack([self.forward_member(tokens, m) for m in range(self.members)])

        def begin_private_continuation(self, *, warm_bank_receipt):
            if self.members != 4 or not warm_bank_receipt or self._in_forward:
                raise ValueError("M4 actual copied warm receipt and idle model required")
            for parameter in self.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            for name in self.intermediate_sites:
                factor = self.core.get_submodule(name)
                factor.R.requires_grad_(True); factor.S.requires_grad_(True)
            self.eval()
            self.stage = "B_private_only"
            self.warm_bank_receipt = warm_bank_receipt

        def _primitive_state(self):
            names = ("K", "base", "n_head", "multi", "d_head", "dprate", "dropout",
                     "dataset", "nlayers", "active_member", "members", "in_features",
                     "out_features", "normalized_shape", "eps", "elementwise_affine",
                     "approximate", "inplace")
            modules = []
            for path, module in self.named_modules():
                values = tuple((name, deepcopy(getattr(module, name))) for name in names if hasattr(module, name))
                modules.append((path, values))
            own = (self.spec, self.site_names, self.intermediate_sites, self.composition_receipt,
                   self.stage, self.warm_bank_receipt, self._in_forward)
            return tuple(modules), own

        def _restore_primitive_state(self, image):
            modules, own = image
            current = dict(self.named_modules())
            if set(current) != {path for path, _ in modules}:
                raise RuntimeError("Structural module mutation is unsupported")
            for path, values in modules:
                for name, value in values:
                    setattr(current[path], name, deepcopy(value))
            (self.spec, self.site_names, self.intermediate_sites, self.composition_receipt,
             self.stage, self.warm_bank_receipt, self._in_forward) = deepcopy(own)

        def stage_b_custody(self, *, audit_receipt, optimizer_resolution_receipt):
            if self.stage != "B_private_only" or not self.warm_bank_receipt:
                raise ValueError("Actual warm-bank factors must exist before Stage B custody")
            names = tuple(f"core.{site}.{factor}" for site in self.intermediate_sites for factor in ("R", "S"))
            hook = ExternalStateHook("native_config_and_member_selection", audit_receipt,
                                     self._primitive_state, self._restore_primitive_state)
            return Custody(names, self.composition_receipt + ":" + self.warm_bank_receipt,
                           audit_receipt, optimizer_resolution_receipt,
                           MOMENT_POLICY, ZERO_POLICY, (hook,))

    return AllLayerPolyFormer()
