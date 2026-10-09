"""Inactive adapters for one exact pinned SeHGNN HGB IMDB model.

No architecture/data/optimizer/driver is constructed at import time. The caller
supplies a natively initialized, finally placed prototype and exact model module.
Only the listed grouped/semantic affine sites receive member-private factors.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence


PINNED_COMMIT = "e92bd37d0b803457339555684f139b4c8f3e160d"
PINNED_MODEL_SHA256 = "0948239c4c4d06dd258cd106fd2fa7b6aaa2a62688fe70dd662413fa1eae532a"
GROUPED_SITES = ("feature_projection.0", "feature_projection.4")
SEMANTIC_SITES = ("semantic_fusion.query", "semantic_fusion.key", "semantic_fusion.value",
                  "fc_after_concat")
SITES = GROUPED_SITES + SEMANTIC_SITES


class AdapterContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdapterConfig:
    enabled: bool = False
    members: int = 4  # M1 is only the prospective native/unit qualification witness.

    def require_enabled(self) -> None:
        if not self.enabled:
            raise AdapterContractError("Inactive SeHGNN adapter; root qualification required")
        if self.members not in (1, 4):
            raise AdapterContractError("This packet supports only qualification M1 or fixed M4")


@dataclass(frozen=True)
class Channel:
    namespace: str
    key: str
    row: int


def channel_order(prototype: Any) -> tuple[Channel, ...]:
    """Do not collapse a feature/label channel that has the same path string."""
    feature, label = tuple(prototype.feat_keys), tuple(prototype.label_feat_keys)
    if feature != tuple(sorted(feature)) or label != tuple(sorted(label)):
        raise AdapterContractError("Native channel lists are not separately sorted")
    if len(set(feature)) != len(feature) or len(set(label)) != len(label):
        raise AdapterContractError("Duplicate key within a native namespace")
    result = tuple(Channel("feature", key, i) for i, key in enumerate(feature))
    result += tuple(Channel("label", key, len(feature)+i) for i, key in enumerate(label))
    if len(result) != prototype.num_channels:
        raise AdapterContractError("Native num_channels does not match both namespaces")
    return result


def _named_parameters(model: Any) -> dict[str, Any]:
    return dict(model.named_parameters(remove_duplicate=False))


def _named_buffers(model: Any) -> dict[str, Any]:
    return dict(model.named_buffers(remove_duplicate=False))


def _unique(values: Sequence[Any]) -> tuple[Any, ...]:
    result, seen = [], set()
    for value in values:
        if id(value) not in seen:
            result.append(value)
            seen.add(id(value))
    return tuple(result)


def _storage(value: Any) -> tuple[str, int]:
    return str(value.device), int(value.untyped_storage().data_ptr())


def _site(model: Any, path: str) -> tuple[Any, str, Any]:
    components = path.split(".")
    owner = model
    for component in components[:-1]:
        owner = owner._modules[component]
    return owner, components[-1], owner._modules[components[-1]]


def _validate_binding(native_module: Any, prototype: Any) -> tuple[Any, dict[str, Any]]:
    path = Path(native_module.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINNED_MODEL_SHA256:
        raise AdapterContractError("Native model bytes differ from exact pinned HGB source")
    if type(prototype) is not native_module.SeHGNN or prototype.dataset != "IMDB":
        raise AdapterContractError("Only this native SeHGNN IMDB class is supported")
    torch = native_module.torch  # Preserve the author's actual provider.
    channels = channel_order(prototype)
    if not channels or type(prototype.semantic_fusion) is not native_module.Transformer:
        raise AdapterContractError("Unexpected native semantic module")
    if (len(prototype.feat_keys), len(prototype.label_feat_keys), len(channels)) != (25, 12, 37):
        raise AdapterContractError("Representative requires all25 feature/all12 native label channels")
    if len(prototype.feature_projection) != 8 or len(prototype.task_mlp) != 5:
        raise AdapterContractError("Require native two-FP/four-task-layer representative")
    if prototype.residual:
        raise AdapterContractError("Representative raw-feature residual flag must remain false")
    for name in GROUPED_SITES:
        native = _site(prototype, name)[2]
        if type(native) is not native_module.LinearPerMetapath:
            raise AdapterContractError("Grouped site is not the pinned LinearPerMetapath")
        if tuple(native.W.shape) != (len(channels), native.cin, native.cout):
            raise AdapterContractError("Native grouped weight orientation differs")
        if tuple(native.bias.shape) != (len(channels), native.cout):
            raise AdapterContractError("Native grouped bias shape differs")
        if (native.cin, native.cout) != (512, 512):
            raise AdapterContractError("Representative grouped widths must remain512")
    for name in SEMANTIC_SITES:
        if type(_site(prototype, name)[2]) is not torch.nn.Linear:
            raise AdapterContractError("Semantic site is not a literal native nn.Linear")
    if prototype.semantic_fusion.n_channels != 512 or prototype.semantic_fusion.num_heads != 1:
        raise AdapterContractError("Representative semantic width/head count changed")
    if (prototype.task_mlp[-1][0].out_features != 5
            or tuple(prototype.task_mlp[-1][1].normalized_shape) != (5,)
            or prototype.task_mlp[-1][1].elementwise_affine):
        raise AdapterContractError("Native five-logit nonaffine task LayerNorm changed")
    return torch, _named_parameters(prototype)


def _adapter_types(native_module: Any) -> tuple[Any, Any]:
    torch, nn = native_module.torch, native_module.nn

    class GroupedFactorAffine(nn.Module):
        """Only the exact W[C,cin,cout] author affine operation is supported."""
        def __init__(self, native: Any) -> None:
            super().__init__()
            self.cin, self.cout = native.cin, native.cout
            self.num_metapaths = native.num_metapaths
            self.W, self.bias = native.W, native.bias
            self.input_factor = nn.Parameter(torch.ones(
                (self.num_metapaths, self.cin), dtype=self.W.dtype, device=self.W.device))
            self.output_factor = nn.Parameter(torch.ones(
                (self.num_metapaths, self.cout), dtype=self.W.dtype, device=self.W.device))
            object.__setattr__(self, "_native_forward", type(native).forward)
            object.__setattr__(self, "_native_reset", type(native).reset_parameters)
            self.train(native.training)

        def forward(self, x: Any) -> Any:
            if x.ndim != 3 or x.shape[1:] != (self.num_metapaths, self.cin):
                raise AdapterContractError("Grouped input no longer has native [B,C,cin] layout")
            # Cast effective factors, preserving native AMP activation/output dtype.
            r = self.input_factor.to(dtype=x.dtype)
            native_value = self._native_forward(self, x*r.unsqueeze(0))
            s = self.output_factor.to(dtype=native_value.dtype)
            bias = self.bias.to(dtype=native_value.dtype).unsqueeze(0)
            # Bias remains outside s. At r=s=1 the correction is identically zero;
            # fast derivatives are still present, without a conditional fast bypass.
            return native_value + (s.unsqueeze(0)-1)*(native_value-bias)

        def reset_parameters(self) -> None:
            # API preservation only. Do not call after bank installation: slow
            # objects are shared, and per-member resets would overwrite them.
            self._native_reset(self)
            nn.init.ones_(self.input_factor)
            nn.init.ones_(self.output_factor)

    class SemanticFactorAffine(nn.Module):
        """Exact native Linear forward with factors over its last input/output axes."""
        def __init__(self, native: Any) -> None:
            super().__init__()
            self.in_features, self.out_features = native.in_features, native.out_features
            self.weight, self.bias = native.weight, native.bias
            self.input_factor = nn.Parameter(torch.ones(
                self.in_features, dtype=self.weight.dtype, device=self.weight.device))
            self.output_factor = nn.Parameter(torch.ones(
                self.out_features, dtype=self.weight.dtype, device=self.weight.device))
            object.__setattr__(self, "_native_forward", type(native).forward)
            object.__setattr__(self, "_native_reset", type(native).reset_parameters)
            self.train(native.training)

        def forward(self, x: Any) -> Any:
            if x.shape[-1] != self.in_features:
                raise AdapterContractError("Semantic input last axis differs from native")
            r = self.input_factor.to(dtype=x.dtype)
            native_value = self._native_forward(self, x*r)
            s = self.output_factor.to(dtype=native_value.dtype)
            bias = 0 if self.bias is None else self.bias.to(dtype=native_value.dtype)
            return native_value + (s-1)*(native_value-bias)

        def reset_parameters(self) -> None:
            self._native_reset(self)
            nn.init.ones_(self.input_factor)
            nn.init.ones_(self.output_factor)

    return GroupedFactorAffine, SemanticFactorAffine


def _bank_type(torch: Any) -> Any:
    class SeHGNNMemberBank(torch.nn.Module):
        def __init__(self, members: Sequence[Any], slow_names: Sequence[str],
                     private_names: Sequence[str], channels: Sequence[Channel]) -> None:
            super().__init__()
            self.members = torch.nn.ModuleList(members)
            self.slow_names = tuple(slow_names)
            self.private_names = tuple(private_names)
            self.channels = tuple(channels)
            self.native_model_sha256 = PINNED_MODEL_SHA256
            self.native_commit = PINNED_COMMIT

        def forward_member(self, index: int, batch: Any, feature_dict: Any,
                           label_dict: Any, mask: Any = None) -> Any:
            return self.members[index](batch, feature_dict, label_dict, mask=mask)

        def slow_parameters(self) -> tuple[Any, ...]:
            named = _named_parameters(self.members[0])
            return _unique([named[name] for name in self.slow_names])

        def private_parameters(self, member: int) -> tuple[Any, ...]:
            named = _named_parameters(self.members[member])
            return tuple(named[name] for name in self.private_names)

        def verify_ownership(self) -> dict[str, Any]:
            maps = [_named_parameters(member) for member in self.members]
            expected = set(self.slow_names) | set(self.private_names)
            if any(set(named) != expected for named in maps):
                raise AdapterContractError("Native/factor parameter coverage changed")
            for name in self.slow_names:
                if any(named[name] is not maps[0][name] for named in maps[1:]):
                    raise AdapterContractError("A native slow parameter is not genuinely shared")
            slow = self.slow_parameters()
            slow_storage = {_storage(p) for p in slow}
            if len(slow_storage) != len(slow):
                raise AdapterContractError("Distinct native slow objects alias one storage")
            private_ids, private_storage = set(), set()
            for named in maps:
                block = [named[name] for name in self.private_names]
                identities, storage = {id(p) for p in block}, {_storage(p) for p in block}
                if len(identities) != len(block) or len(storage) != len(block):
                    raise AdapterContractError("Private factor sites alias each other")
                if identities & private_ids or storage & (private_storage | slow_storage):
                    raise AdapterContractError("Private factor ownership/storage is not disjoint")
                private_ids |= identities
                private_storage |= storage
            buffer_maps = [_named_buffers(member) for member in self.members]
            if any(set(named) != set(buffer_maps[0]) for named in buffer_maps[1:]):
                raise AdapterContractError("Native buffer schema differs across member contexts")
            seen_ids, seen_storage = set(), set()
            for named in buffer_maps:
                buffers = _unique(list(named.values()))
                identities = {id(value) for value in buffers}
                storage = {_storage(value) for value in buffers if value.numel()}
                if identities & seen_ids or storage & seen_storage:
                    raise AdapterContractError("Member native buffers are shared")
                seen_ids |= identities
                seen_storage |= storage
            return {"slow_parameter_objects": len(slow), "private_parameters_per_member": len(self.private_names),
                    "members": len(self.members), "shared_buffers": [],
                    "member_owned_buffer_names": tuple(buffer_maps[0])}

        def verify_shared_state_dict(self, state: Any) -> None:
            # Prevent last-write-wins when distinct keys load into one slow object.
            for name in self.slow_names:
                values = [state[f"members.{m}.{name}"] for m in range(len(self.members))]
                if any(not torch.equal(value, values[0]) for value in values[1:]):
                    raise AdapterContractError("Saved slow aliases have unequal values")

    return SeHGNNMemberBank


def install_member_bank(native_module: Any, prototype: Any,
                        config: AdapterConfig = AdapterConfig()) -> Any:
    """No prototype reset/slow RNG draw; install after placement, before optimizer.

    Parameter memo prevents transient complete slow-weight copies. Modules and
    registered buffers are copied; the original prototype's module tree is intact.
    """
    config.require_enabled()
    torch, original = _validate_binding(native_module, prototype)
    grouped_type, semantic_type = _adapter_types(native_module)
    channels = channel_order(prototype)
    slow_names = tuple(original)
    private_names = tuple(f"{site}.{factor}" for site in SITES
                          for factor in ("input_factor", "output_factor"))
    members = []
    for _ in range(config.members):
        # Use a fresh memo for each member so no module or buffer context is shared.
        memo = {id(parameter): parameter for parameter in _unique(list(original.values()))}
        member = copy.deepcopy(prototype, memo)
        for path in SITES:
            owner, key, native = _site(member, path)
            owner._modules[key] = (grouped_type(native) if path in GROUPED_SITES else semantic_type(native))
        named = _named_parameters(member)
        if set(named) != set(slow_names) | set(private_names):
            raise AdapterContractError("Adapter changed native parameter names or added other sites")
        if any(named[name] is not parameter for name, parameter in original.items()):
            raise AdapterContractError("Adapter did not preserve all native slow objects")
        members.append(member)
    bank = _bank_type(torch)(members, slow_names, private_names, channels)
    bank.verify_ownership()
    return bank
