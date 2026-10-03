"""Native mono normalization and exact feature-gradient pullback preparation."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any


@dataclass(frozen=True)
class CacheIdentity:
    dataset_release_sha256: str
    features_sha256: str
    native_edge_records_sha256: str
    edge_attributes_sha256: str
    removed_units_sha256: str
    fit_role_labels_sha256: str
    view: str
    K: int
    base: str
    normalization_source_sha256: str
    runtime_precision_receipt: str

    def __post_init__(self):
        if (self.view not in ("native", "same_removal", "other_removal") or self.base != "mono" or
                type(self.K) is not int or self.K < 0):
            raise ValueError("Declared native/probe mono view and order required")
        if any(not value for key, value in asdict(self).items() if isinstance(value, str)):
            raise ValueError("Complete feature/graph/mask/source/runtime cache identity required")

    @property
    def fingerprint(self):
        return sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class MonoTokenBank:
    tokens: tuple[Any, ...]
    operator: Any
    identity: CacheIdentity


@dataclass(frozen=True)
class CompleteViewBank:
    banks: tuple[tuple[str, MonoTokenBank], ...]
    complete_nodes: int
    original_features: int
    custody_receipt: str

    def __post_init__(self):
        views = ("native", "same_removal", "other_removal")
        records = dict(self.banks)
        if (set(records) != set(views) or len(self.banks) != 3 or
                type(self.complete_nodes) is not int or type(self.original_features) is not int or
                self.complete_nodes <= 0 or self.original_features <= 0 or not self.custody_receipt):
            raise ValueError("Exactly three fixed complete-node views and cache custody required")
        native = records["native"].identity
        fields = ("dataset_release_sha256", "features_sha256", "native_edge_records_sha256",
                  "edge_attributes_sha256", "fit_role_labels_sha256", "K", "base",
                  "normalization_source_sha256", "runtime_precision_receipt")
        for view, bank in self.banks:
            if bank.identity.view != view or any(getattr(bank.identity, f) != getattr(native, f) for f in fields):
                raise ValueError("View cache changed original features/graph/role/source contract")
            if len(bank.tokens) != native.K + 1 or any(tuple(t.shape) != (self.complete_nodes, self.original_features) for t in bank.tokens):
                raise ValueError("Cache lacks complete per-order original-feature rows")
            if bank.operator.shape != (self.complete_nodes, self.complete_nodes):
                raise ValueError("View operator lacks complete graph nodes")

    @property
    def fingerprint(self):
        value = (self.complete_nodes, self.original_features, self.custody_receipt,
                 tuple((v, b.identity.fingerprint) for v, b in self.banks))
        return sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()

    def forward_view(self, model, view):
        """Readonly source callback for core.guards.complete_views; no retries.

        Cached Tensor values remain mutable Python objects: this descriptor is
        not a hardware readonly guarantee. Custody must qualify immutable input
        artifacts and the exact nonmutating source callback, separately from
        model-state snapshot purity. No cached X/token detach is introduced here.
        """
        import torch
        if view not in dict(self.banks) or model.spec.K != dict(self.banks)[view].identity.K or model.spec.num_features != self.original_features:
            raise ValueError("Model/view differs from complete token bank")
        return model(torch.stack(dict(self.banks)[view].tokens, dim=1))


def build_uncached_mono_tokens(x, edge_index, edge_attr, identity: CacheIdentity):
    """Exact inspected native arithmetic, bypassing author dataset/base/K pickle.

    Retains normalized SciPy->float32 operator construction and uncoalesced COO
    native edge orientation. Native code forces operator float32; reject other
    X dtypes here rather than silently changing normalization/precision. If X
    requires_grad, token construction remains differentiable with respect to X.
    Edge-weight/normalization derivatives through SciPy are not supported.
    """
    import torch
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.utils import to_scipy_sparse_matrix
    from .native_preprocess import sparse_mx_to_torch_sparse_tensor
    if x.dtype != torch.float32 or x.ndim != 2 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Native inspected mono path requires float32 X and COO edge records")
    if edge_attr is not None and edge_attr.requires_grad:
        raise ValueError("SciPy native normalization cannot qualify edge-weight derivatives")
    normalized_edges, norm = gcn_norm(edge_index, edge_attr, num_nodes=x.size(0), dtype=x.dtype)
    matrix = to_scipy_sparse_matrix(normalized_edges, norm, x.size(0))
    operator = sparse_mx_to_torch_sparse_tensor(matrix).to(x.device)
    tokens = [x]
    current = x
    for _ in range(identity.K):
        current = torch.spmm(operator, current)
        tokens.append(current)
    return MonoTokenBank(tuple(tokens), operator, identity)


def pullback_mono_gradients(token_gradients, operator):
    """Exact fixed-operator adjoint with all member/target axes retained.

    g_X=sum_k (P^k)^T g_Tk; reverse recurrence avoids materializing dense P^k.
    Every gradient has [..., complete_nodes, original_features]. Higher-order
    autograd remains enabled; no detach/no_grad/coordinate-space normalization.
    This is the adjoint only, not a FoRDE source objective implementation.
    """
    import torch
    if not token_gradients or any(g.shape != token_gradients[0].shape for g in token_gradients):
        raise ValueError("Parallel complete-feature token gradients required")
    shape = token_gradients[0].shape
    if len(shape) < 2 or operator.shape != (shape[-2], shape[-2]):
        raise ValueError("Operator and complete original-feature node axes differ")
    leading = shape[:-2]
    gradients = [g.reshape(-1, shape[-2], shape[-1]) for g in token_gradients]
    transpose = operator.transpose(0, 1)
    results = []
    for row in range(gradients[0].shape[0]):
        result = gradients[-1][row]
        for k in range(len(gradients) - 2, -1, -1):
            result = gradients[k][row] + torch.spmm(transpose, result)
        results.append(result)
    return torch.stack(results).reshape(*leading, shape[-2], shape[-1])
