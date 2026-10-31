"""Unexecuted wrapper over an injected authorized source Model.

No Torch/source-model import or model construction is performed at module scope.
The factory below is future source only and has not been called.
"""
from dataclasses import dataclass

from .domain import ScheduleDomain


AUTHORIZED_SOURCE_SHA256 = "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303"


@dataclass(frozen=True)
class EncodedGraph:
    """One static H, reused by generation and every teacher-forced branch.

    The caller must keep graph, order, encoder/decoder parameters and H fixed
    during a sampling/scoring objective batch. H retains the common encoder's
    autograd graph when gradients are enabled; this object is not a cache law.
    """

    H: object
    domain: ScheduleDomain
    encoder_recipe_id: str


def make_static_source_encoder(*, torch_api, source_core, audit, ledger):
    """Wrap a fresh externally constructed source Model with dropout exactly 0.

    The core's output dimension is the externally chosen node embedding size.
    Source construction/initialization, graph binding and deterministic kernels
    need prospective qualification. No source module is copied or modified.
    """
    required = ("input_linear", "output_linear", "residual_modules",
                "dropout", "act", "output_normalization")
    if any(not hasattr(source_core, name) for name in required):
        raise ValueError("Injected authorized source Model is required")
    if audit.source_sha256 != AUTHORIZED_SOURCE_SHA256:
        raise ValueError("Encoder audit must identify the authorized source snapshot")
    if ledger.audit != audit or source_core.output_linear.out_features != audit.node_embedding_dimension:
        raise ValueError("Encoder audit, ledger and source embedding shape must agree")
    expected_norm = torch_api.nn.LayerNorm if audit.normalization == "LayerNorm" else torch_api.nn.Identity
    norms = [source_core.output_normalization]
    norms.extend(residual.normalization for residual in source_core.residual_modules)
    hidden_dimension = source_core.input_linear.out_features
    for norm in norms:
        if type(norm) is not expected_norm:
            raise ValueError("Every source normalization must be audited node-wise LayerNorm or Identity")
        if audit.normalization == "LayerNorm" and tuple(norm.normalized_shape) != (hidden_dimension,):
            raise ValueError("LayerNorm must normalize the feature axis of each node only")
    for module in source_core.modules():
        if isinstance(module, torch_api.nn.Dropout) and module.p != 0:
            raise ValueError("All source dropout modules must have p=0")
        dropout = getattr(module, "dropout", None)
        if isinstance(dropout, (float, int)) and dropout != 0:
            raise ValueError("All source attention dropout attributes must be zero")

    class StaticSourceEncoder(torch_api.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = source_core
            self.audit = audit
            self.work_ledger = ledger

        def forward(self, graph, features, domain: ScheduleDomain, *, graph_id: str):
            if graph_id != domain.graph_id or features.shape[0] != domain.n_nodes:
                raise ValueError("Source graph/features must cover the same complete conflict instance")
            self.work_ledger.record_encoding(domain.n_nodes)
            H = self.core(graph, features)
            if H.ndim == 1 and self.audit.node_embedding_dimension == 1:
                H = H.unsqueeze(-1)
            if tuple(H.shape) != (domain.n_nodes, self.audit.node_embedding_dimension):
                raise ValueError("Source encoder must emit one H vector per original node")
            if not torch_api.isfinite(H).all().item():
                raise ArithmeticError("Nonfinite static node embeddings")
            return EncodedGraph(H, domain, self.audit.source_recipe_id)

    return StaticSourceEncoder()
