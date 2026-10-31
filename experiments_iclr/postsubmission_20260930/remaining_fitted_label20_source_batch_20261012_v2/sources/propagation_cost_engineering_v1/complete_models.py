"""Explicit complete-data interfaces; original fixture guards stay unchanged.

Reuse sealed parameter/block structure, centered aggregation, packed/serial
ensemble forwards, dropout, checkpoint handling and CSR adjoint autograd.
Only constructor/input interfaces are independently authorized for complete N.
"""

from __future__ import annotations

import torch
from types import SimpleNamespace
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .contract import verify_impl

verify_impl()
from propagation_cost_impl_v1.csr_operator import FixedCSR, CSRFamily, OperationLedger
from propagation_cost_impl_v1.models import PropagationSAGE, SourcePlainSAGE, _paired_block
from propagation_cost_impl_v1.dropout import dropout_member


class CompleteCSR(FixedCSR):
    """New complete-data constructor; inherits unchanged sparse call/backward."""
    def __init__(self, name, forward_csr, adjoint_csr, ledger, nodes):
        if forward_csr.layout != torch.sparse_csr or adjoint_csr.layout != torch.sparse_csr:
            raise ValueError("Fixed forward and prebuilt adjoint CSR required")
        if forward_csr.shape != (nodes, nodes) or adjoint_csr.shape != (nodes, nodes):
            raise ValueError("Complete operator shape mismatch")
        if forward_csr.dtype != torch.float32 or adjoint_csr.dtype != torch.float32:
            raise ValueError("Frozen float32 graph values required")
        if forward_csr.device != adjoint_csr.device:
            raise ValueError("Operator device mismatch")
        if forward_csr.requires_grad or adjoint_csr.requires_grad:
            raise ValueError("Fixed graph only")
        self.name, self.forward_csr, self.adjoint_csr = name, forward_csr, adjoint_csr
        self.nnz, self.nodes, self.ledger = int(forward_csr._nnz()), int(nodes), ledger


class NoAdjointCSR(CompleteCSR):
    """Detached full mean has no input gradient; do not allocate an unused AT."""
    def __init__(self, name, forward_csr, ledger, nodes):
        if forward_csr.layout != torch.sparse_csr or forward_csr.shape != (nodes, nodes):
            raise ValueError("Fixed complete forward CSR required")
        if forward_csr.dtype != torch.float32 or forward_csr.requires_grad:
            raise ValueError("Fixed float32 CSR required")
        self.name, self.forward_csr = name, forward_csr
        self.adjoint_csr = None
        self.nnz, self.nodes, self.ledger = int(forward_csr._nnz()), int(nodes), ledger

    def __call__(self, features):
        if features.requires_grad or features.shape[0] != self.nodes or features.ndim != 2:
            raise ValueError("NoAdjointCSR accepts only explicitly detached [N,D] states")
        if features.dtype != self.forward_csr.dtype or features.device != self.forward_csr.device:
            raise ValueError("Detached full-mean precision/device mismatch")
        self.ledger.record(self, "forward", features)
        return torch.sparse.mm(self.forward_csr, features)

    def payload_bytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.forward_csr.crow_indices(), self.forward_csr.col_indices(), self.forward_csr.values()))


class CompleteFamily(CSRFamily):
    """Keep only an arm's required GPU operators; no irrelevant baseline tax."""
    def __init__(self, operators, metadata, ledger, device):
        self.operators, self.metadata, self.ledger = dict(operators), dict(metadata), ledger
        self.fingerprint = metadata["P_edge_only_fingerprint"]
        self._shape_only = SimpleNamespace(nodes=metadata["nodes"], nnz=metadata["canonical_edges"],
                                          forward_csr=SimpleNamespace(device=torch.device(device), dtype=torch.float32))
        if any(op.nodes != metadata["nodes"] or op.ledger is not ledger for op in operators.values()):
            raise ValueError("Complete CSR family node/ledger mismatch")

    @property
    def full(self):
        # Naive controls do not propagate P. This descriptor is only for the
        # inherited input check; it stores no unnecessary full graph on GPU.
        return self.operators.get("full", self._shape_only)


class CompleteEnsemble(PropagationSAGE):
    """Inherited computation; replace only the fixture input-size interface."""
    def _check(self, family, features, colors):
        if features.ndim != 2 or features.shape != (family.full.nodes, self.input_dim):
            raise ValueError("Complete feature/node order mismatch")
        if features.dtype != torch.float32 or features.device != family.full.forward_csr.device:
            raise ValueError("Complete graph/state precision or device mismatch")
        if len(colors) != self.layers or any(c not in range(4) for c in colors):
            raise ValueError("Immutable source layer colors required")


def detached_aggregation(family, normalized, color):
    mean = normalized.mean(dim=1).detach()
    deviation = normalized - mean[:, None, :]
    family.ledger.record_dense("detached_post_norm_mean_in_both_branches", normalized)
    common = family.full(mean)
    residual = family.color(color)(deviation.flatten(1)).reshape_as(normalized) * 4.0
    return common[:, None, :] + residual


class DetachedMeanEnsemble(CompleteEnsemble):
    """Prospective ablation: identical local forward, different local adjoint."""
    def __init__(self, *args, **kwargs):
        kwargs["arm"] = "candidate"
        super().__init__(*args, **kwargs)
        self.arm = "detached_current_member_mean"

    def _propagate(self, family, normalized, color):
        return detached_aggregation(family, normalized, color)

    def forward(self, *args, execution="packed", return_trace=False, **kwargs):
        if execution != "packed" or return_trace:
            raise ValueError("Detached coupled arm requires packed states; no diagnostic trace")
        return super().forward(*args, execution="packed", return_trace=False, **kwargs)

    def forward_member(self, *_args, **_kwargs):
        raise ValueError("A serial detached-mean trajectory changes its coupled forward")


class CompleteBASE(SourcePlainSAGE):
    """Source-equivalent single forward with an explicit complete-data check."""
    arm = "single_base"
    def forward(self, family, features, *, colors=(0, 2), dropout_key=None):
        if features.ndim != 2 or features.shape != (family.full.nodes, self.input_dim):
            raise ValueError("Complete BASE feature shape mismatch")
        if features.dtype != torch.float32 or features.device != family.full.forward_csr.device:
            raise ValueError("Complete BASE precision/device mismatch")
        hidden = F.gelu(dropout_member(self.input(features), self.p, self.training,
                                       dropout_key, 0, -1, "input"))
        for layer, (norm, block) in enumerate(zip(self.norm, self.blocks)):
            def step(state, layer=layer, norm=norm, block=block):
                normalized = norm(state)
                return state + _paired_block(
                    block, normalized, family.full(normalized), p=self.p,
                    training=self.training, key=dropout_key, layer=layer, member=0)
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                hidden = checkpoint(
                    step, hidden, use_reentrant=False, preserve_rng_state=False,
                    context_fn=lambda: (family.ledger.scope("checkpoint_forward"),
                                        family.ledger.scope("checkpoint_recompute")))
            else:
                hidden = step(hidden)
        return self.output(self.output_norm(hidden))[None, :, :]


def construct_source_arm(task, arm, *, optimizer_seed=0, checkpoint_blocks=False, naive_path="lazy_two_colors"):
    """Fresh source seed0 tensors, discarded after bounded engineering steps."""
    from propagation_cost_impl_v1._pinned.frozen_models import Model, TABMModel
    kwargs = dict(model_name="SAGE", num_layers=2, input_dim=task["features"],
                  hidden_dim=64, output_dim=task["classes"], hidden_dim_multiplier=1,
                  num_heads=1, normalization="LayerNorm", dropout=0.2)
    if optimizer_seed not in (0, 1, 2):
        raise ValueError("Frozen prospective optimizer seeds only")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(optimizer_seed)
        if arm == "single_base":
            original = Model(**kwargs)
            result = CompleteBASE(task["features"], task["classes"], checkpoint_blocks=checkpoint_blocks)
            result.copy_from_source_model(original)
        else:
            original = TABMModel(**kwargs, tabm_inits=4, device="cpu")
            cls = DetachedMeanEnsemble if arm == "detached_current_member_mean" else CompleteEnsemble
            result = cls(task["features"], task["classes"], arm=arm,
                         naive_path=naive_path, checkpoint_blocks=checkpoint_blocks)
            result.copy_from_source_tabm(original)
    return result


def own_train_loss(logits, train_indices, train_labels):
    if train_labels.ndim != 1 or train_labels.numel() != train_indices.numel():
        raise ValueError("Only aligned official training labels may enter the objective")
    return torch.stack([F.cross_entropy(member[train_indices], train_labels)
                        for member in logits]).mean()


def train_step(model, family, features, train_indices, train_labels, optimizer,
               *, colors, dropout_key, execution, return_loss=False):
    optimizer.zero_grad(set_to_none=True)
    detached_losses = []
    if (not isinstance(model, CompleteBASE) and execution == "serial"
            and model.arm in ("exact_member", "naive_shared_sample")):
        for member in range(4):
            logits = model.forward_member(family, features, member, colors=colors, dropout_key=dropout_key)
            loss = F.cross_entropy(logits[train_indices], train_labels) / 4
            loss.backward()
            if return_loss:
                detached_losses.append(loss.detach())
            del logits, loss
    else:
        kwargs = dict(colors=colors, dropout_key=dropout_key)
        if not isinstance(model, CompleteBASE):
            kwargs["execution"] = execution
        logits = model(family, features, **kwargs)
        loss = own_train_loss(logits, train_indices, train_labels)
        loss.backward()
        if return_loss:
            detached_losses.append(loss.detach())
        del logits, loss
    optimizer.step()
    return torch.stack(detached_losses).sum() if return_loss else None


@torch.no_grad()
def pooled_inference(model, family, features, execution):
    if getattr(model, "native_source", False):
        return model.pooled(family, features)
    if execution == "serial" and model.arm in ("exact_member", "naive_shared_sample"):
        result = None
        for member in range(4):
            logits = model.forward_member(family, features, member, colors=(0, 2))
            if result is None:
                result = logits
            else:
                result.add_(logits)
                del logits
        return result / 4
    kwargs = {} if isinstance(model, CompleteBASE) else {"execution": execution}
    return model(family, features, colors=(0, 2), **kwargs).mean(dim=0)


@torch.no_grad()
def inference_validation_work(model, family, features, validation_indices, execution):
    pooled = pooled_inference(model, family, features, execution)
    # Full official validation-node reductions without any validation targets,
    # synthetic target assignment, classification score, or stored predictions.
    values = pooled[validation_indices]
    label_free_work = values.logsumexp(dim=-1).sum() + values.sum()
    del pooled, values, label_free_work
