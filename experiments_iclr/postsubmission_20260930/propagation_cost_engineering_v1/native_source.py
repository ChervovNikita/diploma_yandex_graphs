"""Original gather/scatter source controls with member-paired dropout modules."""

from __future__ import annotations

from types import SimpleNamespace
import torch
from torch import nn

from propagation_cost_impl_v1._pinned.frozen_models import Model, TABMModel
from propagation_cost_impl_v1.dropout import dropout_member


class _PairedDropout(nn.Module):
    def __init__(self, p, context, layer, site):
        super().__init__()
        self.p, self.context, self.layer, self.site = p, context, layer, site

    def forward(self, features):
        return dropout_member(features, self.p, self.training, self.context["key"],
                              self.context["member"], self.layer, self.site)


class NativeArm(nn.Module):
    """Fresh unchanged source graph path; no extra trainable parameters.

    Replace only parameter-free dropout modules to realize the already
    declared member noise streams. Eval with dropout disabled is the literal
    frozen Model/TABMModel computation. Native source checkpointing is absent.
    """
    native_source = True

    def __init__(self, task, arm, edge_index, optimizer_seed=0):
        super().__init__()
        if arm not in ("exact_member", "single_base"):
            raise ValueError("Native source controls only")
        if optimizer_seed not in (0, 1, 2):
            raise ValueError("Frozen prospective optimizer seeds only")
        self.arm, self.members = arm, 4 if arm == "exact_member" else 1
        self.context = {"key": None, "member": 0}
        self.graph = SimpleNamespace(edge_index=edge_index)
        kwargs = dict(model_name="SAGE", num_layers=2, input_dim=task["features"],
                      hidden_dim=64, output_dim=task["classes"], hidden_dim_multiplier=1,
                      num_heads=1, normalization="LayerNorm", dropout=0.2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(optimizer_seed)
            self.source = TABMModel(**kwargs, tabm_inits=4, device="cpu") if self.members == 4 else Model(**kwargs)
        self.source.dropout = _PairedDropout(0.2, self.context, -1, "input")
        for layer, residual in enumerate(self.source.residual_modules):
            ff = residual.module.feed_forward_module
            ff.dropout_1 = _PairedDropout(0.2, self.context, layer, "ffn_first")
            ff.dropout_2 = _PairedDropout(0.2, self.context, layer, "ffn_second")

    def forward_member(self, family, features, member, *, colors=(0, 2), dropout_key=None):
        self.context.update(key=dropout_key, member=member)
        if self.members == 4:
            return self.source(self.graph, features, member)
        if member != 0:
            raise ValueError("Ordinary BASE has one member")
        return self.source(self.graph, features)

    def forward(self, family, features, *, colors=(0, 2), dropout_key=None, execution="serial"):
        return torch.stack([self.forward_member(family, features, member, colors=colors,
                                                dropout_key=dropout_key)
                            for member in range(self.members)])

    @torch.no_grad()
    def pooled(self, family, features):
        # Streaming raw-logit pooling: never retain an M*N*C source stack.
        result = None
        for member in range(self.members):
            logits = self.forward_member(family, features, member)
            if result is None:
                result = logits
            else:
                result.add_(logits)
                del logits
        return result / self.members


@torch.no_grad()
def transfer_execution_state(training_model, inference_model):
    """Map all trained tensors when score-blind training/inference paths differ."""
    if training_model.arm != inference_model.arm:
        raise ValueError("Execution transfer cannot switch scientific arms")
    source_training = getattr(training_model, "native_source", False)
    source_inference = getattr(inference_model, "native_source", False)
    if not source_training and not source_inference:
        inference_model.load_state_dict(training_model.state_dict())
    elif source_training and source_inference:
        inference_model.source.load_state_dict(training_model.source.state_dict())
    else:
        from propagation_cost_impl_v1.runtime_fixture import parameter_pairs
        source = training_model.source if source_training else inference_model.source
        optimized = inference_model if source_training else training_model
        for _, original, accelerated, transposed in parameter_pairs(source, optimized, training_model.arm == "exact_member"):
            if source_training:
                accelerated.copy_(original.T if transposed else original)
            else:
                original.copy_(accelerated.T if transposed else accelerated)
