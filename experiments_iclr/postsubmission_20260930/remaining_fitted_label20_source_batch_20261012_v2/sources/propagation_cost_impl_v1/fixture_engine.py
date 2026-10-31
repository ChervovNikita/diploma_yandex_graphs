"""Bounded mechanics, not a real-data experiment or a predictive evaluator."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .contract import fixture_shape, refuse_real_pilot
from .models import PropagationSAGE, SourcePlainSAGE


def own_member_ce(logits, targets, train_nodes):
    if logits.ndim != 3 or targets.ndim != 1 or logits.shape[1] != targets.numel():
        raise ValueError("Expected member logits [M,N,C] and targets [N]")
    fixture_shape(logits.shape[1], 0)
    return torch.stack([F.cross_entropy(member[train_nodes], targets[train_nodes])
                        for member in logits], dim=0).mean()


def pooled_logits(logits):
    if logits.ndim != 3:
        raise ValueError("Expected member logits [M,N,C]")
    return logits.mean(dim=0)


def serial_own_ce_step(model, family, features, targets, train_nodes,
                       optimizer, *, colors, dropout_key):
    """Low-memory exact/naive control: M backwards, exactly ONE AdamW step.

    Gradients from CE/M accumulate into all shared tensors. Never update
    shared tensors between members. This equals the own-member mean loss
    gradient in exact arithmetic, subject to floating-point summation order.
    It is not a permitted candidate path because candidate layers are coupled.
    """
    if model.arm not in ("exact_member", "naive_shared_sample"):
        raise ValueError("Serial own-CE control only for uncoupled trajectories")
    fixture_shape(features.shape[0], family.full.nnz, features.shape[1])
    optimizer.zero_grad(set_to_none=True)
    losses = []
    with family.ledger.scope("serial_own_ce_train"):
        for member in range(model.members):
            logits = model.forward_member(family, features, member, colors=colors,
                                          dropout_key=dropout_key)
            loss = F.cross_entropy(logits[train_nodes], targets[train_nodes]) / model.members
            loss.backward()
            losses.append(loss.detach())
            del logits, loss
    optimizer.step()
    return torch.stack(losses).sum()


def make_fixture_sources(input_dim=3, output_dim=2, dim=4, seed=0, dropout=0.0):
    """Construct the frozen source first, then transfer every parameter.

    RNG scoping prevents source/optimized constructor order from being an
    implicit random-stream control. Real input dimensions and data are denied
    by the bounded fixture contract in this packet.
    """
    from ._pinned.frozen_models import Model, TABMModel

    fixture_shape(1, 0, input_dim)
    if not (1 < output_dim <= 16 and dim <= 64):
        raise ValueError("Synthetic fixture dimensions only")
    kwargs = dict(model_name="SAGE", num_layers=2, input_dim=input_dim,
                  hidden_dim=dim, output_dim=output_dim, hidden_dim_multiplier=1,
                  num_heads=1, normalization="LayerNorm", dropout=dropout)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        source_ensemble = TABMModel(**kwargs, tabm_inits=4, device="cpu")
        torch.manual_seed(seed)
        source_single = Model(**kwargs)
        arms = {}
        for arm in ("candidate", "exact_member", "naive_shared_sample"):
            model = PropagationSAGE(input_dim, output_dim, dim, dropout=dropout, arm=arm)
            model.copy_from_source_tabm(source_ensemble)
            arms[arm] = model
        single = SourcePlainSAGE(input_dim, output_dim, dim, dropout=dropout)
        single.copy_from_source_model(source_single)
        arms["single_base"] = single
    return source_ensemble, source_single, arms


def real_pilot(*args, **kwargs):
    return refuse_real_pilot(*args, **kwargs)


if __name__ == "__main__":
    real_pilot()
