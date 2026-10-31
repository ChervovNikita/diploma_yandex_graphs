"""Retain all source operations, replacing only normalized neighbor messages.

This module imports Torch and pinned model blocks. Import it only after the
source, route and explicit engineering/scientific admission guards.
"""
import torch
from torch.nn import functional as F

from .counts import ARMS, DIM, SEEDS, RECIPE, TASKS, MAX_SINGLE_WIDTH, matching_width, ordinary_count, ensemble_count
from propagation_cost_engineering_v1.complete_models import CompleteEnsemble, CompleteBASE
from propagation_cost_impl_v1.models import SourcePlainSAGE
from propagation_cost_impl_v1._pinned.frozen_centered_paths import PlainSAGE
from propagation_cost_impl_v1._pinned.frozen_models import Model, TABMModel


def shared_aggregation(family, normalized, *, detached=False):
    """P mean(LN(h_m)), broadcast; whole neighbor-state gradient may stop.

    Differentiable local adjoint to every member j is P.T sum_m(U_m)/M.
    Detachment removes this entire path, including j's own contribution;
    learned neighbor/FFN maps and the separate root-state path still train.
    """
    mean = normalized.mean(dim=1)
    if detached:
        mean = mean.detach()
    family.ledger.record_dense("detached_shared_normalized_mean" if detached else "differentiable_shared_normalized_mean", normalized)
    common = family.full(mean)
    return common[:, None, :].expand_as(normalized)


class SharedMeanEnsemble(CompleteEnsemble):
    def __init__(self, features, classes, *, detached=False, checkpoint_blocks=False):
        super().__init__(features, classes, dim=DIM, arm="candidate", checkpoint_blocks=checkpoint_blocks)
        self.detached_message = bool(detached)
        self.arm = "detached_shared_mean" if detached else "shared_mean"

    def _propagate(self, family, normalized, color):
        return shared_aggregation(family, normalized, detached=self.detached_message)

    def forward(self, *args, execution="packed", return_trace=False, **kwargs):
        if execution != "packed" or return_trace:
            raise ValueError("Coupled shared-message trajectories require packed states; trace is separate")
        return super().forward(*args, execution="packed", return_trace=False, **kwargs)

    def forward_member(self, *args, **kwargs):
        raise ValueError("A member-at-a-time trajectory changes the recurrent shared mean")


class ParameterMatchedBASE(PlainSAGE):
    """New bounded constructor; original SourcePlainSAGE D<=64 guard remains.

    Reuse the pinned PlainSAGE blocks, source transfer and complete BASE
    forward. No width guard in any existing class is changed or bypassed.
    """
    copy_from_source_model = SourcePlainSAGE.copy_from_source_model
    forward = CompleteBASE.forward

    def __init__(self, features, classes, *, width, checkpoint_blocks=False):
        if type(width) is not int or not 1 <= width <= MAX_SINGLE_WIDTH:
            raise ValueError("Explicit new single width bound is 1..256")
        if width != matching_width(features, classes)["width"]:
            raise ValueError("Matched single width must use the deterministic source formula")
        super().__init__(features, classes, dim=width, layers=2, dropout=0.2)
        self.input_dim, self.output_dim, self.dim = features, classes, width
        self.layers, self.p, self.members = 2, 0.2, 1
        self.checkpoint_blocks, self.arm = bool(checkpoint_blocks), "single_matched"


def construct(task, arm, *, seed=17, checkpoint_blocks=False):
    """Fresh paired source parameters; no old seed/fixture guard is weakened.

    task is the complete task name or a dimensions dict for synthetic tests.
    CPU construction is scoped. Preserve initialized CUDA RNG as well when
    manual_seed seeds it; actual training dropout uses private keyed generators.
    """
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("New packet arm and predeclared fresh seed required")
    spec = TASKS[task] if isinstance(task, str) else task
    features, classes = spec["features"], spec["classes"]
    ordinary_count(features, classes, DIM)
    width = matching_width(features, classes)["width"] if arm == "single_matched" else DIM
    kwargs = dict(model_name="SAGE", num_layers=2, input_dim=features, hidden_dim=width,
                  output_dim=classes, hidden_dim_multiplier=1, num_heads=1,
                  normalization="LayerNorm", dropout=0.2)
    devices = [0] if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if arm.startswith("single_"):
            source = Model(**kwargs)
            model = (ParameterMatchedBASE(features, classes, width=width, checkpoint_blocks=checkpoint_blocks)
                     if arm == "single_matched" else CompleteBASE(features, classes, checkpoint_blocks=checkpoint_blocks))
            model.copy_from_source_model(source)
            model.arm, model.members = arm, 1
            expected = ordinary_count(features, classes, width)
        else:
            source = TABMModel(**kwargs, tabm_inits=4, device="cpu")
            model = (CompleteEnsemble(features, classes, arm="exact_member", checkpoint_blocks=checkpoint_blocks)
                     if arm == "exact_member" else SharedMeanEnsemble(features, classes,
                         detached=arm == "detached_shared_mean", checkpoint_blocks=checkpoint_blocks))
            model.copy_from_source_tabm(source)
            expected = ensemble_count(features, classes)
    if any(p.dtype != torch.float32 for p in model.parameters()) or sum(p.numel() for p in model.parameters()) != expected:
        raise ValueError("Runtime source parameter count or precision differs from exact formula")
    model.parameter_plan = dict(width=width, parameters=expected,
                                matched=matching_width(features, classes) if arm == "single_matched" else None)
    return model


def forward_logits(model, family, features, *, execution="packed", dropout_key=None):
    if execution not in ("packed", "serial"):
        raise ValueError("Unknown execution")
    if model.arm.startswith("single_"):
        if execution != "packed":
            raise ValueError("Single models have one declared complete trajectory")
        return model(family, features, colors=(0, 2), dropout_key=dropout_key)
    return model(family, features, colors=(0, 2), dropout_key=dropout_key, execution=execution)


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"],
                            betas=tuple(RECIPE["betas"]), eps=RECIPE["eps"])


def train_step(model, family, features, train_ids, train_targets, optimizer, *, execution="packed", dropout_key=None, return_loss=False):
    """Full official own-member CE; one AdamW update even for serial exact."""
    if train_targets.ndim != 1 or train_targets.numel() != train_ids.numel():
        raise ValueError("Aligned training targets only")
    if not model.training:
        raise ValueError("Actual training mode required; inference is not train timing")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    if model.arm == "exact_member" and execution == "serial":
        for member in range(4):
            logits = model.forward_member(family, features, member, colors=(0, 2), dropout_key=dropout_key)
            loss = F.cross_entropy(logits[train_ids], train_targets) / 4
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite training objective")
            loss.backward()
            if return_loss:
                losses.append(loss.detach())
            del logits, loss
    else:
        logits = forward_logits(model, family, features, execution=execution, dropout_key=dropout_key)
        loss = torch.stack([F.cross_entropy(row[train_ids], train_targets) for row in logits]).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite training objective")
        loss.backward()
        if return_loss:
            losses.append(loss.detach())
        del logits, loss
    optimizer.step()
    return torch.stack(losses).sum() if return_loss else None


@torch.no_grad()
def pooled_inference(model, family, features, *, execution="packed"):
    if model.training:
        raise ValueError("Dropout-disabled evaluation required")
    if model.arm == "exact_member" and execution == "serial":
        pooled = None
        for member in range(4):
            logits = model.forward_member(family, features, member, colors=(0, 2))
            if pooled is None:
                pooled = logits
            else:
                pooled.add_(logits)
                del logits
        return pooled / 4
    return forward_logits(model, family, features, execution=execution).mean(dim=0)


@torch.no_grad()
def split_logits(model, family, features, indices, *, execution="packed"):
    """One full-node forward; CPU official slices, no label argument."""
    if model.training:
        raise ValueError("Dropout-disabled evaluation required")
    if model.arm == "exact_member" and execution == "serial":
        slices, pooled = [], None
        for member in range(4):
            logits = model.forward_member(family, features, member, colors=(0, 2))
            slices.append(logits[indices].detach().cpu().clone())
            if pooled is None:
                pooled = logits
            else:
                pooled.add_(logits)
                del logits
        return torch.stack(slices), (pooled / 4)[indices].detach().cpu().clone()
    logits = forward_logits(model, family, features, execution=execution)
    return logits[:, indices].detach().cpu().clone(), logits.mean(0)[indices].detach().cpu().clone()
