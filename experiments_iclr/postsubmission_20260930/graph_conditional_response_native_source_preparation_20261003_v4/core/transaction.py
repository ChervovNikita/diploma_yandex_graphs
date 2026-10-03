"""Prepared backtracking interpretation, gated on an explicit rule resolution.

This is scaled AdamW displacement with full proposed moment commitment. It is
not gradient scaling and is not an unchanged ordinary AdamW update at eta<1.
Only Torch's exact AdamW class is supported; numerical qualification is pending.
"""
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Mapping
import math
from time import perf_counter
from typing import Any, Callable
from .groups import ResolutionRequired
from .guards import CostLedger, GuardDecision

SCALES = (1.0, 0.5, 0.25, 0.125, 0.0)
MOMENT_POLICY = "scaled_displacement_full_adamw_moments_once"
ZERO_POLICY = "restore_only_no_guard"


def _torch():
    import torch
    return torch


@dataclass(frozen=True)
class ExternalStateHook:
    name: str
    source_receipt: str
    capture: Callable[[], Any]
    restore: Callable[[Any], None]


@dataclass(frozen=True)
class Custody:
    private_parameter_names: tuple[str, ...]
    private_site_source_receipt: str
    mutable_state_audit_receipt: str
    optimizer_resolution_receipt: str
    moment_policy: str
    zero_policy: str
    external_hooks: tuple[ExternalStateHook, ...] = ()

    def __post_init__(self):
        if (not self.private_parameter_names or
                len(set(self.private_parameter_names)) != len(self.private_parameter_names)):
            raise ValueError("Explicit unique existing private-factor names required")
        if not all((self.private_site_source_receipt, self.mutable_state_audit_receipt,
                    self.optimizer_resolution_receipt)):
            raise ResolutionRequired("Native sites, mutable state, optimizer semantics need receipts")
        if self.moment_policy != MOMENT_POLICY or self.zero_policy != ZERO_POLICY:
            raise ResolutionRequired("This prepared interpretation requires explicit adoption")
        if len({h.name for h in self.external_hooks}) != len(self.external_hooks) or any(
                not h.name or not h.source_receipt for h in self.external_hooks):
            raise ValueError("Every external mutable-state hook needs unique name/source receipt")


def _equal(a: Any, b: Any) -> bool:
    """Exact bit comparison for ordinary strided state; no allclose tolerances."""
    torch = _torch()
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return False
        if (a.dtype, a.device, a.shape, a.layout, a.requires_grad) != (b.dtype, b.device, b.shape, b.layout, b.requires_grad):
            return False
        if a.layout != torch.strided:
            raise ResolutionRequired("Sparse/non-strided state requires a qualified comparator")
        # Flatten first: scalar optimizer step tensors cannot dtype-view directly.
        av = a.detach().contiguous().reshape(-1).view(torch.uint8)
        bv = b.detach().contiguous().reshape(-1).view(torch.uint8)
        return bool(torch.equal(av, bv))
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        return (isinstance(a, Mapping) and isinstance(b, Mapping) and
                type(a) is type(b) and tuple(a.keys()) == tuple(b.keys()) and
                getattr(a, "default_factory", None) == getattr(b, "default_factory", None) and
                all(_equal(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and bool(a == b)


def _parameter_memo(model: Any) -> dict[int, Any]:
    # Full optimizer Python state is copied while preserving actual parameter
    # identities in state keys/param_groups. Deep-copying those would break binding.
    return {id(p): p for p in model.parameters()}


@dataclass
class StateSnapshot:
    model_state: Any
    parameters: dict[str, Any]
    buffers: dict[str, Any]
    gradients: dict[str, Any]
    requires_grad: dict[str, bool]
    training: dict[str, bool]
    optimizer_image: Any
    cpu_rng: Any
    cuda_rng: Any
    external_state: tuple[tuple[str, Any], ...]

    @classmethod
    def capture(cls, model: Any, optimizer: Any, custody: Custody):
        torch = _torch()
        params = dict(model.named_parameters())
        return cls(
            deepcopy(model.state_dict()),
            {n: p.detach().clone() for n, p in params.items()},
            {n: b.detach().clone() for n, b in model.named_buffers()},
            {n: None if p.grad is None else p.grad.detach().clone() for n, p in params.items()},
            {n: p.requires_grad for n, p in params.items()},
            {n: module.training for n, module in model.named_modules()},
            deepcopy(optimizer.__dict__, _parameter_memo(model)),
            torch.get_rng_state().clone(),
            tuple(s.clone() for s in torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else None,
            tuple((h.name, deepcopy(h.capture())) for h in custody.external_hooks),
        )

    def restore_optimizer(self, model: Any, optimizer: Any):
        image = deepcopy(self.optimizer_image, _parameter_memo(model))
        optimizer.__dict__.clear()
        optimizer.__dict__.update(image)

    def restore(self, model: Any, optimizer: Any, custody: Custody):
        torch = _torch()
        with torch.no_grad():
            model.load_state_dict(deepcopy(self.model_state), strict=True)
            params, buffers = dict(model.named_parameters()), dict(model.named_buffers())
            if set(params) != set(self.parameters) or set(buffers) != set(self.buffers):
                raise RuntimeError("Model parameter/buffer registration changed")
            for n, value in self.parameters.items():
                params[n].copy_(value)
                params[n].requires_grad_(self.requires_grad[n])
                params[n].grad = None if self.gradients[n] is None else self.gradients[n].clone()
            for n, value in self.buffers.items():
                buffers[n].copy_(value)
            modules = dict(model.named_modules())
            if set(modules) != set(self.training):
                raise RuntimeError("Model module registration changed")
            for n, flag in self.training.items():
                modules[n].training = flag
            self.restore_optimizer(model, optimizer)
            for hook in custody.external_hooks:
                hook.restore(deepcopy(dict(self.external_state)[hook.name]))
            torch.set_rng_state(self.cpu_rng)
            if self.cuda_rng is not None:
                torch.cuda.set_rng_state_all(list(self.cuda_rng))
            elif torch.cuda.is_initialized():
                raise ResolutionRequired("Guard/step initialized CUDA outside captured state")

    def exact_equal(self, other: "StateSnapshot") -> bool:
        return all(_equal(getattr(self, field), getattr(other, field))
                   for field in self.__dataclass_fields__)


def validate_private_partition(model: Any, optimizer: Any, custody: Custody):
    torch = _torch()
    from torch.optim import optimizer as optimizer_module
    if type(optimizer) is not torch.optim.AdamW:
        raise ResolutionRequired("Only source-qualified exact Torch AdamW is prepared")
    if "step" in optimizer.__dict__:
        raise ResolutionRequired("Instance-patched optimizer step requires extended custody")
    for name in ("_global_optimizer_pre_hooks", "_global_optimizer_post_hooks",
                 "_global_optimizer_step_pre_hooks", "_global_optimizer_step_post_hooks"):
        if getattr(optimizer_module, name, {}):
            raise ResolutionRequired("Global optimizer hooks require extended custody")
    params = dict(model.named_parameters())
    private = set(custody.private_parameter_names)
    if not private <= params.keys():
        raise ValueError("Unknown private factor parameter")
    aliases = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p), []).append(name)
    if any(len(aliases[id(params[n])]) != 1 for n in private):
        raise ResolutionRequired("Aliased private factors need a separately qualified transaction")
    if {n for n, p in params.items() if p.requires_grad} != private:
        raise ValueError("Only intended existing private intermediate factors may require gradients")
    optimizer_params = [p for group in optimizer.param_groups for p in group["params"]]
    if len(set(map(id, optimizer_params))) != len(optimizer_params) or set(map(id, optimizer_params)) != {
            id(params[n]) for n in private}:
        raise ValueError("AdamW must own exactly the intended private factors, once each")
    if any(module.training for module in model.modules()):
        raise ValueError("Stage B must use eval mode for all objectives/guards/statistics")
    for n, p in params.items():
        if p.layout != torch.strided:
            raise ResolutionRequired("Non-strided parameter custody unqualified")
        if n not in private and p.grad is not None:
            raise ValueError("Clear inherited gradients from frozen common/stem/classifier parameters")
    if any(p.grad is None for p in optimizer_params):
        raise ValueError("Every intended factor needs a prepared gradient or explicit missing-gradient rule")
    # Step hooks may mutate state outside the optimizer; no silent custody claim.
    if any(value for key, value in optimizer.__dict__.items()
           if "hook" in key and isinstance(value, Mapping)):
        raise ResolutionRequired("Optimizer hooks require explicit extended custody")


@dataclass(frozen=True)
class TransactionResult:
    scale: float
    accepted_nonzero: bool
    trials: tuple[tuple[float, GuardDecision], ...]
    optimizer_semantics: str


def _assert_proposal_custody(base: StateSnapshot, proposal: StateSnapshot, custody: Custody):
    private = set(custody.private_parameter_names)
    for n, value in base.parameters.items():
        if n not in private and not _equal(value, proposal.parameters[n]):
            raise RuntimeError("Ordinary proposal changed a frozen parameter")
    for field in ("buffers", "gradients", "requires_grad", "training", "cpu_rng", "cuda_rng", "external_state"):
        if not _equal(getattr(base, field), getattr(proposal, field)):
            raise RuntimeError(f"Ordinary proposal unexpectedly changed {field}")
    # state_dict extra state and non-parameter buffers must also remain identical.
    for name, value in base.model_state.items():
        if name not in private and not _equal(value, proposal.model_state[name]):
            raise RuntimeError(f"Ordinary proposal changed frozen state_dict entry {name}")


def guarded_adamw_displacement(model: Any, optimizer: Any, custody: Custody,
                               guard: Callable[[Any, CostLedger], GuardDecision],
                               ledger: CostLedger):
    """One AdamW proposal; scale complete displacement; commit full moments once.

    theta_eta = theta_base + eta*(theta_AdamW - theta_base), including weight decay.
    eta=1 copies the exact proposal; eta=0 or a bitwise-zero displacement restores
    all original state without an extra guard. A nonzero displacement acceptance
    commits the full proposed optimizer image. Zero/rounding handling is part of
    the explicitly required zero-policy resolution, not an inferred native rule.
    A zero/rejected/exception proposal restores model state, nonpersistent buffers,
    grads, requires_grad, modes, optimizer's complete Python state, RNG and audited
    external hooks. Work ledger is never rolled back. This interpretation cannot
    be used until the moment and zero policies have been explicitly resolved.

    Objectives/backward precede this call and must separately be charged. Guards
    run no_grad and must be state-pure. They use complete_views to charge actual
    forwards; a callback cannot hide extra work. This function does not assert
    that the restored base is feasible if the caller supplied an infeasible base.
    """
    transaction_start = perf_counter()
    torch = _torch()
    validate_private_partition(model, optimizer, custody)
    base = StateSnapshot.capture(model, optimizer, custody)
    trials, committed = [], False
    ledger.optimizer_step_attempts += 1
    start = perf_counter()
    try:
        optimizer.step()
        ledger.record_work("adamw_proposal", elapsed_seconds=perf_counter() - start,
                           details={"status": "returned", "scales": SCALES})
        proposal = StateSnapshot.capture(model, optimizer, custody)
        _assert_proposal_custody(base, proposal, custody)
        if all(_equal(base.parameters[n], proposal.parameters[n])
               for n in custody.private_parameter_names):
            base.restore(model, optimizer, custody)
            ledger.discarded_proposals += 1
            return TransactionResult(0.0, False, (), MOMENT_POLICY)
        for scale in SCALES[:-1]:
            base.restore(model, optimizer, custody)
            with torch.no_grad():
                params = dict(model.named_parameters())
                for name in custody.private_parameter_names:
                    if scale == 1.0:
                        params[name].copy_(proposal.parameters[name])
                    else:
                        params[name].copy_(base.parameters[name] + scale *
                                          (proposal.parameters[name] - base.parameters[name]))
            trial_state = StateSnapshot.capture(model, optimizer, custody)
            if all(_equal(base.parameters[n], trial_state.parameters[n])
                   for n in custody.private_parameter_names):
                base.restore(model, optimizer, custody)
                ledger.discarded_proposals += 1
                return TransactionResult(0.0, False, tuple(trials), MOMENT_POLICY)
            with torch.no_grad():
                decision = guard(model, ledger)
            if not isinstance(decision, GuardDecision):
                raise TypeError("Guard must return a validated GuardDecision")
            if not trial_state.exact_equal(StateSnapshot.capture(model, optimizer, custody)):
                raise RuntimeError("Guard mutated model/optimizer/RNG/audited external state")
            trials.append((scale, decision))
            if decision.accepted:
                proposal.restore_optimizer(model, optimizer)
                ledger.committed_proposals += 1
                committed = True
                return TransactionResult(scale, True, tuple(trials), MOMENT_POLICY)
        base.restore(model, optimizer, custody)
        ledger.discarded_proposals += 1
        return TransactionResult(0.0, False, tuple(trials), MOMENT_POLICY)
    except BaseException:
        base.restore(model, optimizer, custody)
        ledger.discarded_proposals += 1
        ledger.record_work("transaction_exception", elapsed_seconds=perf_counter() - start,
                           details={"status": "rolled_back", "trial_count": len(trials)})
        raise
    finally:
        ledger.record_work("transaction_total", elapsed_seconds=perf_counter() - transaction_start,
                           details={"committed_nonzero": committed, "trial_count": len(trials),
                                    "includes_nested_forward_and_proposal_spans": True,
                                    "synchronized_gpu_timing": False})
        if not committed:
            # Rejected transactions should already be restored. A second exact
            # assertion detects incomplete custody without claiming qualification.
            if not base.exact_equal(StateSnapshot.capture(model, optimizer, custody)):
                raise RuntimeError("Post-transaction rollback state differs from base")
