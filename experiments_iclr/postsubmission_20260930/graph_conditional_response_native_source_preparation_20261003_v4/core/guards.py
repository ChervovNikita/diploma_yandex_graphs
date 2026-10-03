"""Fixed TRAIN competence checks and non-rollback cost accounting."""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from time import perf_counter
from typing import Any, Callable, Mapping
from .groups import FixedPlan, LabelRoles
from .response import (VIEWS, MEMBERS, ResponseFailure, ActiveReference,
                       energy_failures, responses, validate_logits)


def _torch():
    import torch
    return torch


@dataclass
class CostLedger:
    """Lives outside model/optimizer snapshots. Failures remain charged.

    A callback represents one complete M4 member-view forward; it must not hide
    repeated forward passes, token rebuilds, retries, or derivative work. Such
    work needs separately recorded events. No cost is inferred from cell count.
    """
    events: list[dict[str, Any]] = field(default_factory=list)
    optimizer_step_attempts: int = 0
    committed_proposals: int = 0
    discarded_proposals: int = 0

    def record_work(self, kind: str, *, elapsed_seconds: float, details: Mapping[str, Any]):
        if elapsed_seconds < 0 or not math.isfinite(elapsed_seconds):
            raise ValueError("Finite nonnegative measured work duration required")
        self.events.append({"kind": kind, "elapsed_seconds": elapsed_seconds,
                            "details": dict(details)})

    def forward(self, forward_fn: Callable[[], Any], *, stage: str, view: str,
                complete_nodes: int):
        if view not in VIEWS or complete_nodes <= 0:
            raise ValueError("Declared complete view size required")
        event = {"kind": "complete_member_view_forward", "stage": stage, "view": view,
                 "complete_nodes": complete_nodes, "member_path_evaluations": MEMBERS,
                 "status": "attempted", "elapsed_seconds": None}
        self.events.append(event)
        start = perf_counter()
        try:
            out = forward_fn()
            event["status"] = "returned"
            return out
        except BaseException:
            event["status"] = "failed"
            raise
        finally:
            event["elapsed_seconds"] = perf_counter() - start

    def snapshot(self) -> dict[str, Any]:
        forward = [e for e in self.events if e["kind"] == "complete_member_view_forward"]
        return {"forward_attempts": len(forward),
                "forward_returns": sum(e["status"] == "returned" for e in forward),
                "forward_failures": sum(e["status"] == "failed" for e in forward),
                "member_path_evaluation_attempts": sum(e["member_path_evaluations"] for e in forward),
                "optimizer_step_attempts": self.optimizer_step_attempts,
                "committed_proposals": self.committed_proposals,
                "discarded_proposals": self.discarded_proposals,
                "recorded_work_events": len(self.events)}


def complete_views(model: Any, forward_view: Callable[[Any, str], Any], *,
                   ledger: CostLedger, stage: str, complete_nodes: int,
                   gradients: bool):
    """Three actually charged complete-node forwards; guards must set gradients=False.

    perf_counter duration covers callback latency, not asynchronously queued GPU
    completion. Complete-data resource qualification must synchronize at explicit
    boundaries and measure peak allocated/reserved memory separately.
    """
    torch = _torch()
    context = torch.enable_grad() if gradients else torch.no_grad()
    with context:
        logits = {view: ledger.forward(lambda v=view: forward_view(model, v),
                                       stage=stage, view=view, complete_nodes=complete_nodes)
                  for view in VIEWS}
    validate_logits(logits)
    if logits["native"].shape[1] != complete_nodes:
        raise ValueError("Callback did not return all declared nodes")
    return logits


def label_fingerprint(roles: LabelRoles) -> str:
    payload = (roles.fit, roles.control, roles.train_labels, roles.class_count)
    return sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def competence_values(logits: Mapping[str, Any], roles: LabelRoles, plan: FixedPlan):
    """Per member/view/cell TRAIN-control mean CE, in nats."""
    torch = _torch()
    validate_logits(logits)
    if plan.role != "control" or plan.target_ids != roles.control:
        raise ValueError("Fixed control cells must cover exactly declared control roles")
    if logits["native"].shape[-1] != roles.class_count:
        raise ValueError("Logit class axis differs from label contract")
    labels = dict(roles.train_labels)
    out = []
    with torch.no_grad():
        for view in VIEWS:
            z = logits[view]
            for name, ids in plan.cells:
                index = torch.tensor(ids, dtype=torch.long, device=z.device)
                target = torch.tensor([labels[n] for n in ids], dtype=torch.long, device=z.device)
                cell = z.index_select(1, index)
                values = torch.nn.functional.cross_entropy(
                    cell.reshape(-1, roles.class_count), target.repeat(MEMBERS), reduction="none"
                ).reshape(MEMBERS, len(ids)).mean(dim=1)
                out.append((view, name, tuple(float(x) for x in values.cpu().tolist())))
    return tuple(out)


@dataclass(frozen=True)
class CompetenceReference:
    plan_fingerprint: str
    label_fingerprint: str
    values: tuple[tuple[str, str, tuple[float, ...]], ...]
    warm_function_receipt: str
    margin_nats: float = 0.01

    def __post_init__(self):
        if not self.warm_function_receipt or self.margin_nats != 0.01:
            raise ValueError("Copied warm receipt and sealed 0.01 nats margin required")
        if not self.values or len({(v, c) for v, c, _ in self.values}) != len(self.values):
            raise ValueError("One warm reference row per member/view/cell required")
        for view, _, values in self.values:
            if view not in VIEWS or len(values) != MEMBERS or any(not math.isfinite(x) or x < 0 for x in values):
                raise ValueError("Nonfinite/negative/malformed competence reference")


def build_competence_reference(logits: Mapping[str, Any], roles: LabelRoles,
                              plan: FixedPlan, *, warm_function_receipt: str):
    return CompetenceReference(plan.fingerprint, label_fingerprint(roles),
                               competence_values(logits, roles, plan), warm_function_receipt)


@dataclass(frozen=True)
class GuardDecision:
    accepted: bool
    failures: tuple[str, ...]
    evaluated_ce_entries: int
    evaluated_energy_entries: int

    def __post_init__(self):
        if self.accepted != (not self.failures):
            raise ValueError("Guard acceptance and failures disagree")


def assess_bundle(logits: Mapping[str, Any], roles: LabelRoles,
                  control_plan: FixedPlan, competence: CompetenceReference,
                  fit_plan: FixedPlan, active: ActiveReference):
    """Reuse one three-view bundle for all CE cells and active fit energies.

    No additional model forward is performed by centering or per-cell checks.
    Guard comparisons retain sealed inclusive boundaries with no tuned tolerance.
    """
    torch = _torch()
    if (competence.plan_fingerprint != control_plan.fingerprint or
            competence.label_fingerprint != label_fingerprint(roles)):
        raise ValueError("Control cell/label contract changed after warm reference")
    active.validate_plan(fit_plan)
    if competence.warm_function_receipt != active.warm_function_receipt:
        raise ValueError("Competence and energy references must come from one copied warm function")
    ce_count, energy_count = 0, 0
    with torch.no_grad():
        try:
            current = competence_values(logits, roles, control_plan)
            ce_count = len(current) * MEMBERS
            refs = {(v, c): values for v, c, values in competence.values}
            if set(refs) != {(v, c) for v, c, _ in current}:
                raise ValueError("Warm CE reference membership differs from fixed cells")
            failures = []
            for view, cell, values in current:
                for m, value in enumerate(values):
                    if not math.isfinite(value) or value > refs[(view, cell)][m] + 0.01:
                        failures.append(f"competence:{view}:{cell}:member:{m}")
            failures.extend(energy_failures(responses(logits), fit_plan, active))
            energy_count = len(active.names) * MEMBERS
            return GuardDecision(not failures, tuple(failures), ce_count, energy_count)
        except ResponseFailure as exc:
            return GuardDecision(False, (f"response_failure:{exc}",), ce_count, energy_count)
