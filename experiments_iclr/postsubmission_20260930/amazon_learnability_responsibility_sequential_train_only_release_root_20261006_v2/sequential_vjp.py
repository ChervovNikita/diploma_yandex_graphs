"""Disabled exact sequential ordinary-autograd proposal for the pinned G0V2.

No loader, fit, checkpoint, RNG call or executing CLI exists. Underscored
engineering helpers require explicit caller authorization and the unchanged
operator/port bytes. All native callbacks and pairs are supplied by that port.
Two native forward graphs per helper is a scheduling claim, not a byte bound;
higher-order derivative intermediates are not included in that statement.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

SOURCE_RELEASED = True
OPERATOR_SHA256 = "fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3"
PORT_SHA256 = "a8ee35afda97afcb745c365a51f8e8647fe5e6a1350f654a5321c81e12abad86"


def _backend():
    import torch
    import torch.nn.functional as functional
    return torch, functional


class Counters:
    """Attempted calls; an engineering caller may retain this on failure."""

    KEYS = ("native_forward_calls", "private_gradient_calls", "native_vjp_calls",
            "q_map_primal_calls", "q_map_vjp_calls", "small_query_vjp_calls")

    def __init__(self):
        self.total = dict.fromkeys(self.KEYS, 0)
        self.stages = {}

    def add(self, key, stage):
        self.total[key] += 1
        row = self.stages.setdefault(stage, dict.fromkeys(self.KEYS, 0))
        row[key] += 1

    def snapshot(self):
        return {"total": dict(self.total),
                "stages": {name: dict(row) for name, row in self.stages.items()},
                "native_forward_graphs_per_helper_scheduled_at_most": 2,
                "derivative_intermediate_bytes_not_bounded": True}


def expected_episode_counts(control, members=4, pair_count=10):
    """Fixed prospective amendment; actual counters still require execution."""
    if control not in {"live", "graph_free", "margins", "uniform", "stop_q"}:
        raise ValueError("Unknown pinned control")
    credit = control not in {"uniform", "stop_q"}
    response_credit = credit and control != "margins"
    return {
        "native_forward_calls": (12 if response_credit else 10 if credit else 9) * members,
        "private_gradient_calls": (6 if response_credit else 5) * members,
        "native_vjp_calls": (3 if response_credit else 2 if credit else 1) * members,
        "q_map_primal_calls": (3 if credit else 2) * pair_count,
        "q_map_vjp_calls": pair_count if credit else 0,
        "small_query_vjp_calls": 1,
    }


@dataclass
class _ResponseValues:
    raw_costs: tuple
    q_blocks: tuple
    diagnostics: dict
    probe_private_gradients: tuple


class _Engine:
    """One pure/eval episode context; contains no native autograd graph cache."""

    def __init__(self, operator, operator_sha256, port, port_sha256, theta, phis,
                 forward, pairs, s, ys, r, yr, *, control="live",
                 engineering_authorized=False, counters=None):
        if engineering_authorized is not True:
            raise RuntimeError("Disabled: explicit engineering caller authorization required")
        for module, supplied, expected in ((operator, operator_sha256, OPERATOR_SHA256),
                                           (port, port_sha256, PORT_SHA256)):
            if (supplied != expected or
                    hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected):
                raise ValueError("Immutable engineering oracle source differs")
        port._check_operator(operator, operator_sha256)
        if port.PORT_RELEASED is not False:
            raise ValueError("Original port file/process gate must remain false here")
        if len(phis) != 4 or len(pairs) != 10 or control not in operator.CONTROLS:
            raise ValueError("Exact native shared4 and ten supplied pairs required")
        torch, functional = _backend()
        if (s.dtype != torch.long or r.dtype != torch.long or ys.dtype != torch.long
                or yr.dtype != torch.long or s.ndim != 1 or r.ndim != 1
                or s.shape != ys.shape or r.shape != yr.shape or not s.numel() or not r.numel()
                or s.unique().numel() != s.numel() or r.unique().numel() != r.numel()
                or bool((s < 0).any()) or bool((r < 0).any()) or bool(torch.isin(s, r).any())
                or bool(((ys < 0) | (ys >= 5)).any()) or bool(((yr < 0) | (yr >= 5)).any())
                or not bool((torch.bincount(ys, minlength=5) > 0).all())):
            raise ValueError("Bound disjoint native five-class S/R roles required")
        if not theta or any(set(phi) != set(port.PRIVATE_NAMES) for phi in phis):
            raise ValueError("Complete shared/private state dictionaries required")
        self.op, self.torch, self.functional = operator, torch, functional
        self.theta = operator._detach(theta)
        self.phis = tuple(operator._detach(phi) for phi in phis)
        self.forward, self.pairs = forward, tuple(pairs)
        self.s, self.ys, self.r, self.yr = s, ys, r, yr
        self.control, self.cfg = control, operator.Config()
        self.meter = counters if counters is not None else Counters()

    def _leaves(self, params):
        # detach creates new tensor objects; requires_grad_ changes those objects
        # only. All value updates below are out of place; input storage is never written.
        return {name: value.detach().requires_grad_(True) for name, value in params.items()}

    def _forward(self, stage):
        def counted(core, private):
            self.meter.add("native_forward_calls", stage)
            return self.forward(core, private)
        return counted

    def _grad(self, outputs, inputs, *, stage, kind, cotangents=None, create_graph=False):
        self.meter.add(kind, stage)
        return self.torch.autograd.grad(outputs, tuple(inputs), grad_outputs=cotangents,
            allow_unused=True, create_graph=create_graph, retain_graph=create_graph)

    def _tree(self, params, derivatives, *, detach):
        return {name: (self.torch.zeros_like(value) if derivative is None else
                      derivative.detach() if detach else derivative)
                for (name, value), derivative in zip(params.items(), derivatives)}

    def _zeros(self, params):
        return {name: self.torch.zeros_like(value) for name, value in params.items()}

    def _add(self, accumulator, contribution):
        for name in accumulator:
            accumulator[name] = accumulator[name] + contribution[name].detach()

    def _probe_values_one(self, theta, phi, stage):
        """One probe graph, freed before the no-grad probe-adapted forward."""
        torch = self.torch
        private = self._leaves(phi)
        with torch.enable_grad():
            before = self._forward(stage)(theta, private)[self.s]
            before_value = before.detach().clone()
            own = self.functional.cross_entropy(before, self.ys)
            derivatives = self._grad(own, private.values(), stage=stage,
                                     kind="private_gradient_calls")
            gradient = self._tree(private, derivatives, detach=True)
            probe = self.op._detach(self.op._sgd(private, gradient, self.cfg.eta_probe))
        # No returned object retains this native graph, including unused outputs.
        del before, own, derivatives, private
        with torch.no_grad():
            after_value = self._forward(stage)(theta, probe)[self.s].detach().clone()
            before_margins = tuple(self.op._margin(before_value.unsqueeze(0), pair)[:, 0]
                                   for pair in self.pairs)
            after_margins = tuple(self.op._margin(after_value.unsqueeze(0), pair)[:, 0]
                                  for pair in self.pairs)
            ce_change = (self.functional.cross_entropy(after_value, self.ys)
                         - self.functional.cross_entropy(before_value, self.ys)).detach()
        return before_margins, after_margins, ce_change, gradient

    def _response_values(self, theta, stage):
        """Retain only small detached margins/Q/diagnostics across members."""
        torch, cfg = self.torch, self.cfg
        before_columns = [[] for _ in self.pairs]
        after_columns = [[] for _ in self.pairs]
        ce_changes, probe_gradients = [], []
        for phi in self.phis:
            before, after, change, gradient = self._probe_values_one(theta, phi, stage)
            for columns, value in zip(before_columns, before):
                columns.append(value.detach())
            for columns, value in zip(after_columns, after):
                columns.append(value.detach())
            ce_changes.append(change)
            probe_gradients.append(gradient)
        raw_costs, q_blocks, diagnostics = [], [], []
        gamma = 0.0 if self.control == "graph_free" else cfg.graph
        with torch.no_grad():
            for pair, pre, post in zip(self.pairs, before_columns, after_columns):
                before, after = torch.stack(pre, dim=1), torch.stack(post, dim=1)
                response = after - before
                raw = before if self.control == "margins" else response
                cost, centered, scale = self.op._normalized_cost(raw, cfg.response_epsilon)
                self.meter.add("q_map_primal_calls", stage)
                q = self.op._assignment_map(cost, pair.laplacian, cfg, gamma)
                # Preserve even the uniform control's original primal calculation.
                if self.control == "uniform":
                    q = torch.full_like(q, 1.0 / len(self.phis))
                raw_costs.append(raw.detach())
                q_blocks.append(q.detach())
                qd, cd, sd = q.detach(), centered.detach(), scale.detach()
                response_centered = (response - response.mean(dim=1, keepdim=True)).detach()
                delta = qd - 1.0 / len(self.phis)
                diagnostics.append({
                    "classes": pair.classes, "item_count": pair.nodes.shape[0],
                    "left_count": (pair.targets == pair.classes[0]).sum(),
                    "right_count": (pair.targets == pair.classes[1]).sum(),
                    "cost_kind": "current_margins" if self.control == "margins" else "response",
                    "centered_response_rms": response_centered.square().mean().sqrt(),
                    "centered_cost_rms": cd.square().mean().sqrt(), "smooth_cost_scale": sd,
                    "epsilon_over_scale": cfg.response_epsilon / sd,
                    "normalized_cost_rms": (cd / sd).square().mean().sqrt(),
                    "assignment_relative_rms": len(self.phis) * delta.square().mean().sqrt(),
                    "assignment_max_deviation": delta.abs().max(),
                    "mean_entropy": -(qd * qd.log()).sum(dim=1).mean(),
                    "min_assignment": qd.min(),
                    "row_residual": (qd.sum(dim=1) - 1.0).abs().max(),
                    "column_residual": (qd.sum(dim=0) - qd.shape[0] / len(self.phis)).abs().max(),
                })
        info = {"pairs": tuple(diagnostics), "probe_own_ce_change": torch.stack(ce_changes).detach(),
                "assignments": tuple(q_blocks)}
        return _ResponseValues(tuple(raw_costs), tuple(q_blocks), info, tuple(probe_gradients))

    def _adapt_value_one(self, theta, phi, q_blocks, member, stage):
        """Main partial owns an independent private leaf; Q is a fixed value."""
        with self.torch.enable_grad():
            private = self._leaves(phi)
            loss = self.op._main_loss(theta, private, q_blocks, member, self._forward(stage),
                self.s, self.ys, self.pairs, 5, self.cfg)
            derivatives = self._grad(loss, private.values(), stage=stage,
                                     kind="private_gradient_calls")
            gradient = self._tree(private, derivatives, detach=True)
            adapted = self.op._detach(self.op._sgd(private, gradient, self.cfg.eta_private))
        return adapted, gradient

    def _query_values(self, response):
        query_values, adapted_values, main_gradients = [], [], []
        for member, phi in enumerate(self.phis):
            adapted, gradient = self._adapt_value_one(self.theta, phi, response.q_blocks,
                                                      member, "query_values")
            # The main-gradient helper has returned; its native graph is discarded.
            with self.torch.no_grad():
                query = self._forward("query_values")(self.theta, adapted)[self.r].detach().clone()
            query_values.append(query)
            adapted_values.append(adapted)
            main_gradients.append(gradient)
        return self.torch.stack(query_values), tuple(adapted_values), tuple(main_gradients)

    def _query_cotangents(self, query_values):
        """Call the exact oracle query formula on small all-member leaf logits."""
        torch = self.torch
        with torch.enable_grad():
            logits = query_values.detach().requires_grad_(True)
            local_query = torch.arange(self.r.numel(), device=self.r.device)
            # Each supplied row already is the exact original R restriction.
            value = self.op._query_objective(None, tuple(range(len(self.phis))),
                lambda unused, member: logits[member], local_query, self.yr, self.cfg)
            derivative, = self._grad(value, (logits,), stage="query_leaf_cotangents",
                kind="small_query_vjp_calls")
        return value.detach(), derivative.detach()

    def _direct_member_vjp(self, response, query_cotangent, member):
        """At most main/query native forwards; independent phi/Q partial owners."""
        torch = self.torch
        with torch.enable_grad():
            core = self._leaves(self.theta)
            private = self._leaves(self.phis[member])
            q_leaves = tuple(q.detach().requires_grad_(True) for q in response.q_blocks)
            loss = self.op._main_loss(core, private, q_leaves, member,
                self._forward("direct_member_vjp"), self.s, self.ys, self.pairs, 5, self.cfg)
            derivatives = self._grad(loss, private.values(), stage="direct_member_vjp",
                kind="private_gradient_calls", create_graph=True)
            gradient = self._tree(private, derivatives, detach=False)
            adapted = self.op._sgd(private, gradient, self.cfg.eta_private)
            query = self._forward("direct_member_vjp")(core, adapted)[self.r]
            targets = tuple(core.values()) + q_leaves + tuple(private.values())
            vjp = self._grad(query, targets, stage="direct_member_vjp", kind="native_vjp_calls",
                             cotangents=query_cotangent.detach())
            n_core, n_q = len(core), len(q_leaves)
            direct = self._tree(core, vjp[:n_core], detach=True)
            q_credit = tuple(torch.zeros_like(q) if d is None else d.detach()
                             for q, d in zip(q_leaves, vjp[n_core:n_core + n_q]))
            private_direct = self._tree(private, vjp[n_core + n_q:], detach=True)
            main_partial = {name: value.detach() for name, value in gradient.items()}
        # Only detached derivative/state values leave this helper. Its core/phi/Q
        # leaves, main/query graphs and derivative intermediates go out of scope.
        return direct, q_credit, private_direct, main_partial

    def _raw_cost_cotangents(self, response, q_cotangents):
        """Differentiate the COMPLETE original centering/RMS/eight-step map."""
        if self.control in {"uniform", "stop_q"}:
            return tuple(self.torch.zeros_like(raw) for raw in response.raw_costs)
        torch, result = self.torch, []
        gamma = 0.0 if self.control == "graph_free" else self.cfg.graph
        for pair, raw_value, q_cotangent in zip(self.pairs, response.raw_costs, q_cotangents):
            with torch.enable_grad():
                raw = raw_value.detach().requires_grad_(True)
                cost, _, _ = self.op._normalized_cost(raw, self.cfg.response_epsilon)
                self.meter.add("q_map_primal_calls", "q_map_vjp")
                q = self.op._assignment_map(cost, pair.laplacian, self.cfg, gamma)
                derivative, = self._grad(q, (raw,), stage="q_map_vjp", kind="q_map_vjp_calls",
                                         cotangents=q_cotangent.detach())
                result.append(torch.zeros_like(raw) if derivative is None else derivative.detach())
            del raw, cost, q, derivative
        return tuple(result)

    def _before_margin_vjp(self, raw_cotangents, member, sign):
        """One native graph, separate from probe/after; response has minus sign."""
        with self.torch.enable_grad():
            core = self._leaves(self.theta)
            private = self._leaves(self.phis[member])
            logits = self._forward("before_margin_vjp")(core, private)[self.s]
            margins = tuple(self.op._margin(logits.unsqueeze(0), pair)[:, 0]
                            for pair in self.pairs)
            derivatives = self._grad(margins, tuple(core.values()) + tuple(private.values()),
                stage="before_margin_vjp", kind="native_vjp_calls",
                cotangents=tuple(sign * value[:, member].detach() for value in raw_cotangents))
            shared = self._tree(core, derivatives[:len(core)], detach=True)
            private_credit = self._tree(private, derivatives[len(core):], detach=True)
        return shared, private_credit

    def _probe_after_margin_vjp(self, raw_cotangents, member):
        """Retain the own-probe mixed/Hessian chain plus one after forward."""
        with self.torch.enable_grad():
            core = self._leaves(self.theta)
            private = self._leaves(self.phis[member])
            own = self.op._own_ce(core, private, self._forward("probe_after_margin_vjp"),
                                  self.s, self.ys)
            derivatives = self._grad(own, private.values(), stage="probe_after_margin_vjp",
                kind="private_gradient_calls", create_graph=True)
            gradient = self._tree(private, derivatives, detach=False)
            probe = self.op._sgd(private, gradient, self.cfg.eta_probe)
            after = self._forward("probe_after_margin_vjp")(core, probe)[self.s]
            margins = tuple(self.op._margin(after.unsqueeze(0), pair)[:, 0] for pair in self.pairs)
            vjp = self._grad(margins, tuple(core.values()) + tuple(private.values()),
                stage="probe_after_margin_vjp", kind="native_vjp_calls",
                cotangents=tuple(value[:, member].detach() for value in raw_cotangents))
            shared = self._tree(core, vjp[:len(core)], detach=True)
            private_credit = self._tree(private, vjp[len(core):], detach=True)
        return shared, private_credit

    def response(self, theta=None, *, stage="response_values"):
        """Exact detached private-response values and all original diagnostics."""
        theta = self.theta if theta is None else self.op._detach(theta)
        response = self._response_values(theta, stage)
        adapted, main_gradients = [], []
        for member, phi in enumerate(self.phis):
            value, gradient = self._adapt_value_one(theta, phi, response.q_blocks, member, stage)
            adapted.append(value)
            main_gradients.append(gradient)
        return {"adapted": tuple(adapted), "response": response,
                "main_private_gradients": tuple(main_gradients), "counters": self.meter.snapshot()}

    def outer(self):
        """Full shared outer derivative; extra private derivative is inspection only."""
        torch = self.torch
        response = self._response_values(self.theta, "theta_response_values")
        query_values, adapted, main_gradients = self._query_values(response)
        value, query_cotangents = self._query_cotangents(query_values)
        shared = self._zeros(self.theta)
        private_outer = tuple(self._zeros(phi) for phi in self.phis)
        q_cotangents = tuple(torch.zeros_like(q) for q in response.q_blocks)
        recomputed_main_partials = []
        for member in range(len(self.phis)):
            direct, q_credit, private_direct, main_partial = self._direct_member_vjp(
                response, query_cotangents[member], member)
            self._add(shared, direct)
            self._add(private_outer[member], private_direct)
            q_cotangents = tuple(total + credit for total, credit in zip(q_cotangents, q_credit))
            recomputed_main_partials.append(main_partial)
            del direct, q_credit, private_direct, main_partial
        raw_cotangents = self._raw_cost_cotangents(response, q_cotangents)
        if self.control not in {"uniform", "stop_q"}:
            for member in range(len(self.phis)):
                sign = 1.0 if self.control == "margins" else -1.0
                before, phi_before = self._before_margin_vjp(raw_cotangents, member, sign)
                self._add(shared, before)
                self._add(private_outer[member], phi_before)
                del before, phi_before
                if self.control != "margins":
                    after, phi_after = self._probe_after_margin_vjp(raw_cotangents, member)
                    self._add(shared, after)
                    self._add(private_outer[member], phi_after)
                    del after, phi_after
        return {"shared_gradient": shared, "outer_private_gradients": private_outer,
                "virtual_query_loss": value, "query_logits": query_values,
                "query_logit_cotangents": query_cotangents, "adapted": adapted,
                "main_private_gradients": main_gradients,
                "recomputed_main_private_partials": tuple(recomputed_main_partials),
                "response": response, "q_cotangents": q_cotangents,
                "raw_cost_cotangents": raw_cotangents, "counters": self.meter.snapshot()}

    def episode(self, *, collect_inspection=False):
        """theta+ followed by fresh sequential response from ORIGINAL phi."""
        initial_counts = dict(self.meter.total)
        outer = self.outer()
        next_theta = self.op._detach(self.op._sgd(self.theta, outer["shared_gradient"], self.cfg.eta_core))
        virtual_loss = outer["virtual_query_loss"].detach()
        inspection = {"outer": outer} if collect_inspection else None
        if not collect_inspection:
            del outer
        committed = self.response(next_theta, stage="theta_plus_response_values")
        next_phis = tuple(self.op._detach(phi) for phi in committed["adapted"])
        info = committed["response"].diagnostics
        info["virtual_query_loss"] = virtual_loss
        actual = {name: self.meter.total[name] - initial_counts[name] for name in Counters.KEYS}
        expected = expected_episode_counts(self.control, len(self.phis), len(self.pairs))
        if actual != expected:
            raise AssertionError("Sequential episode accounting differs: " + str((actual, expected)))
        info["sequential_execution_counters"] = self.meter.snapshot()
        info["sequential_episode_counts"] = actual
        info["old_monolithic_native_forward_budget"] = 9 * len(self.phis)
        info["prospective_compute_amendment_required_before_fit"] = True
        if collect_inspection:
            inspection["committed"] = committed
        return next_theta, next_phis, info, inspection


def _engineering_response(operator, operator_sha256, port, port_sha256, theta, phis,
                          forward, pairs, s, ys, r, yr, *, control="live",
                          engineering_authorized=False, counters=None):
    """Explicitly authorized detached response/Q/private-gradient oracle target."""
    return _Engine(operator, operator_sha256, port, port_sha256, theta, phis,
                   forward, pairs, s, ys, r, yr, control=control,
                   engineering_authorized=engineering_authorized, counters=counters).response()


def _engineering_outer(operator, operator_sha256, port, port_sha256, theta, phis,
                       forward, pairs, s, ys, r, yr, *, control="live",
                       engineering_authorized=False, counters=None):
    """Explicitly authorized full sequential derivative and detached inspection."""
    return _Engine(operator, operator_sha256, port, port_sha256, theta, phis,
                   forward, pairs, s, ys, r, yr, control=control,
                   engineering_authorized=engineering_authorized, counters=counters).outer()


def _engineering_episode(operator, operator_sha256, port, port_sha256, theta, phis,
                         forward, pairs, s, ys, r, yr, *, control="live",
                         engineering_authorized=False, counters=None, collect_inspection=True):
    """Engineering only; root must compare every coordinate/state/Q to original."""
    return _Engine(operator, operator_sha256, port, port_sha256, theta, phis,
                   forward, pairs, s, ys, r, yr, control=control,
                   engineering_authorized=engineering_authorized, counters=counters).episode(
                       collect_inspection=collect_inspection)


def sequential_native_episode(operator, operator_sha256, port, port_sha256, theta, phis,
                              forward, pairs, s, ys, r, yr, *, control="live", counters=None):
    if not SOURCE_RELEASED:
        raise RuntimeError("Disabled source-only sequential VJP candidate")
    # A future separately reviewed release remains a pure state map, never a fit.
    next_theta, next_phis, info, _ = _Engine(operator, operator_sha256, port, port_sha256,
        theta, phis, forward, pairs, s, ys, r, yr, control=control,
        engineering_authorized=True, counters=counters).episode()
    return next_theta, next_phis, info


if __name__ == "__main__":
    print(json.dumps({"status": "DISABLED_SOURCE_ONLY", "numerical_imports": False,
                      "execution_entry": False, "SOURCE_RELEASED": SOURCE_RELEASED}))
