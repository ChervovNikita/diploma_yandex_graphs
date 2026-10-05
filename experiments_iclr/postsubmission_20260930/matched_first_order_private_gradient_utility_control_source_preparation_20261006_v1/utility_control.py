"""Disabled matched first-order private-gradient utility reference.

Only raw allocation utility and its total live credit replace finite response.
No loader/trainer/CLI, automatic release, source mutation or numerical run.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

SOURCE_RELEASED = False
ARM = "first_order_utility_live"
OPERATOR_SHA256 = "fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3"
PORT_SHA256 = "a8ee35afda97afcb745c365a51f8e8647fe5e6a1350f654a5321c81e12abad86"
SEQUENTIAL_SHA256 = "1f9e704a1b95a98cec688d695094058594758a99bac2f057e2fe90e7f9315167"
STANDARD_KEYS = ("native_forward_calls", "private_gradient_calls", "native_vjp_calls",
                 "q_map_primal_calls", "q_map_vjp_calls", "small_query_vjp_calls")
UTILITY_KEYS = ("utility_margin_phi_vjp_calls", "utility_dummy_cotangent_reverse_calls",
                "utility_weighted_margin_private_gradient_calls")
EXPECTED_EPISODE_COUNTS = {"native_forward_calls": 40, "private_gradient_calls": 24,
    "native_vjp_calls": 8, "q_map_primal_calls": 30, "q_map_vjp_calls": 10, "small_query_vjp_calls": 1,
    "utility_margin_phi_vjp_calls": 8, "utility_dummy_cotangent_reverse_calls": 8,
    "utility_weighted_margin_private_gradient_calls": 4}
EXPECTED_RESPONSE_COUNTS = {"native_forward_calls": 12, "private_gradient_calls": 8,
    "native_vjp_calls": 0, "q_map_primal_calls": 10, "q_map_vjp_calls": 0, "small_query_vjp_calls": 0,
    "utility_margin_phi_vjp_calls": 4, "utility_dummy_cotangent_reverse_calls": 4,
    "utility_weighted_margin_private_gradient_calls": 0}


class Counters:
    """Attempted ordinary/utility constructions, not FLOPs or equal-cost units."""
    KEYS = STANDARD_KEYS + UTILITY_KEYS

    def __init__(self):
        self.total = dict.fromkeys(self.KEYS, 0)
        self.stages = {}

    def add(self, key, stage):
        self.total[key] += 1
        row = self.stages.setdefault(stage, dict.fromkeys(self.KEYS, 0))
        row[key] += 1

    def snapshot(self):
        return {"total": dict(self.total), "stages": {k: dict(v) for k, v in self.stages.items()},
                "private_gradient_calls_mean": "Own-CE/main partials only; utility constructions separately counted.",
                "native_forward_graphs_per_helper_scheduled_at_most": 2,
                "derivative_intermediate_bytes_not_bounded": True,
                "no_measured_FLOP_time_memory_equivalence_claim": True}


@dataclass
class _UtilityValues:
    raw_costs: tuple             # Utility costs, never relabeled as finite response.
    q_blocks: tuple
    diagnostics: dict
    probe_private_gradients: tuple
    finite_response_costs: tuple


def _engine(operator, operator_sha256, port, port_sha256, sequential, sequential_sha256,
            theta, phis, forward, pairs, s, ys, r, yr, *, engineering_authorized=False, counters=None):
    if engineering_authorized is not True:
        raise RuntimeError("Disabled: explicit independent engineering caller required")
    for module, supplied, expected in ((operator, operator_sha256, OPERATOR_SHA256),
        (port, port_sha256, PORT_SHA256), (sequential, sequential_sha256, SEQUENTIAL_SHA256)):
        if supplied != expected or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError("Exact unchanged mathematical dependency required")
    if operator.SOURCE_RELEASED is not False or port.PORT_RELEASED is not False or sequential.SOURCE_RELEASED is not False:
        raise ValueError("Original operator/port/sequential gates remain false")
    if tuple(sequential.Counters.KEYS) != STANDARD_KEYS:
        raise ValueError("Reviewed sequential counter convention changed")
    meter = counters if counters is not None else Counters()
    if set(meter.total) != set(Counters.KEYS):
        raise ValueError("Complete ordinary plus separate utility counters required")

    class UtilityEngine(sequential._Engine):
        def _value_grad(self, outputs, inputs, *, stage, kind, cotangents=None,
                        create_graph=False, retain_graph=False):
            self.meter.add(kind, stage)
            return self.torch.autograd.grad(outputs, tuple(inputs), grad_outputs=cotangents,
                allow_unused=True, create_graph=create_graph, retain_graph=retain_graph)

        def _utility_value_one(self, core, phi, stage):
            """One original forward, matrix-free directional product, paid after forward."""
            torch = self.torch
            with torch.enable_grad():
                private = self._leaves(phi)
                before = self._forward(stage)(core, private)[self.s]
                before_value = before.detach().clone()
                own = self.functional.cross_entropy(before, self.ys)
                # Reuse this native graph for the dummy margin VJP. Values only;
                # full derivative ownership is restored by _utility_credit below.
                own_partials = self._value_grad(own, private.values(), stage=stage,
                    kind="private_gradient_calls", retain_graph=True)
                gradient = self._tree(private, own_partials, detach=True)
                margins = tuple(self.op._margin(before.unsqueeze(0), pair)[:, 0] for pair in self.pairs)
                dummy = tuple(torch.zeros_like(b).requires_grad_(True) for b in margins)
                partials = self._value_grad(margins, private.values(), stage=stage,
                    kind="utility_margin_phi_vjp_calls", cotangents=dummy, create_graph=True, retain_graph=True)
                margin_partial = self._tree(private, partials, detach=False)
                dot = sum((margin_partial[n] * gradient[n].detach()).sum() for n in private)
                # This zero anchor preserves exact zero/disconnected directions
                # while keeping the scheduled dummy reverse well-defined.
                dot = dot + sum(0.0 * a.sum() for a in dummy)
                direction = self._value_grad(dot, dummy, stage=stage,
                    kind="utility_dummy_cotangent_reverse_calls")
                utility = tuple((-self.cfg.eta_probe * (torch.zeros_like(a) if d is None else d)).detach()
                                for a, d in zip(dummy, direction))
                probe = self.op._detach(self.op._sgd(private, gradient, self.cfg.eta_probe))
            del before, own, own_partials, margins, dummy, partials, margin_partial, dot, direction, private
            with torch.no_grad():
                after_value = self._forward(stage)(core, probe)[self.s].detach().clone()
                pre = tuple(self.op._margin(before_value.unsqueeze(0), pair)[:, 0] for pair in self.pairs)
                post = tuple(self.op._margin(after_value.unsqueeze(0), pair)[:, 0] for pair in self.pairs)
                ce_change = (self.functional.cross_entropy(after_value, self.ys)
                             - self.functional.cross_entropy(before_value, self.ys)).detach()
            return pre, post, ce_change, gradient, utility

        def _response_values(self, core, stage):
            torch, cfg = self.torch, self.cfg
            pre_columns, post_columns, utility_columns = ([[] for _ in self.pairs] for _ in range(3))
            changes, probe_gradients = [], []
            for phi in self.phis:
                pre, post, change, gradient, utility = self._utility_value_one(core, phi, stage)
                for bank, values in ((pre_columns, pre), (post_columns, post), (utility_columns, utility)):
                    for column, value in zip(bank, values):
                        column.append(value.detach())
                changes.append(change); probe_gradients.append(gradient)
            raw_costs, q_blocks, finite_costs, rows = [], [], [], []
            with torch.no_grad():
                for pair, pre, post, columns in zip(self.pairs, pre_columns, post_columns, utility_columns):
                    before, after, raw = (torch.stack(values, dim=1) for values in (pre, post, columns))
                    finite_response = after - before
                    cost, centered, scale = self.op._normalized_cost(raw, cfg.response_epsilon)
                    self.meter.add("q_map_primal_calls", stage)
                    q = self.op._assignment_map(cost, pair.laplacian, cfg, cfg.graph)
                    raw_costs.append(raw.detach()); q_blocks.append(q.detach()); finite_costs.append(finite_response.detach())
                    response_centered = finite_response - finite_response.mean(dim=1, keepdim=True)
                    delta = q - 1.0 / len(self.phis)
                    rows.append({"classes": pair.classes, "item_count": pair.nodes.shape[0],
                        "left_count": (pair.targets == pair.classes[0]).sum(), "right_count": (pair.targets == pair.classes[1]).sum(),
                        "cost_kind": "first_order_private_gradient_utility", "observed_response_kind": "paid_finite_softplus_margin_response",
                        "centered_response_rms": response_centered.square().mean().sqrt(),
                        "centered_cost_rms": centered.square().mean().sqrt(), "smooth_cost_scale": scale,
                        "epsilon_over_scale": cfg.response_epsilon / scale,
                        "normalized_cost_rms": (centered / scale).square().mean().sqrt(),
                        "assignment_relative_rms": len(self.phis) * delta.square().mean().sqrt(),
                        "assignment_max_deviation": delta.abs().max(), "mean_entropy": -(q * q.log()).sum(dim=1).mean(),
                        "min_assignment": q.min(), "row_residual": (q.sum(dim=1) - 1.0).abs().max(),
                        "column_residual": (q.sum(dim=0) - q.shape[0] / len(self.phis)).abs().max()})
            info = {"pairs": tuple(rows), "probe_own_ce_change": torch.stack(changes).detach(),
                    "assignments": tuple(q_blocks), "utility_raw_costs": tuple(raw_costs),
                    "observed_finite_response_costs": tuple(finite_costs)}
            return _UtilityValues(tuple(raw_costs), tuple(q_blocks), info, tuple(probe_gradients), tuple(finite_costs))

        def _utility_credit(self, raw_cotangents, member):
            """-eta*(Dcore h)^T g -eta*(Dcore g)^T h; BOTH factors live."""
            torch, stage = self.torch, "live_first_order_utility_credit"
            with torch.enable_grad():
                core, private = self._leaves(self.theta), self._leaves(self.phis[member])
                logits = self._forward(stage)(core, private)[self.s]
                own = self.functional.cross_entropy(logits, self.ys)
                partials = self._grad(own, private.values(), stage=stage, kind="private_gradient_calls", create_graph=True)
                g = self._tree(private, partials, detach=False)
                weighted_margin = sum((t[:, member].detach() * self.op._margin(logits.unsqueeze(0), pair)[:, 0]).sum()
                                      for pair, t in zip(self.pairs, raw_cotangents))
                partials_h = self._grad(weighted_margin, private.values(), stage=stage,
                    kind="utility_weighted_margin_private_gradient_calls", create_graph=True)
                h = self._tree(private, partials_h, detach=False)
                signed_dot = -self.cfg.eta_probe * sum((g[n] * h[n]).sum() for n in private)
                signed_dot = signed_dot + sum(0.0 * v.sum() for v in (*core.values(), *private.values()))
                derivatives = self._grad(signed_dot, tuple(core.values()) + tuple(private.values()),
                    stage=stage, kind="native_vjp_calls")
                shared = self._tree(core, derivatives[:len(core)], detach=True)
                private_credit = self._tree(private, derivatives[len(core):], detach=True)
            return shared, private_credit

        def outer(self):
            response = self._response_values(self.theta, "theta_response_values")
            query_values, adapted, main_gradients = self._query_values(response)
            value, query_cotangents = self._query_cotangents(query_values)
            shared = self._zeros(self.theta)
            private_outer = tuple(self._zeros(phi) for phi in self.phis)
            q_cotangents = tuple(self.torch.zeros_like(q) for q in response.q_blocks)
            recomputed_main = []
            for member in range(len(self.phis)):
                direct, q_credit, private_direct, main_partial = self._direct_member_vjp(response, query_cotangents[member], member)
                self._add(shared, direct); self._add(private_outer[member], private_direct)
                q_cotangents = tuple(total + credit for total, credit in zip(q_cotangents, q_credit))
                recomputed_main.append(main_partial)
                del direct, q_credit, private_direct, main_partial
            raw_cotangents = self._raw_cost_cotangents(response, q_cotangents)
            for member in range(len(self.phis)):
                credit, private_credit = self._utility_credit(raw_cotangents, member)
                self._add(shared, credit); self._add(private_outer[member], private_credit)
                del credit, private_credit
            return {"shared_gradient": shared, "outer_private_gradients": private_outer,
                "virtual_query_loss": value, "query_logits": query_values, "query_logit_cotangents": query_cotangents,
                "adapted": adapted, "main_private_gradients": main_gradients,
                "recomputed_main_private_partials": tuple(recomputed_main), "response": response,
                "q_cotangents": q_cotangents, "raw_cost_cotangents": raw_cotangents, "counters": self.meter.snapshot()}

        def episode(self, *, collect_inspection=False):
            initial = dict(self.meter.total)
            outer = self.outer()
            next_theta = self.op._detach(self.op._sgd(self.theta, outer["shared_gradient"], self.cfg.eta_core))
            query_loss = outer["virtual_query_loss"].detach()
            inspection = {"outer": outer} if collect_inspection else None
            if not collect_inspection:
                del outer
            # Inherited response starts from self.phis, the ORIGINAL rows, at theta+.
            committed = self.response(next_theta, stage="theta_plus_response_values")
            next_phis = tuple(self.op._detach(phi) for phi in committed["adapted"])
            info = committed["response"].diagnostics
            actual = {key: self.meter.total[key] - initial[key] for key in Counters.KEYS}
            if actual != EXPECTED_EPISODE_COUNTS:
                raise AssertionError("Utility episode attempted-count schedule differs: " + str(actual))
            info.update(arm=ARM, virtual_query_loss=query_loss, utility_episode_counts=actual,
                utility_execution_counters=self.meter.snapshot(), first_order_is_probe_approximation_not_Hessian_free=True,
                serving="Unchanged mean of four complete member softmax probabilities; no routing.",
                cost_equivalence_or_memory_success_claim=False, frozen_six_arm_pilot_changed=False)
            if collect_inspection:
                inspection["committed"] = committed
            return next_theta, next_phis, info, inspection

    return UtilityEngine(operator, operator_sha256, port, port_sha256, theta, phis, forward, pairs, s, ys, r, yr,
                         control="live", engineering_authorized=True, counters=meter)


def _engineering_response(*args, engineering_authorized=False, counters=None):
    return _engine(*args, engineering_authorized=engineering_authorized, counters=counters).response()


def _engineering_outer(*args, engineering_authorized=False, counters=None):
    return _engine(*args, engineering_authorized=engineering_authorized, counters=counters).outer()


def _engineering_episode(*args, engineering_authorized=False, counters=None, collect_inspection=False):
    return _engine(*args, engineering_authorized=engineering_authorized, counters=counters).episode(collect_inspection=collect_inspection)


def first_order_utility_native_episode(*args, counters=None):
    if not SOURCE_RELEASED:
        raise RuntimeError("Disabled first-order utility reference; no numerical or fit admission")
    theta, phis, info, _ = _engine(*args, engineering_authorized=True, counters=counters).episode()
    return theta, phis, info
