"""Meaningful synthetic/stdlb checks only; no Torch, models or scientific data."""
from __future__ import annotations

import ast
import hashlib
import json
from math import exp, log, isclose
from pathlib import Path
import sys

from source_supply import (Config, ContractError, analytic_source_supply,
                           check_guard_values, project_nonincrease_cone,
                           validate_descent_slope, verify_ownership)


def require_error(function):
    try:
        function()
    except ContractError:
        return
    raise AssertionError("Required contract error was absent")


def gradient_case(mode, target):
    factual = [0.7, -0.4, 1.1]
    probe = [-0.2, 0.5, 0.3]
    peers = [[0.1, -0.1, 0.4], [-0.5, 0.6, 0.2], [0.3, -0.7, 0.8]]
    base = analytic_source_supply(factual, probe, peers, target, mode=mode)
    errors = []
    step = 1e-6
    for which, values in [("factual", factual), ("probe", probe)]:
        for index in range(len(values)):
            plus, minus = values[:], values[:]
            plus[index] += step
            minus[index] -= step
            args_plus = (plus, probe) if which == "factual" else (factual, plus)
            args_minus = (minus, probe) if which == "factual" else (factual, minus)
            value_plus = analytic_source_supply(*args_plus, peers, target, mode=mode)["J"]
            value_minus = analytic_source_supply(*args_minus, peers, target, mode=mode)["J"]
            difference = (value_plus-value_minus)/(2*step)
            exact = base[which+"_gradient"][index]
            errors.append(abs(difference-exact))
            assert isclose(difference, exact, rel_tol=2e-7, abs_tol=2e-9)
    assert all(0 <= r <= 1 for key in ["rho_supply", "rho_absent"] for r in base[key])
    return max(errors), base


class Storage:
    def __init__(self, identity):
        self.identity = identity
    def data_ptr(self):
        return self.identity


class Parameter:
    """Non-tensor identity/storage fixture, not a numerical model."""
    def __init__(self, identity):
        self.storage = Storage(identity)
        self.device, self.requires_grad, self._version, self.grad = "fixture", True, 0, None
    def numel(self):
        return 1
    def untyped_storage(self):
        return self.storage


class Member:
    def __init__(self, parameters):
        self.parameters = parameters
    def named_parameters(self, remove_duplicate=False):
        return self.parameters.items()


def main():
    assert "torch" not in sys.modules
    categorical_error, categorical = gradient_case("categorical_logits", 2)
    bernoulli_error, bernoulli = gradient_case("bernoulli_marginal_logits", [1, 0, 1])

    # Per-label marginal mixture must match averaging sigmoid probabilities,
    # with observed positive AND negative outcomes, then mean BCE.
    factual, probe = [0.7, -0.4, 1.1], [-0.2, 0.5, 0.3]
    peers = [[0.1, -0.1, 0.4], [-0.5, 0.6, 0.2], [0.3, -0.7, 0.8]]
    target = [1, 0, 1]
    sigmoid = lambda x: 1/(1+exp(-x))
    def marginal_score(first):
        mean = [sum(sigmoid(row[c]) for row in [first, *peers])/4 for c in range(3)]
        return -sum(log(p if y else 1-p) for p, y in zip(mean, target))/3
    assert isclose(bernoulli["supply"], marginal_score(factual), abs_tol=1e-12)
    assert isclose(bernoulli["absent"], marginal_score(probe), abs_tol=1e-12)
    joint = -log(sum(
        __import__("math").prod(sigmoid(row[c]) if target[c] else 1-sigmoid(row[c])
                                for c in range(3)) for row in [factual, *peers])/4)/3
    assert abs(joint-bernoulli["supply"]) > 1e-4

    # Changing label1 logits cannot alter label0 responsibilities/derivatives.
    changed = factual[:]
    changed[1] += 2.0
    independent_label = analytic_source_supply(changed, probe, peers, target,
                                               mode="bernoulli_marginal_logits")
    assert independent_label["rho_supply"][0] == bernoulli["rho_supply"][0]
    assert independent_label["factual_gradient"][0] == bernoulli["factual_gradient"][0]

    # A shared pointwise predictor has the SAME private Jacobian in both views.
    # Separate factual/probe derivatives need not vanish, but their sum does.
    same = analytic_source_supply(factual, factual, peers, target,
                                  mode="bernoulli_marginal_logits")
    assert abs(same["J"]) < 1e-14
    assert all(abs(a+b) < 1e-14 for a, b in zip(same["factual_gradient"], same["probe_gradient"]))

    # Better contrast obtained solely by worsening its absent denominator fails.
    reference = {"J": -0.1, "source_j:0": -0.1, "absent:0": 0.8,
                 "full_own:0": 0.7, "full_pool": 0.6, "probe_own:0": 0.9}
    gaming = dict(reference, J=-0.2, **{"source_j:0": -0.2, "absent:0": 0.9})
    risks = ["absent:0", "full_own:0", "full_pool", "probe_own:0"]
    rejected = check_guard_values(reference, gaming, risks, ["source_j:0"], armijo_rhs=-0.11)
    assert not rejected.accepted and "absent:0" in rejected.violations
    supplied = dict(reference, J=-0.2, **{"source_j:0": -0.2, "absent:0": 0.79})
    accepted = check_guard_values(reference, supplied, risks, ["source_j:0"], armijo_rhs=-0.11)
    assert accepted.accepted
    assert supplied["J"]+supplied["absent:0"] < reference["J"]+reference["absent:0"]

    # Cone projection: a feasible face, redundant constraints, and zero direction.
    projected = project_nonincrease_cone([1.0, -1.0], [[1.0, 0.0], [2.0, 0.0]])
    assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(projected, [0.0, -1.0]))
    assert project_nonincrease_cone([1.0], [[1.0]]) == (0.0,)
    for scale in (1e200, 1e-200):
        scaled = project_nonincrease_cone([1.0, -1.0], [[scale, 0.0]])
        assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(scaled, [0.0, -1.0]))
    require_error(lambda: validate_descent_slope(float("nan")))
    require_error(lambda: validate_descent_slope(float("-inf")))
    assert isclose(sum(a*b for a, b in zip([-1.0, 1.0], projected)),
                   -sum(x*x for x in projected), abs_tol=1e-12)

    # Complete identity/storage ownership checks; no numerical model is built.
    common = Parameter(100)
    members = [Member({"common": common, "factor": Parameter(200+i)}) for i in range(4)]
    ownership = verify_ownership(members, ["common"], ["factor"])
    assert len(ownership.shared) == 1 and len(ownership.private) == 4
    cloned = [Member(dict(m.parameters)) for m in members]
    cloned[1].parameters["common"] = Parameter(999)
    require_error(lambda: verify_ownership(cloned, ["common"], ["factor"]))
    aliased = [Member(dict(m.parameters)) for m in members]
    aliased[1].parameters["factor"] = aliased[0].parameters["factor"]
    require_error(lambda: verify_ownership(aliased, ["common"], ["factor"]))
    storage_alias = [Member(dict(m.parameters)) for m in members]
    storage_alias[1].parameters["factor"] = Parameter(200)
    require_error(lambda: verify_ownership(storage_alias, ["common"], ["factor"]))
    require_error(lambda: verify_ownership(members, ["common"], []))
    untied = [Member({"native": Parameter(500+i), "factor": Parameter(600+i)}) for i in range(4)]
    independent = verify_ownership(untied, [], ["factor"], ["native"], require_shared=False)
    assert not independent.shared and len(independent.member_owned_nonsteered) == 4
    untied[1].parameters["native"] = untied[0].parameters["native"]
    require_error(lambda: verify_ownership(untied, [], ["factor"], ["native"], require_shared=False))
    require_error(lambda: Config().require_enabled())

    source = Path(__file__).with_name("source_supply.py")
    parsed = ast.parse(source.read_text())
    # Source audit: peers are explicitly selected by recipient source, then detached.
    source_text = source.read_text()
    assert 'OutputKey("train_probe", k, source)].detach()' in source_text
    assert 'native_forward(m, None, "eval", eval_tokens[m])' in source_text
    assert "torch" not in sys.modules
    result = {
        "schema": "source-supply-synthetic-analytical-verification-v1",
        "status": "passed", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "categorical_central_difference_max_abs_error": categorical_error,
        "bernoulli_marginal_central_difference_max_abs_error": bernoulli_error,
        "checks": ["categorical weighted factual-minus-probe CE derivative",
                   "mixed positive/negative Bernoulli entry derivatives",
                   "mean BCE of per-label probability pool",
                   "no joint-label likelihood mixture or across-label normalization",
                   "pointwise identical-Jacobian cancellation",
                   "absent-pool anti-gaming guard", "cone face/redundancy/zero projection",
                   "cone constraint scale invariance without overflow/underflow loss",
                   "nonfinite descent-slope rejection",
                   "genuine shared identity and private object/storage ownership",
                   "inactive default", "source AST and explicit same-recipient peer view",
                   "explicit all-full serving callback"],
        "torch_imported": False, "models_or_scientific_data_executed": False,
        "qualification_limit": "Synthetic stdlib algebra, scalar guards and ownership fixtures only. No Torch autograd/replay, native views, optimizer, architecture, data, runtime memory or predictive utility qualification.",
        "source_ast_nodes": sum(1 for _ in ast.walk(parsed))
    }
    Path(__file__).with_name("ANALYTICAL_VERIFICATION.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "torch_imported": False,
                      "categorical_max_error": categorical_error, "bernoulli_max_error": bernoulli_error}))


if __name__ == "__main__":
    main()
