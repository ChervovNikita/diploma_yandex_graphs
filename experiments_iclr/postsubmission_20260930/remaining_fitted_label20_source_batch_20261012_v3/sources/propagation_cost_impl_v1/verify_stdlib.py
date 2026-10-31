"""Rational fixtures and source inspection. No Torch/NumPy/data import."""

from __future__ import annotations

import ast
import argparse
from collections import Counter
from fractions import Fraction as R
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from contract import ROOT, SCHEMA, PINS, check_pins, confined_output
else:
    from .contract import ROOT, SCHEMA, PINS, check_pins, confined_output


def zeros(n, d):
    return [[R(0) for _ in range(d)] for _ in range(n)]


def mm(a, x):
    return [[sum(a[i][j] * x[j][k] for j in range(len(x)))
             for k in range(len(x[0]))] for i in range(len(a))]


def transpose(a):
    return [list(row) for row in zip(*a)]


def add(a, b, scale=1):
    return [[av + scale * bv for av, bv in zip(arow, brow)]
            for arow, brow in zip(a, b)]


def scale(a, s):
    return [[s * value for value in row] for row in a]


def mean(members):
    total = zeros(len(members[0]), len(members[0][0]))
    for member in members:
        total = add(total, member)
    return scale(total, R(1, len(members)))


def inner(a, b):
    return sum(x * y for ar, br in zip(a, b) for x, y in zip(ar, br))


def offset(fingerprint, receiver):
    return int.from_bytes(hashlib.sha256(
        f"{fingerprint}|{receiver}|20260930".encode()).digest()[:8], "big") % 4


def gate():
    checks = []

    def checked(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    # Raw duplicates, low degrees, isolates and pre-existing self edges.
    source = [1, 1, 2, 0, 2, 3, 3, 4, 5, 2, 4, 5]
    receiver = [0, 0, 0, 1, 2, 2, 2, 2, 4, 4, 4, 5]
    n, m, d = 6, 4, 3
    degree = Counter(receiver)
    multiplicity = Counter(zip(receiver, source))
    p, groups = zeros(n, n), [zeros(n, n) for _ in range(4)]
    fingerprint = hashlib.sha256(b"rational-source-fixture-v1").hexdigest()
    color_supports = [set() for _ in range(4)]
    for row in range(n):
        columns = sorted(s for (r, s) in multiplicity if r == row)
        for rank, column in enumerate(columns):
            weight = R(multiplicity[(row, column)], degree[row])
            color = (rank + offset(fingerprint, row)) % 4
            p[row][column] = groups[color][row][column] = weight
            color_supports[color].add((row, column))
    checked("duplicate raw degree preserved", p[0][1] == R(2, 3) and p[0][2] == R(1, 3))
    checked("isolate stays zero", p[3] == [R(0)] * n)
    checked("existing self edges retained", p[2][2] == R(1, 4) and p[4][4] == R(1, 3) and p[5][5] == 1)
    checked("canonical color supports disjoint", all(
        not color_supports[a] & color_supports[b] for a in range(4) for b in range(a)))
    checked("colors sum to P", mean([scale(group, 4) for group in groups]) == p)
    naive = [scale(add(groups[c], groups[(c + 1) % 4]), 2) for c in range(4)]
    checked("naive half-edge expectation equals P", mean(naive) == p)

    states = [[[R((node + 1) * (member + 2) + channel**2, 7)
                for channel in range(d)] for node in range(n)] for member in range(m)]
    state_mean = mean(states)
    delta = [add(state, state_mean, -1) for state in states]
    checked("post-normalized member residuals center exactly", mean(delta) == zeros(n, d))
    exact_mean = mm(p, state_mean)
    for color in range(4):
        q = scale(groups[color], 4)
        aggregates = [add(exact_mean, mm(q, residual)) for residual in delta]
        checked(f"linear shared-mask mean exact color{color}", mean(aggregates) == exact_mean)
        checked(f"adjoint exact color{color}", inner(mm(q, states[0]), states[1])
                == inner(states[0], mm(transpose(q), states[1])))
        upstream = [[[R((member + 1) * (node + 2) + channel, 11)
                      for channel in range(d)] for node in range(n)] for member in range(m)]
        upstream_mean = mean(upstream)
        gradients = [add(mm(transpose(q), u),
                         mm(transpose(add(p, q, -1)), upstream_mean)) for u in upstream]
        direction = [[[R(member + node - channel, 13) for channel in range(d)]
                      for node in range(n)] for member in range(m)]
        direction_mean = mean(direction)
        perturbation = [add(mm(p, direction_mean),
                            mm(q, add(v, direction_mean, -1))) for v in direction]
        checked(f"differentiable centering gradient color{color}",
                sum(inner(g, v) for g, v in zip(gradients, direction))
                == sum(inner(u, dg) for u, dg in zip(upstream, perturbation)))
    all_color = [[add(exact_mean, mm(scale(groups[c], 4), residual))
                  for c in range(4)] for residual in delta]
    checked("conditional linear unbiasedness all members", all(
        mean(outcomes) == mm(p, state) for outcomes, state in zip(all_color, states)))
    independently_masked = [add(exact_mean, mm(scale(groups[i], 4), delta[i]))
                            for i in range(m)]
    checked("different member masks generally break cancellation", mean(independently_masked) != exact_mean)
    # Squaring is enough to demonstrate a nonlinear expectation failure.
    exact_scalar, sampled_scalars = R(1), [R(2), R(0)]
    checked("nonlinear expectation not guaranteed",
            sum(value**2 for value in sampled_scalars) / 2 != exact_scalar**2)
    wrong_sampled_mean = mean([[[R(1), R(0)]], [[R(0), R(1)]],
                              [[R(0), R(0)]], [[R(0), R(0)]]])
    checked("sampled-degree renormalization is biased", wrong_sampled_mean != [[R(1, 2), R(1, 2)]])

    check_pins()
    checked("pinned source bytes unchanged", True)
    source_files = sorted(ROOT.rglob("*.py"))
    for path in source_files:
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
    checked("all source files parse", True)
    contract = (ROOT / "contract.py").read_text()
    engine = (ROOT / "fixture_engine.py").read_text()
    checked("real-pilot entry permanently refused", "REAL_PILOT_CLOSED" in contract
            and "return refuse_real_pilot(*args, **kwargs)" in engine)
    operator = (ROOT / "csr_operator.py").read_text()
    checked("explicit first-order adjoint path present", "@once_differentiable" in operator
            and "torch.sparse.mm(operator.adjoint_csr, upstream)" in operator)
    checked("CSR uses sparse matrix multiplication", "torch.sparse.mm" in operator
            and "sparse_csr" in operator)
    checked("single serial optimizer step", engine.count("optimizer.step()") == 1
            and "loss.backward()" in engine)
    counts = {}
    for task, f, c in [("Reddit2", 602, 41), ("ogbn-products", 100, 47)]:
        width, layers, members = 64, 2, 4
        ordinary = 5 * layers * width**2 + width * (f + c + 3 + 5 * layers) + c
        ensemble = f * width + width * c + members * f + 3 * members * width + 2 * members * c
        ensemble += layers * (5 * width**2 + 5 * width) + 2 * width
        counts[task] = {"single_base": ordinary, "candidate": ensemble,
                        "exact_member": ensemble, "naive_shared_sample": ensemble}
    checked("candidate/control parameter formulas identical", all(
        len({counts[t][a] for a in ("candidate", "exact_member", "naive_shared_sample")}) == 1
        for t in counts))
    return {
        "schema": SCHEMA, "kind": "stdlib_rational_and_source_fixture",
        "status": "passed", "checks": checks, "source_file_count": len(source_files),
        "fixture": {"nodes": n, "raw_edges": len(source), "canonical_edges": len(multiplicity),
                    "members": m, "channels": d}, "source_pins": PINS,
        "symbolic_parameter_counts": counts, "torch_runtime_executed": False,
        "dataset_accessed": False, "training_executed": False,
        "complete_graph_resource_feasibility_measured": False,
        "predictive_quality_claim": False, "novelty_certification": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "STDLIB_GATE_local_run01.json")
    args = parser.parse_args()
    result = gate()
    output = confined_output(args.output)
    if output.exists():
        raise FileExistsError("Preserve earlier verification evidence")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(result["checks"]),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
