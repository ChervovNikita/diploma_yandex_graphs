"""Authorized two-node exact arithmetic only; no model or data imports."""
from fractions import Fraction as F
from math import comb
from pathlib import Path
import hashlib
import json

BASE = Path(__file__).resolve().parent
SPEC = json.loads((BASE / "EXACT_CHECK_SPEC.json").read_text())
BINDINGS = json.loads((BASE / "BEFORE_EXECUTION_BINDINGS.json").read_text())
for binding in BINDINGS["files"]:
    raw = (BASE / binding["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == binding["sha256"]


def add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scale(a, c):
    return [[c * x for x in row] for row in a]


def mul(a, b):
    return [[sum((a[i][r] * b[r][j] for r in range(len(b))), F(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def power(a, k):
    out = [[F(1), F(0)], [F(0), F(1)]]
    for _ in range(k):
        out = mul(out, a)
    return out


def serial(a):
    if isinstance(a, list):
        return [serial(x) for x in a]
    return str(a)


P = [[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]]
I = [[F(1), F(0)], [F(0), F(1)]]
X = [[F(x)] for x in SPEC["features"]]
s = F(SPEC["shared_scalar_s"])
members = [F(x) for x in SPEC["member_b"]]
probes = [F(x) for x in SPEC["rational_probe_points"]]
orders = SPEC["cached_orders"]
assert probes == [F(0), F(1, 2), F(1)]
assert orders == [0, 1, 2]
assert mul(P, P) == P
cached = [mul(power(P, k), X) for k in orders]
derivative_P = add(I, scale(P, F(-1)))
derivative_P2_minus_P = add(
    add(mul(derivative_P, P), mul(P, derivative_P)),
    scale(derivative_P, F(-1)),
)
derivatives = []
for b in members:
    direct = mul(scale(derivative_P2_minus_P, b), X)
    expected = scale(X, -b)
    assert direct == expected
    derivatives.append({"b": str(b), "dz_dt_at_zero": serial(direct)})

points = []
for t in probes:
    pt = add(scale(P, 1 - t), scale(I, t))
    w = (1 - t) / (1 + t)
    realized = scale([[F(1), w], [w, F(1)]], F(1) / (1 + w))
    assert realized == pt
    assert mul(pt, X) == scale(X, t)
    nonlinear_term = add(mul(pt, pt), scale(pt, F(-1)))
    assert nonlinear_term == scale(add(I, scale(P, F(-1))), t * t - t)
    members_at_t = []
    maps = []
    for b in members:
        full_map = add(scale(I, s), scale(nonlinear_term, b))
        logits = mul(full_map, X)
        assert logits == scale(X, s + b * (t * t - t))
        if t in (F(0), F(1)):
            assert full_map == scale(I, s)
            assert logits == scale(X, s)
        maps.append(full_map)
        members_at_t.append({"b": str(b), "full_feature_map": serial(full_map),
                             "logits": serial(logits)})
    if t == F(1, 2):
        assert len({tuple(tuple(row) for row in mul(a, X)) for a in maps}) == len(members)
    mean_map = scale(I, F(0))
    for a in maps:
        mean_map = add(mean_map, scale(a, F(1, len(members))))
    mean_b = sum(members, F(0)) / len(members)
    assert mean_map == add(scale(I, s), scale(nonlinear_term, mean_b))
    token_records = []
    for k in orders:
        mixed = scale(X, F(0))
        for j in range(k + 1):
            mixed = add(mixed, scale(cached[j], F(comb(k, j)) * (1 - t) ** j * t ** (k - j)))
        direct = mul(power(pt, k), X)
        assert direct == mixed
        token_records.append({"order": k, "direct": serial(direct), "cached_mix": serial(mixed)})
    points.append({"t": str(t), "nonloop_edge_weight_w": str(w), "normalized_P": serial(pt),
                   "members": members_at_t, "tokens": token_records,
                   "mean_feature_map": serial(mean_map), "mean_logits": serial(mul(mean_map, X))})

print(json.dumps({"schema": "two-node-exact-fraction-check-output-v1", "status": "pass",
                  "arithmetic": "stdlib fractions.Fraction; exact rational equality",
                  "preexecution_bindings_verified": True,
                  "spec_sha256": hashlib.sha256((BASE / "EXACT_CHECK_SPEC.json").read_bytes()).hexdigest(),
                  "points": points, "derivatives": derivatives,
                  "limitations": [SPEC["probability_limit"], SPEC["identification_limit"],
                                   "Finite two-node algebra check only; no predictive utility, numerical stability, general implementation, or novelty certification."]}, indent=2))
