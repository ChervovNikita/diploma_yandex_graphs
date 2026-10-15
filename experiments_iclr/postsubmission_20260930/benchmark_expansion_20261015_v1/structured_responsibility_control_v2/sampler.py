"""Persistent positive endpoint and marginal-matched query weights, version 2.

Pure stdlib. No native sampler, graph, model, labels, or global RNG is accessed.
Analytic covariance describes ideal independent Exp(1) latent variables; the
hash implementation is a deterministic finite pseudorandom approximation.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import operator

# NCN's native 65,536 positives PLUS 65,536 negatives must fit as one batch.
# The cap bounds materialization; it never licenses dropping or splitting groups.
QUERY_LIMIT = 262_144
MEMBER_LIMIT = 16
COVARIANCE_QUERY_LIMIT = 256
HASH_NAMESPACE = b"link-resp-v2"
SCHEMA = "structured-responsibility-v2"


def _integer(value, name, *, nonnegative=True):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True)
class QueryBatch:
    """Immutable persistent semantic identities, in the source's original order.

    scope identifies a fixed dataset/node-ID/relation-ID namespace; keep it the
    same across minibatches, epochs, and replays. It must not include batch/epoch,
    row position, source label, or a fit outcome. Directed reversals are distinct
    query identities, although this endpoint field is intentionally role agnostic.
    """
    keys: tuple
    scope: str
    directed: bool = False

    def __post_init__(self):
        if not isinstance(self.scope, str) or not self.scope.strip() or len(self.scope) > 512:
            raise ValueError("A nonempty stable scope of at most 512 characters is required")
        if not isinstance(self.directed, bool):
            raise ValueError("directed must be bool")
        if not isinstance(self.keys, tuple) or not 1 <= len(self.keys) <= QUERY_LIMIT:
            raise ValueError(f"QueryBatch requires 1..{QUERY_LIMIT} immutable keys")
        canonical = []
        for key in self.keys:
            if not isinstance(key, tuple) or len(key) != 3:
                raise ValueError("Each query key must be the tuple (u,v,relation)")
            u, v, relation = (_integer(value, "query ID") for value in key)
            if not self.directed and u > v:
                u, v = v, u
            canonical.append((u, v, relation))
        object.__setattr__(self, "keys", tuple(canonical))


def make_query_batch(endpoints, relation_ids=None, *, scope, directed=False):
    """Canonicalize caller-supplied existing source rows without sampling any."""
    if not 1 <= len(endpoints) <= QUERY_LIMIT:
        raise ValueError(f"Expected 1..{QUERY_LIMIT} existing native queries")
    if relation_ids is None:
        relation_ids = (0,) * len(endpoints)
    if len(relation_ids) != len(endpoints):
        raise ValueError("Relation and endpoint row counts differ")
    keys = []
    for pair, relation in zip(endpoints, relation_ids):
        if len(pair) != 2:
            raise ValueError("Each existing query must have exactly two endpoints")
        keys.append((pair[0], pair[1], relation))
    return QueryBatch(tuple(keys), scope, directed)


def binary_groups(groups, count):
    result = tuple(_integer(value, "source label") for value in groups)
    if len(result) != count or any(value not in (0, 1) for value in result):
        raise ValueError("Aligned native binary source labels are required")
    return result


def authenticate(batch, groups, *, role="TRAIN"):
    """Caller must compute this from each actual route capture's ordered rows."""
    if not isinstance(batch, QueryBatch) or role != "TRAIN":
        raise ValueError("An explicit QueryBatch and TRAIN role are required")
    labels = binary_groups(groups, len(batch.keys))
    payload = [SCHEMA, batch.scope, batch.directed, batch.keys, labels]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _exponential(seed, *parts):
    raw = json.dumps([seed, *parts], separators=(",", ":")).encode()
    number = int.from_bytes(hashlib.blake2b(raw, digest_size=8, person=HASH_NAMESPACE).digest(), "big")
    unit = ((number >> 12) + 0.5) / (2 ** 52)
    return -math.log(unit)


def _law(mode, floor):
    if mode not in ("endpoint", "query_iid", "unit"):
        raise ValueError("mode must be endpoint, query_iid, or unit")
    if isinstance(floor, bool) or not math.isfinite(floor) or floor < 0:
        raise ValueError("floor must be finite and nonnegative")


def raw_responsibilities(batch, seed, *, mode, members=4, floor=0.25):
    """Return [member, source row], with no minibatch/group normalization.

    query_iid is independent across distinct semantic queries in the ideal law,
    not across duplicated rows. Nonself queries use two independent Exp draws;
    loops use one. Endpoint latent keys are (scope,member,relation,node).
    The finite 52-bit midpoint implementation has 0 < Exp <= 53*log(2).
    """
    if not isinstance(batch, QueryBatch):
        raise ValueError("An explicit QueryBatch is required")
    _law(mode, floor)
    seed = _integer(seed, "seed", nonnegative=False)
    members = _integer(members, "members")
    if not 1 <= members <= MEMBER_LIMIT:
        raise ValueError(f"members must lie in 1..{MEMBER_LIMIT}")
    if mode == "unit":
        return tuple((1.0,) * len(batch.keys) for _ in range(members))
    result = []
    for member in range(members):
        cache = {}
        def draw(*identity):
            if identity not in cache:
                cache[identity] = _exponential(seed, batch.scope, member, *identity)
            return cache[identity]
        row = []
        for u, v, relation in batch.keys:
            if mode == "endpoint":
                value = 0.5 * (draw("endpoint", relation, u) + draw("endpoint", relation, v))
            elif u == v:
                value = draw("query", batch.directed, relation, u, v, "loop")
            else:
                value = 0.5 * (draw("query", batch.directed, relation, u, v, "a")
                               + draw("query", batch.directed, relation, u, v, "b"))
            row.append(float(floor) + value)
        result.append(tuple(row))
    return tuple(result)


def half_incidence_rows(batch):
    """Sparse B rows; a loop accumulates both halves into coefficient one."""
    if not isinstance(batch, QueryBatch):
        raise ValueError("An explicit QueryBatch is required")
    rows = []
    for u, v, relation in batch.keys:
        row = {}
        for node in (u, v):
            identity = (relation, node)
            row[identity] = row.get(identity, 0.0) + 0.5
        rows.append(row)
    return tuple(rows)


def ideal_raw_covariance(batch, *, mode):
    """Small synthetic/audit matrices only; never allocate graph-scale Q by Q."""
    _law(mode, 0.0)
    if len(batch.keys) > COVARIANCE_QUERY_LIMIT:
        raise ValueError(f"Dense covariance audit is limited to {COVARIANCE_QUERY_LIMIT} rows")
    incidence = half_incidence_rows(batch)
    result = []
    for i, left in enumerate(incidence):
        row = []
        for j, right in enumerate(incidence):
            if mode == "unit":
                value = 0.0
            elif mode == "endpoint":
                value = sum(coefficient * right.get(key, 0.0) for key, coefficient in left.items())
            else:
                value = (1.0 if batch.keys[i][0] == batch.keys[i][1] else 0.5) if batch.keys[i] == batch.keys[j] else 0.0
            row.append(value)
        result.append(tuple(row))
    return tuple(result)
