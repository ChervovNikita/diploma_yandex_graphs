"""TRAIN query identities, deterministic responsibility and bounded contrast plans.

No graph/negative sampler, model, numerical library, or global RNG is touched.
The caller supplies exactly the rows already used by the native task update.
"""
import hashlib
import heapq
import json
import math


def _hash(seed, *parts):
    raw = json.dumps([int(seed), *parts], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8, person=b'link-ext-v1').digest(), 'big')


def query_keys(endpoints, relation_ids=None, directed=False):
    if relation_ids is None:
        relation_ids = [0] * len(endpoints)
    if len(relation_ids) != len(endpoints):
        raise ValueError('Relation/query row counts differ')
    result = []
    for pair, relation in zip(endpoints, relation_ids):
        if len(pair) != 2:
            raise ValueError('Every native query must have exactly two endpoints')
        u, v = map(int, pair)
        if u < 0 or v < 0:
            raise ValueError('Nonnegative native endpoint IDs required')
        if not directed and u > v:
            u, v = v, u
        result.append((u, v, int(relation)))
    return result


def authenticate(keys, groups, role):
    if role != 'TRAIN':
        raise ValueError('Objective construction accepts TRAIN queries only')
    if not keys or len(keys) != len(groups) or any(int(g) not in (0, 1) for g in groups):
        raise ValueError('Aligned nonempty native binary query groups required')
    raw = json.dumps([keys, [int(g) for g in groups]], separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def raw_responsibilities(keys, seed, mode='endpoint', members=4, floor=.25):
    """Fixed positive weights; normalization belongs to the native loss groups.

    Endpoint weights correlate responsibility across queries incident to a
    TRAIN node. Query-IID weights are the matched unstructured control.
    The source sampler's binary label is deliberately absent from the hash.
    """
    if mode not in ('endpoint', 'query') or floor < 0:
        raise ValueError('Unknown responsibility law or negative floor')
    def exponential(*parts):
        # 52-bit cell midpoints remain representably inside (0,1) in FP64.
        unit = ((_hash(seed, *parts) >> 12) + .5) / (2 ** 52)
        return -math.log(unit)
    result = []
    for member in range(members):
        row = []
        endpoint_cache = {}
        def endpoint_value(relation, endpoint):
            key = (relation, endpoint)
            if key not in endpoint_cache:
                endpoint_cache[key] = exponential('endpoint', member, relation, endpoint)
            return endpoint_cache[key]
        for u, v, relation in keys:
            value = (.5 * (endpoint_value(relation, u) + endpoint_value(relation, v))
                     if mode == 'endpoint' else exponential('query', member, u, v, relation))
            row.append(floor + value)
        result.append(row)
    return result


def contrast_plan(keys, groups, seed, mode='endpoint', anchor_limit=1024, negative_limit=16):
    """Only existing native query rows enter the auxiliary denominator.

    Positives are the SAME row/semantic query in another existing member view.
    Endpoint negatives have the same relation, opposite sampled/observed label,
    and at least one common endpoint. No new pair or graph edge is generated.
    """
    if mode not in ('endpoint', 'global', 'cosine') or anchor_limit < 2 or negative_limit < 1:
        raise ValueError('Unsupported bounded contrast plan')
    if len(keys) != len(groups):
        raise ValueError('Native query/group rows differ')
    groups = [int(g) for g in groups]
    label_rows = {g: [i for i, value in enumerate(groups) if value == g] for g in (0, 1)}
    # A declared balanced AUXILIARY anchor subset; native task loss uses ALL rows.
    chosen = []
    for group in (0, 1):
        chosen += heapq.nsmallest(anchor_limit // 2, label_rows[group],
                                 key=lambda i: (_hash(seed, 'anchor', keys[i]), i))
    buckets = {}
    global_rows = {}
    for i, (u, v, relation) in enumerate(keys):
        if mode == 'global':
            global_rows.setdefault((relation, groups[i]), []).append(i)
        if mode == 'endpoint':
            for endpoint in set((u, v)):
                buckets.setdefault((relation, endpoint, groups[i]), []).append(i)
    for pool in list(buckets.values()) + list(global_rows.values()):
        pool.sort(key=lambda j: (keys[j], j))
    def sample_existing(pools, anchor):
        # Bind each cycle independently, avoiding late-bound generator closures.
        def cycle(pool, start, stride):
            return (pool[(start+step*stride) % len(pool)] for step in range(len(pool)))
        streams = []
        for index, pool in enumerate(pools):
            if pool:
                start = _hash(seed, 'start', keys[anchor], index) % len(pool)
                stride = 1 + _hash(seed, 'stride', keys[anchor], index) % len(pool)
                while math.gcd(stride, len(pool)) != 1:
                    stride = stride % len(pool) + 1
                streams.append(cycle(pool, start, stride))
        chosen_rows, seen = [], set()
        while streams and len(chosen_rows) < negative_limit:
            for stream in tuple(streams):
                try:
                    row = next(stream)
                except StopIteration:
                    streams.remove(stream)
                    continue
                if row not in seen and keys[row] != keys[anchor]:
                    seen.add(row)
                    chosen_rows.append(row)
                    if len(chosen_rows) == negative_limit:
                        break
        return chosen_rows
    anchors, alternatives = [], []
    uncovered = {0: 0, 1: 0}
    for i in chosen:
        u, v, relation = keys[i]
        if mode == 'cosine':
            candidates = []
        elif mode == 'endpoint':
            candidates = sample_existing([buckets.get((relation, u, 1-groups[i]), ()),
                                          buckets.get((relation, v, 1-groups[i]), ())], i)
        else:
            candidates = sample_existing([global_rows.get((relation, 1-groups[i]), ())], i)
        if mode != 'cosine':
            if not candidates:
                uncovered[groups[i]] += 1
                continue
        anchors.append(i)
        alternatives.append(candidates)
    coverage = {str(g): {'candidate_anchors': sum(groups[i] == g for i in chosen),
                         'used_anchors': sum(groups[i] == g for i in anchors),
                         'no_compatible_native_alternative': uncovered[g]} for g in (0, 1)}
    return {'mode': mode, 'anchors': anchors, 'alternatives': alternatives,
            'coverage_by_native_label': coverage, 'anchor_limit': anchor_limit,
            'negative_limit': negative_limit, 'native_query_count': len(keys),
            'alternative_count': sum(map(len, alternatives)),
            'selection': 'deterministic hash-cyclic bounded existing-query buckets',
            'positive_pair_rule': 'same authenticated query in another existing route',
            'sampled_negative_is_verified_nonlink': False,
            'new_graph_or_query_pairs_created': 0}


def match_endpoint_support(reference, control):
    """Match auxiliary anchor coverage and negative count for decisive controls."""
    if reference['mode'] != 'endpoint' or control['mode'] not in ('global', 'cosine'):
        raise ValueError('Endpoint reference and cosine/global control required')
    by_anchor = dict(zip(control['anchors'], control['alternatives']))
    alternatives = []
    for anchor, local_rows in zip(reference['anchors'], reference['alternatives']):
        if anchor not in by_anchor:
            raise ValueError('Control cannot cover an endpoint-reference anchor')
        rows = by_anchor[anchor]
        if control['mode'] == 'global' and len(rows) < len(local_rows):
            raise ValueError('Global control has fewer alternatives than endpoint reference')
        alternatives.append(rows[:len(local_rows)] if control['mode'] == 'global' else [])
    result = dict(control)
    result.update(anchors=list(reference['anchors']), alternatives=alternatives,
                  coverage_by_native_label=reference['coverage_by_native_label'],
                  alternative_count=sum(map(len, alternatives)),
                  support_matched_to='endpoint plan with same seed/anchor/alternative budget')
    return result
