"""Bounded factual query contexts from a cached simple native TRAIN graph."""
from __future__ import annotations
import hashlib
import heapq
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tensor_pin(value):
    x = value.detach().cpu().contiguous()
    return {'shape': list(x.shape), 'dtype': str(x.dtype), 'bytes_sha256': hashlib.sha256(x.numpy().tobytes()).hexdigest()}


def select(planning, keys, groups, *, seed, nodes, role='TRAIN'):
    """Authenticate all source rows; AUX keeps unique unambiguous nonself keys."""
    signature = planning.authenticate(keys, groups, role)
    if type(seed) is not int or type(nodes) is not int or nodes < 1:
        raise ValueError('Actual native fit seed and node count required')
    first, seen_groups, self_rows = {}, {}, 0
    for row, (key, group) in enumerate(zip(keys, groups)):
        u, v, relation = key
        if not (0 <= u <= v < nodes) or relation < 0:
            raise ValueError('Canonical actual query IDs required')
        if u == v:
            self_rows += 1
            continue
        first.setdefault((key, int(group)), row)
        seen_groups.setdefault(key, set()).add(int(group))
    conflicts = {key for key, values in seen_groups.items() if len(values) > 1}
    selected, unique_counts = [], {}
    for group in (0, 1):
        rows = [row for (key, g), row in first.items() if g == group and key not in conflicts]
        unique_counts[str(group)] = len(rows)
        selected += heapq.nsmallest(512, rows, key=lambda i: (planning._hash(seed, 'anchor', keys[i]), i))
    selected.sort()
    chosen_keys = [keys[i] for i in selected]
    ids = [((relation*nodes+u)*nodes+v) for u, v, relation in chosen_keys]
    if any(value > 2**63-1 for value in ids) or len(ids) != len(set(ids)):
        raise ValueError('Collision-free signed-int64 semantic IDs required')
    conflicting_rows = sum(key in conflicts for key in keys)
    eligible_rows = len(keys)-self_rows-conflicting_rows
    unique_nonconflicting = sum(unique_counts.values())
    receipt = {'schema': 'context-affinity-link-cohort-v1', 'source_authentication': signature,
        'source_records': len(keys), 'source_group_counts': {str(g): sum(int(x) == g for x in groups) for g in (0, 1)},
        'self_rows_omitted_from_AUX': self_rows, 'conflicting_semantic_objects': len(conflicts),
        'conflicting_source_rows_omitted_from_AUX': conflicting_rows,
        'same_group_duplicate_rows_omitted_from_AUX': eligible_rows-unique_nonconflicting,
        'unique_nonconflicting_by_group': unique_counts, 'selected_rows': selected,
        'selected_keys': [list(key) for key in chosen_keys], 'selected_groups': [int(groups[i]) for i in selected],
        'selected_object_ids': ids, 'limit_per_source_group': 512,
        'selected_by_group': {str(g): sum(int(groups[i]) == g for i in selected) for g in (0, 1)},
        'ordering': 'native row order after frozen link anchor hash selection',
        'native_F_rows_removed': 0, 'new_query_pairs': 0, 'sampled_negative_is_verified_nonlink': False}
    receipt['sha256'] = digest(receipt)
    return selected, ids, receipt


class TrainGraph:
    """One CSR per fit; query-specific contexts are never globally cached."""
    def __init__(self, torch, positive, nodes):
        import numpy as np
        edge = positive.detach().cpu().contiguous()
        if edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype != torch.long or nodes < 1:
            raise ValueError('Complete native TRAIN positive [2,E] records required')
        if ((edge < 0) | (edge >= nodes)).any():
            raise ValueError('TRAIN endpoints outside actual graph')
        raw = edge.numpy(); mask = raw[0] != raw[1]
        src, dst = raw[0, mask], raw[1, mask]
        row, col = np.concatenate((src, dst)), np.concatenate((dst, src))
        if nodes*nodes > 2**63-1:
            raise ValueError('Signed-int64 collision-free CSR row/column encoding required')
        # Sort/unique int64 coordinates directly: no bounded integer sum can
        # overflow while collapsing arbitrarily repeated native records.
        encoded = np.unique(row*np.int64(nodes)+col)
        rows, self.indices = encoded//nodes, encoded % nodes
        self.indptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(np.bincount(rows, minlength=nodes), dtype=np.int64)))
        self.degree = np.diff(self.indptr); self.nodes = nodes
        self.receipt = {'schema': 'simple-native-TRAIN-target-CSR-v1', 'source_TRAIN_records': tensor_pin(edge),
            'nodes': nodes, 'source_loop_records_removed': int((~mask).sum()),
            'simple_undirected_edges': len(self.indices)//2, 'CSR_directed_entries': len(self.indices),
            'CSR_indptr_sha256': hashlib.sha256(self.indptr.tobytes()).hexdigest(),
            'CSR_indices_sha256': hashlib.sha256(self.indices.tobytes()).hexdigest(),
            'degree_sha256': hashlib.sha256(self.degree.tobytes()).hexdigest(),
            'target_population': 'all original TRAIN positive records only; loops removed and duplicates coalesced',
            'query_exclusion': 'only current pair, equally for both source groups',
            'full_dense_adjacency': False, 'unbounded_query_context_cache': False}
        self.receipt['sha256'] = digest(self.receipt)

    def contexts(self, torch, keys, dtype):
        import numpy as np
        a, b, details = [], [], []
        for u, v, relation in keys:
            if not (0 <= u < v < self.nodes) or relation < 0:
                raise ValueError('Bounded unique nonself semantic queries required')
            nu = self.indices[self.indptr[u]:self.indptr[u+1]]
            nv = self.indices[self.indptr[v]:self.indptr[v+1]]
            i = np.searchsorted(nu, v); present = bool(i < len(nu) and nu[i] == v)
            du, dv = int(self.degree[u])-int(present), int(self.degree[v])-int(present)
            witnesses = np.intersect1d(nu, nv, assume_unique=True)
            witnesses = witnesses[(witnesses != u) & (witnesses != v)]
            degree = self.degree[witnesses]
            if len(degree) and (degree < 2).any():
                raise ValueError('Distinct common witness requires degree at least two')
            aa = float((1/np.log(degree.astype(np.float64))).sum()) if len(degree) else 0.
            ra = float((1/degree.astype(np.float64)).sum()) if len(degree) else 0.
            a.append(sorted((math.log1p(du), math.log1p(dv))))
            b.append([math.log1p(len(witnesses)), math.log1p(aa), math.log1p(ra)])
            details.append({'pair_present_in_full_TRAIN_graph': present, 'degree_without_pair': sorted((du, dv)),
                'CN': len(witnesses), 'AA_sum': aa, 'RA_sum': ra})
        A = torch.tensor(a, dtype=dtype).reshape(len(keys), 2)
        B = torch.tensor(b, dtype=dtype).reshape(len(keys), 3)
        receipt = {'schema': 'own-query-excluded-link-context-v1', 'target_graph_sha256': self.receipt['sha256'],
            'queries': [list(key) for key in keys], 'query_statistics': details, 'A': tensor_pin(A), 'B': tensor_pin(B),
            'A_coordinates': 'sorted log1p endpoint degrees after own-pair exclusion',
            'B_coordinates': ['log1p(CN)', 'log1p(AA_sum)', 'log1p(RA_sum)'],
            'witness_degrees': 'unchanged by excluding own pair, because witnesses differ from both endpoints',
            'zero_A': int((A.norm(dim=-1) <= 1e-12).sum()), 'zero_B': int((B.norm(dim=-1) <= 1e-12).sum())}
        receipt['sha256'] = digest(receipt)
        return A, B, receipt


def target_diagnostics(torch, a, b, labels, ids, relations):
    compatible = labels[:, None].eq(labels[None, :]) & relations[:, None].eq(relations[None, :]) & ~ids[:, None].eq(ids[None, :])
    support = compatible.sum(-1); eligible = support > 0
    info = {'objects': len(labels), 'supported_anchors': int(eligible.sum()), 'omitted_anchors': int((~eligible).sum()),
        'zero_A': int((a.norm(dim=-1) <= 1e-12).sum()), 'zero_B': int((b.norm(dim=-1) <= 1e-12).sum()),
        'by_source_group': {str(g): {'objects': int((labels == g).sum()),
            'supported': int((eligible & (labels == g)).sum()), 'zero_B': int(((b.norm(dim=-1) <= 1e-12) & (labels == g)).sum())} for g in (0, 1)}}
    if not eligible.any():
        return info
    targets = []
    for name, value in (('A', a), ('B', b)):
        z = torch.nn.functional.normalize(value, dim=-1, eps=1e-12)
        q = ((z[eligible] @ z.T)/.2).masked_fill(~compatible[eligible], -torch.inf).softmax(-1)
        uniform = compatible[eligible].to(q)/support[eligible, None]
        info[name+'_nonuniform_anchors_at_1e_8'] = int(((q-uniform).abs().max(-1).values > 1e-8).sum())
        targets.append(q)
    difference = (targets[0]-targets[1]).abs()
    info.update(A_B_target_max_abs_difference=float(difference.max()), A_B_target_mean_abs_difference=float(difference.mean()),
        distinct_target_anchors_at_1e_8=int((difference.max(-1).values > 1e-8).sum()))
    return info
