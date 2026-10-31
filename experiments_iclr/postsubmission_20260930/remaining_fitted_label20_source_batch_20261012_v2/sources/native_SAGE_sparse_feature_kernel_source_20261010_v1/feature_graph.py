"""Prospective sparse support/kernel helper; caller supplies Torch. No runtime on import."""
from contextlib import contextmanager
import hashlib
import inspect
import math
from pathlib import Path
from types import MethodType


K = 20
RANK = 16
TEMPERATURE = .2
NORMALIZE_EPS = 1e-12
METRIC_FLOOR = 1e-4
KERNEL_SEED_OFFSET = 5000011


def build_support(torch, raw_x, original_edges):
    """One label-free, deterministic incoming raw-cosine top20 candidate union.

    Rows of raw_x are the existing canonical node IDs. Stable descending sort
    retains ascending source ID for equal scores. Original loops are retained;
    feature candidates exclude self. Duplicate original edges are rejected,
    since silently deduplicating them could change the original mean operator.
    """
    if raw_x.shape != (11701, 300) or raw_x.dtype != torch.float32:
        raise ValueError('Existing full11701x300 float32 X required')
    if original_edges.ndim != 2 or original_edges.shape[0] != 2 or original_edges.dtype != torch.long:
        raise ValueError('Original COO int64 edge_index required')
    if original_edges.device != raw_x.device or not torch.isfinite(raw_x).all().item():
        raise ValueError('Finite same-device original graph/features required')
    n = raw_x.shape[0]
    if original_edges.numel() == 0 or original_edges.min().item() < 0 or original_edges.max().item() >= n:
        raise ValueError('Nonempty original node-index support required')
    with torch.no_grad():
        original_keys = original_edges[1] * n + original_edges[0]
        if torch.unique(original_keys).numel() != original_keys.numel():
            raise ValueError('Original prepared graph must have unique directed pairs')
        unit = torch.nn.functional.normalize(raw_x, dim=-1, eps=NORMALIZE_EPS)
        sources, destinations = [], []
        for start in range(0, n, 256):
            stop = min(n, start + 256)
            scores = unit[start:stop] @ unit.transpose(0, 1)
            local_rows = torch.arange(stop-start, device=raw_x.device)
            node_rows = torch.arange(start, stop, device=raw_x.device)
            scores[local_rows, node_rows] = float('-inf')
            nearest = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :K]
            sources.append(nearest.reshape(-1))
            destinations.append(node_rows.repeat_interleave(K))
        candidates = torch.stack([torch.cat(sources), torch.cat(destinations)])
        candidate_keys = candidates[1] * n + candidates[0]
        keys = torch.unique(torch.cat([original_keys, candidate_keys]), sorted=True)
        new_edges = int((~torch.isin(keys, original_keys)).sum().item())
        if new_edges == 0:
            raise ValueError('Feature candidates add no genuinely absent directed message path')
        support = torch.stack([keys.remainder(n), keys.div(n, rounding_mode='floor')])
        metadata = {'nodes': n, 'raw_feature_width': raw_x.shape[1], 'candidate_k': K,
                    'candidate_directed_pairs': n*K, 'original_directed_pairs': original_edges.shape[1],
                    'union_directed_pairs': support.shape[1], 'new_directed_pairs': new_edges,
                    'original_self_loops': int(original_edges[0].eq(original_edges[1]).sum().item()),
                    'union_self_loops': int(support[0].eq(support[1]).sum().item()),
                    'raw_cosine_dot_MACs': n*n*raw_x.shape[1], 'raw_cosine_scores': n*n,
                    'sorted_rows': n, 'stable_sort_candidate_entries': n*n,
                    'construction': 'incoming top20 raw cosine; self excluded; source-ID ties; sorted union',
                    'labels_or_predictions_used': False}
    return support, metadata


def install(torch, body, model, members, mode, kernel_seed):
    """Attach one registered kernel and replace only native SAGE incoming mean.

    Install after the original dense-factor wrapping and before .to/optimizer.
    The original SAGEConv objects and lin_l/lin_r Parameter objects are retained.
    Closures reference the kernel without registering duplicate module aliases.
    """
    if mode not in ('private', 'common', 'fixed') or members not in (1, 4):
        raise ValueError('One fixed private/common/fixed kernel recipe, M1 or M4')
    if hasattr(model, '_feature_kernel') or body.model_name != 'SAGE':
        raise ValueError('Fresh native SAGE body/model required')
    convs = [r.module.conv for r in body.residual_modules]
    if len(convs) != 2:
        raise ValueError('Exact native two-SAGE-layer recipe required')
    for conv in convs:
        if (conv.__class__.__name__ != 'SAGEConv' or conv.__class__.__module__ != 'torch_geometric.nn.conv.sage_conv'
                or conv.aggr != 'mean' or conv.flow != 'source_to_target'
                or conv.project or conv.normalize or not conv.root_weight
                or getattr(conv, 'node_dim', -2) != -2
                or getattr(conv, 'decomposed_layers', 1) != 1 or getattr(conv, 'explain', False)
                or getattr(conv, '_feature_mean_installed', False)):
            raise ValueError('Only original homogeneous nonprojected/unnormalized mean SAGE is supported')
        if conv.lin_l.bias is None or conv.lin_r.bias is not None:
            raise ValueError('Preserve biased neighbor and bias-free root maps')
    native_path = inspect.getsourcefile(type(convs[0]))
    if native_path is None:
        raise ValueError('Materialized original PyG SAGE source is required for custody')
    native_bytes = Path(native_path).read_bytes()
    native_binding = {'path': str(Path(native_path).resolve()), 'bytes': len(native_bytes),
                      'sha256': hashlib.sha256(native_bytes).hexdigest(),
                      'class': type(convs[0]).__module__ + '.SAGEConv'}
    nn = torch.nn

    class Kernel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode, self.members, self.seed = mode, members, kernel_seed
            self.rows = members if mode == 'private' else 1
            self._current_weights, self._current_edges = None, None
            self.counts = {'contexts': 0, 'projection_calls': 0, 'projection_MACs': 0,
                           'metric_route_rows': 0, 'normalized_coordinates': 0,
                           'cosine_pairs': 0, 'cosine_MACs': 0, 'row_softmax_entries': 0,
                           'fixed_uniform_entries': 0, 'weighted_mean_calls': 0,
                           'incoming_edge_visits': 0, 'weighted_message_coordinates': 0}
            if mode == 'fixed':
                self.register_parameter('projection', None)
                self.register_parameter('metric_raw', None)
            else:
                generator = torch.Generator(device='cpu').manual_seed(kernel_seed)
                value = torch.empty(300, RANK, dtype=torch.float32, device='cpu')
                value.uniform_(-1/math.sqrt(300), 1/math.sqrt(300), generator=generator)
                self.projection = nn.Parameter(value)
                # softplus(raw)+floor starts at one; local construction consumes no global RNG.
                raw = math.log(math.expm1(1-METRIC_FLOOR))
                self.metric_raw = nn.Parameter(torch.full((self.rows, RANK), raw, dtype=torch.float32))

        @contextmanager
        def context(self, raw_x, edges):
            if self._current_weights is not None:
                raise ValueError('One complete factual forward context at a time')
            n, e = raw_x.shape[0], edges.shape[1]
            src, dst = edges[0], edges[1]
            self.counts['contexts'] += 1
            if self.mode == 'fixed':
                degree = torch.bincount(dst, minlength=n).to(raw_x.dtype)
                if (degree == 0).any().item():
                    raise ValueError('Feature candidate support must cover every destination')
                weights = (degree[dst].reciprocal()).unsqueeze(0)
                self.counts['fixed_uniform_entries'] += e
            else:
                embedding = raw_x @ self.projection
                metric = torch.nn.functional.softplus(self.metric_raw) + METRIC_FLOOR
                q = torch.nn.functional.normalize(embedding[None] * metric[:, None], dim=-1, eps=NORMALIZE_EPS)
                scores = (q[:, dst] * q[:, src]).sum(-1) / TEMPERATURE
                if not torch.isfinite(scores).all().item():
                    raise ValueError('Finite kernel scores required; no clamp or repair')
                index = dst[None].expand(self.rows, -1)
                maxima = scores.new_full((self.rows, n), float('-inf'))
                maxima.scatter_reduce_(1, index, scores.detach(), reduce='amax', include_self=True)
                numerators = (scores - maxima.gather(1, index)).exp()
                totals = scores.new_zeros((self.rows, n)).scatter_add_(1, index, numerators)
                weights = numerators / totals.gather(1, index)
                self.counts['projection_calls'] += 1
                self.counts['projection_MACs'] += n*300*RANK
                self.counts['metric_route_rows'] += self.rows
                self.counts['normalized_coordinates'] += self.rows*n*RANK
                self.counts['cosine_pairs'] += self.rows*e
                self.counts['cosine_MACs'] += self.rows*e*RANK
                self.counts['row_softmax_entries'] += self.rows*e
            self._current_weights, self._current_edges = weights, edges
            try:
                yield
            finally:
                # No learned-weight cache survives an update, validation or restoration.
                self._current_weights, self._current_edges = None, None

        def mean(self, hidden, edges, member):
            if self._current_weights is None or edges is not self._current_edges:
                raise ValueError('Weighted SAGE requires this same full live feature-support context')
            if hidden.ndim != 2 or hidden.dtype != torch.float32 or hidden.shape[0] != 11701:
                raise ValueError('Original homogeneous float32 node states required')
            if member < 0 or member >= self.members:
                raise ValueError('Existing native member index is out of range')
            weights = self._current_weights[member if self.mode == 'private' else 0]
            src, dst = edges[0], edges[1]
            result = hidden.new_zeros(hidden.shape)
            result.index_add_(0, dst, hidden[src] * weights[:, None])
            self.counts['weighted_mean_calls'] += 1
            self.counts['incoming_edge_visits'] += edges.shape[1]
            self.counts['weighted_message_coordinates'] += edges.shape[1]*hidden.shape[1]
            return result

        def describe(self):
            return {'mode': self.mode, 'members': self.members, 'metric_rows': self.rows,
                    'rank': RANK, 'temperature': TEMPERATURE, 'normalize_eps': NORMALIZE_EPS,
                    'metric_floor': METRIC_FLOOR, 'kernel_seed': self.seed,
                    'native_PyG_source': native_binding,
                    'registered_parameters': sum(p.numel() for p in self.parameters()),
                    'projection_shared_within_bank': self.mode != 'fixed',
                    'weights_reused_across_native_layers': 2,
                    'score_inputs': 'raw X only; no labels, predictions, learned native hidden inputs',
                    'counters': dict(self.counts)}

    kernel = Kernel()
    model.add_module('_feature_kernel', kernel)
    for conv in convs:
        def weighted_forward(this, x, edge_index, size=None, _kernel=kernel):
            if size is not None or not torch.is_tensor(x) or x.ndim != 2:
                raise ValueError('No bipartite/custom-size/fused SAGE path in this extension')
            member = getattr(this.lin_l, 'member', 0)
            mean = _kernel.mean(x, edge_index, member)
            return this.lin_l(mean) + this.lin_r(x)
        conv.forward = MethodType(weighted_forward, conv)
        conv._feature_mean_installed = True
    return kernel
