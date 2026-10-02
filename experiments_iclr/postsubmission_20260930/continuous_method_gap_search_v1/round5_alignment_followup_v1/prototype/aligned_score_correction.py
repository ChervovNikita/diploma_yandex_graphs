"""SOURCE-ONLY PROTOTYPE: authored, never imported or executed in this phase.

Frozen teacher interface: FP32 logits [K,N,C], canonical int64 undirected
edges [2,E] with u<v, fixed APS uniforms [N,C], fixed node permutations [N,K].
No dataset, teacher fitting, model loading, file access or launch entry point.
Native CF-GNN remains a separately pinned baseline; this module is a declared
score-space variant, not a reproduction of its TPS training implementation.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class SourceLabels:
    """Only one admitted source role's labels, never a full graph label vector."""
    nodes: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class FixedInputs:
    point_probabilities: torch.Tensor
    node_features: torch.Tensor
    base_scores: torch.Tensor
    edges: torch.Tensor
    degree: torch.Tensor
    aligned_features: torch.Tensor
    shuffled_features: torch.Tensor
    members: int
    classes: int


def _check_source_roles(roles, pool_nodes, nodes, classes):
    if pool_nodes.dtype != torch.int64 or pool_nodes.ndim != 1:
        raise ValueError('Pool nodes must be an int64 vector')
    if bool(((pool_nodes < 0) | (pool_nodes >= nodes)).any()):
        raise ValueError('Pool index outside score table')
    if pool_nodes.unique().numel() != pool_nodes.numel():
        raise ValueError('Pool nodes must be unique')
    occupied = torch.zeros(nodes, dtype=torch.bool, device=pool_nodes.device)
    occupied[pool_nodes] = True
    for role in roles:
        if role.nodes.dtype != torch.int64 or role.labels.dtype != torch.int64:
            raise ValueError('Role node and class indices must be int64')
        if role.nodes.device != pool_nodes.device or role.labels.device != pool_nodes.device:
            raise ValueError('Role tensors must share device')
        if bool(((role.nodes < 0) | (role.nodes >= nodes)).any()):
            raise ValueError('Source node index outside score table')
        if role.nodes.ndim != 1 or role.labels.shape != role.nodes.shape:
            raise ValueError('Role needs one class label per node')
        if role.nodes.numel() == 0 or role.nodes.unique().numel() != role.nodes.numel():
            raise ValueError('Role must be nonempty with unique nodes')
        if bool((occupied[role.nodes]).any()):
            raise ValueError('Source roles must be pairwise disjoint and outside pool')
        if bool(((role.labels < 0) | (role.labels >= classes)).any()):
            raise ValueError('Invalid class index')
        occupied[role.nodes] = True


def randomized_aps(probabilities, uniforms):
    """Class ties by class index; fixed label-independent per-node/class uniforms."""
    order = probabilities.argsort(dim=-1, descending=True, stable=True)
    sorted_p = probabilities.gather(1, order)
    sorted_u = uniforms.gather(1, order)
    sorted_scores = sorted_p.cumsum(-1) - sorted_p + sorted_u * sorted_p
    return torch.zeros_like(probabilities).scatter(1, order, sorted_scores)


def _lexicographic_member_multiset(member_probabilities):
    # [K,N,C] -> [N,K,C]; stable successive class sorts give exact multiset.
    vectors = member_probabilities.permute(1, 0, 2)
    for class_index in reversed(range(vectors.shape[2])):
        order = vectors[:, :, class_index].argsort(dim=1, stable=True)
        vectors = vectors.gather(1, order.unsqueeze(-1).expand_as(vectors))
    return vectors.flatten(1)


def _edge_covariance(deviations, edges, variance, chunk_edges):
    features = []
    for block in edges.split(chunk_edges, dim=1):
        u, v = block
        covariance = (deviations[:, u] * deviations[:, v]).sum(-1).mean(0)
        correlation = covariance / ((variance[u] + 1e-8) * (variance[v] + 1e-8)).sqrt()
        features.append(torch.stack((covariance, correlation), dim=1))
    if not features:
        return deviations.new_empty((0, 2))
    return torch.cat(features, dim=0)


def prepare_fixed_inputs(logits, edges, aps_uniforms, node_permutations,
                         chunk_edges=65536, feature_mode='all'):
    """No labels. Node/member randomness must be committed before source fitting."""
    if logits.ndim != 3 or logits.dtype != torch.float32:
        raise ValueError('Expected FP32 logits [K,N,C]')
    k, n, c = logits.shape
    if k < 2 or c < 2 or not bool(torch.isfinite(logits).all()):
        raise ValueError('Need finite multi-member classification logits')
    if edges.dtype != torch.int64 or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('Expected canonical int64 edges [2,E]')
    if bool(((edges < 0) | (edges >= n)).any()) or bool((edges[0] >= edges[1]).any()):
        raise ValueError('Canonical undirected edges require 0<=u<v<N')
    if edges.T.unique(dim=0).shape[0] != edges.shape[1]:
        raise ValueError('Duplicate edges must be resolved by precommitted preprocessing')
    if aps_uniforms.shape != (n, c) or not bool(torch.isfinite(aps_uniforms).all()):
        raise ValueError('Expected fixed finite APS uniforms [N,C]')
    if bool(((aps_uniforms < 0) | (aps_uniforms > 1)).any()):
        raise ValueError('APS uniforms outside [0,1]')
    expected = torch.arange(k, device=logits.device).expand(n, k)
    if node_permutations.dtype != torch.int64 or node_permutations.shape != (n, k):
        raise ValueError('Expected fixed node permutations [N,K]')
    if not torch.equal(node_permutations.sort(dim=1).values, expected):
        raise ValueError('Every node must have a member permutation')
    if chunk_edges <= 0:
        raise ValueError('Positive edge chunk size required')
    if feature_mode not in ('all', 'aligned', 'shuffled', 'none', 'pooled'):
        raise ValueError('Unknown covariance preparation mode')
    with torch.no_grad():
        detached_logits = logits.detach()
        pstar = detached_logits.mean(0).softmax(-1)
        pk = detached_logits.softmax(-1)
        pbar = pk.mean(0)
        degree = torch.bincount(edges.flatten(), minlength=n)
        multiset = pk.new_zeros((n,k*c)) if feature_mode == 'pooled' else _lexicographic_member_multiset(pk)
        node_features = torch.cat((pstar, pbar, multiset,
                                   degree.float().log1p().unsqueeze(-1)), dim=1)
        aligned = pk.new_zeros((edges.shape[1], 2))
        shuffled = torch.zeros_like(aligned)
        if feature_mode in ('all', 'aligned', 'shuffled'):
            deviations = pk - pbar.unsqueeze(0)
            variance = deviations.square().sum(-1).mean(0)
        if feature_mode in ('all', 'aligned'):
            aligned = _edge_covariance(deviations, edges, variance, chunk_edges)
        if feature_mode in ('all', 'shuffled'):
            by_node = deviations.permute(1, 0, 2)
            shuffled_d = by_node.gather(1, node_permutations.unsqueeze(-1).expand_as(by_node))
            shuffled = _edge_covariance(shuffled_d.permute(1, 0, 2), edges, variance, chunk_edges)
        return FixedInputs(pstar, node_features, randomized_aps(pstar, aps_uniforms),
                           edges, degree, aligned, shuffled, k, c)


class SymmetricEdgeGate(nn.Module):
    def __init__(self, node_dimension):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2 * node_dimension + 2, 32),
                                     nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
                                     nn.Linear(32, 1))

    def forward(self, left, right, edge_features):
        uv = self.network(torch.cat((left, right, edge_features), dim=1))
        vu = self.network(torch.cat((right, left, edge_features), dim=1))
        return ((uv + vu) * 0.5).tanh().squeeze(-1)


def corrected_scores(inputs, gate, arm, eta=0.5, chunk_edges=65536):
    """Exactly one signed step. All arms use equal gate dimensions and work."""
    if arm not in ('pooled', 'marginal', 'aligned', 'shuffled'):
        raise ValueError('Unknown information arm')
    features = inputs.node_features
    if arm == 'pooled':
        # Same pstar, pbar, degree; no full nodewise member multiset.
        features = features.clone()
        start = 2 * inputs.classes
        features[:, start:start + inputs.members * inputs.classes] = 0
    edge_features = inputs.aligned_features if arm == 'aligned' else inputs.shuffled_features
    if arm in ('pooled', 'marginal'):
        edge_features = torch.zeros_like(inputs.aligned_features)
    accumulator = torch.zeros_like(inputs.base_scores)
    for start in range(0, inputs.edges.shape[1], chunk_edges):
        u, v = inputs.edges[:, start:start + chunk_edges]
        rho = gate(features[u], features[v], edge_features[start:start + chunk_edges])
        difference = inputs.base_scores[v] - inputs.base_scores[u]
        contribution = rho.unsqueeze(-1) * difference
        accumulator.index_add_(0, u, contribution)
        accumulator.index_add_(0, v, -contribution)
    return inputs.base_scores + eta * accumulator / inputs.degree.clamp_min(1).unsqueeze(-1)


def source_quantile(scores, role, alpha=0.1):
    """Piecewise-differentiable kth score; equal-score subgradient by node index."""
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0,1)')
    values = scores[role.nodes, role.labels]
    n = values.numel()
    rank = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or rank > n:
        return scores.new_tensor(float('inf'))
    node_order = role.nodes.argsort(stable=True)
    sorted_values = values[node_order].sort(stable=True).values
    return sorted_values[rank - 1]


def _development_metrics(scores, correction_calibration, development, alpha):
    threshold = source_quantile(scores, correction_calibration, alpha)
    sets = scores[development.nodes] <= threshold
    coverage = sets.gather(1, development.labels.unsqueeze(1)).float().mean()
    return float(coverage), float(sets.sum(1).float().mean())


@dataclass(frozen=True)
class FitResult:
    frozen_scores: torch.Tensor
    selected_epoch: int | None
    selected_zero_diffusion: bool
    no_source_feasible_choice: bool
    source_trace: tuple
    selected_state: dict | None


def fit_source_gate(inputs, gate, arm, correction_calibration, correction_targets,
                    development, pool_nodes, alpha=0.1, max_epochs=2000,
                    min_epochs=200, patience=200, learning_rate=0.001, trace_sink=None):
    """Future source fit only. No final calibration/test label argument exists.

    Caller supplies identical committed gate initialization to all four arms.
    Fallback APS is an eligible source choice from epoch0, so trained correction
    never replaces a smaller feasible APS set merely because it is feasible.
    """
    _check_source_roles((correction_calibration, correction_targets, development),
                        pool_nodes, inputs.base_scores.shape[0], inputs.classes)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=learning_rate, weight_decay=0)
    best_state = None
    best_epoch = None
    best_size = float('inf')
    trace = []
    with torch.no_grad():
        coverage, size = _development_metrics(inputs.base_scores, correction_calibration,
                                              development, alpha)
        trace.append((0, None, coverage, size))
        if trace_sink is not None:
            trace_sink(trace[-1])
        if coverage >= 1 - alpha:
            best_size = size
            best_epoch = 0
    last_improvement = 0
    for epoch in range(1, max_epochs + 1):
        gate.train()
        optimizer.zero_grad(set_to_none=True)
        scores = corrected_scores(inputs, gate, arm)
        threshold = source_quantile(scores, correction_calibration, alpha)
        if not bool(torch.isfinite(threshold)):
            raise ValueError('Correction-calibration source count cannot supply finite quantile')
        soft_sets = torch.sigmoid((threshold - scores[correction_targets.nodes]) / 0.05)
        target_coverage = soft_sets.gather(1, correction_targets.labels.unsqueeze(1)).mean()
        loss = soft_sets.mean() + 10 * torch.relu((1 - alpha) - target_coverage).square()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite source objective; record failed arm without hidden retry')
        loss.backward()
        nonzero_gradients = sum(int(p.grad is not None and bool((p.grad != 0).any()))
                                for p in gate.parameters())
        optimizer.step()
        gate.eval()
        with torch.no_grad():
            scores = corrected_scores(inputs, gate, arm)
            coverage, size = _development_metrics(scores, correction_calibration, development, alpha)
        trace.append((epoch, float(loss.detach()), coverage, size, nonzero_gradients))
        if trace_sink is not None:
            trace_sink(trace[-1])
        if coverage >= 1 - alpha and size < best_size:
            best_size = size
            best_epoch = epoch
            last_improvement = epoch
            best_state = {name: value.detach().clone() for name, value in gate.state_dict().items()}
        if epoch >= min_epochs and epoch - last_improvement >= patience:
            break
    gate.eval()
    for parameter in gate.parameters():
        parameter.requires_grad_(False)
    if best_state is None:
        frozen_scores = inputs.base_scores.detach().clone()
    else:
        gate.load_state_dict(best_state)
        with torch.no_grad():
            frozen_scores = corrected_scores(inputs, gate, arm).detach().clone()
    return FitResult(frozen_scores, best_epoch, best_state is None, best_epoch is None,
                     tuple(trace), best_state)


def calibrate_frozen_scores(frozen_scores, pool_nodes, calibration_nodes, calibration_labels, alpha=0.1):
    """Only permitted final-calibration operation after method/score freeze."""
    if frozen_scores.requires_grad:
        raise ValueError('Final calibration requires frozen score table')
    if calibration_nodes.dtype != torch.int64 or calibration_labels.dtype != torch.int64:
        raise ValueError('Calibration node/class indices must be int64')
    if calibration_nodes.ndim != 1 or calibration_labels.shape != calibration_nodes.shape:
        raise ValueError('One calibration class label per node')
    if calibration_nodes.unique().numel() != calibration_nodes.numel():
        raise ValueError('Calibration nodes must be unique')
    if not bool(torch.isin(calibration_nodes, pool_nodes).all()):
        raise ValueError('Final calibration nodes must lie in the committed pool')
    if bool(((calibration_labels < 0) | (calibration_labels >= frozen_scores.shape[1])).any()):
        raise ValueError('Invalid calibration class')
    role = SourceLabels(calibration_nodes, calibration_labels)
    return source_quantile(frozen_scores, role, alpha).detach()


def prediction_sets(frozen_scores, pool_nodes, calibration_nodes, test_nodes, threshold):
    """Final test labels are not accepted; they belong to separate frozen reporting."""
    if test_nodes.dtype != torch.int64 or test_nodes.ndim != 1:
        raise ValueError('Test nodes must be an int64 vector')
    if test_nodes.unique().numel() != test_nodes.numel():
        raise ValueError('Test nodes must be unique')
    if not bool(torch.isin(test_nodes, pool_nodes).all()):
        raise ValueError('Final test nodes must lie in the committed pool')
    if bool(torch.isin(test_nodes, calibration_nodes).any()):
        raise ValueError('Final calibration and test nodes must be disjoint')
    return frozen_scores[test_nodes] <= threshold


def fixed_diffusion(inputs, edge_coefficients=None, node_coefficients=None):
    """APS/DAPS/HeAD literal score transforms, no fitting or final labels."""
    if (edge_coefficients is None) == (node_coefficients is None):
        raise ValueError('Supply exactly one coefficient interface')
    u, v = inputs.edges
    difference = inputs.base_scores[v] - inputs.base_scores[u]
    accumulator = torch.zeros_like(inputs.base_scores)
    if edge_coefficients is not None:
        contribution = edge_coefficients.unsqueeze(1) * difference
        accumulator.index_add_(0, u, contribution)
        accumulator.index_add_(0, v, -contribution)
    else:
        accumulator.index_add_(0, u, node_coefficients[u].unsqueeze(1) * difference)
        accumulator.index_add_(0, v, -node_coefficients[v].unsqueeze(1) * difference)
    return inputs.base_scores + accumulator / inputs.degree.clamp_min(1).unsqueeze(1)


def head_variant_scores(inputs, variant):
    """One frozen deployed variant only; no member features/family search."""
    p, u, v = inputs.point_probabilities, *inputs.edges
    c = inputs.classes
    if variant == 'aps':
        return inputs.base_scores
    if variant == 'daps':
        return fixed_diffusion(inputs, node_coefficients=torch.full_like(inputs.degree, .5, dtype=torch.float32))
    if variant == 'head_signed':
        same = (p.argmax(1)[u] == p.argmax(1)[v]).float()
        same_count = torch.zeros_like(inputs.degree, dtype=torch.float32)
        same_count.index_add_(0, u, same)
        same_count.index_add_(0, v, same)
        hard_homophily = (same_count + 1) / (inputs.degree + 2)
        return fixed_diffusion(inputs, node_coefficients=.5*(2*hard_homophily-1))
    if variant == 'head_edge':
        compatibility = ((p[u]*p[v]).sum(1)-1/c)/(1-1/c)
        return fixed_diffusion(inputs, edge_coefficients=.5*compatibility)
    if variant == 'head_v3':
        soft_sum = torch.zeros_like(inputs.degree, dtype=torch.float32)
        dot = (p[u]*p[v]).sum(1)
        soft_sum.index_add_(0, u, dot)
        soft_sum.index_add_(0, v, dot)
        soft_homophily = soft_sum/inputs.degree.clamp_min(1)
        centered_homophily = (soft_homophily-1/c)/(1-1/c)
        confidence = (p.max(1).values-1/c)/(1-1/c)
        coefficient = .5+.5*centered_homophily*confidence
        return fixed_diffusion(inputs, node_coefficients=coefficient)
    raise ValueError('Unknown frozen HeAD variant')


def head_family_scores(inputs):
    """Source-family selection only; literal pinned .5 constants."""
    return {name: head_variant_scores(inputs, name) for name in
            ('aps', 'daps', 'head_signed', 'head_edge', 'head_v3')}
