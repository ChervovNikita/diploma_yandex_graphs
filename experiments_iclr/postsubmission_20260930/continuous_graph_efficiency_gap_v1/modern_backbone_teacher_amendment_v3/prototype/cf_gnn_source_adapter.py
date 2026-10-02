"""SOURCE-ONLY CF-GNN baseline port. Never imported/executed in this phase.

Architecture and TPS objective follow author commit3564a3d6d9bd4ff69e36c57672f8272c5fb0ff39.
Declared differences: fixed cached same-teacher probabilities; explicit A/B/D
roles; source-development APS selection; independently randomized final APS.
No author pretrained model, optimal-parameter pickle, data loader or launcher.
"""
import math
import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv
from aligned_score_correction import (_check_source_roles, randomized_aps,
                                      _development_metrics, FitResult)


class PooledGraphCorrection(nn.Module):
    def __init__(self, classes, backbone):
        super().__init__()
        if backbone == 'GCN':
            self.first = GCNConv(classes, 64, cached=True, normalize=True)
            self.second = GCNConv(64, classes, cached=True, normalize=True)
        elif backbone == 'GraphSAGE':
            self.first = SAGEConv(classes, 64, aggr='sum')
            self.second = SAGEConv(64, classes, aggr='sum')
        else:
            raise ValueError('Only the two precommitted correction backbones')

    def forward(self, probabilities, directed_edges):
        values = torch.nn.functional.dropout(probabilities, p=0.5, training=self.training)
        values = self.first(values, directed_edges).relu()
        values = torch.nn.functional.dropout(values, p=0.5, training=self.training)
        return self.second(values, directed_edges)


def native_source_loss(logits, calibration, targets, epoch, alpha=0.1):
    """Same native TPS computation for fitting and explicit branch qualification."""
    n = calibration.nodes.numel()
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    if not 0 <= 1 - q_level <= 1:
        raise ValueError('Insufficient A nodes for native TPS training quantile')
    probabilities = logits.softmax(1)
    threshold = torch.quantile(probabilities[calibration.nodes, calibration.labels],
                               1 - q_level, interpolation='higher')
    inclusion = torch.sigmoid((probabilities[targets.nodes] - threshold) / 0.1)
    prediction_loss = torch.nn.functional.cross_entropy(logits[targets.nodes], targets.labels)
    size_loss = inclusion.sum(1).relu().mean()
    return prediction_loss if epoch <= 1000 else prediction_loss + size_loss


def fit_cf_source_corrector(inputs, corrector, aps_uniforms,
                            correction_calibration, correction_targets, development,
                            pool_nodes, alpha=0.1, max_epochs=5000,
                            min_epochs=2000, patience=200, trace_sink=None):
    """Future training operation only. Class loss B; TPS quantile A; selection D."""
    _check_source_roles((correction_calibration, correction_targets, development),
                        pool_nodes, inputs.base_scores.shape[0], inputs.classes)
    directed = torch.cat((inputs.edges, inputs.edges.flip(0)), dim=1)
    optimizer = torch.optim.Adam(corrector.parameters(), lr=0.001, weight_decay=5e-4)
    a, b = correction_calibration, correction_targets
    n = a.nodes.numel()
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    if not 0 <= 1 - q_level <= 1:
        raise ValueError('Insufficient A nodes for native TPS training quantile')
    best_size = float('inf')
    best_epoch = None
    best_state = None
    last_improvement = 0
    trace = []
    with torch.no_grad():
        coverage, size = _development_metrics(inputs.base_scores, a, development, alpha)
        trace.append((0, None, coverage, size))
        if trace_sink is not None:
            trace_sink(trace[-1])
        if coverage >= 1 - alpha:
            best_size, best_epoch = size, 0
    for epoch in range(1, max_epochs + 1):
        corrector.train()
        optimizer.zero_grad(set_to_none=True)
        logits = corrector(inputs.point_probabilities, directed)
        loss = native_source_loss(logits, a, b, epoch, alpha)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Record failed native arm; no hidden outcome-dependent retry')
        loss.backward()
        optimizer.step()
        corrector.eval()
        with torch.no_grad():
            probabilities = corrector(inputs.point_probabilities, directed).softmax(1)
            scores = randomized_aps(probabilities, aps_uniforms)
            coverage, size = _development_metrics(scores, a, development, alpha)
        trace.append((epoch, float(loss.detach()), coverage, size))
        if trace_sink is not None:
            trace_sink(trace[-1])
        if coverage >= 1 - alpha and size < best_size:
            best_size, best_epoch, last_improvement = size, epoch, epoch
            best_state = {key: value.detach().clone() for key, value in corrector.state_dict().items()}
        if epoch >= min_epochs and epoch - last_improvement >= patience:
            break
    corrector.eval()
    for parameter in corrector.parameters():
        parameter.requires_grad_(False)
    if best_state is None:
        scores = inputs.base_scores.detach().clone()
        probabilities = inputs.point_probabilities.detach().clone()
    else:
        corrector.load_state_dict(best_state)
        with torch.no_grad():
            probabilities = corrector(inputs.point_probabilities, directed).softmax(1).detach()
            scores = randomized_aps(probabilities, aps_uniforms).detach()
    result = FitResult(scores, best_epoch, best_state is None, best_epoch is None,
                       tuple(trace), best_state)
    return result, probabilities
