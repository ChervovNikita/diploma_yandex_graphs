"""Five small frozen-output operators. Caller supplies torch; stdlib import only."""
import math


def features(torch, probability, neighbor):
    """[M,N,C] -> [M,N,5], with no label or member/class identity input."""
    members, _, classes = probability.shape
    if members == 1:
        peer, neighbor_peer = probability, neighbor
    else:
        peer = (probability.sum(0, keepdim=True) - probability) / (members - 1)
        neighbor_peer = (neighbor.sum(0, keepdim=True) - neighbor) / (members - 1)
    log_p = probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
    entropy = -(probability * log_p).sum(-1) / math.log(classes)
    top = probability.topk(2, dim=-1).values
    return torch.stack((entropy, top[..., 0] - top[..., 1],
                        (probability * peer).sum(-1),
                        (probability * neighbor).sum(-1),
                        (probability * neighbor_peer).sum(-1)), dim=-1)


def make_operator(torch, rule, members, classes):
    """Return one specified operator; never construct or update a backbone."""
    nn, F = torch.nn, torch.nn.functional

    class Reliability(nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator(device='cpu').manual_seed(11709)
            self.W = nn.Parameter(torch.empty(8, 5, dtype=torch.float64)
                                  .uniform_(-0.1, 0.1, generator=generator))
            self.b = nn.Parameter(torch.zeros(8, dtype=torch.float64))
            self.a = nn.Parameter(torch.zeros(8, dtype=torch.float64))
            if sum(p.numel() for p in self.parameters()) != 56:
                raise ValueError('Exactly56 scorer parameters required')

        def forward(self, log_p, probability, context):
            score = torch.tanh(F.linear(context, self.W, self.b)) @ self.a
            log_alpha = score.log_softmax(0)
            log_q = torch.logsumexp(log_p + log_alpha[..., None], dim=0)
            shrink = 0.01 * (log_alpha.exp() * (log_alpha + math.log(members))).sum(0).mean()
            return log_q, shrink

    class Temperature(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_T = nn.Parameter(torch.zeros(
                1 if rule == 'temperature_global' else members, dtype=torch.float64))

        def forward(self, log_p, probability, context):
            scaled = (log_p / self.log_T.exp().reshape(-1, 1, 1)).log_softmax(-1)
            return torch.logsumexp(scaled, dim=0) - math.log(members), self.log_T.sum() * 0

    class LinearStacker(nn.Module):
        def __init__(self):
            super().__init__()
            self.B = nn.Parameter(torch.zeros(classes, members * classes, dtype=torch.float64))
            self.c = nn.Parameter(torch.zeros(classes, dtype=torch.float64))

        def forward(self, log_p, probability, context):
            own = probability.transpose(0, 1).reshape(probability.shape[1], -1)
            q = F.linear(own, self.B, self.c).log_softmax(-1)
            shrink = 0.01 * (self.B.square().sum() + self.c.square().sum()) / (self.B.numel() + self.c.numel())
            return q, shrink

    if rule in ('reliability_graph', 'reliability_self'):
        return Reliability()
    if rule in ('temperature_global', 'temperature_member'):
        return Temperature()
    if rule == 'linear_stacking':
        return LinearStacker()
    raise ValueError('Unknown fixed operator')


def fit_fold(torch, rule, log_p, probability, context, labels, fit_ids, held_ids):
    """Exactly500 updates; held labels never choose an epoch, rule or setting."""
    model = make_operator(torch, rule, probability.shape[0], probability.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    train_log, train_p, train_f = log_p[:, fit_ids], probability[:, fit_ids], context[:, fit_ids]
    train_y = labels[fit_ids]
    trace = []
    for update in range(1, 501):
        optimizer.zero_grad(set_to_none=True)
        log_q, penalty = model(train_log, train_p, train_f)
        loss = torch.nn.functional.nll_loss(log_q, train_y) + penalty
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite fixed aggregator objective')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite/missing small-head gradient')
        optimizer.step()
        if update in (1, 100, 200, 300, 400, 500):
            trace.append({'update': update, 'fusion_fit_objective_before_update': loss.item()})
    model.eval()
    with torch.no_grad():
        held_log, _ = model(log_p[:, held_ids], probability[:, held_ids], context[:, held_ids])
    if not torch.isfinite(held_log).all():
        raise FloatingPointError('Nonfinite aggregator endpoint')
    return held_log, {'updates': 500, 'parameters': sum(p.numel() for p in model.parameters()),
                      'fit_nodes': len(fit_ids), 'held_nodes': len(held_ids), 'trace': trace,
                      'selected_endpoint': 'fixed_final_update_no_heldout_selector',
                      'state': {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()}}
