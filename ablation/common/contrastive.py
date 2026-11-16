"""Task-unit InfoNCE and explicit shared/private gradient assignment.

Views are the existing four route preheads. Only TRAIN units enter this helper.
Sampling has a private CPU RNG, so it never changes the backbone dropout stream.
"""
import random
import torch
from torch.nn import functional as F

COEFFICIENT = 0.05
TEMPERATURE = 0.2
ANCHORS = 1024
ALTERNATIVES = 16


def unit_contrastive_loss(preheads, labels, seed, temperature=TEMPERATURE,
                          anchor_limit=ANCHORS, negative_limit=ALTERNATIVES):
    if preheads.ndim != 3 or preheads.shape[0] != 4:
        raise ValueError('Four aligned route preheads [4, training units, width] required')
    labels = labels.detach().reshape(-1).cpu()
    if labels.numel() != preheads.shape[1] or temperature <= 0:
        raise ValueError('Labels must align with units and temperature must be positive')
    if anchor_limit < 1 or negative_limit < 1:
        raise ValueError('Positive sampling limits required')
    values = labels.tolist()
    if len(set(values)) < 2:
        return preheads.sum() * 0, {'anchors': 0, 'ordered_pairs': 12, 'active': False}
    rng = random.Random(int(seed))
    anchors = rng.sample(range(len(values)), min(anchor_limit, len(values)))
    pools = {label: [i for i, other in enumerate(values) if other != label]
             for label in set(values)}
    alternatives = [rng.sample(pools[values[i]], min(negative_limit, len(pools[values[i]])))
                    for i in anchors]
    width = max(map(len, alternatives))
    padded = [row + [row[0]] * (width - len(row)) for row in alternatives]
    valid = torch.tensor([[j < len(row) for j in range(width)] for row in alternatives],
                         device=preheads.device, dtype=torch.bool)
    features = F.normalize(preheads, p=2, dim=-1, eps=1e-12)
    records = []
    for member in range(4):
        for view in range(4):
            if member == view:
                continue
            anchor = features[member, anchors]
            positive = (anchor * features[view, anchors]).sum(-1, keepdim=True)
            negative = (anchor[:, None, :] * features[view, padded]).sum(-1)
            negative = negative.masked_fill(~valid, float('-inf'))
            scores = torch.cat((positive, negative), dim=-1) / temperature
            records.append(torch.logsumexp(scores, dim=-1) - scores[:, 0])
    return torch.stack(records).mean(), {
        'anchors': len(anchors), 'ordered_pairs': 12, 'active': True,
        'temperature': temperature, 'negative_limit': negative_limit,
        'sampler': 'same TRAIN object positive; different TRAIN class alternatives',
    }


def private_gradient_step(model, optimizer, task_loss, auxiliary, private_parameters,
                          coefficient=COEFFICIENT):
    private = tuple(p for p in private_parameters if p.requires_grad)
    private_ids = {id(p) for p in private}
    shared = tuple(p for p in model.parameters() if p.requires_grad and id(p) not in private_ids)
    owned = [id(p) for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
    if not private or not shared or len(owned) != len(set(owned)):
        raise ValueError('Disjoint shared/private optimizer blocks required')
    if set(owned) != {id(p) for p in shared + private}:
        raise ValueError('Gradient blocks must partition optimizer parameters')
    if not torch.isfinite(task_loss) or not torch.isfinite(auxiliary):
        raise FloatingPointError('Non-finite contrastive or task objective')
    shared_grad = torch.autograd.grad(task_loss, shared, retain_graph=True, allow_unused=True)
    private_grad = torch.autograd.grad(task_loss + coefficient * auxiliary, private, allow_unused=True)
    for parameter, gradient in zip(shared + private, shared_grad + private_grad):
        if gradient is not None and not torch.isfinite(gradient).all():
            raise FloatingPointError('Non-finite contrastive gradient')
        parameter.grad = gradient
    optimizer.step()
