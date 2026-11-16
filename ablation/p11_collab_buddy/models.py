"""Native BUDDY with original BEBlock factors at branch inputs and output."""

import sys

import torch
import torch.nn.functional as F

from .protocol import REFERENCE, namespace

# The pinned reference uses absolute src imports within its package.
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))
from src.models.elph import BUDDY
from models import BEBlock
from ablation.p09_molecular.normalization import install_private_batchnorm, select_batchnorm_route


class GNNMBuddy(BUDDY):
    """Four routes share BUDDY's feature MLP, normalizers and linear weights."""

    def __init__(self, num_features, members=4, private_bn=False):
        super().__init__(namespace(), num_features, node_embedding=None)
        self.members = members
        # BUDDY has parallel structure and node-feature entry projections.
        self.label_lin_layer = self.route_projection(self.label_lin_layer, True)
        self.lin_feat = self.route_projection(self.lin_feat, True)
        self.lin = self.route_projection(self.lin, False)
        if private_bn:
            install_private_batchnorm(self)

    def route_projection(self, native, is_first):
        block = BEBlock(native.in_features, native.out_features, is_first, 'cpu', self.members)
        with torch.no_grad():
            block.W.weight.copy_(native.weight)
            block.B.copy_(native.bias.expand(self.members, -1))
        return block

    def forward(self, sf, node_features, src_degree, dst_degree, route=0, return_prehead=False):
        select_batchnorm_route(self, route)
        sf = self._append_degree_normalised(sf, src_degree, dst_degree)
        x = self.label_lin_layer(sf, route)
        x = self.bn_labels(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.label_dropout, training=self.training)
        features = self.lin_feat(node_features, route)
        features = features[:, 0, :] * features[:, 1, :]
        features = self.lin_out(features)
        features = self.bn_feats(features)
        features = F.relu(features)
        features = F.dropout(features, p=self.feature_dropout, training=self.training)
        x = torch.cat([x, features.to(torch.float)], dim=1)
        logits = self.lin(x, route)
        return (logits, x) if return_prehead else logits


def build_model(variant, num_features):
    if variant == 'native':
        return BUDDY(namespace(), num_features, node_embedding=None)
    if variant in ('gnnm', 'gnnm_contrast', 'gnnm_private_bn', 'gnnm_private_bn_contrast'):
        return GNNMBuddy(num_features, private_bn='private_bn' in variant)
    raise ValueError(f'Unknown BUDDY variant {variant}')


def route_logits(model, inputs, variant):
    if variant == 'native':
        return model(*inputs).reshape(1, -1)
    return torch.stack([model(*inputs, route=route).reshape(-1) for route in range(4)])


def contrastive_step(model, inputs, targets, endpoints, seed, optimizer):
    """Historical same-query InfoNCE updates private factors only."""
    from .historical_contrast.objectives import build_objectives, step_blocks
    from .historical_contrast.planning import authenticate, query_keys
    outputs = [model(*inputs, route=route, return_prehead=True) for route in range(4)]
    logits = torch.stack([value[0].reshape(-1) for value in outputs])
    preheads = torch.stack([value[1] for value in outputs])
    groups = targets.detach().cpu().long().tolist()
    queries = endpoints.detach().cpu().tolist()
    signature = authenticate(query_keys(queries), groups, 'TRAIN')
    losses = [F.binary_cross_entropy_with_logits(row, targets) for row in logits]
    terms = build_objectives(
        torch, family='BUDDY', logits=logits, preheads=preheads,
        endpoints=queries, groups=groups, native_member_losses=losses,
        seed=seed, contrast='global', responsibility='none', aux_credit='private',
        coefficient=0.05, temperature=0.2, anchor_limit=1024, negative_limit=16,
        route_query_signatures=[signature] * 4, match_endpoint_control_support=True)
    private = [parameter for block in model.modules() if isinstance(block, BEBlock)
               for parameter in (block.R, block.S, block.B)]
    private_ids = {id(parameter) for parameter in private}
    shared = [parameter for parameter in model.parameters() if id(parameter) not in private_ids]
    def finite_gradients():
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise FloatingPointError('Non-finite BUDDY gradient')
    step_blocks(torch, optimizer, shared, private, terms, clip=finite_gradients)
    return terms['native_own_loss'], {
        'auxiliary_loss': float(terms['query_auxiliary_loss'].detach()),
        'anchors': len(terms['diagnostics']['contrast_plan']['anchors']),
        'native_examples': len(groups)}
