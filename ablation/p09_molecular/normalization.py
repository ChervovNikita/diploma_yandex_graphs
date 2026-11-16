"""Route-private BatchNorm buffers with exactly shared learned affine weights."""

import torch
from torch import nn
from torch.nn import functional as F


class RouteBatchNorm(nn.Module):
    """Keep the native BN update law and one buffer bank for each route."""

    def __init__(self, native, members=4):
        super().__init__()
        if not isinstance(native, nn.BatchNorm1d) or not native.track_running_stats:
            raise ValueError('Tracked BatchNorm1d required')
        self.num_features = native.num_features
        self.eps = native.eps
        self.momentum = native.momentum
        self.weight = native.weight
        self.bias = native.bias
        self.members = members
        self.route = 0
        self.register_buffer('running_mean', native.running_mean.repeat(members, 1))
        self.register_buffer('running_var', native.running_var.repeat(members, 1))
        self.register_buffer('num_batches_tracked', native.num_batches_tracked.repeat(members))

    def forward(self, value):
        route = self.route
        if not 0 <= route < self.members:
            raise ValueError('BatchNorm route index is outside the bank')
        factor = self.momentum or 0.0
        if self.training:
            self.num_batches_tracked[route].add_(1)
            if self.momentum is None:
                factor = 1.0 / float(self.num_batches_tracked[route])
        return F.batch_norm(value, self.running_mean[route], self.running_var[route],
                            self.weight, self.bias, self.training, factor, self.eps)


def install_private_batchnorm(module, members=4):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            setattr(module, name, RouteBatchNorm(child, members))
        else:
            install_private_batchnorm(child, members)


def select_batchnorm_route(module, route):
    for child in module.modules():
        if isinstance(child, RouteBatchNorm):
            child.route = route
