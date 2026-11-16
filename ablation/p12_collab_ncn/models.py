"""Keep the native NCN graph and predictor, routing only its endpoint affines."""
import torch
from torch import nn

from .source import CNLinkPredictor, GCN


class RoutedGCN(nn.Module):
    """R before the native projection, S/B after native graph propagation."""
    def __init__(self, conv, members, signs=True):
        super().__init__()
        self.conv = conv
        self.member = 0
        self.R = nn.Parameter(torch.ones(members, conv.in_channels))
        self.S = nn.Parameter(torch.ones(members, conv.out_channels))
        bias = conv.bias.detach().clone()
        self.B = nn.Parameter(bias.repeat(members, 1))
        self.conv.register_parameter("bias", None)
        if signs:
            with torch.no_grad():
                self.R.bernoulli_(0.5).mul_(2).sub_(1)

    def forward(self, x, adj):
        return self.conv(x * self.R[self.member], adj) * self.S[self.member] + self.B[self.member]


class RoutedLinear(nn.Module):
    """The original GNNM affine formula, preserving the source shared weight."""
    def __init__(self, linear, members):
        super().__init__()
        self.member = 0
        self.W = linear
        self.R = nn.Parameter(torch.ones(members, linear.in_features))
        self.S = nn.Parameter(torch.ones(members, linear.out_features))
        self.B = nn.Parameter(linear.bias.detach().clone().repeat(members, 1))
        self.W.register_parameter("bias", None)

    def forward(self, x):
        return self.W(x * self.R[self.member]) * self.S[self.member] + self.B[self.member]


class NCN(nn.Module):
    def __init__(self, input_dim, config, route_signs=True):
        super().__init__()
        self.members = config["members"]
        self.encoder = GCN(
            input_dim, config["hiddim"], config["hiddim"], config["mplayers"],
            config["gnndp"], config["ln"], config["res"], -1,
            config["model"], config["jk"], config["gnnedp"],
            xdropout=config["xdp"], taildropout=config["tdp"])
        self.predictor = CNLinkPredictor(
            config["hiddim"], config["hiddim"], 1, config["nnlayers"],
            config["predp"], config["preedp"], config["lnnn"],
            cndeg=config["cndeg"], use_xlin=config["use_xlin"],
            tailact=config["tailact"], twolayerlin=config["twolayerlin"],
            beta=config["beta"])
        if self.members > 1:
            self.encoder.convs[0] = RoutedGCN(self.encoder.convs[0], self.members, route_signs)
            self.predictor.lin[-1] = RoutedLinear(self.predictor.lin[-1], self.members)

    def select_member(self, member):
        if self.members > 1:
            self.encoder.convs[0].member = member
            self.predictor.lin[-1].member = member
