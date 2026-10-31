"""BUDDY sign_k=0 ports. Graph evidence is supplied by one common cache."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from guards import verify_source_pins

HERE = Path(__file__).resolve().parent


def verified_source(name):
    pins = verify_source_pins(HERE)
    if f"vendor/{name}" not in pins:
        raise RuntimeError(f"Unpinned source: {name}")
    path = HERE / "vendor" / name
    return path.read_text()


def native_buddy_class():
    """Execute only the exact saved BUDDY class, avoiding unrelated PyG imports.

    SIGN/embedding/RA execution is prohibited by native_args. The class bytes and
    AST are from the immutable author source; this is not a rewritten baseline.
    """
    tree = ast.parse(verified_source("models_elph.py.txt"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BUDDY")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"torch": torch, "F": F, "Linear": nn.Linear}
    exec(compile(module, "pinned_author_BUDDY", "exec"), namespace)
    return namespace["BUDDY"]


def native_args(config, width):
    return SimpleNamespace(
        use_feature=True, label_dropout=config["label_dropout"],
        feature_dropout=config["feature_dropout"], propagate_embeddings=False,
        add_normed_features=True, use_RA=False, sign_k=0, sign_dropout=0.5,
        max_hash_hops=2, hidden_channels=width,
    )


def single(config, width):
    return native_buddy_class()(native_args(config, width), num_features=128, node_embedding=None)


class FactorLinear(nn.Module):
    """y_m = ((x_m * r_m) W^T) * s_m + b; W and b shared."""
    def __init__(self, source, members):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.bias = nn.Parameter(source.bias.detach().clone())
        self.r = nn.Parameter(torch.ones(members, source.in_features))
        self.s = nn.Parameter(torch.ones(members, source.out_features))

    def forward(self, x):
        return F.linear(x * self.r[:, None, :], self.weight) * self.s[:, None, :] + self.bias


class FactorizedBUDDY(nn.Module):
    """Four native nonlinear predictors with shared linear maps/private factors.

    Every learned native linear map is modulated. Endpoint products are taken
    inside each member. BatchNorm affine parameters and running states are
    private. No member axis is folded into a BatchNorm batch.
    """
    def __init__(self, native, members=4, factor_seed=None):
        super().__init__()
        self.members = members
        self.label_dropout = native.label_dropout
        self.feature_dropout = native.feature_dropout
        self.node_embedding = None
        self.label_lin_layer = FactorLinear(native.label_lin_layer, members)
        self.lin_feat = FactorLinear(native.lin_feat, members)
        self.lin_out = FactorLinear(native.lin_out, members)
        self.lin = FactorLinear(native.lin, members)
        import copy
        self.bn_labels = nn.ModuleList([copy.deepcopy(native.bn_labels) for _ in range(members)])
        self.bn_feats = nn.ModuleList([copy.deepcopy(native.bn_feats) for _ in range(members)])
        # Native BUDDY registers this unused BN even when use_RA=False. Keep it
        # for honest parameter/state accounting and identity source parity.
        self.bn_RA = nn.ModuleList([copy.deepcopy(native.bn_RA) for _ in range(members)])
        if factor_seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(factor_seed)
            with torch.no_grad():
                for layer in (self.label_lin_layer, self.lin_feat, self.lin_out, self.lin):
                    layer.r.copy_(torch.randint(0, 2, layer.r.shape, generator=generator) * 2 - 1)
            # Disclosed from-scratch BE-style sign initialization: private R
            # are random signs; S are one. No residual-guided initializer.

    def forward(self, sf, node_features, src_degree, dst_degree):
        normalizer = torch.sqrt(src_degree * dst_degree).unsqueeze(1)
        normalized = torch.nan_to_num(sf / normalizer, nan=0.0, posinf=0.0, neginf=0.0)
        structural = torch.cat([sf, normalized], dim=1)
        structural = self.label_lin_layer(structural.unsqueeze(0).expand(self.members, -1, -1))
        structural = torch.stack([bn(structural[m]) for m, bn in enumerate(self.bn_labels)])
        structural = F.dropout(F.relu(structural), p=self.label_dropout, training=self.training)
        # Project both endpoints with the same member map, then form its product.
        shape = node_features.shape
        endpoints = self.lin_feat(node_features.reshape(-1, shape[-1]).unsqueeze(0).expand(self.members, -1, -1))
        endpoints = endpoints.reshape(self.members, shape[0], 2, -1)
        features = self.lin_out(endpoints[:, :, 0] * endpoints[:, :, 1])
        features = torch.stack([bn(features[m]) for m, bn in enumerate(self.bn_feats)])
        features = F.dropout(F.relu(features), p=self.feature_dropout, training=self.training)
        # Native BUDDY casts this branch before concatenation, including when
        # a float64 model is used by the strict identity/gradient CPU gate.
        return self.lin(torch.cat([structural, features.to(torch.float)], dim=-1)).squeeze(-1).transpose(0, 1)


class IndependentBUDDY(nn.Module):
    def __init__(self, config, width, seeds):
        super().__init__()
        self.models = nn.ModuleList()
        for seed in seeds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.models.append(single(config, width))

    def forward(self, sf, node_features, src_degree, dst_degree):
        return torch.cat([model(sf, node_features, src_degree, dst_degree) for model in self.models], dim=1)


def parameter_count(width, members=1, factorized=False):
    # Includes native unused BN_RA parameters; F=128, normalized structure D=16.
    return width * width + 155 * width + 1133 if factorized else members * (width * width + 133 * width + 323)


def matched_width(target, maximum=4096):
    return min(range(2, maximum + 1), key=lambda h: (abs(parameter_count(h) - target), h))


def member_seeds(seed):
    return [seed + 10000 * m for m in range(4)]


def make_model(config, arm, seed):
    torch.manual_seed(seed)
    width = config["width"]
    if arm == "native1024":
        return single(config, 1024)
    if arm == "single256":
        return single(config, width)
    if arm == "factorized4":
        return FactorizedBUDDY(single(config, width), 4, factor_seed=seed + 90000)
    if arm == "independent4":
        return IndependentBUDDY(config, width, member_seeds(seed))
    if arm == "matched_single":
        return single(config, matched_width(parameter_count(width, factorized=True)))
    raise ValueError(arm)


def member_logits(model, batch):
    result = model(*batch)
    return result.reshape(result.shape[0], -1)
