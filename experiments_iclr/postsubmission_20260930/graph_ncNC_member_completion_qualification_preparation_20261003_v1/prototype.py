"""Source-prepared NCNC depth-1 completion twins. No experiment entry point.

NCNC completion ancestry: Wang, Yang and Zhang, arXiv:2302.00890v4.
Rank-one sharing ancestry: Wen, Tran and Ba, arXiv:2002.06715.
Own portable implementation; native comparison bytes are private evidence.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from graph_ops import Graph, enumerate_neighbors, feature_sum


@dataclass(frozen=True)
class Recipe:
    # Author collab recipe/defaults, not new selected hyperparameters or seeds.
    features: int = 128
    hidden: int = 64
    input_dropout: float = .25
    encoder_dropout: float = .1
    encoder_edge_dropout: float = .25
    decoder_dropout: float = .3
    scale: float = 2.5
    offset: float = 6.0
    alpha: float = 1.05
    pt: float = .1
    encoder_lr: float = .0082
    decoder_lr: float = .0037
    completion_depth: int = 1
    candidate_split_size: int = -1
    member_count: int = 4
    native_target_mask: bool = True

    def validate(self):
        if self.completion_depth != 1 or not self.native_target_mask:
            raise ValueError("Only depth1/minibatch-mask semantics prepared")
        if self.candidate_split_size == 0 or self.candidate_split_size < -1:
            raise ValueError("Invalid native splitsize")
        if self.features <= 0 or self.hidden <= 0 or self.member_count <= 0:
            raise ValueError("Invalid dimensions")
        for p in (self.input_dropout, self.encoder_dropout, self.encoder_edge_dropout, self.decoder_dropout):
            if not 0 <= p < 1:
                raise ValueError("Invalid dropout")
        if not 0 < self.pt < 1 or self.alpha < 0:
            raise ValueError("Invalid fixed clamp")


class SharedEncoder(nn.Module):
    """Native collab one-layer GCN/LN/dropout/ReLU mathematical operation.

    res=True is inactive when 128 input features become width64. For equal
    dimensions it is active, matching the native dimension-conditional rule.
    """
    def __init__(self, recipe):
        super().__init__()
        self.recipe = recipe
        self.weight = nn.Parameter(torch.empty(recipe.hidden, recipe.features))
        self.bias = nn.Parameter(torch.zeros(recipe.hidden))
        nn.init.xavier_uniform_(self.weight)  # PyG GCNConv glorot map
        self.norm = nn.LayerNorm(recipe.hidden)

    def forward(self, x, graph):
        x = F.dropout(x, self.recipe.input_dropout, self.training)
        row, col, values = graph.adjacency_for_encoder(self.training, self.recipe.encoder_edge_dropout)
        # PyG SparseTensor GCN normalization uses row sums, both-side inverse
        # square-root degree, then biased projected-feature aggregation.
        degree = torch.zeros(graph.nodes, device=x.device, dtype=values.dtype).index_add(0, row, values)
        inv = degree.rsqrt()
        inv = torch.where(torch.isfinite(inv), inv, torch.zeros_like(inv))
        normed = values * inv[row] * inv[col]
        projected = F.linear(x, self.weight)
        result = projected.new_zeros((graph.nodes, self.recipe.hidden))
        result = result.index_add(0, row, projected[col] * normed.to(projected.dtype)[:, None])
        result = result + self.bias
        result = F.relu(F.dropout(self.norm(result), self.recipe.encoder_dropout, self.training))
        return result + x if result.shape[-1] == x.shape[-1] else result


class FactorLinear(nn.Module):
    def __init__(self, inputs, outputs, members):
        super().__init__()
        native = nn.Linear(inputs, outputs)
        self.weight, self.bias = native.weight, native.bias
        self.r = nn.Parameter(torch.ones(members, inputs))
        self.s = nn.Parameter(torch.ones(members, outputs))

    def forward_member(self, x, member):
        return F.linear(x * self.r[member], self.weight) * self.s[member] + self.bias


class PrivateNorm(nn.Module):
    def __init__(self, width, members):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(members, width))
        self.bias = nn.Parameter(torch.zeros(members, width))
        self.width = width

    def forward_member(self, x, member):
        return F.layer_norm(x, (self.width,), self.weight[member], self.bias[member], 1e-5)


class MemberMLP(nn.Module):
    """Explicit native module-index mapping, with private normalization axis."""
    def __init__(self, layout, width, members, dropout):
        super().__init__()
        self.layout, self.dropout = layout, dropout
        self.ops = nn.ModuleDict()
        for index, kind, output in layout:
            if kind == "linear":
                self.ops[str(index)] = FactorLinear(width, output, members)
                width = output
            elif kind == "norm":
                self.ops[str(index)] = PrivateNorm(width, members)

    def forward_member(self, x, member):
        for index, kind, _ in self.layout:
            if kind in ("linear", "norm"):
                x = self.ops[str(index)].forward_member(x, member)
            elif kind == "dropout":
                x = F.dropout(x, self.dropout, self.training)
            elif kind == "relu":
                x = F.relu(x)
            elif kind == "sigmoid":
                x = torch.sigmoid(x)
            else:
                raise RuntimeError("Unknown native MLP operation")
        return x


def clamp_completion(scores, scale, offset, alpha, pt):
    p0 = torch.sigmoid(scale * (scores - offset))
    return alpha * pt * p0 / (pt * p0 + 1 - p0)


def route_weights(weights, mode):
    if mode == "private":
        return weights
    if mode == "pooled_after_clamp":
        return weights.mean(0, keepdim=True).expand_as(weights)
    raise ValueError("Only exact private or pooled-after-clamp twin allowed")


class CompletionDecoder(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        self.recipe, self.members = recipe, recipe.member_count
        d, m, p = recipe.hidden, self.members, recipe.decoder_dropout
        # use_xlin=True, tailact=True, twolayerlin=False, lnnn=True.
        self.xlin = MemberMLP([(0,"linear",d),(1,"dropout",0),(2,"relu",0),
                              (3,"linear",d),(4,"norm",0),(5,"dropout",0),(6,"relu",0)],d,m,p)
        self.xcnlin = MemberMLP([(0,"linear",d),(1,"dropout",0),(2,"relu",0),
                                (3,"linear",d),(4,"norm",0),(5,"dropout",0),(6,"relu",0)],d,m,p)
        self.xijlin = MemberMLP([(0,"linear",d),(1,"norm",0),(2,"dropout",0),(3,"relu",0)],d,m,p)
        self.lin = MemberMLP([(0,"linear",d),(1,"norm",0),(2,"dropout",0),
                             (3,"relu",0),(8,"linear",1)],d,m,p)
        # Retain native unused fixed-pt ptlin parameters and state honestly.
        self.ptlin = MemberMLP([(0,"linear",d),(1,"relu",0),(2,"linear",1),(3,"sigmoid",0)],d,m,p)
        self.beta = nn.Parameter(torch.ones(m))
        for name in ("scale", "offset", "alpha", "pt"):
            self.register_buffer(name, torch.tensor([getattr(recipe, name)]))

    def decode(self, endpoint_product, common_features, member):
        # Native order matters for training dropout: xijlin precedes xcnlin.
        endpoint = self.xijlin.forward_member(endpoint_product, member)
        common = self.xcnlin.forward_member(common_features, member)
        return self.lin.forward_member(common * self.beta[member] + endpoint, member).flatten()

    def depth_zero(self, x, graph, query, member):
        endpoint = x[query[:, 0]] * x[query[:, 1]]
        transformed = x + self.xlin.forward_member(x, member)
        neighbors = enumerate_neighbors(graph, query)
        common = feature_sum(transformed, neighbors.common, len(query))
        return self.decode(endpoint, common, member)

    def completion_scores(self, transformed, graph, query, member):
        # Entire native recursive forward, including second full-node xlin and
        # active training dropout, stays under no_grad. Empty queries still
        # execute the full native path (no silent RNG/work shortcut).
        with torch.no_grad():
            size = self.recipe.candidate_split_size
            if size < 0:
                return self.depth_zero(transformed, graph, query, member)
            chunks = [self.depth_zero(transformed, graph, query[start:start + size], member)
                      for start in range(0, len(query), size)]
            return torch.cat(chunks) if chunks else transformed.new_empty(0)

    def forward(self, h, graph, queries, mode, return_details=False):
        neighbors = enumerate_neighbors(graph, queries)
        lq, ln = neighbors.left
        rq, rn = neighbors.right
        left_query = torch.stack((queries[lq, 1], ln), dim=1)
        right_query = torch.stack((queries[rq, 0], rn), dim=1)
        products, transformed, left, right, scores = [], [], [], [], []
        # Both twins use this identical phase-A member/candidate call schedule.
        # Per-member native xlin -> left recursion -> right recursion is kept;
        # only the cross-member final decode interleaving is standardized.
        for member in range(self.members):
            products.append(h[queries[:, 0]] * h[queries[:, 1]])
            hm = h + self.xlin.forward_member(h, member)
            transformed.append(hm)
            ls = self.completion_scores(hm, graph, left_query, member)
            rs = self.completion_scores(hm, graph, right_query, member)
            scores.append((ls, rs))
            left.append(clamp_completion(ls, self.scale, self.offset, self.alpha, self.pt))
            right.append(clamp_completion(rs, self.scale, self.offset, self.alpha, self.pt))
        raw_left, raw_right = torch.stack(left), torch.stack(right)
        routed_left, routed_right = route_weights(raw_left, mode), route_weights(raw_right, mode)
        outputs = []
        # Native xlin/downstream feature graph was not detached with scores.
        for member in range(self.members):
            hm = transformed[member]
            common = feature_sum(hm, neighbors.common, len(queries))
            common = common + feature_sum(hm, neighbors.left, len(queries), routed_left[member])
            common = common + feature_sum(hm, neighbors.right, len(queries), routed_right[member])
            outputs.append(self.decode(products[member], common, member))
        logits = torch.stack(outputs, dim=1)
        if return_details:
            return logits, {"neighbors":neighbors,"transformed":transformed,
                            "scores":scores,"raw_left":raw_left,"raw_right":raw_right,
                            "routed_left":routed_left,"routed_right":routed_right}
        return logits


class CompletionTwin(nn.Module):
    def __init__(self, recipe=None):
        super().__init__()
        self.recipe = recipe or Recipe()
        self.recipe.validate()
        self.encoder = SharedEncoder(self.recipe)
        self.decoder = CompletionDecoder(self.recipe)

    def forward(self, x, graph, queries, mode, return_details=False):
        return self.decoder(self.encoder(x, graph), graph, queries, mode, return_details)

    @staticmethod
    def serve(logits):
        # One unchanged serving pool in all qualification and future variants.
        return logits.mean(dim=1)


def training_batch_loss(model, x, train_pairs, record_ids, negative_pairs, mode):
    if negative_pairs.shape != (len(record_ids), 2):
        raise ValueError("Require one supplied native negative per selected record")
    model.train()
    graph = Graph.mask_train_batch(train_pairs, record_ids, len(x))
    h = model.encoder(x, graph)  # one shared encoder pass per native minibatch
    pos = model.decoder(h, graph, train_pairs[record_ids], mode)
    neg = model.decoder(h, graph, negative_pairs, mode)
    # Native positive/negative log-sigmoid sum, twice a balanced BCE mean.
    return -F.logsigmoid(pos).mean() - F.logsigmoid(-neg).mean()


def native_optimizer(model):
    return torch.optim.Adam([{"params":model.encoder.parameters(),"lr":model.recipe.encoder_lr},
                             {"params":model.decoder.parameters(),"lr":model.recipe.decoder_lr}])


def permute_members(model, order):
    if sorted(order) != list(range(model.recipe.member_count)):
        raise ValueError("Not a full member permutation")
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, FactorLinear):
                module.r.copy_(module.r[order].clone())
                module.s.copy_(module.s[order].clone())
            elif isinstance(module, PrivateNorm):
                module.weight.copy_(module.weight[order].clone())
                module.bias.copy_(module.bias[order].clone())
        model.decoder.beta.copy_(model.decoder.beta[order].clone())


def copy_native_unit_member(model, native_encoder, native_decoder):
    """Copy exact saved native values; no factor/random initialization choice."""
    if model.recipe.member_count != 1:
        raise ValueError("Native identity reference is one member with unit factors")
    with torch.no_grad():
        model.encoder.weight.copy_(native_encoder.convs[0].lin.weight)
        model.encoder.bias.copy_(native_encoder.convs[0].bias)
        model.encoder.norm.load_state_dict(native_encoder.lins[0][0].state_dict())
        for name in ("xlin","xcnlin","xijlin","lin","ptlin"):
            target, source = getattr(model.decoder,name), getattr(native_decoder,name)
            for index, operation in target.ops.items():
                original = source[int(index)]
                operation.weight.copy_(original.weight if isinstance(operation,FactorLinear) else original.weight[None])
                operation.bias.copy_(original.bias if isinstance(operation,FactorLinear) else original.bias[None])
                if isinstance(operation,FactorLinear):
                    operation.r.fill_(1); operation.s.fill_(1)
        model.decoder.beta.copy_(native_decoder.beta)
        for name in ("scale","offset","alpha","pt"):
            getattr(model.decoder,name).copy_(getattr(native_decoder,name))


def configuration(model):
    return asdict(model.recipe)
