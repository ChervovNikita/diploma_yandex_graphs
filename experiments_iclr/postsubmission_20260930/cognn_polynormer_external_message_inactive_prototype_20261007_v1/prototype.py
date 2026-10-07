"""Inactive, attributed external-message graft; no numerical imports at import.

Numerical calls are disabled in this source-only task. This file does not patch
or replace the native model, GAT class, or parameter objects.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

RUNTIME_ENABLED = False
PYG_GAT_SHA256 = "a7b2353003394ab433f909c0dbd7e5a06b7ae1289939857245ae3062f1eb5901"


@dataclass(frozen=True)
class ActionNoise:
    receive: Any  # Constant [N,2] Gumbels, owned by this forward's caller.
    send: Any


@dataclass(frozen=True)
class GateTensors:
    member: int
    view: int
    layer: int
    receive: Any
    send: Any
    receive_soft: Any
    send_soft: Any
    temperature: Any
    edge_gate: Any
    attention: Any  # Native softmax/dropout alpha, before gating.


@dataclass(frozen=True)
class ForwardRecord:
    logits: Any
    gates: tuple


def build_runtime():
    """Deliberately fails before importing Torch/PyG in this inactive packet."""
    if RUNTIME_ENABLED is not True:
        raise RuntimeError("Inactive source prototype: runtime qualification is disabled")

    import inspect
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from torch import nn
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import scatter

    if torch_geometric.__version__ != "2.7.0":
        raise RuntimeError("Only the retained PyG2.7.0 source is supported")
    source = inspect.getsourcefile(GATConv)
    if source is None or sha256(Path(source).read_bytes()).hexdigest() != PYG_GAT_SHA256:
        raise RuntimeError("GATConv runtime source does not match the retained pin")

    def plain_support(x, edge_index):
        if x.ndim != 2:
            raise ValueError("Require homogeneous [N,D] features")
        if type(edge_index) is not torch.Tensor or edge_index.layout != torch.strided:
            raise TypeError("Require plain dense COO edge_index; no SparseTensor/EdgeIndex")
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Require long [2,E] support")
        if edge_index.device != x.device:
            raise ValueError("Support and features must share a device")
        # Do not sort, filter, coalesce, add or remove edges, including loops.

    def check_gat(native):
        if type(native) is not GATConv:
            raise TypeError("Require the pinned native GATConv, not a subclass")
        if native.lin is None or native.lin_src is not None or native.lin_dst is not None:
            raise ValueError("Only homogeneous shared-projection GAT is supported")
        if native.add_self_loops or native.edge_dim is not None:
            raise ValueError("Require prepared support, add_self_loops=False, no edge attributes")
        if native.flow != "source_to_target" or native.node_dim != 0 or native.aggr != "add":
            raise ValueError("Require native source-to-target additive GAT")
        if native.explain or native.decomposed_layers != 1:
            raise ValueError("Explain/decomposed execution is outside this prototype")
        if native._forward_hooks or native._forward_pre_hooks:
            raise ValueError("GAT forward hooks require a separate qualification")

    def edge_gates(edge_index, receive, send):
        """All listed self edges keep1; external products retain ST graphs."""
        u, v = edge_index
        external = receive.index_select(0, v) * send.index_select(0, u)
        return torch.where(u == v, torch.ones_like(external), external)

    def gat(native, x, edge_index, receive, send):
        """Scoped native forward; gating is AFTER its original softmax/dropout.

        No cache, state assignment, edge filtering, resoftmax, or new PyG
        signature. Existing dense maps may already contain private factors.
        """
        plain_support(x, edge_index)
        check_gat(native)
        if receive.shape != (x.shape[0],) or send.shape != (x.shape[0],):
            raise ValueError("Receive/send must be [N] tensors")
        if any(t.device != x.device or t.dtype != x.dtype for t in (receive, send)):
            raise ValueError("Gate dtype/device must match the current state")
        heads, channels = native.heads, native.out_channels
        residual = None if native.res is None else native.res(x)
        projected = native.lin(x).view(-1, heads, channels)
        alpha_src = (projected * native.att_src).sum(dim=-1)
        alpha_dst = (projected * native.att_dst).sum(dim=-1)
        alpha = native.edge_updater(
            edge_index, alpha=(alpha_src, alpha_dst), edge_attr=None, size=None)
        gate = edge_gates(edge_index, receive, send)
        effective_alpha = alpha * gate.unsqueeze(-1)
        out = native.propagate(
            edge_index, x=(projected, projected), alpha=effective_alpha, size=None)
        out = out.view(-1, heads * channels) if native.concat else out.mean(dim=1)
        if residual is not None:
            out = out + residual
        if native.bias is not None:
            out = out + native.bias
        return out, alpha, gate

    def st_keep(logits, temperature, gumbels):
        if logits.shape != gumbels.shape or logits.shape[-1] != 2:
            raise ValueError("Binary logits and explicit Gumbels must be [N,2]")
        if gumbels.requires_grad or gumbels.device != logits.device or gumbels.dtype != logits.dtype:
            raise ValueError("Noise must be a constant tensor with matching dtype/device")
        # Clone caller noise to prevent later panel writes from aliasing this draw.
        soft = F.softmax((logits + gumbels.clone()) / temperature, dim=-1)
        hard = F.one_hot(soft.argmax(dim=-1), num_classes=2).to(soft.dtype)
        straight_through = hard - soft.detach() + soft
        # Keep is author class0; off is class1. .eval() does not switch sampler.
        return straight_through[:, 0], soft[:, 0]

    class MemberPolicy(nn.Module):
        """One-layer MeanGNN receive/send and learned linear temperature.

        Reused across local depth. No policy dropout or factor traversal here;
        initialization/temperature are prototype choices, not a scientific recipe.
        """
        def __init__(self, width, tau0):
            super().__init__()
            if tau0 <= 0:
                raise ValueError("tau0 must be positive")
            self.receive = nn.Linear(2 * width, 2)
            self.send = nn.Linear(2 * width, 2)
            self.temperature = nn.Linear(width, 1, bias=False)
            self.tau0 = float(tau0)

        def sample(self, x, edge_index, noise):
            plain_support(x, edge_index)
            u, v = edge_index
            neighbor = scatter(x.index_select(0, u), v, dim=0,
                               dim_size=x.shape[0], reduce="mean")
            policy_input = torch.cat((x, neighbor), dim=-1)
            temperature = (F.softplus(self.temperature(x)) + self.tau0).reciprocal()
            receive, receive_soft = st_keep(self.receive(policy_input), temperature, noise.receive)
            send, send_soft = st_keep(self.send(policy_input), temperature, noise.send)
            return receive, send, receive_soft, send_soft, temperature

    class PolicyBank(nn.Module):
        def __init__(self, width, members, tau0):
            super().__init__()
            if members < 1:
                raise ValueError("Need a nonempty member bank")
            self.members = nn.ModuleList([MemberPolicy(width, tau0) for _ in range(members)])

    def make_policies(width, members, tau0, seed, *, dtype=None, device=None):
        """CPU constructor RNG is isolated; no native reset or CUDA seed call."""
        live_cpu_state = torch.get_rng_state()
        private = torch.Generator(device="cpu").manual_seed(seed)
        try:
            torch.set_rng_state(private.get_state())
            bank = PolicyBank(width, members, tau0)
        finally:
            torch.set_rng_state(live_cpu_state)
        move = {}
        if dtype is not None:
            move["dtype"] = dtype
        if device is not None:
            move["device"] = device
        return bank.to(**move) if move else bank

    def forward(native, policies, x, edge_index, *, member, view, noise_panel=None,
                force_all_keep=False):
        """Functional graft of the pinned native Polynormer forward.

        Caller owns the existing factor member context and provides a constant
        ActionNoise for each (member,view,layer). Returned records keep every
        graph live; no module attribute holds current gates/member/view/noise.
        """
        plain_support(x, edge_index)
        if native._forward_hooks or native._forward_pre_hooks:
            raise ValueError("Native top-level hooks require separate qualification")
        if member < 0 or member >= len(policies.members):
            raise IndexError("Invalid explicit member")
        x = F.dropout(x, p=native.in_drop, training=native.training)
        x = native.lin_in(x)
        x = F.dropout(x, p=native.dropout, training=native.training)
        x_local = 0
        records = []
        for layer, local_conv in enumerate(native.local_convs):
            if native.pre_ln:
                x = native.pre_lns[layer](x)
            if force_all_keep:
                receive = send = x.new_ones(x.shape[0])
                receive_soft = send_soft = temperature = None
            else:
                if noise_panel is None:
                    raise ValueError("Explicit per-forward noise panel required")
                noise = noise_panel[(member, view, layer)]
                receive, send, receive_soft, send_soft, temperature = policies.members[member].sample(
                    x, edge_index, noise)
            h = F.relu(native.h_lins[layer](x))
            neighbor, attention, gate = gat(local_conv, x, edge_index, receive, send)
            x = neighbor + native.lins[layer](x)  # Original own/root route.
            x = F.relu(x)
            x = F.dropout(x, p=native.dropout, training=native.training)
            beta = (F.sigmoid(native.betas[layer]).unsqueeze(0) if native.beta < 0
                    else native.betas[layer].unsqueeze(0))
            x = (1 - beta) * native.lns[layer](h * x) + beta * x
            x_local = x_local + x
            records.append(GateTensors(member, view, layer, receive, send,
                                       receive_soft, send_soft, temperature, gate, attention))
        logits = (native.pred_global(native.global_attn(native.ln(x_local)))
                  if native._global else native.pred_local(x_local))
        return ForwardRecord(logits, tuple(records))

    return SimpleNamespace(gat=gat, forward=forward, edge_gates=edge_gates,
                           st_keep=st_keep, make_policies=make_policies)
