"""Native NCN/F4 plus shared symmetric nonseparable residual interaction.

No import-time numerical dependency. Original builders remain available but
are excluded from default fit commands. New private parameters are only r/a;
existing base private factors, LayerNorm affine, and beta remain intact.
"""
from time import monotonic
import models_base
import topology

NEW_ARMS = ("structural_private", "structural_tied", "informed_single",
            "informed_independent4", "structure_blind", "count_only")
BASE_ARMS = ("shared_f4", "capable_single", "ordinary_native4")
ARMS = BASE_ARMS + NEW_ARMS


def score_bank(native, predictor, h, support, edges, member):
    # Exact donor F4 operations, bias placement, private LayerNorm and dropout.
    adjacency = predictor.dropadj(support)
    left, right = h[edges[0]], h[edges[1]]
    context_matrix = native.adjoverlap(adjacency, adjacency, edges, False, cnsampledeg=-1)
    transformed = h + predictor.xlin.forward_member(h, member)
    common = native.spmm_add(context_matrix, transformed)
    pair = predictor.xijlin.forward_member(left*right, member)
    context = predictor.xcnlin.forward_member(common, member)
    return predictor.lin.forward_member(context*predictor.beta[member]+pair, member)


def build(native, donor_heads, *, arm, seed, factor_seed, device, input_dim,
          structural_seed=None, message_width=32, message_hidden=64, chunk_size=4096):
    torch, nn = native.torch, native.nn
    if arm not in ARMS:
        raise ValueError("Explicit reviewed architecture required")
    base_arm = ("capable_single" if arm == "informed_single" else
                "ordinary_native4" if arm == "informed_independent4" else
                "shared_f4" if arm in NEW_ARMS else arm)
    base, base_shared, base_private, base_partition = models_base.build(native, donor_heads,
        arm=base_arm, seed=seed, factor_seed=factor_seed, device=device, input_dim=input_dim)
    if arm in BASE_ARMS:
        return base, base_shared, base_private, base_partition
    if type(structural_seed) is not int or not 0 <= structural_seed < 2**63:
        raise ValueError("Prospectively fixed structural initialization seed required")
    if (message_width, message_hidden) != (32, 64) or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Fixed message width32/hidden64 and positive chunk size required")
    kind = "blind" if arm == "structure_blind" else "count" if arm == "count_only" else "structural"

    class Residual(nn.Module):
        def __init__(self, rows):
            super().__init__()
            self.psi = nn.Sequential(nn.Linear(3*256, message_hidden), nn.ReLU(),
                                     nn.Linear(message_hidden, message_width))
            self.r = nn.Parameter(torch.zeros(rows, 4))
            self.a = nn.Parameter(torch.zeros(rows, message_width))

        def forward(self, h, info, members, charge):
            qcount, witnesses = len(info["queries"]), len(info["query_ids"])
            selected = [0 if self.r.shape[0] == 1 else member for member in members]
            residual = h.new_zeros((qcount, len(members)))
            if not witnesses:
                # Connected exact-zero derivatives ensure ordinary Adam advances
                # every structural block's moments even for no-witness batches.
                anchor = sum(parameter.sum()*0 for parameter in self.parameters())
                charge("empty_branch_anchor_parameter_elements", sum(p.numel() for p in self.parameters()))
                charge("empty_branch_calls", 1)
                return residual+anchor
            r, a = self.r[selected], self.a[selected]
            endpoints = None if kind == "count" else torch.tensor(info["queries"], dtype=torch.long, device=h.device)
            if endpoints is not None: charge("query_endpoint_device_rows", qcount)
            for start in range(0, witnesses, chunk_size):
                stop = min(start+chunk_size, witnesses)
                q = torch.tensor(info["query_ids"][start:stop], dtype=torch.long, device=h.device)
                xi = h.new_tensor(info["gates"][start:stop])
                if kind == "count":
                    counts = h.new_tensor(info["statistics"][start:stop])
                    # Same nominal Psi dimensions; only five known count features
                    # are active. No neighbor-pair embedding interaction.
                    features = torch.cat((counts, h.new_zeros((stop-start, 3*256-5))), 1)
                else:
                    w = torch.tensor(info["left"][start:stop], dtype=torch.long, device=h.device)
                    z = torch.tensor(info["right"][start:stop], dtype=torch.long, device=h.device)
                    uv = endpoints[q]
                    features = torch.cat((h[w]+h[z], h[w]*h[z], h[uv[:, 0]]*h[uv[:, 1]]), 1)
                messages = self.psi(features)  # computed once, reused by all selected gates
                gate = torch.sigmoid(xi @ r.t())
                # Equivalent to a_m dot sum(g_m Psi), with scalar scatter memory.
                residual = residual.index_add(0, q, gate*(messages @ a.t()))
                charge("Psi_rows", stop-start); charge("gate_rows", (stop-start)*len(members))
                charge("readout_rows", (stop-start)*len(members)); charge("message_chunks", 1)
            return residual

    class Structural(nn.Module):
        def __init__(self, backbone, branches):
            super().__init__(); self.base = backbone; self.branches = nn.ModuleList(branches)
            self.member_count = backbone.member_count
            self._work = {}

        @property
        def routes(self):
            # Delegate without registering duplicate encoders. Existing ordinary
            # native4 sum/times4 loss scaling detects only independent routes.
            if base_arm != "ordinary_native4":
                raise AttributeError("Shared/single architecture has no independent routes")
            return self.base.routes

        def charge(self, key, value):
            self._work[key] = self._work.get(key, 0)+value

        def drain_work(self):
            value, self._work = self._work, {}
            return value

        def forward(self, features, support, positive, negative, route=None):
            if route is not None and (type(route) is not int or route not in range(self.member_count)):
                raise ValueError("Route outside declared bank")
            members = list(range(self.member_count)) if route is None else [route]
            all_edges = torch.cat((positive, negative), 1)
            neighbors, queries, transfer = topology.from_support(support, all_edges, len(features))
            started = monotonic()
            info = topology.enumerate_queries(neighbors, queries, kind)
            for key, value in {**transfer, **info["work"]}.items(): self.charge(key, value)
            self.charge("witness_enumeration_seconds", monotonic()-started)
            self.charge("wrapper_forward_calls", 1)
            if base_arm == "ordinary_native4":
                p_rows, n_rows = [], []
                for member in members:
                    row = self.base.routes[member]
                    h = row.encoder(features, support); self.charge("encoder_forward_count", 1)
                    p, n = row.predictor(h, support, positive), row.predictor(h, support, negative)
                    delta = self.branches[member](h, info, [0], self.charge)
                    p_rows.append(p+delta[:positive.shape[1]]); n_rows.append(n+delta[positive.shape[1]:])
                return torch.cat(p_rows, 1), torch.cat(n_rows, 1)
            h = self.base.encoder(features, support); self.charge("encoder_forward_count", 1)
            if base_arm == "capable_single":
                p, n = self.base.predictor(h, support, positive), self.base.predictor(h, support, negative)
            else:
                p = torch.cat([score_bank(native, self.base.predictor, h, support, positive, m) for m in members], 1)
                n = torch.cat([score_bank(native, self.base.predictor, h, support, negative, m) for m in members], 1)
            delta = self.branches[0](h, info, members, self.charge)
            return p+delta[:positive.shape[1]], n+delta[positive.shape[1]:]

    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        branches = []
        for member in range(4 if base_arm == "ordinary_native4" else 1):
            torch.random.default_generator.manual_seed(structural_seed+5*member)
            if device.type == "cuda":
                with torch.cuda.device(device): torch.cuda.manual_seed(structural_seed+5*member)
            branches.append(Residual(1 if base_arm != "shared_f4" or arm == "structural_tied" else 4))
    model = Structural(base, branches).to(device)
    private = tuple("base."+name for name in base_private) + tuple(
        name for name, _ in model.named_parameters() if name.startswith("branches.") and name.endswith((".r", ".a")))
    names = dict(model.named_parameters())
    shared = tuple(name for name in names if name not in private)
    if len(set(shared) & set(private)) or set(shared) | set(private) != set(names):
        raise ValueError("Structural parameter partition is not exhaustive")
    if len({id(parameter) for parameter in model.parameters()}) != len(names):
        raise ValueError("Unexpected duplicate parameter registration")
    partition = {"arm": arm, "base_arm": base_arm, "input_dim": input_dim,
        "shared": list(shared), "private": list(private), "base_partition": base_partition,
        "parameters": [{"name": name, "shape": list(value.shape), "numel": value.numel(),
                        "role": "private" if name in private else "shared"} for name, value in names.items()],
        "structural": {"kind": kind, "message_width": message_width, "message_hidden": message_hidden,
            "chunk_size": chunk_size, "structural_seed": structural_seed,
            "Psi_parameter_sharing": "independent_per_native_route" if base_arm == "ordinary_native4" else "one_shared_Psi",
            "gate_readout_rows": [branch.r.shape[0] for branch in branches],
            "zero_readout_initialization": True, "initial_Psi_gate_gradients": "exact_zero_until_readout_moves",
            "tied_effective_capacity_smaller": arm == "structural_tied",
            "count_effective_inputs": ["CN", "LCL", "CAR", "CRA", "CAA"] if kind == "count" else None,
            "ordinary_role_labels_only": True,
            "chunking_limit": "bounds_one_chunk_temporaries; autograd_saved_activations_still_scale_with_all_witnesses"}}
    return model, shared, private, partition
