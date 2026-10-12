"""Attributed orthogonal-pair and rank-one additive maps; no trainer or owner."""


def install_corrections(torch, factors, body, *, backbone, kind, members, seed, cfg):
    """Extend selected already-factorized maps before device/optimizer creation."""
    if backbone not in ("GAT", "SAGE") or kind not in ("paired_graph", "rank1_lora_graph", "paired_local"):
        raise ValueError("Explicit native GAT/SAGE correction and placement required")
    graph, local = [], []
    for i, residual in enumerate(body.residual_modules):
        module = residual.module
        for name in (("lin",) if backbone == "GAT" else ("lin_l", "lin_r")):
            graph.append((module.conv, name, f"body.residual_modules.{i}.module.conv.{name}"))
        name = "linear_2" if backbone == "GAT" else "linear_1"
        local.append((module.feed_forward_module, name, f"body.residual_modules.{i}.module.feed_forward_module.{name}"))
    for parent, name, _ in graph:
        source = getattr(parent, name)
        if not isinstance(source, factors.FactorLinear) or source.weight.shape[0] != source.weight.shape[1]:
            raise ValueError("Reviewed square native graph maps required")
    graph_budget = sum(2 * members * getattr(parent, name).weight.shape[1] for parent, name, _ in graph)
    selected = local if kind == "paired_local" else graph
    paired = kind != "rank1_lora_graph"
    nn, F = torch.nn, torch.nn.functional

    class CorrectedLinear(factors.FactorLinear):
        def __init__(self, source, directions):
            nn.Module.__init__(self)
            # Retain native slow and common diagonal Parameter objects.
            self.weight, self.bias, self.r, self.s = source.weight, source.bias, source.r, source.s
            self.member = 0
            if paired:
                self.u1 = nn.Parameter(directions.clone())
                self.u2 = nn.Parameter(directions.clone())  # Equal values, separate storage.
            else:
                self.lora_a = nn.Parameter(directions.clone())
                self.lora_b = nn.Parameter(torch.zeros(members, source.weight.shape[0], dtype=torch.float32))

        def forward(self, x):
            m, z = self.member, x * self.r[self.member]
            if paired:
                u1, u2 = self.u1[m], self.u2[m]
                norm2 = torch.stack((u1.square().sum(), u2.square().sum()))
                if not (torch.isfinite(norm2) & (norm2 > 0)).all().item():
                    raise FloatingPointError("Householder normals must be finite and nonzero")
                # Row implementation of column Q=H(u1)H(u2): apply u2, then u1.
                z = z - 2 * (z * u2).sum(-1, keepdim=True) / norm2[1] * u2
                z = z - 2 * (z * u1).sum(-1, keepdim=True) / norm2[0] * u1
                y = F.linear(z, self.weight)
            else:
                a, b = self.lora_a[m], self.lora_b[m]
                if not (torch.isfinite(a).all() & torch.isfinite(b).all()).item():
                    raise FloatingPointError("LoRA vectors must be finite")
                y = F.linear(z, self.weight) + (z * a).sum(-1, keepdim=True) * b
            y = y * self.s[m]
            return y if self.bias is None else y + self.bias

    sites, total = [], 0
    for ordinal, (parent, name, path) in enumerate(selected):
        source = getattr(parent, name)
        if not isinstance(source, factors.FactorLinear):
            raise ValueError("Common diagonal factors must be installed first")
        incoming, outgoing = source.weight.shape[1], source.weight.shape[0]
        rows, seeds = [], []
        for member in range(members):
            orientation_seed = seed + cfg.get("correction_seed_offset", 5000081) + cfg.get("member_seed_stride", 1000003) * member + 1009 * ordinal
            generator = torch.Generator(device="cpu").manual_seed(orientation_seed)
            vector = torch.randn(incoming, generator=generator, dtype=torch.float32)
            radius = vector.norm()
            if not (torch.isfinite(radius) & (radius > 0)).item():
                raise FloatingPointError("Initial normal direction must be finite and nonzero")
            rows.append(vector / radius)
            seeds.append(orientation_seed)
        replacement = CorrectedLinear(source, torch.stack(rows))
        setattr(parent, name, replacement)
        extra = members * (2 * incoming if paired else incoming + outgoing)
        total += extra
        sites.append(dict(path=path, correction=kind, input_dim=incoming, output_dim=outgoing,
                          orientation_seeds=seeds, extra_private_parameters=extra))
    if total != graph_budget:
        raise ValueError("Graph, local-pair and rank-one extra parameter counts must match")
    return sites
