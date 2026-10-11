"""Matched boundary-only FactorLinear control; caller supplies native modules.

This code imports/loads no framework, constructs no graph and runs no fit.
It reuses the saved native adapter and member contexts without rewriting SAGE.
"""


def wrap_native_boundary(torch, factors, body, adapter, factor_seeds):
    """Four current-native routes; only stem/head receive existing r/s factors."""
    if adapter.body is not body or getattr(body, "model_name", None) != "SAGE":
        raise ValueError("Exact fresh native SAGE Model/adapter required")
    if (len(factor_seeds) != 4 or len(set(factor_seeds)) != 4
            or any(type(seed) is not int or seed < 0 for seed in factor_seeds)):
        raise ValueError("Four distinct prospectively bound nonnegative integer factor seeds required")
    if torch.get_default_dtype() != torch.float32 or list(body.named_buffers()):
        raise ValueError("Fresh native float32 body without running buffers required")
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in body.parameters()):
        raise ValueError("Install on fresh CPU body before device/optimizer")
    if any(isinstance(module, factors.FactorLinear) for module in body.modules()):
        raise ValueError("Already factorized body forbidden")
    native = list(body.named_parameters(remove_duplicate=False))
    native_ids = {id(parameter) for _, parameter in native}
    if len(native_ids) != len(native):
        raise ValueError("Tied native parameter aliases need a separate contract")
    for name in ("input_linear", "output_linear"):
        source = getattr(body, name)
        if not isinstance(source, torch.nn.Linear):
            raise ValueError("Exact native stem/head dense maps required")
        setattr(body, name, factors.FactorLinear(source, 4))
    if not native_ids.issubset({id(parameter) for parameter in body.parameters()}):
        raise ValueError("Native parameter ownership changed")
    with torch.no_grad():
        for row, seed in enumerate(factor_seeds):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            signs = torch.randint(0, 2, (body.input_linear.weight.shape[1],),
                                  generator=generator) * 2 - 1
            body.input_linear.r[row].copy_(signs.to(body.input_linear.r))

    class BoundaryRoutes(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body, self.members = body, 4
            self.factorized_maps = 2

        def check_optimizer_ownership(self, optimizer):
            expected = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
            actual = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError("All registered parameters must be owned exactly once")

        def forward(self, batch, *, route_scope=None):
            if any(key in batch for key in ("y", "labels", "train_mask", "valid_mask", "test_mask")):
                raise ValueError("Label-free factual native batch required")
            if self.training and route_scope is None:
                raise ValueError("Existing persistent member RNG scope required")
            if (batch["x"].ndim != 2 or batch["edge_index"].ndim != 2
                    or batch["edge_index"].shape[0] != 2):
                raise ValueError("Complete factual feature matrix and COO support required")
            adapter.check(batch)
            versions = batch["x"]._version, batch["edge_index"]._version
            rows = []
            for member in range(4):
                if route_scope is None:
                    with factors.member_context(self.body, member):
                        rows.append(adapter.native(batch))
                else:
                    with route_scope(member), factors.member_context(self.body, member):
                        rows.append(adapter.native(batch))
            if versions != (batch["x"]._version, batch["edge_index"]._version):
                raise ValueError("Native route mutated public graph/features")
            ids = batch.get("ids", torch.arange(len(batch["x"]), device=batch["x"].device))
            return (torch.stack([row[0][ids] for row in rows]),
                    torch.stack([row[1][ids] for row in rows]))

    return BoundaryRoutes()


if __name__ == "__main__":
    raise SystemExit("Source-only boundary control: root reviews and qualifies before any native execution")
