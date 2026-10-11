"""Injected-runtime path residual; import has no torch/model/data dependency.

Caller supplies its exact existing torch and FactorLinear provider. Two shared
maps retain private vectors. No trainer, loss, router, sampler or admission here.
"""
import math

MODES = ("ordered", "marginal", "order_breaking")
WALKS_PER_NODE = 8
POSITIONS = 4


def validate_walk_tensors(torch, node_count, indices, swap_mask, device):
    """Root calls once after converting the common frozen sampler table.

    This checks tensor geometry, root-row identity and bounds, not edge custody
    or the sampler digest. The safe provider owns those separate bindings.
    """
    if indices.dtype != torch.long or tuple(indices.shape) != (node_count, 8, 4):
        raise ValueError("Exact original-row [N,8,4] int64 walk table required")
    if swap_mask.dtype != torch.bool or tuple(swap_mask.shape) != (node_count, 8):
        raise ValueError("One frozen Boolean intermediate-swap mask per tuple")
    if indices.device != device or swap_mask.device != device:
        raise ValueError("Common walk metadata and native stem device must match")
    if int(indices.min()) < 0 or int(indices.max()) >= node_count:
        raise ValueError("Walk IDs outside full native row order")
    roots = torch.arange(node_count, device=device)[:, None].expand(-1, 8)
    if not torch.equal(indices[:, :, 0], roots):
        raise ValueError("Every original walk must begin at its native row")


def make_path_block(torch, factors, *, width, members, init_seed):
    """Make 4h->h->h maps on CPU, restoring caller constructor RNG exactly.

    No new forward dropout draws. The second source map is exactly zero; its
    FactorLinear vectors start at one. Initial equality assumes finite inputs.
    """
    if type(width) is not int or width <= 0 or members not in (1, 4):
        raise ValueError("Positive width and existing M1/M4 factor geometry")
    if type(init_seed) is not int or init_seed < 0:
        raise ValueError("Root must supply a bound nonnegative constructor seed")
    if torch.get_default_dtype() != torch.float32 or torch.empty(0).device.type != "cpu":
        raise ValueError("Existing float32 CPU construction convention required")
    with torch.random.fork_rng(devices=[]):
        first = torch.nn.Linear(4 * width, width, device="cpu", dtype=torch.float32)
        second = torch.nn.Linear(width, width, device="cpu", dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(init_seed)
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(first.weight, a=math.sqrt(5), generator=generator)
            first.bias.uniform_(-1 / math.sqrt(4 * width), 1 / math.sqrt(4 * width),
                                generator=generator)
            second.weight.zero_()
            second.bias.zero_()
        first = factors.FactorLinear(first, members)
        second = factors.FactorLinear(second, members)
    nn = torch.nn

    class PathResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.first, self.second = first, second
            self.width = width
            self.work = {"route_calls": 0, "stem_node_rows_reused": 0,
                         "walk_slots_gathered": 0, "first_dense_rows": 0,
                         "second_dense_rows": 0, "first_dense_MACs": 0,
                         "second_dense_MACs": 0, "order_mask_tuple_visits": 0}

        def forward(self, stem, indices, swap_mask, *, mode):
            if mode not in MODES or stem.ndim != 2 or stem.shape[1] != width:
                raise ValueError("Bound path mode and complete [N,h] native nonlinear stem")
            n = stem.shape[0]
            if stem.dtype != torch.float32 or indices.dtype != torch.long:
                raise ValueError("Native float32 stem and int64 tuple data required")
            if tuple(indices.shape) != (n, 8, 4) or tuple(swap_mask.shape) != (n, 8):
                raise ValueError("All modes use the identical full [N,8,4] table")
            if swap_mask.dtype != torch.bool or indices.device != stem.device or swap_mask.device != stem.device:
                raise ValueError("Frozen mask dtype/device or tuple device differs")
            versions = (indices._version, swap_mask._version, stem._version)
            selected = indices
            if mode == "order_breaking":
                selected = indices.clone()
                selected[:, :, 1] = torch.where(swap_mask, indices[:, :, 2], indices[:, :, 1])
                selected[:, :, 2] = torch.where(swap_mask, indices[:, :, 1], indices[:, :, 2])
                self.work["order_mask_tuple_visits"] += n * 8
            gathered = stem[selected]
            if mode == "marginal":
                # Same tuples and position marginals, pairing removed pre-map.
                inputs = gathered.mean(dim=1).reshape(n, 4 * width)
                pooled = nn.functional.gelu(self.first(inputs))
                first_rows = n
            else:
                inputs = gathered.reshape(n, 8, 4 * width)
                pooled = nn.functional.gelu(self.first(inputs)).mean(dim=1)
                first_rows = n * 8
            residual = self.second(pooled)
            if versions != (indices._version, swap_mask._version, stem._version):
                raise ValueError("Path operation mutated caller-owned stem or metadata")
            self.work["route_calls"] += 1
            self.work["stem_node_rows_reused"] += n
            self.work["walk_slots_gathered"] += n * 8 * 4
            self.work["first_dense_rows"] += first_rows
            self.work["second_dense_rows"] += n
            self.work["first_dense_MACs"] += first_rows * 4 * width * width
            self.work["second_dense_MACs"] += n * width * width
            return residual

    return PathResidual()
