"""Six exact native affine sites; no Torch/model imports at module import."""
import copy
import hashlib

from joint_contracts import FAST_RANK1_PER_MEMBER, NATIVE_SCALARS, require


def dictionary(torch, inputs, rank, seeds, site, group, dtype, device):
    columns = []
    for seed in seeds[:rank]:
        value = int.from_bytes(hashlib.sha256((str(seed) + ':' + site + ':' + str(group)).encode()).digest()[:8], 'big') % 2**63
        generator = torch.Generator(device='cpu').manual_seed(value)
        column = torch.randn(inputs, generator=generator, dtype=torch.float32)
        norm = column.norm()
        require(torch.isfinite(column).all().item() and torch.isfinite(norm).item() and norm.item() > 0, 'Finite nonzero one-shot input dictionary; no redraw')
        columns.append((column / norm).to(dtype=dtype, device=device))
    return torch.stack(columns, dim=1)


def install(native_module, qualified_adapter, prototype, rank, seeds):
    """Own module/buffer context; preserve native W/bias names and object IDs."""
    torch, original = qualified_adapter._validate_binding(native_module, prototype)
    require(rank in (1, 4) and len(seeds) >= rank, 'Only the declared rank1/rank4 source')
    nn = native_module.nn

    class GroupedAdditive(nn.Module):
        def __init__(self, native, site):
            super().__init__()
            self.cin, self.cout, self.num_metapaths = native.cin, native.cout, native.num_metapaths
            self.W, self.bias = native.W, native.bias
            self.v = nn.Parameter(torch.stack([dictionary(torch, self.cin, rank, seeds, site, group, self.W.dtype, self.W.device) for group in range(self.num_metapaths)]))
            self.u = nn.Parameter(torch.zeros(self.num_metapaths, rank, self.cout, dtype=self.W.dtype, device=self.W.device))
            object.__setattr__(self, '_native_forward', type(native).forward)
            self.train(native.training)

        def forward(self, x):
            require(x.ndim == 3 and tuple(x.shape[1:]) == (self.num_metapaths, self.cin), 'Exact native grouped axes')
            value = self._native_forward(self, x)
            latent = torch.einsum('bci,cir->bcr', x, self.v.to(dtype=x.dtype))
            correction = torch.einsum('bcr,cro->bco', latent, self.u.to(dtype=latent.dtype))
            return value + correction.to(dtype=value.dtype)

        def reset_parameters(self):
            raise RuntimeError('Native slow weights cannot be reset after adapter installation')

    class SemanticAdditive(nn.Module):
        def __init__(self, native, site):
            super().__init__()
            self.in_features, self.out_features = native.in_features, native.out_features
            self.weight, self.bias = native.weight, native.bias
            self.v = nn.Parameter(dictionary(torch, self.in_features, rank, seeds, site, 0, self.weight.dtype, self.weight.device))
            self.u = nn.Parameter(torch.zeros(rank, self.out_features, dtype=self.weight.dtype, device=self.weight.device))
            object.__setattr__(self, '_native_forward', type(native).forward)
            self.train(native.training)

        def forward(self, x):
            require(x.shape[-1] == self.in_features, 'Exact native semantic input axis')
            value = self._native_forward(self, x)
            latent = x @ self.v.to(dtype=x.dtype)
            correction = latent @ self.u.to(dtype=latent.dtype)
            return value + correction.to(dtype=value.dtype)

        def reset_parameters(self):
            raise RuntimeError('Native slow weights cannot be reset after adapter installation')

    for site in qualified_adapter.SITES:
        owner, name, native = qualified_adapter._site(prototype, site)
        owner._modules[name] = (GroupedAdditive(native, site) if site in qualified_adapter.GROUPED_SITES else SemanticAdditive(native, site))
    private_names = tuple(site + '.' + factor for site in qualified_adapter.SITES for factor in ('v', 'u'))
    named = dict(prototype.named_parameters(remove_duplicate=False))
    require(set(named) == set(original) | set(private_names) and all(named[name] is value for name, value in original.items()), 'Only rank factors added; native slow names/identity retained')
    require(sum(value.numel() for name, value in named.items() if name in private_names) == rank * FAST_RANK1_PER_MEMBER, 'Measured exact six-site private scalar budget')
    return prototype, tuple(original), private_names


def shared_additive(rt, adapter, prototype, rank, adapter_seeds, copied=False):
    require(rank == 1, 'Only shared rank1; rank4 is one complete capacity single')
    original = dict(prototype.named_parameters(remove_duplicate=False))
    members = []
    for member in range(4):
        clone = copy.deepcopy(prototype, {id(value): value for value in original.values()})
        seed = adapter_seeds[0 if copied else member]
        clone, slow_names, private_names = install(rt['model_module'], adapter, clone, rank, (seed,))
        members.append(clone)
    bank = adapter._bank_type(rt['torch'])(members, slow_names, private_names, adapter.channel_order(prototype))
    bank.verify_ownership()
    require(sum(value.numel() for value in bank.slow_parameters()) == NATIVE_SCALARS, 'All complete native slow tensors retained')
    return bank
