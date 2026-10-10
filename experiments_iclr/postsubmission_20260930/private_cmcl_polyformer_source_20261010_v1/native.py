"""Guarded exact author ASTs and the existing BE factor module; source only."""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
from types import ModuleType, SimpleNamespace

from .caps import CLOSED
from .plan import NATIVE, MEMBERS

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(key, caps=CLOSED):
    caps.require('source_bound')
    row = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())['dependencies'][key]
    path = (PHASE/row['path']).resolve(strict=True)
    require(path.is_relative_to(PHASE) and path.is_file() and path.stat().st_size == row['bytes']
            and sha(path) == row['sha256'], 'Unchanged pinned project source: '+key)
    return path


def _definitions(path, names, environment, caps=CLOSED):
    caps.require('source_bound', 'runtime')
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(PHASE), 'Native definitions remain inside the project phase')
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    require(set(n.name for n in nodes) == set(names), 'Complete exact native definitions')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), environment)
    return environment


def native_class(caps=CLOSED):
    caps.require('source_bound', 'model', 'runtime')
    import torch
    from torch import nn
    from torch.nn import functional as F
    env = dict(torch=torch, nn=nn, F=F, Linear=nn.Linear)
    _definitions(source_path('polyformer_block', caps), ('PolyAttn', 'FFNNetwork', 'FFN', 'PolyFormerBlock'), env, caps)
    _definitions(source_path('polyformer_model', caps), ('PolyFormer',), env, caps)
    return env['PolyFormer']


def preprocess(x, edges, caps=CLOSED):
    caps.require('source_bound', 'data', 'runtime')
    import numpy as np
    import torch
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.utils import to_scipy_sparse_matrix
    env = dict(np=np, torch=torch, gcn_norm=gcn_norm, to_scipy_sparse_matrix=to_scipy_sparse_matrix)
    _definitions(source_path('polyformer_utils', caps), ('sparse_mx_to_torch_sparse_tensor', 'mono_base'), env, caps)
    return env['mono_base'](2, x, edges, None)


def factors(caps=CLOSED):
    caps.require('source_bound', 'model', 'runtime')
    path = source_path('internal_factors', caps)
    module = ModuleType('_private_cmcl_existing_factor_maps')
    module.__file__ = str(path)
    # Explicit source compile avoids writing pycache into a sealed predecessor.
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module


@contextmanager
def seeded(seed, device, caps=CLOSED):
    caps.require('source_bound', 'model', 'runtime')
    import numpy as np
    import torch
    cpu, np_state, py_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    devices = [] if device.type != 'cuda' else [device.index]
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(seed); np.random.seed(seed % 2**32)
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            if devices:
                torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(seed).get_state(), device.index)
            yield
        finally:
            random.setstate(py_state); np.random.set_state(np_state); torch.set_rng_state(cpu)


def build(seed, device, caps=CLOSED):
    caps.require('source_bound', 'model', 'runtime')
    import torch
    with seeded(seed, device, caps):
        model = native_class(caps)(None, SimpleNamespace(**NATIVE)).to(device)
    original = dict(model.named_parameters())
    factor_module = factors(caps)
    sites = factor_module.install_factors(model, MEMBERS)
    model.to(device)  # Existing factors are created on CPU; original W objects stay.
    named = dict(model.named_parameters())
    require(sites == 23 and all(named[n] is p for n, p in original.items()), 'All23 native affine sites; original shared parameters preserved')
    private = tuple(p for n, p in named.items() if n.endswith(('.r', '.s')))
    shared = tuple(p for n, p in named.items() if not n.endswith(('.r', '.s')))
    require(len(private) == 46 and all(p.shape[0] == MEMBERS for p in private)
            and {id(p) for p in private}.isdisjoint({id(p) for p in shared})
            and len({id(p) for p in (*shared, *private)}) == len(named), 'Disjoint full shared/BE-factor ownership')
    require(sum(p.numel() for p in shared) == 2069875
            and sum(p.numel() for p in private) == 55772,
            'Unshrunk native parameter count and complete M4 factor rows')
    require(all(p.ndim == 2 and p.is_contiguous() and p.stride(0) == p.shape[1] for p in private)
            and len({int(p.untyped_storage().data_ptr()) for p in (*shared, *private)}) == len(named),
            'Separate parameter storage; private member rows are nonoverlapping contiguous slices')
    require(not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()), 'Native LayerNorm-only representative; no repeated running-buffer updates')
    require(all(p.dtype == torch.float32 and p.device == device for p in model.parameters()), 'Full native FP32 model and factors')
    require(all(torch.equal(p.detach(), torch.ones_like(p)) for p in private), 'Unit factors, no fitted or randomized initial router')
    return model, factor_module, shared, private
