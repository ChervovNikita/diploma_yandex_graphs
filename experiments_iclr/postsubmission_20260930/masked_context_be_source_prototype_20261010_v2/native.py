"""Hash-bound native class loading without author module side effects."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(key):
    row = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())['dependencies'][key]
    path = (PHASE/row['path']).resolve(strict=True)
    if not path.is_relative_to(PHASE) or not path.is_file() or sha(path) != row['sha256']:
        raise ValueError('Pinned project source changed: '+key)
    return path


def _definitions(path, names, environment):
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if set(node.name for node in nodes) != set(names):
        raise ValueError('Required native definitions absent')
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), environment)
    return environment


def native_class():
    # Exact class ASTs execute without imports, global seeding or training/main.
    import torch
    from torch import nn
    from torch.nn import functional as F
    env = dict(torch=torch, nn=nn, F=F, Linear=nn.Linear)
    _definitions(source_path('polyformer_block'), ('PolyAttn', 'FFNNetwork', 'FFN', 'PolyFormerBlock'), env)
    _definitions(source_path('polyformer_model'), ('PolyFormer',), env)
    return env['PolyFormer']


def native_preprocess(x, edge_index, K=2):
    # AST extraction retains the exact author gcn_norm->SciPy->float32->spmm path.
    import numpy as np
    import torch
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.utils import to_scipy_sparse_matrix
    env = dict(np=np, torch=torch, gcn_norm=gcn_norm, to_scipy_sparse_matrix=to_scipy_sparse_matrix)
    _definitions(source_path('polyformer_utils'), ('sparse_mx_to_torch_sparse_tensor', 'mono_base'), env)
    return env['mono_base'](K, x, edge_index, None)


def factors_module():
    path = source_path('internal_factors')
    spec = importlib.util.spec_from_file_location('_masked_context_existing_factors', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@contextmanager
def seeded(seed, device):
    """Native constructor stream separate from persistent dropout streams."""
    import numpy as np
    import torch
    cpu_rng, np_rng, py_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    devices = [] if device.type != 'cuda' else [device.index]
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(seed); np.random.seed(seed % (2**32))
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            if devices:
                generator = torch.Generator(device=device).manual_seed(seed)
                torch.cuda.set_rng_state(generator.get_state(), device.index)
            yield
        finally:
            random.setstate(py_rng); np.random.set_state(np_rng); torch.set_rng_state(cpu_rng)


def construct(recipe, seed, device):
    # No second reset after factors have been installed by the caller.
    with seeded(seed, device):
        return native_class()(None, SimpleNamespace(**recipe)).to(device)
