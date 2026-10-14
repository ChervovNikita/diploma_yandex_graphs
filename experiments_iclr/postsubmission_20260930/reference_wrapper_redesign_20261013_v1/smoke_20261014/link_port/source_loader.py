"""Load exact saved native source definitions without unrelated entrypoints."""
import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'native_sources'


def checked_source(relative):
    path = SOURCE / relative
    rows = json.loads((SOURCE / 'MANIFEST.json').read_text())['files']
    row = next(r for r in rows if r['snapshot_path'].endswith('/native_sources/' + relative))
    raw = path.read_bytes()
    if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Frozen native source changed: ' + relative)
    return path, raw, row


def full_module(relative, name, aliases=None):
    path, raw, row = checked_source(relative)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    prior = {k: sys.modules.get(k) for k in aliases or {}}
    sys.modules[name] = module
    try:
        for key, value in (aliases or {}).items():
            sys.modules[key] = value
        exec(compile(raw, str(path), 'exec'), module.__dict__)
    finally:
        for key, value in prior.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module, row


def exact_definition(relative, definition, namespace, method=None):
    """Isolate an unchanged original AST node; no transforms or substitutions."""
    path, raw, row = checked_source(relative)
    tree = ast.parse(raw, filename=str(path))
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == definition)
    if method is not None:
        node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    unit = ast.Module(body=[node], type_ignores=[])
    scope = dict(namespace)
    scope['__file__'] = str(path)
    scope['__name__'] = '_redesign_exact_native_definition'
    exec(compile(unit, str(path), 'exec'), scope)
    result = scope[method or definition]
    segment = ast.get_source_segment(raw.decode(), node)
    return result, {'source': row, 'definition': definition, 'method': method,
                    'definition_sha256': hashlib.sha256(segment.encode()).hexdigest(),
                    'unchanged_AST_node': True}


def training_rules():
    path = ROOT / 'runtime' / 'training_rules.py'
    spec = importlib.util.spec_from_file_location('_redesign_link_training_rules', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ncn():
    utils, urow = full_module('ncn/ncn__utils.py', '_redesign_native_ncn_utils')
    model, mrow = full_module('ncn/ncn__model.py', '_redesign_native_ncn_model', {'utils': utils})
    return utils, model, [urow, mrow]


def load_buddy(torch):
    import torch.nn.functional as F
    import torch_sparse
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.utils import add_self_loops, negative_sampling, to_undirected
    # No SIGN/embedding/RA replacement exists here. Those native branches are
    # excluded by the fixed Collab arguments before construction.
    cls, proof = exact_definition('buddy/buddy_src_models_elph.py', 'BUDDY',
                                 {'torch': torch, 'F': F, 'Linear': torch.nn.Linear, 'gcn_norm': gcn_norm})
    propagation, pproof = exact_definition('buddy/buddy_src_datasets_elph.py', 'HashDataset',
        {'torch': torch, 'torch_sparse': torch_sparse, 'gcn_norm': gcn_norm}, '_generate_sign_features')
    hashes, hrow = full_module('buddy/buddy_src_hashing.py', '_redesign_native_buddy_hashing')
    helper_namespace = {'torch': torch, 'to_undirected': to_undirected,
                        'add_self_loops': add_self_loops, 'negative_sampling': negative_sampling}
    filter_year, fproof = exact_definition('buddy/buddy_src_data.py', 'filter_by_year', helper_namespace)
    get_pos, _ = exact_definition('buddy/buddy_src_data.py', 'get_ogb_pos_edges', helper_namespace)
    negative_namespace = helper_namespace | {'get_ogb_pos_edges': get_pos}
    get_negs, nproof = exact_definition('buddy/buddy_src_data.py', 'get_ogb_train_negs', negative_namespace)
    loss, lproof = exact_definition('buddy/buddy_src_runners_train.py', 'bce_loss',
                                  {'torch': torch, 'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss})
    return cls, propagation, hashes.ElphHashes, filter_year, get_negs, loss, [proof, pproof, hrow, fproof, nproof, lproof]
