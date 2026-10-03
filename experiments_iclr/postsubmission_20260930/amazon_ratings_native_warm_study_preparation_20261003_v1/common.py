"""Explicit admissions and source/data custody. Numerical imports are deferred."""
import ast
from contextlib import redirect_stdout
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

PACKET = Path(__file__).resolve().parent
PHASE = PACKET.parent
V4 = PHASE/'graph_conditional_response_native_source_preparation_20261003_v4'
AUTHOR = PHASE/'coordinate_source_independent_review_v1/strong_backbones_v1/sources/polyformer_code/node_classification'
BLOCKS = ((0, 17), (1, 29), (2, 43))
RECIPES = ('source_defaults', 'roman_mono')
EXPECTED_CASES = tuple((j, s, r) for j, s in BLOCKS for r in RECIPES)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def object_sha(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def confined(value):
    p = Path(value)
    p = p if p.is_absolute() else PHASE/p
    require(p.is_absolute() and p.is_relative_to(PHASE) and '..' not in p.parts, 'Project-confined path required')
    require(not any(q.is_symlink() for q in (p, *p.parents) if q.is_relative_to(PHASE)), 'Symlink path forbidden')
    return p


def read(value):
    return json.loads(confined(value).read_text())


def record(value):
    p = confined(value)
    b = p.read_bytes()
    return {'path': str(p.relative_to(PHASE)), 'sha256': sha_bytes(b), 'bytes': len(b)}


def verify(rec):
    require(set(rec) == {'path', 'sha256', 'bytes'}, 'Exact descriptor schema required')
    p = confined(rec['path'])
    require(record(p) == dict(rec, path=str(p.relative_to(PHASE))), 'Bound artifact changed: '+str(p))
    return p


def write(path, value):
    p = confined(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def fresh_directory(path):
    p = confined(path)
    require(not p.is_relative_to(PACKET) and not p.is_relative_to(V4), 'Execution output must be outside sealed source')
    p.mkdir(parents=True, exist_ok=False)
    return p


def verify_sources():
    manifest = read(PACKET/'MANIFEST.json')
    require(manifest['schema'] == 'amazon_native_warm_source_manifest_v1' and
            manifest['payload_file_count'] == len(manifest['payload']), 'Prepared manifest schema/count differs')
    actual_payload = sorted(str(p.relative_to(PACKET)) for p in PACKET.rglob('*')
                            if p.is_file() and p.name != 'MANIFEST.json')
    require(sorted(r['path'] for r in manifest['payload']) == actual_payload, 'Prepared payload inventory differs')
    for row in manifest['payload']:
        p = confined(PACKET/row['path'])
        require(p.is_relative_to(PACKET), 'Prepared payload path escapes packet')
        require(record(p)['sha256'] == row['sha256'] and p.stat().st_size == row['bytes'], 'Prepared payload changed')
    bindings = read(PACKET/'SOURCE_BINDINGS.json')
    for row in bindings['files']:
        verify(row)
    # Only engineering status/count fields are inspected; no predictive result is imported.
    check = read(verify(bindings['native_cpu_receipt']))
    require(check['status'] == 'passed' and len(check['family_records']) == 8 and
            all(r['status'] == 'passed' for r in check['family_records']), 'Root native8 engineering passage required')
    require(check['packet_manifest_sha256'] == bindings['v4_manifest_sha256'], 'Native qualification source differs')
    return record(PACKET/'MANIFEST.json')


def admission(path, kind, output, *, data=None):
    source = verify_sources()
    p = confined(path)
    require(not p.is_relative_to(PACKET), 'Root release must be outside sealed source')
    a = read(p)
    require(a['schema'] == 'amazon_native_warm_release_v1' and a['kind'] == kind and
            a['execution_authorized'] is True and a['root_observed_source_review'] is True and
            a['automatic_retry_authorized'] is False and a['test_labels_authorized'] is False and
            a['packet_manifest'] == source and a['output'] == str(confined(output).relative_to(PHASE)),
            'Exact separate source-reviewed root release required')
    verify(a['source_review'])
    if data is not None:
        require(a['data_manifest'] == record(data), 'Release binds a different data projection')
    return a, record(p)


def official_reference(value):
    require(isinstance(value, dict) and set(value) == {'repository', 'commit', 'blob_path', 'download_url'} and
            value['repository'] == 'https://github.com/yandex-research/heterophilous-graphs' and
            re.fullmatch('[0-9a-f]{40}', value['commit']) is not None and
            value['blob_path'] == 'data/amazon_ratings.npz' and
            value['download_url'] == 'https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/'+
                                     value['commit']+'/data/amazon_ratings.npz',
            'Root-resolved commit-pinned official Amazon NPZ reference required')


def import_native():
    for prefix in ('core', 'native_source'):
        for name, module in tuple(sys.modules.items()):
            if name == prefix or name.startswith(prefix+'.'):
                require(hasattr(module, '__file__') and Path(module.__file__).resolve().is_relative_to(V4),
                        'Conflicting native/core module already imported')
    sys.path.insert(0, str(V4))
    return importlib.import_module('native_source.adapter'), importlib.import_module('native_source.tokens')


def runtime_file_rows():
    import torch
    import numpy
    import scipy
    import torch_geometric
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.utils import to_scipy_sparse_matrix, add_remaining_self_loops, to_undirected, scatter
    from torch_geometric.datasets import HeterophilousGraphDataset
    import torch.optim._functional
    import scipy.sparse._sparsetools
    numpy_binary = importlib.import_module('numpy.core._multiarray_umath')
    source = [torch.__file__, numpy.__file__, scipy.__file__, torch_geometric.__file__,
              inspect.getfile(gcn_norm), inspect.getfile(to_scipy_sparse_matrix),
              inspect.getfile(add_remaining_self_loops), inspect.getfile(to_undirected),
              inspect.getfile(HeterophilousGraphDataset), torch.optim._functional.__file__,
              inspect.getfile(torch.optim.AdamW), inspect.getfile(inspect.unwrap(torch.optim.AdamW.step)),
              inspect.getfile(torch.optim.Adam), inspect.getfile(inspect.unwrap(torch.optim.Adam.step)),
              inspect.getfile(torch.func.functional_call), inspect.getfile(scatter)]
    binary = [sys.executable, torch._C.__file__, numpy_binary.__file__, scipy.sparse._sparsetools.__file__]
    rows = {}
    for kind, paths in (('source', source), ('binary', binary)):
        for value in paths:
            p = Path(value).resolve(); b = p.read_bytes()
            rows[str(p)] = {'path': str(p), 'sha256': sha_bytes(b), 'size': len(b), 'kind': kind}
    pins = ((inspect.getfile(gcn_norm), 'nn__conv__gcn_conv.py'),
            (inspect.getfile(to_scipy_sparse_matrix), 'utils__convert.py'),
            (inspect.getfile(add_remaining_self_loops), 'utils__loop.py'),
            (inspect.getfile(to_undirected), 'utils__undirected.py'),
            (inspect.getfile(HeterophilousGraphDataset), 'datasets__heterophilous_graph_dataset.py'))
    for actual, name in pins:
        require(Path(actual).read_bytes() == (V4/'pyg_source_pins'/name).read_bytes(), 'Pinned PyG source differs')
    return [rows[p] for p in sorted(rows)]


def configure_runtime(device):
    # Ordinary in-process library settings, never an OS isolation/supervision wrapper.
    require(device == 'cpu' or device.startswith('cuda:'), 'Explicit cpu or indexed CUDA device required')
    qualified_python = read(PACKET/'SOURCE_BINDINGS.json')['qualified77_interpreter']
    require(str(Path(sys.executable).absolute()) == qualified_python['path'] and
            sha_bytes(Path(sys.executable).read_bytes()) == qualified_python['sha256'] and
            sys.version_info[:2] == (3, 12), 'Root native-qualified77 interpreter required')
    if device.startswith('cuda:'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        require(os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8', ':16:8'), 'Deterministic cuBLAS workspace required')
    import torch
    import numpy
    import scipy
    import torch_geometric
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    versions = {'torch': torch.__version__, 'numpy': numpy.__version__, 'scipy': scipy.__version__,
                'torch_geometric': torch_geometric.__version__}
    require(versions == {'torch': '2.7.1', 'numpy': '1.26.4', 'scipy': '1.16.0', 'torch_geometric': '2.7.0'},
            'Source-qualified77 package versions required')
    hardware = {'device': device}
    if device.startswith('cuda:'):
        require(torch.cuda.is_available(), 'Requested CUDA unavailable')
        torch.cuda.set_device(torch.device(device))
        prop = torch.cuda.get_device_properties(torch.device(device))
        hardware.update(name=prop.name, total_memory=prop.total_memory,
                        uuid=str(getattr(prop, 'uuid', 'unavailable')), cuda_build=torch.version.cuda)
    settings = {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
                'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                'default_device': str(torch.get_default_device()), 'default_dtype': str(torch.get_default_dtype()),
                'float32_matmul_precision': torch.get_float32_matmul_precision(),
                'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    return torch, {'versions': versions, 'python_version': sys.version, 'files': runtime_file_rows(),
                   'settings': settings, 'hardware': hardware}


def runtime(device, release):
    torch, actual = configure_runtime(device)
    supplied = read(verify(release['runtime_receipt']))
    require(supplied['schema'] == 'amazon_native_warm_runtime_v1' and
            supplied['packet_manifest'] == release['packet_manifest'] and supplied['runtime'] == actual,
            'Actual numerical files/settings/device/source differ from root runtime receipt')
    return torch, actual


def load_public(data_manifest, *, with_validation=False):
    import numpy as np
    import torch
    d = read(data_manifest)
    require(d['schema'] == 'amazon_native_train_validation_projection_v1' and d['dataset'] == 'amazon-ratings' and
            d['class_schema'] == [0, 1, 2, 3, 4] and d['split_count'] == 10 and
            d['test_label_artifacts'] == [] and d['raw_all_node_label_payload_decoded'] is True and
            d['test_labels_used_for_fitting_grouping_selection_or_scoring'] is False,
            'Official block-specific TRAIN/VAL projection required')
    require(d['packet_manifest'] == verify_sources(), 'Data projection belongs to a different prepared source')
    verify(d['raw_release'])
    producer, producer_record = admission(d['producer_release']['path'], 'data_projection', confined(data_manifest).parent)
    official_reference(d['official_release_reference'])
    require(producer_record == d['producer_release'] and producer['raw_release'] == d['raw_release'] and
            producer['official_release_reference'] == d['official_release_reference'] and
            producer['runtime_device'] == 'cpu' and producer['raw_label_payload_decode_disclosed'] is True and
            producer['class_schema'] == d['class_schema'], 'Data producer release/source/raw authority differs')
    producer_runtime = read(verify(producer['runtime_receipt']))
    require(producer_runtime['schema'] == 'amazon_native_warm_runtime_v1' and
            producer_runtime['packet_manifest'] == d['packet_manifest'] and
            producer_runtime['runtime'] == d['runtime'], 'Data producer runtime receipt differs')
    with np.load(verify(d['public_graph']), allow_pickle=False) as arrays:
        require(set(arrays.files) == {'features', 'edge_index', 'train_mask', 'val_mask', 'test_mask'},
                'Public graph must contain exactly public arrays and no labels')
        x = torch.from_numpy(arrays['features'].copy())
        edge = torch.from_numpy(arrays['edge_index'].copy())
        masks = {name: torch.from_numpy(arrays[name].copy()) for name in ('train_mask', 'val_mask', 'test_mask')}
    n = x.shape[0]
    require(x.ndim == 2 and x.dtype == torch.float32 and torch.isfinite(x).all().item(), 'Native float32 public features required')
    require(edge.dtype == torch.int64 and edge.ndim == 2 and edge.shape[0] == 2 and
            edge.numel() > 0 and int(edge.min()) >= 0 and int(edge.max()) < n, 'Native edge schema invalid')
    require(all(m.dtype == torch.bool and tuple(m.shape) == (n, 10) for m in masks.values()), 'Native [nodes,10] masks required')
    for j, _ in BLOCKS:
        require((masks['train_mask'][:, j].to(torch.int8)+masks['val_mask'][:, j].to(torch.int8)+
                 masks['test_mask'][:, j].to(torch.int8) == 1).all().item(), 'Official roles must be disjoint/complete per block')
    # Compile the exact commit-pinned helper AST without importing its unrelated module globals.
    tree = ast.parse((AUTHOR/'utils.py').read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'heter_fixed_splits')
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(AUTHOR/'utils.py'), 'exec'), namespace)
    official = {}
    for j, s in BLOCKS:
        holder = SimpleNamespace(**masks); target = SimpleNamespace()
        with redirect_stdout(io.StringIO()):
            selected = namespace['heter_fixed_splits']([holder], target, j)
        train_ids = selected.train_mask.nonzero(as_tuple=False).flatten().tolist()
        val_ids = selected.val_mask.nonzero(as_tuple=False).flatten().tolist()
        require(train_ids and val_ids, 'Empty official TRAIN/VAL role')
        train = load_labels(d['train_labels'][str(j)], train_ids)
        validation = load_labels(d['validation_labels'][str(j)], val_ids) if with_validation else None
        official[j] = {'train_ids': tuple(train_ids), 'train_labels': train,
                       'validation_ids': tuple(val_ids), 'validation_labels': validation, 'seed': s}
    return d, x, edge, official


def load_labels(descriptor, expected_ids):
    import numpy as np
    with np.load(verify(descriptor), allow_pickle=False) as a:
        require(set(a.files) == {'ids', 'labels'}, 'Only compact authorized role labels accepted')
        ids, values = a['ids'].tolist(), a['labels'].tolist()
        require(a['ids'].dtype == np.int64 and a['labels'].dtype == np.int64, 'Compact integer role arrays required')
    require(ids == list(expected_ids) and len(values) == len(ids) and
            all(type(y) is int and 0 <= y < 5 for y in values), 'Compact label rows differ from official role')
    return tuple(values)


def fit_control(official, split, seed):
    import_native()
    module = importlib.import_module('train_roles')
    ids = module.NativeTrainIds(official['train_ids'], split, 'root-bound-official-projection')
    prefix = f'condresp-amazon-v1|split={split}|opt={seed}'
    roles = module.fixed_stratified_roles(ids, official['train_labels'], class_count=5,
            role_seed=prefix+'|roles', explicit_role_split_receipt='prospective_floor4n_over5_v1')
    require(roles.fit and roles.control, 'Empty fit/control role')
    return roles


def preserved(data, release):
    verify_sources(); verify(release['data_manifest'])
    for key in ('raw_release', 'producer_release', 'public_graph'):
        verify(data[key])
    for key in ('train_labels', 'validation_labels'):
        for rec in data[key].values():
            verify(rec)


def plain(value):
    if is_dataclass(value):
        return {'dataclass': type(value).__name__, 'fields': plain(asdict(value))}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    require(value is None or type(value) in (str, int, float, bool), 'Unqualified native primitive')
    return value
