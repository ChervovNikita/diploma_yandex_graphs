"""Source/data/admission custody. Numerical imports occur only after a root gate."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

PACKET = Path(__file__).resolve().parent
PHASE = PACKET.parent
BLOCKS = ((0, 17), (1, 29), (2, 43))


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def object_sha(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def confined(value):
    p = Path(value)
    p = p if p.is_absolute() else PHASE / p
    require(p.is_relative_to(PHASE) and '..' not in p.parts, 'Workspace phase-confined path required')
    require(not any(q.is_symlink() for q in (p, *p.parents) if q.is_relative_to(PHASE)), 'Symlink forbidden')
    return p


def read(value):
    return json.loads(confined(value).read_text())


def record(value):
    p = confined(value)
    h = hashlib.sha256()
    size = 0
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
            size += len(chunk)
    return {'path': str(p.relative_to(PHASE)), 'sha256': h.hexdigest(), 'bytes': size}


def verify(row):
    require(type(row) is dict and set(row) == {'path', 'sha256', 'bytes'}, 'Exact file descriptor required')
    p = confined(row['path'])
    require(record(p) == row, 'Bound artifact changed: ' + str(p))
    return p


def write(path, value):
    p = confined(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return record(p)


def verify_sources():
    m = read(PACKET / 'MANIFEST.json')
    s = read(PACKET / 'SEAL.json')
    require(m['schema'] == 'amazon_polynormer_source_manifest_v2' and
            s['schema'] == 'amazon_polynormer_source_seal_v2' and
            s['manifest'] == record(PACKET / 'MANIFEST.json'), 'Source manifest/seal differs')
    actual = sorted(str(p.relative_to(PACKET)) for p in PACKET.rglob('*')
                    if p.is_file() and p.name not in ('MANIFEST.json', 'SEAL.json'))
    require(actual == sorted(r['path'] for r in m['payload']), 'Exact source payload inventory differs')
    for row in m['payload']:
        p = confined(PACKET / row['path'])
        require(p.is_relative_to(PACKET) and record(p)['sha256'] == row['sha256'] and
                p.stat().st_size == row['bytes'], 'Source payload changed')
    for row in read(PACKET / 'SOURCE_BINDINGS.json')['read_or_hashed_files']:
        # Array/checkpoint bodies are not provenance source rows.
        require(Path(row['path']).suffix not in ('.npz', '.npy', '.pt', '.pth'), 'Payload in source provenance')
        verify(row)
    return {'manifest': record(PACKET / 'MANIFEST.json'), 'seal': record(PACKET / 'SEAL.json')}


def gate(path, kind, output):
    source = verify_sources()
    p = confined(path)
    require(not p.is_relative_to(PACKET), 'Root release must be outside sealed source')
    a = read(p)
    require(a['schema'] == 'amazon_polynormer_root_release_v2' and a['kind'] == kind and
            a['execution_authorized'] is True and a['independent_source_review_passed'] is True and
            a['automatic_retry_authorized'] is False and a['test_labels_authorized'] is False and
            a['source'] == source and a['output'] == str(confined(output).relative_to(PHASE)) and
            a['self_path'] == str(p.relative_to(PHASE)),
            'Separate exact source-reviewed root release required')
    review = read(verify(a['source_review']))
    require(review['status'] == 'passed' and review['source'] == source,
            'Independent source review does not pass this exact packet')
    for row in a['custody_inputs']:
        verify(row)
    ckeys = ('runtime_receipt', 'consumer_release', 'registry', 'qualification_freeze',
             'resource_admission', 'closure_freeze', 'attempt_registry')
    require(all(a[k] in a['custody_inputs'] for k in ckeys if k in a), 'Admission omits a required custody input')
    require(type(a['caps']['wall_seconds']) in (int, float) and a['caps']['wall_seconds'] > 0 and
            all(type(a['caps'][k]) is int and a['caps'][k] > 0 for k in
                ('rss_bytes', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')),
            'Prospective whole-process caps required')
    interpreter(a)
    if kind in ('qualify', 'fit', 'evaluate'):
        validate_consumer(a)
    if kind in ('qualify', 'fit', 'all'):
        require('attempt_registry' in a, 'Complete preserved attempt registry required')
        attempts = read(verify(a['attempt_registry']))
        require(attempts['schema'] == 'amazon_polynormer_attempt_registry_v2' and
                attempts['source'] == source and attempts['preserved_failures'] == record(PACKET / 'PRESERVED_FAILURES.json') and
                attempts['all_prior_failed_incomplete_and_superseded_attempts_disclosed'] is True,
                'Attempt lineage/source/preserved failures differ')
        for row in attempts['prior_attempt_artifacts']:
            require(row in a['custody_inputs'], 'Prior attempt artifact omitted from admission custody')
            verify(row)
    if kind == 'consumer_prepare':
        validate_projection(a)
    return a, record(p)


def preserve(a, admission_record):
    require(verify_sources() == a['source'], 'Source custody changed')
    verify(admission_record)
    verify(a['source_review'])
    for row in a['custody_inputs']:
        verify(row)
    interpreter(a)
    if 'consumer_release' in a:
        validate_consumer(a)
    if a['kind'] == 'consumer_prepare':
        validate_projection(a)


def interpreter(a):
    row = a['interpreter']
    p = Path(row['path']).resolve()
    require(p.is_absolute() and set(row) == {'path', 'sha256', 'bytes'}, 'Exact interpreter descriptor required')
    b = p.read_bytes()
    require(digest(b) == row['sha256'] and len(b) == row['bytes'], 'Admitted interpreter changed')


def runtime_file(path, kind):
    p = Path(path).resolve()
    h = hashlib.sha256()
    size = 0
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
            size += len(chunk)
    return {'path': str(p), 'sha256': h.hexdigest(), 'bytes': size, 'kind': kind}


def validate_consumer(a):
    v = read(verify(a['consumer_release']))
    design = read(PACKET / 'DESIGN.json')['data_projection']
    require(v['schema'] == 'amazon_polynormer_consumer_release_v2' and v['source'] == a['source'] and
            v['execution_authorized'] is True and v['test_labels_authorized'] is False and
            v['existing_data_manifest'] == design['existing_manifest'], 'Exact new consumer release required')
    d = read(verify(v['existing_data_manifest']))
    require(d['public_graph'] == design['public_graph_binding_from_manifest'] and
            d['train_labels'] == design['official_TRAIN_bindings_from_manifest'] and
            d['validation_labels'] == design['official_VAL_bindings_from_manifest'] and
            v['producer_release'] == d['producer_release'] and v['producer_packet'] == d['packet_manifest'] and
            v['raw_descriptor'] == d['raw_release'] and d['test_label_artifacts'] == [],
            'Consumer/data/producer identity differs')
    rows = [v['existing_data_manifest'], d['producer_release'], d['packet_manifest'], d['public_graph'],
            *d['train_labels'].values(), *d['validation_labels'].values(), *v['prior_role_receipts'].values()]
    require(set(v['prior_role_receipts']) == {'0', '1', '2'} and
            all(row in a['custody_inputs'] for row in rows), 'Complete consumer custody/role coverage omitted')
    for row in rows:
        verify(row)
    expected_roles = v['preprocessing_identity']['roles']
    require([(r['split'], r['block_seed']) for r in expected_roles] == list(BLOCKS),
            'Consumer preprocessing omits exact three-block role identities')
    for expected in expected_roles:
        prior = read(verify(v['prior_role_receipts'][str(expected['split'])]))
        require(prior['schema'] == 'amazon_fixed_role_identity_v1' and
                prior['data_manifest'] == v['existing_data_manifest'] and prior['role_record'] == expected and
                prior['prior_artifact_comparison_satisfied'] is True, 'Unreviewed/inconsistent prior role identity')


def validate_projection(a):
    design = read(PACKET / 'DESIGN.json')['data_projection']
    require(a['existing_data_manifest'] == design['existing_manifest'] and a['device'] == 'cpu',
            'Bound CPU-only existing projection preparation required')
    d = read(verify(a['existing_data_manifest']))
    require(d['public_graph'] == design['public_graph_binding_from_manifest'] and
            d['train_labels'] == design['official_TRAIN_bindings_from_manifest'] and
            d['validation_labels'] == design['official_VAL_bindings_from_manifest'] and
            d['test_label_artifacts'] == [] and d['raw_all_node_label_payload_decoded'] is True,
            'Existing exact public/TRAIN/VAL projection differs')
    rows = [a['existing_data_manifest'], d['producer_release'], d['packet_manifest'], d['public_graph'],
            *d['train_labels'].values(), *d['validation_labels'].values()]
    require(all(row in a['custody_inputs'] for row in rows), 'Consumer preparation omits exact producer/data custody')
    for row in rows:
        verify(row)
    return d


def imported_source(relative, name):
    path = PACKET / relative
    require(name not in sys.modules, 'Fresh source module name required')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime(device, a, *, capture=False, configure=True):
    """Called after gate; no package install, device fallback or runtime mixing."""
    import inspect
    require(device == 'cpu' or (device.startswith('cuda:') and device[5:].isdigit()), 'Explicit device required')
    if configure and device.startswith('cuda:'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        require(os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8', 'Fixed cuBLAS workspace differs')
    import torch
    import numpy
    import scipy
    import torch_geometric
    from torch_geometric.nn import GATConv, MessagePassing
    from torch_geometric.nn.dense.linear import Linear as PyGLinear
    from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, scatter, coalesce, softmax
    import scipy.sparse._sparsetools
    numpy_binary = __import__('numpy.core._multiarray_umath', fromlist=['__file__'])
    if configure:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cpu')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    require(torch.get_default_dtype() == torch.float32, 'FP32 default required')
    hardware = {'device': device, 'host': platform.node(), 'machine': platform.machine()}
    if device.startswith('cuda:'):
        require(torch.cuda.is_available(), 'Admitted CUDA unavailable')
        torch.cuda.set_device(torch.device(device))
        props = torch.cuda.get_device_properties(torch.device(device))
        hardware.update(name=props.name, total_memory=props.total_memory,
                        uuid=str(getattr(props, 'uuid', 'unavailable')), cuda_build=str(torch.version.cuda))
    paths = [(sys.executable, 'binary'), (torch._C.__file__, 'binary'),
             (numpy_binary.__file__, 'binary'), (scipy.sparse._sparsetools.__file__, 'binary')]
    objects = (torch, numpy, scipy, torch_geometric, GATConv, MessagePassing, PyGLinear,
               torch.nn.Linear, torch.nn.LayerNorm,
               torch.optim.Adam, inspect.unwrap(torch.optim.Adam.step), to_undirected,
               remove_self_loops, add_self_loops, scatter, coalesce, softmax)
    paths += [(inspect.getfile(o), 'source') for o in objects]
    paths += [(str(p), 'binary') for p in (Path(torch.__file__).parent / 'lib').glob('*.so*') if p.is_file()]
    for prefix in ('torch_scatter', 'torch_sparse', 'pyg_lib'):
        package = sys.modules.get(prefix)
        if package is not None and getattr(package, '__file__', None):
            paths += [(str(p), 'binary') for p in Path(package.__file__).parent.rglob('*.so*') if p.is_file()]
    files = {}
    for name, kind in paths:
        row = runtime_file(name, kind)
        files[row['path']] = row
    actual = {'python': sys.version, 'versions': {k: str(v.__version__) for k, v in
              (('torch', torch), ('numpy', numpy), ('scipy', scipy), ('torch_geometric', torch_geometric))},
              'files': [files[k] for k in sorted(files)], 'hardware': hardware,
              'settings': {'deterministic': torch.are_deterministic_algorithms_enabled(),
                'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
                'precision': torch.get_float32_matmul_precision(), 'dtype': str(torch.get_default_dtype()),
                'default_device': str(torch.empty(0).device),
                'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
    require(actual['versions'] == a['expected_versions'], 'Prospectively admitted package versions differ')
    require(actual['versions'] in [p['versions'] for p in read(PACKET / 'DEPENDENCIES.json')['admissible_existing_profiles'].values()],
            'Runtime differs from the two prospectively declared existing profiles')
    matching_profile = next(p for p in read(PACKET / 'DEPENDENCIES.json')['admissible_existing_profiles'].values()
                            if p['versions'] == actual['versions'])
    require(platform.python_version() == matching_profile['python'], 'Declared existing Python profile differs')
    require(actual['settings']['deterministic'] is True and actual['settings']['warn_only'] is False and
            actual['settings']['precision'] == 'highest' and actual['settings']['matmul_tf32'] is False and
            actual['settings']['default_device'] == 'cpu' and
            actual['settings']['cudnn_tf32'] is True and actual['settings']['cudnn_deterministic'] is True and
            actual['settings']['cudnn_benchmark'] is False and actual['settings']['threads'] == 1 and
            actual['settings']['interop_threads'] == 1 and
            (device == 'cpu' or actual['settings']['cublas_workspace'] == ':4096:8'),
            'Actual deterministic/precision flags differ from fixed prospective recipe')
    if not capture:
        receipt = read(verify(a['runtime_receipt']))
        require(receipt['schema'] == 'amazon_polynormer_runtime_v2' and receipt['source'] == a['source'] and
                receipt['runtime'] == actual, 'Exact admitted runtime files/settings/device differ')
    return actual


def fit_id(row):
    return 'split%d_%s%s_seed%d' % (row['split'], row['kind'],
                '' if row['member'] is None else '_member%d' % row['member'], row['seed'])


def schedule():
    return read(PACKET / 'DESIGN.json')['physical_fit_schedule']


def role_record(split, seed, train_ids, labels, val_ids):
    by_class = {}
    for node, label in zip(train_ids, labels):
        by_class.setdefault(label, []).append(node)
    fit, control = set(), set()
    role_seed = f'condresp-amazon-v1|split={split}|opt={seed}|roles'
    for y, nodes in sorted(by_class.items()):
        order = sorted(nodes, key=lambda n: (digest(f'{role_seed}|fit_control|{y}|{n}'.encode()), n))
        cut = 4 * len(nodes) // 5
        fit.update(order[:cut])
        control.update(order[cut:])
    ordered_fit = tuple(n for n in train_ids if n in fit)
    ordered_control = tuple(n for n in train_ids if n in control)
    require(ordered_fit and ordered_control and not fit.intersection(control) and
            fit.union(control) == set(train_ids), 'Exact fit/control partition invalid')
    rec = {'split': split, 'block_seed': seed, 'role_seed': role_seed,
           'FIT': {'count': len(ordered_fit), 'ordered_node_sha256': object_sha(list(ordered_fit))},
           'control': {'count': len(ordered_control), 'ordered_node_sha256': object_sha(list(ordered_control))},
           'VAL': {'count': len(val_ids), 'ordered_node_sha256': object_sha(list(val_ids))}}
    return ordered_fit, ordered_control, rec


def load_data(a, device, *, evaluation=False, prepare=False):
    """Reads only public NPZ plus compact official TRAIN/VAL; TEST has no reader."""
    import numpy as np
    import torch
    from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
    design = read(PACKET / 'DESIGN.json')['data_projection']
    if prepare:
        require(a['kind'] == 'consumer_prepare' and not evaluation and device == 'cpu', 'Separate CPU preparation gate required')
        d = validate_projection(a)
        consumer = {'schema': 'amazon_polynormer_consumer_release_v2', 'source': a['source'],
                    'execution_authorized': True, 'test_labels_authorized': False,
                    'existing_data_manifest': a['existing_data_manifest'], 'producer_release': d['producer_release'],
                    'producer_packet': d['packet_manifest'], 'raw_descriptor': d['raw_release']}
    else:
        consumer = read(verify(a['consumer_release']))
    require(consumer['schema'] == 'amazon_polynormer_consumer_release_v2' and
            consumer['source'] == a['source'] and consumer['execution_authorized'] is True and
            consumer['test_labels_authorized'] is False and
            consumer['existing_data_manifest'] == design['existing_manifest'], 'New consumer release required')
    d = read(verify(consumer['existing_data_manifest']))
    require(d['public_graph'] == design['public_graph_binding_from_manifest'] and
            d['train_labels'] == design['official_TRAIN_bindings_from_manifest'] and
            d['validation_labels'] == design['official_VAL_bindings_from_manifest'] and
            d['test_label_artifacts'] == [] and d['raw_all_node_label_payload_decoded'] is True,
            'Existing exact producer/data projection binding differs')
    require(consumer['producer_release'] == d['producer_release'] and
            consumer['producer_packet'] == d['packet_manifest'] and
            consumer['raw_descriptor'] == d['raw_release'], 'Consumer/producer custody differs')
    for row in (d['producer_release'], d['packet_manifest']):
        verify(row)
    # Raw descriptor remains bound metadata; the all-node raw label NPZ is never opened.
    with np.load(verify(d['public_graph']), allow_pickle=False) as z:
        require(set(z.files) == {'features', 'edge_index', 'train_mask', 'val_mask', 'test_mask'},
                'Label-free exact public schema required')
        x = torch.from_numpy(z['features'].copy())
        edge = torch.from_numpy(z['edge_index'].copy())
        masks = {k: torch.from_numpy(z[k].copy()) for k in ('train_mask', 'val_mask', 'test_mask')}
    require(x.dtype == torch.float32 and tuple(x.shape) == (24492, 300) and
            torch.isfinite(x).all().item(), 'Exact finite raw Amazon features required')
    require(edge.dtype == torch.int64 and edge.ndim == 2 and edge.shape[0] == 2 and
            edge.numel() and int(edge.min()) >= 0 and int(edge.max()) < 24492, 'Native edges invalid')
    require(all(m.dtype == torch.bool and tuple(m.shape) == (24492, 10) for m in masks.values()),
            'Native nodes-by-ten masks required')
    official = {}
    for j, seed in BLOCKS:
        require((sum(m[:, j].to(torch.int8) for m in masks.values()) == 1).all().item(),
                'Official roles overlap or omit nodes')
        ids = tuple(masks['train_mask'][:, j].nonzero().flatten().tolist())
        val_ids = tuple(masks['val_mask'][:, j].nonzero().flatten().tolist())
        require(len(ids) == 12246 and len(val_ids) == 6123, 'Fixed native role cardinality differs')
        labels = compact_labels(d['train_labels'][str(j)], ids)
        val_labels = compact_labels(d['validation_labels'][str(j)], val_ids)
        fit, control, rec = role_record(j, seed, ids, labels, val_ids)
        if not prepare:
            prior = read(verify(consumer['prior_role_receipts'][str(j)]))
            require(prior['schema'] == 'amazon_fixed_role_identity_v1' and prior['role_record'] == rec and
                    prior['data_manifest'] == consumer['existing_data_manifest'] and
                    prior['prior_artifact_comparison_satisfied'] is True,
                    'Roles differ from independently bound prior role identity')
        label_map = dict(zip(ids, labels))
        official[j] = {'fit': fit, 'fit_labels': tuple(label_map[n] for n in fit),
                       'val': val_ids, 'val_labels': val_labels, 'role_record': rec}
        if evaluation:
            official[j].update(control=control, control_labels=tuple(label_map[n] for n in control))
    edge = to_undirected(edge)
    edge, _ = remove_self_loops(edge)
    edge, _ = add_self_loops(edge, num_nodes=24492)
    preprocessing = {'public_graph': d['public_graph'], 'features': 'raw FP32 unchanged',
                     'edge_recipe': 'to_undirected/remove_self_loops/add_self_loops',
                     'edge_shape': list(edge.shape), 'edge_logical_sha256': digest(edge.numpy().tobytes()),
                     'roles': [official[j]['role_record'] for j, _ in BLOCKS]}
    if not prepare:
        require(preprocessing == consumer['preprocessing_identity'], 'Prospectively bound preprocessing differs')
    return x.to(device), edge.to(device), official, preprocessing


def compact_labels(row, ids):
    import numpy as np
    with np.load(verify(row), allow_pickle=False) as z:
        require(set(z.files) == {'ids', 'labels'} and z['ids'].dtype == np.int64 and
                z['labels'].dtype == np.int64 and z['ids'].ndim == z['labels'].ndim == 1,
                'Compact exact integer label schema required')
        require(z['ids'].tolist() == list(ids), 'Compact row order differs')
        labels = tuple(z['labels'].tolist())
    require(len(labels) == len(ids) and all(type(y) is int and 0 <= y < 5 for y in labels), 'Labels invalid')
    return labels
