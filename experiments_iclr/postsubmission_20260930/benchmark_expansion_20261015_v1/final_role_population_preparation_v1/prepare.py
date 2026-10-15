"""Allocation-only native TEST population preparation; never construct a model.

The supervisor uses stdlib only. Seven fresh, sequential CPU workers load the
existing native providers and split definitions. This is TEST metadata access,
including incidental label deserialization, and is not TEST scoring.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = ROOT / 'experiments_iclr/postsubmission_20260930'
EXPANSION = PHASE / 'benchmark_expansion_20261015_v1'
DESIGN = PHASE / 'reference_wrapper_redesign_20261013_v1'
HERE = EXPANSION / 'final_role_population_preparation_v1'
GIB = 1024 ** 3
DATASETS = ('citeseer', 'pubmed', 'coauthor-cs', 'coauthor-physics',
            'ogbg-molhiv', 'ogbg-molbace', 'ogbl-collab')
LIMITS = {'CPU_cores': 2, 'tensor_threads': 2, 'interop_threads': 1,
          'supervisor_address_space_bytes': 2 * GIB,
          'worker_address_space_bytes': 12 * GIB, 'worker_RSS_stop_bytes': 8 * GIB,
          'minimum_initial_available_bytes': 12 * GIB, 'minimum_live_available_bytes': 4 * GIB,
          'worker_wall_seconds': 900, 'total_wall_seconds': 3600,
          'disk_reservation_bytes': GIB, 'sample_seconds': 1, 'workers_concurrent': 1}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def guard():
    # Must precede all project reads/writes, including source verification.
    require(socket.gethostname() == 'anogena-2-0', 'Excluded route: wrong hostname')
    ids = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines()
    require(ids == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Excluded route: wrong GPU identity')
    require(Path(__file__).resolve().parent == HERE, 'Use the published allocation source copy')
    require(sys.byteorder == 'little', 'Native ID byte contract requires little-endian allocation')
    return {'hostname': socket.gethostname(), 'GPU_UUIDs': ids, 'checked_at_UTC': utc(),
            'GPU_usage': 'nvidia-smi identity query only; no CUDA work'}


def inside(path, root=ROOT):
    path = Path(path).resolve()
    require(path.is_relative_to(root), 'Path escapes allowed root: ' + str(path))
    return path


def receipt(path):
    path = inside(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def verify(pin):
    require(set(pin) == {'path', 'bytes', 'sha256'}, 'Exact receipt keys required')
    require(receipt(pin['path']) == pin, 'Changed source/input: ' + pin['path'])
    return Path(pin['path'])


def clean_pin(pin):
    require(pin.get('exists', True) is True, 'Previously missing native input')
    return {key: pin[key] for key in ('path', 'bytes', 'sha256')}


def write_json(path, value):
    path = inside(path)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return receipt(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_authority(expected):
    pin = receipt(HERE / 'MANIFEST.json')
    require(pin['sha256'] == expected, 'Unexpected population source manifest')
    manifest = read_json(pin['path'])
    require(manifest['schema'] == 'native-final-role-population-source-v1', 'Wrong source manifest')
    own_pins = []
    for item in manifest['files']:
        path = inside(HERE / item['relative_path'], HERE)
        actual = receipt(path)
        require(actual == {'path': str(path), 'bytes': item['bytes'], 'sha256': item['sha256']},
                'Population source bytes changed: ' + str(path))
        own_pins.append(actual)
    sources = read_json(HERE / 'SOURCE_BINDINGS.json')
    for item in sources['files']:
        verify(item)
    env = read_json(EXPANSION / 'expanded_dispatcher_v2/PYTHON_ENVIRONMENT.json')
    require(sys.executable == env['lexical_executable'] and sys.prefix == env['prefix'],
            'Use the qualified lexical .venv/bin/python interpreter')
    require(str(Path(sys.executable).resolve()) == env['resolved_executable']
            and receipt(sys.executable)['sha256'] == env['binary_sha256'], 'Python binary changed')
    verify(env['pyvenv_config'])
    return pin, own_pins + sources['files']


def load_inputs():
    roster = read_json(DESIGN / 'ROSTER.json')['recipes']
    bindings = read_json(EXPANSION / 'job_binding_preparation_v1/CELL_BINDINGS.json')['cells']
    require(len(roster) == len(bindings) == 27 and {r['id'] for r in roster} == set(bindings),
            'Exactly the frozen 27 presets required')
    require(Counter(bindings[r['id']]['family'] for r in roster) == {'node': 8, 'ogb': 16, 'link': 3},
            'Frozen family coverage changed')
    require({r['dataset'] for r in roster} == set(DATASETS), 'Frozen dataset set changed')
    for recipe in roster:
        cell = bindings[recipe['id']]
        for key in ('dataset', 'native_fixed_config', 'official_split_bindings', 'native_seed_offsets'):
            require(recipe[key] == cell[key], 'Recipe/cell mismatch: ' + recipe['id'] + '/' + key)
        keys = {str(i) for i in range(10)} if cell['family'] == 'node' else {'0'}
        require(set(recipe['official_split_bindings']) == keys, 'Missing native split binding')
    policy = read_json(HERE / 'TOLERANCES.json')
    require(policy['rtol'] == 1e-4 and policy['atol'] == 1e-5
            and policy['frozen_before_TEST_metadata'] is True, 'Frozen reconstruction criterion changed')
    return roster, bindings, policy


def definitions(path, names, namespace):
    """Execute exact source AST definitions, excluding imports and constructors."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    chosen = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    require(len(chosen) == len(names) and {node.name for node in chosen} == set(names), 'Native definitions absent')
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def resource_state(pid=None):
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(meminfo['MemAvailable'].split()[0]) * 1024
    # The allocation uses the cgroup v2 mount. Also inspect every visible ancestor
    # of the process cgroup so an inherited tighter limit cannot be overlooked.
    entries = [line.split(':', 2) for line in Path('/proc/self/cgroup').read_text().splitlines()]
    rows = [parts[2] for parts in entries if parts[:2] == ['0', '']]
    require(len(rows) == 1, 'cgroup v2 resource accounting required')
    mount = Path('/sys/fs/cgroup')
    candidate = mount / rows[0].lstrip('/')
    if not (candidate / 'memory.current').is_file():
        candidate = mount  # The mount may already be the process namespace root.
    scopes = []
    while candidate.is_relative_to(mount):
        if (candidate / 'memory.current').is_file():
            def scalar(name):
                value = (candidate / name).read_text().strip()
                return None if value == 'max' else int(value)
            current, maximum = scalar('memory.current'), scalar('memory.max')
            if maximum is not None:
                available = min(available, max(0, maximum - current))
            scopes.append({'path': str(candidate), 'current_bytes': current, 'maximum_bytes': maximum,
                           'peak_bytes': scalar('memory.peak') if (candidate / 'memory.peak').exists() else None,
                           'events': dict(line.split() for line in (candidate / 'memory.events').read_text().splitlines())})
        if candidate == mount:
            break
        candidate = candidate.parent
    require(scopes, 'Visible cgroup memory accounting required')
    result = {'available_bytes': available, 'cgroups': scopes,
              'cgroup_scope': 'shared allocation/ancestors, includes peers and page cache'}
    if pid is not None:
        try:
            status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
            result.update(pid=pid, RSS_bytes=int(status.get('VmRSS', '0 kB').split()[0]) * 1024,
                          RSS_high_water_bytes=int(status.get('VmHWM', '0 kB').split()[0]) * 1024,
                          threads=int(status['Threads']))
        except FileNotFoundError:
            result['process_exited_during_sample'] = True
    return result


def admission(output):
    state = resource_state()
    require(state['available_bytes'] >= LIMITS['minimum_initial_available_bytes'], 'Insufficient CPU RAM headroom')
    require(shutil.disk_usage(output).free >= LIMITS['disk_reservation_bytes'], 'Insufficient output disk headroom')
    cpus = sorted(os.sched_getaffinity(0))
    require(len(cpus) >= LIMITS['CPU_cores'], 'Two available CPU cores required')
    return {'memory': state, 'selected_CPU_affinity': cpus[:2], 'limits': LIMITS,
            'resource_envelope_measured': False, 'admitted_at_UTC': utc()}


def existing_native_provider(factory, **kwargs):
    """Call the original provider; refuse only a request to replace its data.

    This is a narrowly scoped provider lifecycle check. It does not intercept
    filesystem access, network calls, process creation, prompts or runtime caches.
    No dataset subclass or replacement data object is introduced.
    """
    from unittest.mock import patch
    def missing_native_input(*args, **unused):
        raise FileNotFoundError('Native provider requested download/process; the complete existing store is required')
    with patch.object(factory, 'download', missing_native_input), patch.object(factory, 'process', missing_native_input):
        return factory(**kwargs)


def access(output, action, **details):
    with (output / 'ACCESS.jsonl').open('a') as stream:
        stream.write(json.dumps({'at_UTC': utc(), 'action': action, **details}, sort_keys=True) + '\n')


def ordered_array(output, name, value, torch):
    require(value.device.type == 'cpu' and value.dtype == torch.long, 'CPU int64 identity required')
    value = value.detach().contiguous()
    path = output / (name + '.i64')
    with path.open('xb') as stream:
        if value.numel():
            stream.write(memoryview(value.numpy()).cast('B'))
    pin = receipt(path)
    return {'file': pin, 'dtype': 'int64', 'byteorder': 'little', 'shape': list(value.shape),
            'tensor_sha256': pin['sha256'], 'format': 'raw contiguous native int64; no header'}


def native_pins(cell):
    return {name: clean_pin(pin) for name, pin in cell['data_role_receipts'].items()
            if name != 'native_polynomial_basis_cache'}


def node_population(dataset, recipe, cell, output, torch, np):
    from torch_geometric.datasets import Coauthor, Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    pins = native_pins(cell)
    require(set(pins) == {'shared_processed_graph_container'}, 'Wrong native node input set')
    for pin in pins.values():
        verify(pin)
    names = {'citeseer': 'CiteSeer', 'pubmed': 'PubMed', 'coauthor-cs': 'CS', 'coauthor-physics': 'Physics'}
    root, name = inside(cell['data_root']) / dataset, names[dataset]
    require(root / name / 'processed/data.pt' == Path(pins['shared_processed_graph_container']['path']),
            'Native node factory path differs from admitted input')
    access(output, 'before_native_processed_provider_load', dataset=dataset, TEST_accessed=True,
           scope='full transductive graph and labels deserialized; class labels used only for exact native stratification')
    factory = Planetoid if dataset in ('citeseer', 'pubmed') else Coauthor
    source = existing_native_provider(factory, root=str(root), name=name, transform=NormalizeFeatures())
    data = source[0]
    scope = {'torch': torch, 'np': np}
    defs = definitions(DESIGN / 'native_sources/polyformer/node_classification/utils.py',
                       ['index_to_mask', 'take_rest', 'random_splits'], scope)
    count, classes = int(data.num_nodes), int(source.num_classes)
    require(data.y.ndim == 1 and len(data.y) == count, 'Native node labels/graph mismatch')
    per_class, validation = int(round(.6 * len(data.y) / classes)), int(round(.2 * len(data.y)))
    populations = {}
    for index in range(10):
        seed = recipe['official_split_bindings'][str(index)]['source_split_seed']
        require(type(seed) is int and 0 <= seed < 2 ** 32, 'Literal native split seed required')
        data = defs['random_splits'](data, classes, per_class, validation, seed)
        train, valid, test = data.train_mask, data.val_mask, data.test_mask
        require(all(mask.dtype == torch.bool and tuple(mask.shape) == (count,) for mask in (train, valid, test)),
                'Native source mask schema changed')
        require(not (train & valid).any().item() and not (train & test).any().item()
                and not (valid & test).any().item() and (train | valid | test).all().item(),
                'Native masks must form the complete disjoint partition')
        ids = torch.nonzero(test, as_tuple=False).flatten()
        require(len(ids) > 0, 'Empty native TEST population')
        artifact = ordered_array(output, f'split_{index}_TEST_ids', ids, torch)
        populations[str(index)] = {'records': len(ids), 'ids_sha256': artifact['tensor_sha256'],
                                   'ordered_ids': artifact, 'literal_split_seed': seed,
                                   'TRAIN_records': int(train.sum()), 'VALID_records': int(valid.sum()),
                                   'whole_graph_nodes': count, 'split_partition_verified': True}
    return {'populations': populations, 'source_receipts': list(pins.values()),
            'provider': type(source).__module__ + '.' + type(source).__name__,
            'source_num_classes_used_for_stratification': classes,
            'per_class_TRAIN_request': per_class, 'VALID_request': validation,
            'native_provider_graph_deserialized': True, 'serving_topology_constructed': False,
            'polynomial_basis_accessed': False,
            'topology_note': 'Graph input byte-pinned; topology and basis are left to the unchanged serving runtime.'}


def csv_ids(path, torch):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as stream:
        rows = list(csv.reader(stream))
    require(rows and all(len(row) == 1 for row in rows), 'One integer per official scaffold CSV row required')
    values = [int(row[0]) for row in rows]
    require(len(set(values)) == len(values), 'Repeated official graph ID')
    return torch.tensor(values, dtype=torch.long)


def molecular_population(dataset, recipe, cell, output, torch, np):
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.loader import DataLoader
    pins = native_pins(cell)
    require(set(pins) == {'shared_processed_graph_container', 'TRAIN_indices', 'VALID_indices'},
            'Wrong native molecular input set')
    for pin in pins.values():
        verify(pin)
    native = Path(pins['shared_processed_graph_container']['path']).parents[1]
    require(native == inside(cell['data_root']) / dataset.replace('-', '_'), 'Native molecular root changed')
    # The original OGB constructor offers to replace a cache when its release
    # marker is missing. Check the installed provider's own version metadata
    # first, without overriding input() or sandboxing normal runtime behavior.
    master = Path(inspect.getfile(PygGraphPropPredDataset)).parent / 'master.csv'
    pins['installed_OGB_master_metadata'] = receipt(master)
    with master.open(newline='') as stream:
        reader = csv.DictReader(stream)
        key_column = reader.fieldnames[0]
        versions = [row[dataset] for row in reader if row[key_column] == 'version']
    require(len(versions) == 1 and versions[0].isdecimal(), 'Installed native OGB version metadata required')
    marker = native / ('RELEASE_v' + versions[0] + '.txt')
    require(marker.is_file(), 'Existing OGB release marker required; do not update or replace the native cache')
    pins['native_OGB_release_marker'] = receipt(marker)
    split = native / 'split/scaffold'
    # Serving v2 consumes the original TEST CSV, not a newly synthesized PT file.
    test_path = split / 'test.csv.gz'
    require(test_path.is_file(), 'Original scaffold test.csv.gz is required; no download/fallback')
    access(output, 'before_official_TEST_CSV_hash_and_read', dataset=dataset, TEST_accessed=True,
           scope='complete official scaffold TEST graph index order; no scores')
    test_pin = receipt(test_path)
    pins['TEST_indices'] = test_pin
    roles = {role: csv_ids(verify(pins[key]), torch) for role, key in
             (('train', 'TRAIN_indices'), ('valid', 'VALID_indices'), ('test', 'TEST_indices'))}
    # Native session prefers an existing train/valid PT. Verify it gives exactly
    # the admitted CSV order; never ignore a shadow role used by actual serving.
    for role in ('train', 'valid'):
        pt = split / (role + '.pt')
        if pt.exists():
            pins[role + '_native_preferred_PT'] = receipt(pt)
            value = torch.load(pt, map_location='cpu', weights_only=True)
            if isinstance(value, dict) and role in value:
                value = value[role]
            require(torch.is_tensor(value) and value.dtype == torch.long
                    and torch.equal(value.detach().cpu().reshape(-1), roles[role]),
                    'Native preferred PT and admitted scaffold CSV disagree')
    access(output, 'before_native_processed_provider_load', dataset=dataset, TEST_accessed=True,
           scope='full molecule container including label tensors deserialized; labels neither inspected nor exported')
    source = existing_native_provider(PygGraphPropPredDataset, name=dataset, root=str(native.parent))
    require(source.name == dataset and source.num_tasks == 1 and 'classification' in source.task_type
            and source.transform is None, 'Exact original molecular provider required')
    total = len(source)
    for ids in roles.values():
        require((ids >= 0).all().item() and (ids < total).all().item(), 'Scaffold graph ID outside provider')
    combined = torch.cat([roles[role] for role in ('train', 'valid', 'test')])
    require(len(combined) == total and len(torch.unique(combined)) == total,
            'Native scaffold roles must partition all provider graphs')
    defs = definitions(EXPANSION / 'final_selected_scoring_v2/role_serving.py', ['MoleculeRoleView'], {})
    view = defs['MoleculeRoleView'](source, roles['test'], torch)
    loader = DataLoader(view, batch_size=32, shuffle=False, num_workers=0)
    eligible, skipped, batches = [], [], []
    for index, batch in enumerate(loader):
        ids = batch.smoke_graph_id.detach().cpu()
        expected = roles['test'][index * 32:(index + 1) * 32]
        require(torch.equal(ids, expected), 'Native packed-batch graph order changed')
        skip = batch.x.shape[0] == 1  # Exact main_pyg.eval and v2 role-serving law.
        (skipped if skip else eligible).append(ids)
        batches.append({'batch_index': index, 'graph_ids': ids.tolist(),
                        'packed_x_rows': int(batch.x.shape[0]), 'native_skipped': skip})
    served = torch.cat(eligible) if eligible else torch.empty(0, dtype=torch.long)
    omitted = torch.cat(skipped) if skipped else torch.empty(0, dtype=torch.long)
    require(len(served) > 0 and len(served) + len(omitted) == len(roles['test']), 'Incomplete molecular enumeration')
    official = ordered_array(output, 'official_TEST_ids', roles['test'], torch)
    evaluated = ordered_array(output, 'eligible_TEST_ids', served, torch)
    skipped_artifact = ordered_array(output, 'native_skipped_TEST_ids', omitted, torch)
    batch_pin = write_json(output / 'BATCHES.json', {'batch_size': 32, 'shuffle': False, 'num_workers': 0,
                                                  'skip_law': 'packed batch.x.shape[0] == 1', 'batches': batches})
    return {'populations': {'0': {'records': len(roles['test']), 'ids_sha256': evaluated['tensor_sha256'],
                                  'ordered_ids': evaluated, 'official_ordered_ids': official,
                                  'native_skipped_ids': skipped_artifact, 'eligible_records': len(served),
                                  'native_skipped_records': len(omitted), 'packed_batch_evidence': batch_pin}},
            'source_receipts': list(pins.values()), 'test_indices': test_pin,
            'provider': type(source).__module__ + '.' + type(source).__name__,
            'whole_provider_graphs': total, 'TRAIN_records': len(roles['train']), 'VALID_records': len(roles['valid']),
            'complete_scaffold_partition_verified': True}


def link_population(dataset, recipe, cell, output, torch, np):
    pins = native_pins(cell)
    require(set(pins) == {'graph', 'train', 'valid'}, 'Exact admitted native link files required')
    for pin in pins.values():
        verify(pin)
    folder = inside(cell['data_root']) / 'split/time'
    require(Path(pins['train']['path']) == folder / 'train.pt'
            and Path(pins['valid']['path']) == folder / 'valid.pt', 'Native time-split paths differ')
    access(output, 'before_official_TEST_query_hash_and_read', dataset=dataset, TEST_accessed=True,
           scope='all official positive/negative query endpoints and role membership; no predictions or scores')
    pins['test'] = receipt(folder / 'test.pt')
    split = torch.load(verify(pins['test']), map_location='cpu', weights_only=False)
    require(isinstance(split, dict) and {'edge', 'edge_neg'} <= set(split), 'Official Collab query schema required')
    for key in ('edge', 'edge_neg'):
        value = split[key]
        require(torch.is_tensor(value) and value.device.type == 'cpu' and value.dtype == torch.long
                and value.ndim == 2 and value.shape[1] == 2 and value.shape[0] > 0
                and ((value >= 0) & (value < 235868)).all().item(), 'Complete native Collab endpoint tensor required')
    positive, negative = len(split['edge']), len(split['edge_neg'])
    # Use original author concatenation, never sorting, sampling, deduplicating,
    # year-filtering TEST queries or constructing HashDataset/adjacency/model.
    defs = definitions(DESIGN / 'native_sources/buddy/buddy_src_data.py',
                       ['get_ogb_pos_edges', 'make_obg_supervision_edges'], {'torch': torch})
    labels, edge_index = defs['make_obg_supervision_edges']({'test': split}, 'test')
    endpoints = edge_index.t().contiguous()
    require(torch.equal(endpoints[:positive], split['edge'])
            and torch.equal(endpoints[positive:], split['edge_neg']), 'Native positive-then-negative order changed')
    require(len(labels) == positive + negative, 'Native supervision population mismatch')
    ids = ordered_array(output, 'TEST_ids', torch.arange(positive + negative, dtype=torch.long), torch)
    query = ordered_array(output, 'TEST_query_endpoints', endpoints, torch)
    return {'populations': {'0': {'records': positive + negative, 'ids_sha256': ids['tensor_sha256'],
                                  'positive': positive, 'negative': negative,
                                  'ordered_ids': ids, 'query_endpoints': query,
                                  'query_order': 'all edge rows, then all edge_neg rows; original file order'}},
            'source_receipts': list(pins.values()), 'data_files': pins,
            'provider': 'original admitted OGB temporal TEST PT; exact author supervision concatenation',
            'graph_container_deserialized': False, 'serving_topology_constructed': False,
            'BUDDY_hash_features_constructed': False,
            'topology_note': 'BUDDY year>=2007 applies to TRAIN only. Frozen contracts retain architecture-specific TEST topology.'}


def worker(dataset, output, source_sha):
    route = guard()
    source_pin, _ = source_authority(source_sha)
    roster, bindings, _ = load_inputs()
    require(dataset in DATASETS and output.parent.parent == HERE and output.name == dataset
            and inside(output, HERE) == output,
            'Worker output scope differs')
    output.mkdir()
    admit = admission(output)
    write_json(output / 'ADMISSION.json', {'route': route, **admit})
    os.sched_setaffinity(0, set(admit['selected_CPU_affinity']))
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS['worker_address_space_bytes'], LIMITS['worker_address_space_bytes']))
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        import torch_geometric
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden before Python starts')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        require(not torch.cuda.is_initialized(), 'CUDA was unexpectedly initialized')
        distributions = {}
        for name in ('torch', 'numpy', 'torch-geometric', 'ogb', 'scipy', 'pandas', 'scikit-learn', 'torch-sparse', 'torch-scatter'):
            try:
                dist = importlib.metadata.distribution(name)
                record_path = next((dist.locate_file(file) for file in (dist.files or [])
                                    if str(file).endswith('.dist-info/RECORD')), None)
                distributions[name] = {'version': dist.version,
                                       'installed_RECORD': receipt(record_path) if record_path else None}
            except importlib.metadata.PackageNotFoundError:
                distributions[name] = {'installed': False}
        write_json(output / 'DEPENDENCIES.json', {'python': sys.version, 'executable': sys.executable,
                   'distributions': distributions, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
                   'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
                   'torch_source': receipt(torch.__file__), 'pyg_source': receipt(torch_geometric.__file__)})
        from torch_geometric.datasets import Coauthor, Planetoid
        from torch_geometric.transforms import NormalizeFeatures
        from torch_geometric.loader import DataLoader
        from torch_geometric.data import Dataset, InMemoryDataset, Batch
        from ogb.graphproppred import PygGraphPropPredDataset
        provider_sources = [receipt(path) for path in sorted({inspect.getfile(obj) for obj in
                            (Coauthor, Planetoid, NormalizeFeatures, DataLoader, Dataset,
                             InMemoryDataset, Batch, PygGraphPropPredDataset)})]
        write_json(output / 'INSTALLED_PROVIDER_SOURCES.json', provider_sources)
        recipes = [r for r in roster if r['dataset'] == dataset]
        recipe, cell = recipes[0], bindings[recipes[0]['id']]
        for other in recipes[1:]:
            other_cell = bindings[other['id']]
            require(native_pins(other_cell) == native_pins(cell) and other_cell['data_root'] == cell['data_root'],
                    'Presets sharing a population use different admitted inputs')
            require(other['official_split_bindings'] == recipe['official_split_bindings'] or dataset == 'ogbl-collab',
                    'Shared node/molecular populations have different split recipes')
        functions = {'node': node_population, 'ogb': molecular_population, 'link': link_population}
        result = functions[cell['family']](dataset, recipe, cell, output, torch, np)
        result['source_receipts'] += provider_sources + [receipt(output / 'DEPENDENCIES.json'),
                                                        receipt(output / 'INSTALLED_PROVIDER_SOURCES.json')]
        for pin in result['source_receipts']:
            verify(pin)  # Fail closed if an input changed during provider enumeration.
        require(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized during population preparation')
        result.update(schema='native-TEST-dataset-population-v1', dataset=dataset, family=cell['family'],
                      source_manifest=source_pin, preset_ids=[r['id'] for r in recipes], status='complete',
                      TEST_accessed=True, TEST_access_kind='population_metadata_and_declared_incidental_deserialization',
                      TEST_scored=False, TEST_predictions_created=False, model_constructed=False,
                      optimizer_constructed=False, checkpoint_accessed=False,
                      wall_seconds=time.monotonic() - started,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      resources_at_end=resource_state(os.getpid()), CUDA_initialized=torch.cuda.is_initialized())
        write_json(output / 'RESULT.json', result)
    except BaseException as error:
        write_json(output / 'FAILURE.json', {'status': 'failed', 'error_type': type(error).__name__,
                   'error': str(error), 'traceback': traceback.format_exc(),
                   'wall_seconds': time.monotonic() - started, 'TEST_scored': False,
                   'TEST_metadata_access': 'See ACCESS.jsonl; may be partial; never infer no TEST access from failure',
                   'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
        raise


def metadata_module(path):
    spec = importlib.util.spec_from_file_location('_final_population_contracts', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def finish(output, roster, bindings, policy, source_pin, source_receipts):
    contracts_api = metadata_module(EXPANSION / 'final_selected_scoring_v2/contracts.py')
    # A shared 1.1 GB graph must not be streamed 20 times just because two node
    # presets each have ten split proofs. Verify each unique byte receipt once
    # here; all repeated uses must retain the same filesystem identity/times.
    checked = {}
    def checked_once(pin, root, *, json_metadata=False):
        path = contracts_api.inside(pin['path'], root)
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        key = (pin['path'], pin['bytes'], pin['sha256'])
        if key not in checked:
            contracts_api.verify(pin, root)
            after = path.stat()
            require(identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    'Source changed during final verification')
            checked[key] = identity
        require(checked[key] == identity, 'Verified source changed during final contract validation')
        return read_json(path) if json_metadata else path
    authority_fn = definitions(EXPANSION / 'final_selected_scoring_v2/backend.py', ['provider_authority'],
                               {'verify': checked_once, 'require': contracts_api.require})['provider_authority']
    (output / 'contracts').mkdir()
    (output / 'provider_evidence').mkdir()
    results = {dataset: read_json(output / dataset / 'RESULT.json') for dataset in DATASETS}
    index, evidence_count = {}, 0
    for recipe in roster:
        preset = recipe['id']
        cell, native = bindings[preset], results[recipe['dataset']]
        require(native['status'] == 'complete', 'Every unique dataset population must complete')
        result_pin = receipt(output / recipe['dataset'] / 'RESULT.json')
        contract = {'schema': 'native-final-TEST-role-contract-v2', 'role': 'TEST',
                    'family': cell['family'], 'preset_id': preset, 'dataset': recipe['dataset'],
                    'no_TEST_driven_selection': True, 'calibration': 'none',
                    'rtol': policy['rtol'], 'atol': policy['atol'],
                    'tolerance_authority': receipt(HERE / 'TOLERANCES.json'), 'populations': {}}
        if cell['family'] == 'node':
            contract.update(topology='unchanged_native_transductive_graph', index_source='same_literal_split_test_mask')
        elif cell['family'] == 'ogb':
            contract.update(topology='official_scaffold_TEST_graphs', batch_size=32, test_indices=native['test_indices'])
        else:
            architecture = recipe['architecture'].upper()
            contract.update(architecture=architecture,
                            topology='weighted_year2007_TRAIN_plus_weighted_VALID' if architecture == 'BUDDY' else 'unweighted_TRAIN_plus_VALID',
                            evaluator={'name': 'ogbl-collab', 'metric': 'hits@50', 'K': 50}, data_files=native['data_files'])
        for split_index, population in native['populations'].items():
            population = dict(population)
            ordered_pins = [value['file'] for value in population.values() if isinstance(value, dict) and 'file' in value]
            if 'packed_batch_evidence' in population:
                ordered_pins.append(population['packed_batch_evidence'])
            proof = {'schema': 'native-final-TEST-population-evidence-v2', 'role': 'TEST',
                     'complete_official_population': True, 'dataset': recipe['dataset'], 'preset_id': preset,
                     'split_index': int(split_index), **population,
                     'source_receipts': [source_pin, *source_receipts, *native['source_receipts'], result_pin, *ordered_pins],
                     'TEST_accessed': True, 'TEST_scored': False,
                     'population_scope': 'complete source population; molecular records include explicitly source-skipped graph IDs',
                     'topology_inference_or_model_qualification': False}
            if cell['family'] == 'link':
                proof['query_file'] = native['data_files']['test']
            elif cell['family'] == 'ogb':
                proof['test_indices'] = native['test_indices']
            pin = write_json(output / 'provider_evidence' / f'{preset}__split{split_index}.json', proof)
            contract['populations'][split_index] = {**population, 'provider_evidence': pin}
            evidence_count += 1
        contracts_api.validate_role_contract(contract)
        for key in contract['populations']:
            identity = {'preset_id': preset, 'dataset': recipe['dataset'], 'split_index': int(key)}
            if cell['family'] == 'node':
                identity['literal_split_seed'] = recipe['official_split_bindings'][key]['source_split_seed']
            selected = contracts_api.selected_role_contract(contract, identity)
            authority_fn(selected, identity, ROOT)
        index[preset] = write_json(output / 'contracts' / (preset + '.json'), contract)
    require(len(index) == 27 and evidence_count == 99, 'Missing final role coverage')
    return {'schema': 'native-final-TEST-role-contract-index-v1', 'contracts': index,
            'counts': {'presets': 27, 'provider_evidence_populations': 99, 'unique_dataset_populations': 43,
                       'node_presets': 8, 'node_literal_splits_per_preset': 10, 'molecular_presets': 16, 'link_presets': 3},
            'source_manifest': source_pin, 'TEST_accessed': True, 'TEST_scored': False,
            'source_only_preparation_permission_does_not_authorize_scoring': True}


def supervise(output_name, source_sha):
    route = guard()
    source_pin, sources = source_authority(source_sha)
    roster, bindings, policy = load_inputs()
    require(output_name.replace('_', '').isalnum() and output_name.startswith('allocation_population_'),
            'Use a fresh allocation_population_* child name')
    output = inside(HERE / output_name, HERE)
    require(not output.exists(), 'Preserve previous outputs; automatic retry/overwrite forbidden')
    output.mkdir()
    started, active = time.monotonic(), None
    try:
        resource.setrlimit(resource.RLIMIT_AS, (LIMITS['supervisor_address_space_bytes'],
                                              LIMITS['worker_address_space_bytes']))
        def deadline(signum, frame):
            raise TimeoutError('Frozen total wall limit reached')
        def terminated(signum, frame):
            raise InterruptedError('Population supervisor received SIGTERM')
        signal.signal(signal.SIGALRM, deadline)
        signal.signal(signal.SIGTERM, terminated)
        signal.setitimer(signal.ITIMER_REAL, LIMITS['total_wall_seconds'])
        write_json(output / 'START.json', {'route': route, 'source_manifest': source_pin, 'limits': LIMITS,
                   'started_at_UTC': utc(), 'TEST_metadata_access_planned': True, 'TEST_scoring_planned': False})
        for dataset in DATASETS:
            require(time.monotonic() - started < LIMITS['total_wall_seconds'], 'Total wall deadline reached')
            admitted = admission(output)
            os.sched_setaffinity(0, set(admitted['selected_CPU_affinity']))
            write_json(output / (dataset + '_ADMISSION.json'), admitted)
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       NUMEXPR_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                       PYTHONNOUSERSITE='1', TOKENIZERS_PARALLELISM='false')
            env.pop('PYTHONPATH', None)
            command = [sys.executable, '-B', str(HERE / 'prepare.py'), '--source-manifest-sha', source_sha,
                       '--output-name', output_name, '--worker', dataset]
            launched = time.monotonic()
            with (output / (dataset + '_WORKER.log')).open('xb') as log:
                active = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with (output / (dataset + '_RESOURCE_SAMPLES.jsonl')).open('x') as samples:
                    while active.poll() is None:
                        state = resource_state(active.pid)
                        samples.write(json.dumps({'elapsed_seconds': time.monotonic() - launched, **state}) + '\n')
                        samples.flush()
                        require(state.get('RSS_bytes', 0) <= LIMITS['worker_RSS_stop_bytes'], 'Worker RSS stop bound exceeded')
                        require(state['available_bytes'] >= LIMITS['minimum_live_available_bytes'], 'Allocation RAM headroom fell below bound')
                        require(time.monotonic() - launched < LIMITS['worker_wall_seconds']
                                and time.monotonic() - started < LIMITS['total_wall_seconds'], 'Population preparation wall limit reached')
                        time.sleep(LIMITS['sample_seconds'])
                returncode = active.wait()
                active = None
            write_json(output / (dataset + '_EXIT.json'), {'returncode': returncode,
                       'wall_seconds': time.monotonic() - launched, 'no_retry': True})
            require(returncode == 0, 'Population worker failed; preserve its log and partial access ledger: ' + dataset)
        index = finish(output, roster, bindings, policy, source_pin, sources)
        index_pin = write_json(output / 'ROLE_CONTRACT_INDEX.json', index)
        write_json(output / 'RESULT.json', {'status': 'complete', 'index': index_pin, 'counts': index['counts'],
                   'source_manifest': source_pin, 'TEST_accessed': True, 'TEST_scored': False,
                   'models_constructed': 0, 'training_updates': 0, 'predictions_created': 0,
                   'wall_seconds': time.monotonic() - started, 'limits': LIMITS,
                   'resource_evidence': 'Per-worker admission, one-second RSS/cgroup samples, ru_maxrss and exit receipts',
                   'limitations': 'No model, topology, forward parity, score, checkpoint, scientific or campaign qualification'})
        files = [receipt(path) for path in sorted(output.rglob('*')) if path.is_file()]
        final = write_json(output / 'OUTPUT_MANIFEST.json', {'schema': 'native-final-population-output-manifest-v1',
                                                           'source_manifest': source_pin, 'files': files})
        print(json.dumps({'status': 'complete', 'output_manifest': final, 'index': index_pin}, sort_keys=True))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stopped_worker = None
        if active is not None and active.poll() is None:
            stopped_worker = {'pid': active.pid, 'dataset': dataset, 'reason': str(error)}
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=5)
            stopped_worker['returncode'] = active.returncode
        write_json(output / 'FAILURE.json', {'status': 'failed', 'error_type': type(error).__name__,
                   'error': str(error), 'traceback': traceback.format_exc(), 'wall_seconds': time.monotonic() - started,
                   'TEST_scored': False, 'TEST_metadata_access': 'Possibly partial; inspect preserved worker ACCESS.jsonl',
                   'terminated_own_worker': stopped_worker,
                   'no_retry': True, 'partial_outputs_are_not_complete_role_authority': True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest-sha', required=True)
    parser.add_argument('--output-name', required=True)
    parser.add_argument('--worker', choices=DATASETS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(len(args.source_manifest_sha) == 64 and all(c in '0123456789abcdef' for c in args.source_manifest_sha),
            'External exact source-manifest SHA256 required')
    if args.worker:
        # guard runs inside worker before this lexical path is resolved or read.
        worker(args.worker, HERE / args.output_name / args.worker, args.source_manifest_sha)
    else:
        supervise(args.output_name, args.source_manifest_sha)


if __name__ == '__main__':
    main()
