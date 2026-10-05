"""Metadata custody first; never imports original model/training/loader modules."""
import hashlib
import json
from pathlib import Path

N = 24492
C = 5
SOURCE = 'amazon_polynormer_paired_family_source_preparation_20261003_v6'
EXECUTION = 'amazon_polynormer_paired_family_execution_root_20261003_v3'
FAMILIES = ('gnnm_boundary_4', 'independent_author_4_same_width')
SEEDS = ((0, 17, (17, 1026, 2035, 3044)),
         (1, 29, (29, 1038, 2047, 3056)),
         (2, 43, (43, 1052, 2061, 3070)))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def confined(phase, value):
    base = Path(phase).resolve()
    p = Path(value)
    p = p if p.is_absolute() else base / p
    require('..' not in p.parts and p.is_relative_to(base), 'Phase-confined path required')
    require(not any(q.is_symlink() for q in (p, *p.parents) if q.is_relative_to(base)),
            'Symlink forbidden')
    return p


def record(phase, value):
    p = confined(phase, value)
    h = hashlib.sha256()
    size = 0
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
            size += len(chunk)
    return {'path': str(p.relative_to(Path(phase).resolve())), 'sha256': h.hexdigest(),
            'bytes': size}


def verify(phase, row):
    require(set(row) == {'path', 'sha256', 'bytes'}, 'Exact descriptor required')
    require(record(phase, row['path']) == row, 'Changed artifact: ' + row['path'])
    return confined(phase, row['path'])


def read_json(path):
    return json.loads(Path(path).read_text())


def fixed_schedule():
    rows = []
    for split, block_seed, seeds in SEEDS:
        rows.append({'split': split, 'block_seed': block_seed, 'kind': FAMILIES[0],
                     'member': None, 'seed': block_seed})
        rows.extend({'split': split, 'block_seed': block_seed, 'kind': 'native_independent',
                     'member': m, 'seed': seed} for m, seed in enumerate(seeds))
    return rows


def fit_id(row):
    member = '' if row['member'] is None else '_member%d' % row['member']
    return 'split%d_%s%s_seed%d' % (row['split'], row['kind'], member, row['seed'])


def frozen_metadata(phase, descriptor):
    """Bind existing success freeze and JSON metadata only; no directory scans/checkpoints."""
    path = verify(phase, descriptor)
    freeze = read_json(path)
    require(freeze['schema'] == 'amazon_polynormer_success_freeze_v2' and
            freeze['status'] == 'success', 'Existing physical success freeze required')
    rows = {r['relative']: r['descriptor'] for r in freeze['files']}
    require(len(rows) == len(freeze['files']), 'Duplicate freeze paths')
    for name in ('TERMINAL.json', 'RESULT.json'):
        require(name in rows and confined(phase, rows[name]['path']) == path.parent / name,
                'Freeze metadata location differs')
        verify(phase, rows[name])
    terminal = read_json(path.parent / 'TERMINAL.json')
    require(terminal['physical_exit_code'] == 0 and terminal['status'] == 'success',
            'Physical exit success missing')
    return freeze, rows, read_json(path.parent / 'RESULT.json')


def prepare(phase, closure_descriptor, protocol_descriptor):
    """Return a complete bound inventory before any numerical import/array decode."""
    phase = Path(phase).resolve()
    packet = Path(__file__).resolve().parent
    binding = read_json(packet / 'SOURCE_BINDINGS.json')
    for row in binding['original_metadata']:
        verify(phase, row)
    for row in read_json(packet / 'MANIFEST.json')['payload']:
        require(record(phase, packet / row['path'])['sha256'] == row['sha256'],
                'Implementation source changed')
    require(protocol_descriptor == binding['protocol'], 'Exact reviewed V2 protocol required')
    protocol = read_json(verify(phase, protocol_descriptor))
    correction = read_json(verify(phase, binding['reference_correction']))
    require(correction, 'Bound V2 reference correction required')
    # A v1 configuration is intentionally not admissible after the root review amendments.
    require(protocol['budget']['MLP_fits'] == 90 and
            protocol['budget']['calibration_fits'] == 36 and
            protocol['optimizer']['steps'] == 150,
            'Bind reviewed amended protocol with 90 MLP/36 calibration fits')
    consumer = read_json(phase / EXECUTION / 'V6_CONSUMER_RELEASE_v1.json')
    registry = read_json(phase / EXECUTION / 'registry/REGISTRY.json')
    source = consumer['source']
    require(source == binding['original_source'] and
            consumer['test_labels_authorized'] is False and
            registry['schema'] == 'amazon_polynormer_registry_v2', 'Original source/cohort differs')
    expected = fixed_schedule()
    require([r['row'] for r in registry['physical_fits']] == expected and
            [r['id'] for r in registry['physical_fits']] == [fit_id(r) for r in expected] and
            len(registry['families']) == 9, 'Complete fixed 15/9 registry differs')
    data = read_json(verify(phase, consumer['existing_data_manifest']))
    require(data['schema'] == 'amazon_native_train_validation_projection_v1' and
            data['test_label_artifacts'] == [] and data['class_schema'] == list(range(C)) and
            data['public_graph'] == consumer['preprocessing_identity']['public_graph'],
            'Exact public/VALID projection differs')
    # TRAIN and raw-all-label descriptors are inert metadata: never verified/opened here.
    for key in ('producer_release', 'packet_manifest'):
        verify(phase, data[key])
    _, _, closure = frozen_metadata(phase, closure_descriptor)
    registry_descriptor = record(phase, phase / EXECUTION / 'registry/REGISTRY.json')
    consumer_descriptor = record(phase, phase / EXECUTION / 'V6_CONSUMER_RELEASE_v1.json')
    require(closure['schema'] == 'amazon_polynormer_cohort_closure_v2' and
            closure['status'] == 'complete' and closure['source'] == source and
            closure['registry'] == registry_descriptor and
            closure['families'] == registry['families'] and
            [r['id'] for r in closure['physical_fits']] == [fit_id(r) for r in expected],
            'Bound full-15 complete closure required')
    results = {}
    payloads = [data['public_graph']]
    for row, closed in zip(registry['physical_fits'], closure['physical_fits']):
        _, files, result = frozen_metadata(phase, closed['freeze'])
        require(files['RESULT.json'] == closed['result'] and
                confined(phase, files['RESULT.json']['path']).parent == confined(phase, row['output']) and
                result['schema'] == 'amazon_polynormer_physical_fit_v2' and
                result['status'] == 'complete' and result['report_eligible'] is True and
                result['bindings']['source'] == source and result['bindings']['registry'] == registry_descriptor and
                result['bindings']['consumer_release'] == consumer_descriptor and
                result['bindings']['fit_id'] == row['id'] and result['bindings']['row'] == row['row'] and
                result['bindings']['preprocessing'] == consumer['preprocessing_identity'] and
                result['actual_optimizer_updates'] == 2700 and
                result['final_portable_replay']['bitwise_full_next_step'] is True and
                result['selected_logits'] == files['SELECTED_LOGITS.pt'],
                'Frozen own selected-logit lineage/replay differs: ' + row['id'])
        require(result['selected_checkpoint'] in result['checkpoint_retention']['retained_checkpoints'],
                'Selected checkpoint not retained in metadata')
        results[row['id']] = result
        payloads.append(result['selected_logits'])
    require(len(results) == 15 and sum(r['actual_optimizer_updates'] for r in results.values()) == 40500,
            'Incomplete unique fit work')
    payloads.extend(data['validation_labels'][str(s)] for s, _, _ in SEEDS)
    # Hash original public archive bytes without materializing its feature/TRAIN/TEST members.
    for row in payloads:
        verify(phase, row)
    return {'phase': phase, 'consumer': consumer, 'registry': registry, 'data': data,
            'results': results, 'payloads': payloads, 'closure': closure_descriptor,
            'protocol': protocol_descriptor, 'input_bytes': sum(r['bytes'] for r in payloads)}


def load_visible(state):
    """Lazy NPZ: deserialize only edge_index and val_mask, never features/other masks."""
    import numpy as np
    phase, data = state['phase'], state['data']
    with np.load(confined(phase, data['public_graph']['path']), allow_pickle=False) as archive:
        require(set(archive.files) == {'features', 'edge_index', 'train_mask', 'val_mask', 'test_mask'},
                'Public archive schema differs')
        edge = archive['edge_index'].copy()
        val = archive['val_mask'].copy()
    require(edge.dtype == np.int64 and edge.ndim == 2 and edge.shape[0] == 2 and
            edge.size and edge.min() >= 0 and edge.max() < N,
            'Visible edge schema differs')
    require(val.dtype == np.bool_ and val.shape == (N, 10), 'Official VALID mask differs')
    # Equivalent sorted coalesced to_undirected/remove_self_loops/add_self_loops.
    pair = np.concatenate((edge, edge[::-1]), axis=1)
    keys = np.unique(pair[0] * N + pair[1])
    u, v = keys // N, keys % N
    keep = u != v
    canonical = np.concatenate((np.vstack((u[keep], v[keep])),
                                np.vstack((np.arange(N), np.arange(N)))), axis=1).astype(np.int64)
    identity = state['consumer']['preprocessing_identity']
    require(list(canonical.shape) == identity['edge_shape'] and
            hashlib.sha256(canonical.tobytes()).hexdigest() == identity['edge_logical_sha256'],
            'Canonical PyG-equivalent edge identity differs')
    return canonical, val


def load_validation(state, val_mask, split):
    import numpy as np
    ids = np.flatnonzero(val_mask[:, split]).astype(np.int64)
    require(len(ids) == 6123, 'Expected complete VALID population')
    with np.load(confined(state['phase'], state['data']['validation_labels'][str(split)]['path']),
                 allow_pickle=False) as archive:
        require(set(archive.files) == {'ids', 'labels'}, 'Compact VALID schema differs')
        saved_ids, y = archive['ids'].copy(), archive['labels'].copy()
    require(saved_ids.dtype == y.dtype == np.int64 and saved_ids.ndim == y.ndim == 1 and
            np.array_equal(saved_ids, ids) and y.shape == ids.shape and
            y.min() >= 0 and y.max() < C, 'Compact official VALID order/labels differ')
    expected = state['consumer']['preprocessing_identity']['roles'][split]['VAL']
    require(sha_object(ids.tolist()) == expected['ordered_node_sha256'], 'Ordered VALID identity differs')
    return ids, y


def load_bank(state, family):
    import torch
    pieces = []
    provenance = []
    for ident in family['physical_fit_references']:
        result = state['results'][ident]
        saved = torch.load(confined(state['phase'], result['selected_logits']['path']),
                           map_location='cpu', weights_only=True)
        require(set(saved) == {'raw_logits', 'bindings', 'selected_checkpoint'} and
                saved['bindings'] == result['bindings'] and
                saved['selected_checkpoint'] == result['selected_checkpoint'], 'Saved logit binding differs')
        z = saved['raw_logits']
        expected = 4 if result['bindings']['row']['kind'] == FAMILIES[0] else 1
        require(z.dtype == torch.float32 and tuple(z.shape) == (expected, N, C) and
                z.device.type == 'cpu' and torch.isfinite(z).all().item(), 'Saved FP32 logit schema differs')
        pieces.append(z)
        provenance.append({'id': ident, 'selected_logits': result['selected_logits'],
                           'selected_checkpoint': result['selected_checkpoint'],
                           'selected_stage': result['selection']['stage']})
    z = torch.cat(pieces, dim=0)
    require(z.shape[0] == (1 if family['family'] == 'single_author' else 4), 'Complete member bank differs')
    return z, provenance
