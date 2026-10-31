"""Prepare node inputs on the allocation; no model fitting or score computation.

Examples and exact provider/role rules are in docs/NODE_PROVIDER_ROLE_CATALOGUE.json.
TRAIN/SELECT targets and sealed CAL/evaluate targets are separate files. Existing
packets are reused only after identity and file-hash verification; no reset.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import random
import socket
import subprocess
import uuid


ALLOCATION_ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
ALLOCATION_HOST = 'anogena-2-0'
ALLOCATION_GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
ORIGINAL_FIVE = {'romanempire', 'amazonratings', 'minesweeper', 'questions', 'tolokers'}


def normalized(value):
    return ''.join(c for c in str(value).lower() if c.isalnum())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_allocation():
    if socket.gethostname().split('.')[0] != ALLOCATION_HOST:
        raise RuntimeError('Data preparation is restricted to the verified anogena-2 allocation')
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True)
    uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if uuids != [ALLOCATION_GPU]:
        raise RuntimeError('Allocation GPU identity differs; no project data was opened')


def confined(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(ALLOCATION_ROOT):
        raise ValueError('Scientific input/output paths must stay in the allocation repository')
    return path


def split_validation(labels, valid_ids, seed):
    """Divide official validation once by class; never remove TRAIN labels."""
    groups = defaultdict(list)
    for i in sorted(valid_ids):
        groups[int(labels[i])].append(i)
    select, cal = [], []
    for label, ids in sorted(groups.items()):
        if len(ids) < 2:
            raise ValueError('Official validation cannot provide disjoint SELECT/CAL for every class')
        random.Random(seed + 1000003 * label).shuffle(ids)
        cut = (len(ids) + 1) // 2
        select.extend(ids[:cut])
        cal.extend(ids[cut:])
    return sorted(select), sorted(cal)


def class_stratified_roles(labels, seed, counts):
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[int(label)].append(i)
    result = {role: [] for role in ('train', 'select', 'cal', 'evaluate')}
    required = counts['train'] + counts['select'] + counts['cal']
    for label, ids in sorted(groups.items()):
        if len(ids) <= required:
            raise ValueError('Insufficient class support for the frozen TRAIN/SELECT/CAL/evaluate rule')
        random.Random(seed + 1000003 * label).shuffle(ids)
        a, b, c = counts['train'], counts['train'] + counts['select'], required
        result['train'].extend(ids[:a])
        result['select'].extend(ids[a:b])
        result['cal'].extend(ids[b:c])
        result['evaluate'].extend(ids[c:])
    return {k: sorted(v) for k, v in result.items()}


def check_roles(roles, labels):
    if set(roles) != {'train', 'select', 'cal', 'evaluate'}:
        raise ValueError('Exactly four role authorities are required')
    seen = set()
    classes = set(labels)
    for role, ids in roles.items():
        if not ids or ids != sorted(set(ids)) or min(ids) < 0 or max(ids) >= len(labels):
            raise ValueError('Empty, duplicate, unsorted or out-of-range role IDs: ' + role)
        if seen.intersection(ids):
            raise ValueError('Role overlap: ' + role)
        if role in {'train', 'select', 'cal'} and {labels[i] for i in ids} != classes:
            raise ValueError('Missing class support: ' + role)
        seen.update(ids)
    return {'role_counts': {k: len(v) for k, v in roles.items()},
            'role_class_counts': {k: dict(sorted(Counter(labels[i] for i in v).items())) for k, v in roles.items()},
            'unassigned_context_nodes': len(labels) - len(seen)}


def mask_ids(torch, mask, nodes):
    mask = torch.as_tensor(mask).cpu()
    if mask.ndim != 1 or mask.numel() != nodes or not torch.all((mask == 0) | (mask == 1)).item():
        raise ValueError('Provider mask must be one binary entry per node')
    return mask.to(torch.bool).nonzero(as_tuple=False).flatten().tolist()


def mask_column(torch, masks, index, nodes):
    masks = torch.as_tensor(masks).cpu()
    if masks.ndim == 1:
        if index != 0:
            raise ValueError('Provider has only one split')
        return mask_ids(torch, masks, nodes)
    if masks.ndim != 2:
        raise ValueError('Provider mask must have one or two dimensions')
    if masks.shape[0] == nodes:
        if index >= masks.shape[1]:
            raise ValueError('Split is outside the provider mask')
        return mask_ids(torch, masks[:, index], nodes)
    if masks.shape[1] == nodes:
        if index >= masks.shape[0]:
            raise ValueError('Split is outside the provider mask')
        return mask_ids(torch, masks[index], nodes)
    raise ValueError('Provider split-mask shape does not align with nodes')


def provider_source_files(provider, dataset):
    rows = []
    module = importlib.import_module(provider['implementation_module'])
    location = Path(module.__file__).resolve()
    rows.append({'kind': 'installed_provider_source', 'path': str(location), 'sha256': digest(location)})
    for section, paths in (('raw', dataset.raw_paths), ('processed', dataset.processed_paths)):
        for value in paths:
            path = Path(value).resolve()
            if not path.is_file():
                raise ValueError('Actual provider file missing from custody: ' + str(path))
            rows.append({'kind': 'provider_' + section + '_file', 'path': str(path),
                         'bytes': path.stat().st_size, 'sha256': digest(path)})
    for name in ('pre_transform.pt', 'pre_filter.pt'):
        path = Path(dataset.processed_dir) / name
        if path.is_file():
            rows.append({'kind': 'provider_processing_metadata', 'path': str(path.resolve()),
                         'bytes': path.stat().st_size, 'sha256': digest(path)})
    return rows


def expected_provider_paths(provider, datasets, cache):
    """Read bound provider path properties without invoking download/process.

    Planetoid keeps its name case; Amazon/Coauthor store lowercase names. WikiCS
    uses a root-level directory and an undirected processed filename. These are
    the same constructor attributes used by the pinned PyG provider classes.
    """
    kind = provider['kind']
    cls = getattr(datasets, kind)
    probe = cls.__new__(cls)
    probe.root = str(cache)
    if kind == 'Planetoid':
        probe.name, probe.split = provider['name'], 'public'
    elif kind in {'Amazon', 'Coauthor'}:
        probe.name = provider['name'].lower()
    elif kind == 'WikiCS':
        probe.is_undirected = True
    else:
        raise ValueError('Unsupported provider layout')
    return [Path(p) for p in probe.raw_paths], [Path(p) for p in probe.processed_paths]


def load_provider(args, row, torch):
    import torch_geometric
    from torch_geometric import datasets
    if torch_geometric.__version__ != args.expected_pyg_version:
        raise ValueError('Installed PyG version differs from the explicitly bound provider version')
    provider = row['provider']
    cache = confined(args.cache_root) / row['dataset_id']
    if provider['kind'] == 'pinned_filtered_npz':
        import numpy as np
        if not args.source_npz or not args.source_sha256 or not args.provider_revision:
            raise ValueError('Filtered provider requires an actual local NPZ, exact SHA256 and pinned revision')
        source = confined(args.source_npz)
        if digest(source) != args.source_sha256:
            raise ValueError('Filtered provider archive SHA256 differs')
        with np.load(source, allow_pickle=False) as data:
            required = {'node_features', 'node_labels', 'edges', 'train_masks', 'val_masks', 'test_masks'}
            if set(data.files) != required:
                raise ValueError('Filtered NPZ must have the pinned six-field author schema')
            values = {k: torch.as_tensor(data[k].copy()) for k in required}
        values['x'], values['y'] = values.pop('node_features'), values.pop('node_labels')
        edge = values.pop('edges')
        if edge.ndim != 2 or edge.shape[1] != 2:
            raise ValueError('Author filtered edges must have shape E by 2')
        values['edge_index'] = edge.t().contiguous()
        return values, [{'kind': 'pinned_filtered_npz', 'path': str(source), 'sha256': args.source_sha256,
                         'provider_revision': args.provider_revision}], torch_geometric.__version__
    raw_paths, processed_paths = expected_provider_paths(provider, datasets, cache)
    if not args.download and not all(path.is_file() for path in raw_paths):
        raise ValueError('Actual provider raw cache incomplete; pass --download for the authorized preparation download')
    existing_processed = bool(processed_paths) and all(path.is_file() for path in processed_paths)
    cache.mkdir(parents=True, exist_ok=True)
    kind = provider['kind']
    if kind == 'Planetoid':
        dataset = datasets.Planetoid(str(cache), name=provider['name'], split='public', transform=None)
    elif kind == 'WikiCS':
        dataset = datasets.WikiCS(str(cache), is_undirected=True, transform=None)
    elif kind == 'Amazon':
        dataset = datasets.Amazon(str(cache), name=provider['name'], transform=None)
    elif kind == 'Coauthor':
        dataset = datasets.Coauthor(str(cache), name=provider['name'], transform=None)
    else:
        raise ValueError('Provider is blocked or unsupported by this exporter')
    data = dataset[0]
    values = {'x': data.x.detach().cpu(), 'y': data.y.detach().cpu(),
              'edge_index': data.edge_index.detach().cpu()}
    for key in ('train_mask', 'val_mask', 'stopping_mask', 'test_mask'):
        value = getattr(data, key, None)
        if value is not None:
            values[key] = value.detach().cpu()
    source_files = provider_source_files(provider, dataset)
    source_files.append({'kind': 'provider_processing_origin',
                         'value': 'existing_cached_artifact_preserved' if existing_processed else 'processed_by_bound_current_provider',
                         'cache_custody_note': 'Existing caches require root provenance audit; no old processing-version claim is inferred from the current import'})
    return values, source_files, torch_geometric.__version__


def reused_packet(output, request_hash):
    custody = json.loads((output / 'CUSTODY.json').read_text())
    if custody['request_hash'] != request_hash or custody.get('status') != 'prepared':
        raise ValueError('Existing packet has different custody or incomplete preparation; no overwrite')
    for row in custody['files']:
        target = (output / row['path']).resolve()
        if not target.is_relative_to(output) or digest(target) != row['sha256']:
            raise ValueError('Existing packet file binding differs; no reset/reuse')
    return {'status': 'reused_exact_custody', 'output': str(output),
            'request_hash': request_hash, 'split_count': len(custody['splits']), 'fits': 0, 'scores': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogue', default=str(Path(__file__).parent / 'docs/NODE_PROVIDER_ROLE_CATALOGUE.json'))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--exposure-audit', required=True, help='Root-bound JSON: datasets keyed by canonical dataset_id')
    parser.add_argument('--evaluate-purpose', choices=['development', 'protected_confirmation'], default='development')
    parser.add_argument('--expected-pyg-version', default='2.7.0')
    parser.add_argument('--splits', nargs='+', type=int, help='Default: full registered provider split roster')
    parser.add_argument('--source-npz')
    parser.add_argument('--source-sha256')
    parser.add_argument('--provider-revision')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--reuse-existing', action='store_true')
    args = parser.parse_args(argv)
    if normalized(args.dataset) in ORIGINAL_FIVE:
        raise ValueError('Original-paper datasets are excluded from this new-dataset preparation catalogue')
    verify_allocation()
    catalogue_path, audit_path = confined(args.catalogue), confined(args.exposure_audit)
    output, cache = confined(args.output), confined(args.cache_root)
    catalogue = json.loads(catalogue_path.read_text())
    matches = [r for r in catalogue['rows'] if normalized(args.dataset) in
               {normalized(r['dataset_id']), *(normalized(x) for x in r.get('aliases', []))}]
    if len(matches) != 1:
        raise ValueError('Dataset must resolve to exactly one canonical provider row')
    row = matches[0]
    if not row['exporter_supported']:
        raise ValueError('Dataset exporter blocked: ' + '; '.join(row['blockers']))
    audit = json.loads(audit_path.read_text())['datasets'][row['dataset_id']]
    if not audit.get('evidence'):
        raise ValueError('Prior evaluation exposure needs saved evidence/custody, not an inferred unused claim')
    if args.evaluate_purpose == 'protected_confirmation' and (
            audit.get('evaluation_access_status') != 'verified_unused' or
            audit.get('previously_evaluated') is not False or
            not audit.get('whole_project_access_audit_complete')):
        raise ValueError('Protected confirmation requires verified whole-project unused-evidence custody')
    splits = args.splits if args.splits is not None else row['split_ids']
    if not splits or len(set(splits)) != len(splits) or any(i not in row['split_ids'] for i in splits):
        raise ValueError('Split roster must be an explicit unique subset of registered provider splits')
    request = {'dataset_id': row['dataset_id'], 'provider': row['provider'], 'role_rule': row['role_rule'],
               'split_ids': splits, 'graph_policy': catalogue['graph_policy'],
               'expected_pyg_version': args.expected_pyg_version, 'catalogue_sha256': digest(catalogue_path),
               'exporter_sha256': digest(__file__), 'exposure_audit_sha256': digest(audit_path),
               'evaluation_exposure': audit, 'evaluate_purpose': args.evaluate_purpose,
               'source_sha256': args.source_sha256, 'provider_revision': args.provider_revision,
               'cache_root': str(cache)}
    request_hash = hashlib.sha256(canonical(request).encode()).hexdigest()
    if output.exists():
        if not args.reuse_existing:
            raise ValueError('Output already exists; explicit --reuse-existing verifies custody without resetting it')
        print(json.dumps(reused_packet(output, request_hash), indent=2))
        return
    import torch
    from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected
    values, provider_files, pyg_version = load_provider(args, row, torch)
    x, raw_y = values['x'], values['y'].flatten()
    if x.ndim != 2 or raw_y.ndim != 1 or x.shape[0] != raw_y.numel() or not torch.isfinite(x).all().item():
        raise ValueError('Finite two-dimensional node features and one target per node required')
    if not torch.isfinite(raw_y).all().item() or not torch.equal(raw_y, raw_y.to(torch.long).to(raw_y.dtype)):
        raise ValueError('Provider targets must be finite integer class IDs')
    raw_classes = sorted(set(raw_y.tolist()))
    if len(raw_classes) < 2 or raw_classes[0] < 0:
        raise ValueError('At least two nonnegative observed class IDs required; unknown targets are not a class')
    label_map = {int(label): i for i, label in enumerate(raw_classes)}
    labels = [label_map[int(v)] for v in raw_y.tolist()]
    y = torch.tensor(labels, dtype=torch.long)
    edges = values['edge_index']
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.numel() == 0:
        raise ValueError('Nonempty two-row provider edge index required')
    if not torch.equal(edges, edges.to(torch.long).to(edges.dtype)):
        raise ValueError('Provider edge IDs must be integers')
    edges = edges.to(torch.long)
    nodes = len(labels)
    if edges.min().item() < 0 or edges.max().item() >= nodes:
        raise ValueError('Provider edge endpoints outside node range')
    native_edges = edges.shape[1]
    edges, _ = remove_self_loops(edges)
    edges = to_undirected(edges, num_nodes=nodes)
    edges, _ = add_self_loops(edges, num_nodes=nodes)
    graph = {'x': x.to(torch.float32).contiguous(), 'edge_index': edges.contiguous()}
    pending = output.with_name(output.name + '.preparing-' + uuid.uuid4().hex[:12])
    pending.mkdir(parents=True, exist_ok=False)
    (pending / 'sealed').mkdir()
    files, split_rows = [], []
    try:
        def save(relative, obj):
            path = pending / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(obj, path)
            files.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': digest(path)})
        save('graph.pt', graph)
        rule = row['role_rule']
        for split in splits:
            seed = rule['role_seed_base'] + split
            if rule['kind'] == 'planetoid_public_validation_division':
                train = mask_ids(torch, values['train_mask'], nodes)
                valid = mask_ids(torch, values['val_mask'], nodes)
                select, cal = split_validation(labels, valid, seed)
                roles = {'train': train, 'select': select, 'cal': cal,
                         'evaluate': mask_ids(torch, values['test_mask'], nodes)}
            elif rule['kind'] == 'wikics_native_four_masks':
                roles = {'train': mask_column(torch, values['train_mask'], split, nodes),
                         'select': mask_column(torch, values['stopping_mask'], split, nodes),
                         'cal': mask_column(torch, values['val_mask'], split, nodes),
                         'evaluate': mask_ids(torch, values['test_mask'], nodes)}
            elif rule['kind'] == 'class_stratified_20_30_30':
                roles = class_stratified_roles(labels, seed, rule['per_class'])
            elif rule['kind'] == 'filtered_official_validation_division':
                train = mask_column(torch, values['train_masks'], split, nodes)
                valid = mask_column(torch, values['val_masks'], split, nodes)
                select, cal = split_validation(labels, valid, seed)
                roles = {'train': train, 'select': select, 'cal': cal,
                         'evaluate': mask_column(torch, values['test_masks'], split, nodes)}
            else:
                raise ValueError('Unknown frozen role rule')
            info = check_roles(roles, labels)
            targets = {}
            endpoint_inputs = {}
            for role, ids in roles.items():
                relative = ('sealed/' if role in {'cal', 'evaluate'} else '') + 'split_' + str(split) + '/' + role + '.pt'
                index = torch.tensor(ids, dtype=torch.long)
                save(relative, {'ids': index, 'y': y[index]})
                targets[role] = relative
                if role in {'cal', 'evaluate'}:
                    ids_relative = 'ids/split_' + str(split) + '/' + role + '_ids.pt'
                    save(ids_relative, index)
                    endpoint_inputs[role + '_ids'] = ids_relative
                    endpoint_inputs[role + '_targets'] = relative
            split_rows.append({'split_id': split, 'role_seed': seed, 'roles': targets, **info,
                               'evaluate_purpose': args.evaluate_purpose,
                               'node_inputs': {'graph': 'graph.pt', 'train': targets['train'], 'select': targets['select']},
                               'endpoint_inputs': endpoint_inputs})
        custody = {'schema': 'node-provider-role-custody-v1', 'status': 'prepared', 'request_hash': request_hash,
                   'request': request, 'dataset_id': row['dataset_id'], 'provider_pyg_version': pyg_version,
                   'provider_files': provider_files, 'provider_graph_edges': native_edges,
                   'model_graph': {'nodes': nodes, 'features': x.shape[1], 'classes': len(raw_classes),
                                   'edges': edges.shape[1], 'self_edges': nodes,
                                   'class_label_map': label_map,
                                   'feature_transform': 'no_additional_transform; existing_cached_preprocessing_needs_root_custody'},
                   'splits': split_rows, 'files': files, 'benchmark_fits': 0, 'predictive_scores': 0,
                   'training_target_access': 'TRAIN_and_SELECT_files_only',
                   'heldout_target_access': 'sealed_files_custodian_only_until_global_search_selection_closes',
                   'new_to_original_paper': True, 'unused_confirmation_claimed': args.evaluate_purpose == 'protected_confirmation'}
        (pending / 'CUSTODY.json').write_text(json.dumps(custody, indent=2, allow_nan=False) + '\n')
        if output.exists():
            raise ValueError('Output appeared during preparation; preserve both packets, do not overwrite')
        pending.rename(output)
        print(json.dumps({'status': 'prepared', 'output': str(output), 'request_hash': request_hash,
                          'split_count': len(split_rows), 'nodes': nodes, 'features': x.shape[1],
                          'classes': len(raw_classes), 'fits': 0, 'scores': 0}, indent=2))
    except Exception as exc:
        (pending / 'PREPARATION_FAILURE.json').write_text(json.dumps({'status': 'failed', 'error': str(exc),
                'request_hash': request_hash, 'benchmark_fits': 0, 'predictive_scores': 0}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
