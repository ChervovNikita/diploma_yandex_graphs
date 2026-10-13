"""Public raw-to-TRAIN/VALID conversion from explicit local official files.

No download, hostname, GPU inventory, private approval or TEST scoring path.
Published reference hashes identify source bytes; they are not author approvals.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
from data_interface import _data, _sha
from raw_helpers import numeric_rows, pack_molecules

ROOT = Path(__file__).resolve().parent


def fingerprint(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype)).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def wikics(path, pin):
    import torch
    from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
    if _sha(path) != pin['raw_json_sha256'] or path.stat().st_size != pin['raw_bytes']:
        raise ValueError('Official WikiCS raw JSON changed')
    # JSON necessarily decodes the mixed public raw label list; only selected
    # TRAIN/VALID labels become numeric role tensors, never TEST trainer inputs.
    with path.open() as source:
        value = json.load(source)
    x = torch.tensor(value['features'], dtype=torch.float32)
    edges = [(node, neighbour) for node, neighbours in enumerate(value['links']) for neighbour in neighbours]
    edge = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge = to_undirected(edge, num_nodes=11701)  # PyG WikiCS(is_undirected=True).
    edge = to_undirected(edge, num_nodes=11701)  # Exact author post-loader operation.
    edge, _ = remove_self_loops(edge); edge, _ = add_self_loops(edge, num_nodes=11701)
    train_mask = torch.tensor(value['train_masks'], dtype=torch.bool).t().contiguous()[:, 0]
    valid_mask = (torch.tensor(value['val_masks'], dtype=torch.bool).t().contiguous()
                  | torch.tensor(value['stopping_masks'], dtype=torch.bool).t().contiguous())[:, 0]
    test_mask = torch.tensor(value['test_mask'], dtype=torch.bool)
    if (train_mask & valid_mask).any() or (train_mask & test_mask).any() or (valid_mask & test_mask).any():
        raise ValueError('Official split0 masks overlap')
    train_ids = train_mask.nonzero().flatten(); valid_ids = valid_mask.nonzero().flatten()
    tensors = {'x': x, 'edge_index': edge, 'train_ids': train_ids, 'valid_ids': valid_ids,
               'train_y': torch.tensor([value['labels'][int(i)] for i in train_ids], dtype=torch.long),
               'valid_y': torch.tensor([value['labels'][int(i)] for i in valid_ids], dtype=torch.long)}
    for key, tensor in tensors.items():
        expected = pin['projected_raw_tensor_fingerprints'][key]
        if list(tensor.shape) != expected['shape'] or str(tensor.dtype) != expected['dtype']:
            raise ValueError('Original WikiCS role/graph shape changed: ' + key)
        if hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() != expected['contiguous_raw_bytes_sha256']:
            raise ValueError('Complete original WikiCS tensor differs: ' + key)
    arrays = {'train': {key: tensors[key].numpy() for key in ('x', 'edge_index')}, 'valid': {}}
    arrays['train'].update(ids=train_ids.numpy(), y=tensors['train_y'].numpy())
    arrays['valid'].update(ids=valid_ids.numpy(), y=tensors['valid_y'].numpy())
    return arrays, {'raw_json_sha256': pin['raw_json_sha256'], 'complete_original_tensor_fingerprints_matched': True,
                    'mixed_public_raw_JSON_labels_decoded': True, 'TEST_labels_in_role_files': False}, {'split_index': 0}


def collab(path, pin):
    import numpy as np
    import pandas as pd
    import torch
    if _sha(path) != pin['archive_sha256']:
        raise ValueError('Official Collab archive changed; refuse pickle deserialization')
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Repeated archive member')
        def read(name):
            if name not in pin['members']:
                raise ValueError('Nonallowlisted or TEST member')
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != pin['members'][name]:
                raise ValueError('Official source member changed: ' + name)
            return payload
        # Trusted exact upstream TRAIN/VALID NumPy dictionaries; no TEST pickle.
        train = torch.load(io.BytesIO(read('collab/split/time/train.pt')), map_location='cpu', weights_only=False)
        valid = torch.load(io.BytesIO(read('collab/split/time/valid.pt')), map_location='cpu', weights_only=False)
        x = pd.read_csv(io.BytesIO(gzip.decompress(read('collab/raw/node-feat.csv.gz'))), header=None).values.astype(np.float32)
        raw = np.loadtxt(io.BytesIO(gzip.decompress(read('collab/raw/edge.csv.gz'))), delimiter=',', dtype=np.int64)
    if set(train) != {'edge', 'weight', 'year'} or set(valid) != {'edge', 'weight', 'year', 'edge_neg'}:
        raise ValueError('Exact official temporal split dictionary required')
    def array(value):
        return value.numpy() if isinstance(value, torch.Tensor) else value
    train = {key: array(value) for key, value in train.items()}
    valid = {key: array(value) for key, value in valid.items()}
    actual = {'TRAIN_edge': fingerprint(train['edge']), 'VALID_edge': fingerprint(valid['edge']),
              'VALID_negative': fingerprint(valid['edge_neg']), 'raw_features': fingerprint(x),
              'TRAIN_year': fingerprint(train['year']), 'VALID_year': fingerprint(valid['year'])}
    if actual != pin['expected_public_array_digests']:
        raise ValueError('Complete ordered original Collab arrays/years differ')
    train_pairs = train['edge'].min(1) * 235868 + train['edge'].max(1)
    raw_pairs = raw.min(1) * 235868 + raw.max(1)
    valid_pairs = valid['edge'].min(1) * 235868 + valid['edge'].max(1)
    if raw.shape != train['edge'].shape or not np.array_equal(np.sort(raw_pairs), np.sort(train_pairs)):
        raise ValueError('Complete duplicate-preserving raw/TRAIN correspondence')
    arrays = {'train': {'x': x, 'positive': train['edge'], 'positive_year': train['year']},
              'valid': {'positive': valid['edge'], 'positive_year': valid['year'], 'negative': valid['edge_neg']}}
    metadata = {'valid_negative_count': 100000, 'valid_negative_self_pair_records': 1,
                'temporal_roles': {'train_max': 2017, 'valid_only': 2018}}
    origin = {'archive_sha256': pin['archive_sha256'], 'complete_original_array_digests_matched': actual,
              'historical_VALID_pair_overlap_records': int(np.isin(valid_pairs, train_pairs).sum()),
              'VALID_pairs_used_to_filter_TRAIN_support': False, 'TEST_members_deserialized': False}
    return arrays, origin, metadata


def molhiv(path, pin):
    import numpy as np
    for name, expected in pin['files'].items():
        if _sha(path / name) != expected:
            raise ValueError('Official Molhiv raw/split file changed: ' + name)
    train_ids = np.array([int(row[0]) for row in numeric_rows(path / 'split/scaffold/train.csv.gz').values()], dtype=np.int64)
    valid_ids = np.array([int(row[0]) for row in numeric_rows(path / 'split/scaffold/valid.csv.gz').values()], dtype=np.int64)
    if len(train_ids) != 32901 or len(valid_ids) != 4113 or len(np.unique(np.r_[train_ids, valid_ids])) != 37014:
        raise ValueError('Complete disjoint official scaffold IDs required')
    arrays = pack_molecules(path / 'raw', train_ids, valid_ids)
    return arrays, {'raw_split_files_sha256': pin['files'], 'all_public_graph_count_metadata_parsed': True,
                    'unselected_raw_bytes_traversed': True, 'TEST_target_values_parsed': False,
                    'TEST_features_or_labels_in_roles': False}, {'split_kind': 'official_scaffold'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='task', required=True)
    wiki = sub.add_parser('wikics'); wiki.add_argument('--raw-json', type=Path, required=True)
    link = sub.add_parser('collab'); link.add_argument('--archive', type=Path, required=True)
    molecule = sub.add_parser('molhiv'); molecule.add_argument('--raw-root', type=Path, required=True)
    for task_parser in (wiki, link, molecule):
        task_parser.add_argument('--output', type=Path, required=True, help='Fresh numeric role directory')
    args = parser.parse_args(); started = time.monotonic(); args.output.mkdir(parents=True, exist_ok=False)
    try:
        import numpy as np
        import torch
        pin = json.loads((ROOT / 'PUBLIC_DATA_PINS.json').read_text())[args.task]
        source = args.raw_json if args.task == 'wikics' else args.archive if args.task == 'collab' else args.raw_root
        arrays, origin, metadata = {'wikics': wikics, 'collab': collab, 'molhiv': molhiv}[args.task](source, pin)
        _data().check_projection(args.task, {key: torch.from_numpy(value) for key, value in arrays['train'].items()},
                                 {key: torch.from_numpy(value) for key, value in arrays['valid'].items()}, metadata)
        files = {}
        for role, value in arrays.items():
            path = args.output / (role + '.npz')
            np.savez(path, **{key: np.ascontiguousarray(array) for key, array in value.items()})
            # Read original role arrays back for full ordered field equality.
            loaded = _data().load_npz(path, _data().ROLE_KEYS[args.task][role])
            if any(not np.array_equal(loaded[key].numpy(), array) for key, array in value.items()):
                raise ValueError('Serialized role array differs: ' + role)
            files[role] = {'path': path.name, 'sha256': _sha(path),
                           'array_fingerprints': {key: fingerprint(array) for key, array in value.items()}}
        if torch.cuda.is_initialized():
            raise ValueError('Data-only conversion unexpectedly initialized CUDA')
        record = {'schema': 'portable-public-raw-to-roles-v1', 'complete': True, 'task': args.task,
                  'source_bytes_matched_public_pins': True, 'origin': origin, 'roles': files,
                  'complete_original_role_array_serialization_equal': True, 'author_approval_claimed': False,
                  'TEST_scoring': False, 'seconds': time.monotonic() - started,
                  'converter_sha256': _sha(__file__), 'source_numeric_helpers_sha256': _sha(ROOT / 'raw_helpers.py')}
        (args.output / 'DATA.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    except BaseException as error:
        (args.output / 'FAILURE.json').write_text(json.dumps({'complete': False, 'error_type': type(error).__name__,
            'error': str(error), 'seconds': time.monotonic() - started, 'automatic_retry': False}, indent=2) + '\n')
        raise
    print(json.dumps({'complete': True, 'task': args.task, 'role_directory': str(args.output), 'TEST_scoring': False}))


if __name__ == '__main__':
    main()
