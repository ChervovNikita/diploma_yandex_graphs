"""Native NumPy split practice; role choices never inspect label values."""
import argparse
import importlib
import json
from pathlib import Path
from role_loader import (HERE, INPUT_NAMES, ROLE_SCHEMA, SPLIT_RULE, bindings,
                         permitted_files, read, require, sha, source_gate)


def development_ids(path):
    ids = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            # label.dat is the authorized development file; the fourth field is opaque.
            fields = line.rstrip('\r\n').split('\t', 3)
            require(len(fields) == 4 and int(fields[2]) == 0, 'Development movie record schema')
            node_id = int(fields[0])
            require(node_id >= 0, 'Nonnegative development ID')
            ids.append(node_id)
    require(ids and len(set(ids)) == len(ids), 'Unique released development movie IDs')
    return sorted(ids)


def partition(np, ids, seed, class_dim, actual):
    """Exact native split operations on a sorted int64 development-ID array."""
    require(type(seed) is int and 0 <= seed < 2**32, 'Native RandomState seed')
    shuffled = np.asarray(ids, dtype=np.int64).copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(shuffled)
    split = len(ids) // 5
    require(0 < split < len(ids), 'Nonempty complete development partitions')
    valid = sorted(int(node) for node in shuffled[:split])
    train = sorted(int(node) for node in shuffled[split:])
    return {'schema': ROLE_SCHEMA, 'dataset': 'HGB IMDB', 'seed': seed,
            'split_rule': SPLIT_RULE, 'class_dim': class_dim,
            'train_ids': train, 'valid_ids': valid, 'input_files': actual}


def build_all(np, root, seeds, class_dim, expected_input_files=None):
    """Hash the large text inputs twice for the entire fixed role cohort."""
    files = permitted_files(root)
    actual = bindings(files)
    if expected_input_files is not None:
        require(actual == expected_input_files, 'Actual root-authorized input member identities')
    ids = development_ids(files['label.dat'])
    result = [partition(np, ids, seed, class_dim, actual) for seed in seeds]
    require(bindings(files) == actual, 'Input identities unchanged through the complete role freeze')
    return result


def build(np, root, seed, class_dim):
    """Single-descriptor library entry; no Torch/model import or TEST input."""
    return build_all(np, root, [seed], class_dim)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.build:
        print(json.dumps({'inactive': True, 'NumPy_or_dataset_access': False}))
        return
    require(args.release and args.input_root and args.output, 'Explicit separate root role-freeze release')
    source_gate()
    cfg = read(args.release)
    require(cfg['enabled'] is True and cfg['root_source_review_approved'] is True
            and cfg['actual_archive_authority_and_development_scope_verified'] is True
            and cfg['action'] == 'build_fixed_IMDB_TRAIN_VALID_roles'
            and cfg['source_seal_sha256'] == sha(HERE / 'SEAL.json') and cfg['seeds'] == list(range(1, 11)), 'Pinned ten-seed role freeze')
    require(str(args.input_root.resolve()) == cfg['input_root'] and str(args.output.resolve()) == cfg['output_directory']
            and not args.output.exists(), 'Fresh bound role output')
    # Numerical import occurs only after explicit separate release. Never executed in preparation.
    np = importlib.import_module('numpy')
    require(np.__version__ == cfg['numpy_version'], 'Recorded existing NumPy implementation')
    spec = read(HERE / 'SOURCE_EXPECTATIONS.json')
    frozen = build_all(np, args.input_root, cfg['seeds'], spec['class_dim'])
    args.output.mkdir(parents=True, exist_ok=False)
    for role in frozen:
        with (args.output / ('seed' + str(role['seed']) + '.json')).open('x') as stream:
            json.dump(role, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
