"""Read only explicit HGB IMDB inputs and development roles; no TEST file API."""
import argparse
from array import array
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT_NAMES = ('node.dat', 'link.dat', 'label.dat')
ROLE_KEYS = {'schema', 'dataset', 'seed', 'split_rule', 'class_dim',
             'train_ids', 'valid_ids', 'input_files'}
ROLE_SCHEMA = 'explicit-HGB-IMDB-TRAIN-VALID-roles-v1'
SPLIT_RULE = 'NumPy RandomState(seed).shuffle(sorted development IDs); first floor(0.2*N) VALID'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def permitted_files(root):
    """Explicit filenames only: never list, stat, hash or open another member."""
    root = Path(root).resolve(strict=True)
    require(root.is_dir(), 'Explicit unpacked input directory required')
    files = {}
    for name in INPUT_NAMES:
        path = root / name
        require(not path.is_symlink(), 'Input aliases are refused')
        path = path.resolve(strict=True)
        require(path.parent == root and path.name == name and path.is_file(), 'Exact allowed input member')
        files[name] = path
    return files


def bindings(files):
    return {name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
            for name, path in files.items()}


def nodes(path, spec):
    types = spec['node_types']
    counts = [int(row['count']) for row in types]
    offsets, total = [], 0
    for count in counts:
        offsets.append(total)
        total += count
    features = [None if row['raw_feature_width'] is None else array('f', [0.0]) * (count * row['raw_feature_width'])
                for row, count in zip(types, counts)]
    seen = bytearray(total)
    for line_number, line in enumerate(path.open(encoding='utf-8'), 1):
        fields = line.rstrip('\r\n').split('\t')
        require(len(fields) in (3, 4), 'node.dat schema at row ' + str(line_number))
        node_id, type_id = int(fields[0]), int(fields[2])
        require(0 <= type_id < len(types) and 0 <= node_id < total and not seen[node_id], 'Unique typed node ID')
        require(offsets[type_id] <= node_id < offsets[type_id] + counts[type_id], 'Native contiguous type blocks')
        seen[node_id] = 1
        width = types[type_id]['raw_feature_width']
        if width is None:
            require(len(fields) == 3, 'Identity feature type has no released dense attributes')
        else:
            require(len(fields) == 4, 'All dense feature rows must be present')
            values = [float(value) for value in fields[3].split(',')]
            require(len(values) == width and all(math.isfinite(value) for value in values), 'Finite full feature width')
            converted = array('f', values)
            require(all(math.isfinite(value) for value in converted), 'Finite native float32 conversion')
            start = (node_id - offsets[type_id]) * width
            features[type_id][start:start + width] = converted
    require(all(seen), 'Complete expected source node population')
    return counts, offsets, features


def node_type(node_id, counts, offsets):
    require(type(node_id) is int and 0 <= node_id < sum(counts), 'Graph endpoint within complete node population')
    for type_id, (offset, count) in enumerate(zip(offsets, counts)):
        if offset <= node_id < offset + count:
            return type_id
    raise ValueError('Unreachable node type')


def links(path, counts, offsets, spec):
    expected = [tuple(value) for value in spec['raw_relation_semantic_order']]
    order, relation_meta, raw_counts, pairs, duplicates = [], {}, Counter(), {}, Counter()
    for line_number, line in enumerate(path.open(encoding='utf-8'), 1):
        fields = line.rstrip('\r\n').split('\t')
        require(len(fields) == 4, 'link.dat schema at row ' + str(line_number))
        head, tail, relation_id, weight = int(fields[0]), int(fields[1]), int(fields[2]), float(fields[3])
        require(relation_id >= 0 and math.isfinite(weight) and weight == 1.0, 'Native binary unit-weight input scope')
        semantic = (node_type(head, counts, offsets), node_type(tail, counts, offsets))
        require(semantic in expected and head != tail, 'Only the six released typed cross relations')
        if relation_id not in relation_meta:
            relation_meta[relation_id] = semantic
            order.append(relation_id)
            pairs[relation_id] = set()
        require(relation_meta[relation_id] == semantic, 'One endpoint-type pair per raw relation ID')
        raw_counts[relation_id] += 1
        pair = (head, tail)
        duplicates[relation_id] += int(pair in pairs[relation_id])
        pairs[relation_id].add(pair)
    require([relation_meta[key] for key in order] == expected, 'Native first-appearance relation order MD DM MA AM MK KM')
    semantic_pairs = {relation_meta[key]: pairs[key] for key in order}
    for left, right in ((0, 1), (0, 2), (0, 3)):
        forward = semantic_pairs[left, right]
        reverse = semantic_pairs[right, left]
        require(forward and reverse and {(tail, head) for head, tail in forward} == reverse, 'Complete exact reverse support pairing')
    movie_directors = Counter(head for head, _ in semantic_pairs[0, 1])
    require(all(movie_directors[node_id] == 1 for node_id in range(counts[0])), 'Native one unique director per movie')
    rows = []
    for relation_id in order:
        head_type, tail_type = relation_meta[relation_id]
        rows.append({'raw_relation_id': relation_id, 'raw_head_type': head_type, 'raw_tail_type': tail_type,
                     'raw_records': raw_counts[relation_id], 'unique_pairs': len(pairs[relation_id]),
                     'duplicates_coalesced': duplicates[relation_id],
                     'CSR_row_type': head_type, 'CSR_column_type': tail_type,
                     'message_source_type': tail_type, 'message_destination_type': head_type})
    canonical = {key: tuple(sorted(value)) for key, value in pairs.items()}
    return rows, canonical


def development_labels(path, movie_count, class_dim):
    labels = {}
    for line_number, line in enumerate(path.open(encoding='utf-8'), 1):
        fields = line.rstrip('\r\n').split('\t')
        require(len(fields) == 4, 'label.dat development schema at row ' + str(line_number))
        node_id, type_id = int(fields[0]), int(fields[2])
        require(type_id == 0 and 0 <= node_id < movie_count and node_id not in labels, 'Unique development movie label ID')
        classes = [int(value) for value in fields[3].split(',')]
        require(classes and len(classes) == len(set(classes)) and all(0 <= value < class_dim for value in classes), 'Fixed public multilabel schema')
        labels[node_id] = tuple(int(class_id in classes) for class_id in range(class_dim))
    require(labels, 'Nonempty released development pool')
    return labels


def check_role(role, labels, class_dim, actual):
    require(set(role) == ROLE_KEYS and role['schema'] == ROLE_SCHEMA and role['dataset'] == 'HGB IMDB', 'Explicit TRAIN/VALID role schema only')
    require(type(role['seed']) is int and 0 <= role['seed'] < 2**32 and role['split_rule'] == SPLIT_RULE, 'Pinned native split practice')
    require(role['class_dim'] == class_dim and role['input_files'] == actual, 'Exact public classes and frozen allowed input identities')
    train, valid = role['train_ids'], role['valid_ids']
    for ids in (train, valid):
        require(isinstance(ids, list) and ids and all(type(value) is int for value in ids)
                and ids == sorted(set(ids)), 'Explicit nonempty sorted unique role IDs')
    require(not set(train) & set(valid) and set(train) | set(valid) == set(labels), 'Disjoint roles exactly cover released development pool')
    require(len(valid) == len(labels) // 5, 'Native floor20percent VALID count')
    require(all(any(labels[node][j] for node in train) and any(not labels[node][j] for node in train)
                for j in range(class_dim)), 'Every TRAIN binary task has both classes')
    return train, valid


@dataclass
class RoleData:
    """No global truth matrix or TEST values exist in this object."""
    counts: tuple
    offsets: tuple
    features: tuple
    relation_rows: tuple
    canonical_relation_pairs: dict
    train_ids: tuple
    valid_ids: tuple
    train_labels: tuple
    valid_labels: tuple
    unassigned_movie_ids: tuple
    class_dim: int
    input_bindings: dict

    def with_roles(self, role):
        """Validate another frozen split on the same once-loaded development pool."""
        labels = dict(zip(self.train_ids, self.train_labels))
        labels.update(zip(self.valid_ids, self.valid_labels))
        train, valid = check_role(role, labels, self.class_dim, self.input_bindings)
        return RoleData(self.counts, self.offsets, self.features, self.relation_rows, self.canonical_relation_pairs,
                        tuple(train), tuple(valid), tuple(labels[node] for node in train),
                        tuple(labels[node] for node in valid), self.unassigned_movie_ids,
                        self.class_dim, self.input_bindings)

    def training_label_source(self):
        """Propagation input only. Non-TRAIN zeros are unavailable-label placeholders."""
        source = array('f', [0.0]) * (self.counts[0] * self.class_dim)
        for node_id, row in zip(self.train_ids, self.train_labels):
            source[node_id * self.class_dim:(node_id + 1) * self.class_dim] = array('f', row)
        return source

    def report(self):
        return {'dataset': 'HGB IMDB', 'node_counts': self.counts, 'offsets': self.offsets,
                'feature_widths': [None if value is None else len(value) // count for value, count in zip(self.features, self.counts)],
                'keyword_features': 'implicit identity; no dense eye allocated here',
                'relations': self.relation_rows, 'TRAIN_count': len(self.train_ids), 'VALID_count': len(self.valid_ids),
                'unassigned_movie_count': len(self.unassigned_movie_ids), 'class_dim': self.class_dim,
                'TRAIN_positive_counts': [sum(row[j] for row in self.train_labels) for j in range(self.class_dim)],
                'VALID_positive_counts': [sum(row[j] for row in self.valid_labels) for j in range(self.class_dim)],
                'input_files': self.input_bindings, 'TEST_membership_known': False, 'TEST_file_access': False,
                'TEST_labels_present': False, 'full_y_present': False, 'training_label_source_uses_TRAIN_only': True,
                'official_archive_authority_verified': False, 'backbone_numerically_qualified': False}


def load(root, role_path, spec):
    files = permitted_files(root)
    role_path = Path(role_path)
    require(not role_path.is_symlink() and role_path.name.endswith('.json')
            and role_path.resolve(strict=True) not in files.values(), 'Explicit JSON role descriptor, never a dataset member')
    role = read(role_path)
    require(set(role) == ROLE_KEYS and role['schema'] == ROLE_SCHEMA and role['dataset'] == 'HGB IMDB', 'Explicit TRAIN/VALID role schema only')
    require(type(role['seed']) is int and 0 <= role['seed'] < 2**32 and role['split_rule'] == SPLIT_RULE, 'Pinned native split practice')
    class_dim = spec['class_dim']
    require(role['class_dim'] == class_dim and set(role['input_files']) == set(INPUT_NAMES), 'Only allowed inputs and public class dimension')
    actual = bindings(files)
    require(actual == role['input_files'], 'Exact frozen node/link/development-label input identities')
    counts, offsets, features = nodes(files['node.dat'], spec)
    relation_rows, canonical = links(files['link.dat'], counts, offsets, spec)
    labels = development_labels(files['label.dat'], counts[0], class_dim)
    train, valid = check_role(role, labels, class_dim, actual)
    require(bindings(files) == actual, 'Inputs unchanged through complete validation')
    unassigned = tuple(node_id for node_id in range(counts[0]) if node_id not in labels)
    return RoleData(tuple(counts), tuple(offsets), tuple(features), tuple(relation_rows), canonical,
                    tuple(train), tuple(valid), tuple(labels[node] for node in train),
                    tuple(labels[node] for node in valid), unassigned, class_dim, actual)


def source_gate():
    seal = read(HERE / 'SEAL.json')
    require(seal['source_only'] is True and seal['execution_enabled'] is False
            and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact inactive source seal')
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed sealed source')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--roles', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.validate:
        print(json.dumps({'inactive': True, 'dataset_or_model_access': False}))
        return
    require(args.release and args.input_root and args.roles and args.output, 'Explicit external release and fresh schema output required')
    source_gate()
    release = read(args.release)
    require(release['enabled'] is True and release['root_source_review_approved'] is True
            and release['actual_archive_authority_and_development_scope_verified'] is True
            and release['action'] == 'validate_IMDB_TRAIN_VALID_schema'
            and release['source_seal_sha256'] == sha(HERE / 'SEAL.json'), 'Explicit source-reviewed schema release')
    require(str(args.input_root.resolve()) == release['input_root'] and str(args.roles.resolve()) == release['roles_path']
            and str(args.output.resolve()) == release['output_path'] and not args.output.exists(), 'Exact bound inputs and fresh output')
    data = load(args.input_root, args.roles, read(HERE / 'SOURCE_EXPECTATIONS.json'))
    with args.output.open('x') as stream:
        json.dump(data.report(), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
