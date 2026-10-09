"""Synthetic input/role isolation checks only; no real data or NumPy import."""
import builtins
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import build_roles
import role_loader

HERE = Path(__file__).resolve().parent


class FakeArray(list):
    def copy(self):
        return FakeArray(self)


class FakeRandomState:
    def __init__(self, seed, calls):
        calls.append(('RandomState', seed))
        self.calls = calls

    def shuffle(self, values):
        self.calls.append(('shuffle', len(values)))
        values.reverse()


class FakeNP:
    """Checks API sequencing and label independence, not native RNG parity."""
    int64 = 'int64'

    def __init__(self):
        self.calls = []
        self.random = self

    def asarray(self, values, dtype):
        assert dtype == self.int64
        self.calls.append(('asarray', tuple(values)))
        return FakeArray(values)

    def RandomState(self, seed):
        return FakeRandomState(seed, self.calls)


def fixture(folder):
    root = folder / 'input'
    root.mkdir()
    spec = {'class_dim': 2,
            'node_types': [{'count': 7, 'raw_feature_width': 3}, {'count': 2, 'raw_feature_width': 2},
                           {'count': 2, 'raw_feature_width': 2}, {'count': 3, 'raw_feature_width': None}],
            'raw_relation_semantic_order': [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0]]}
    nodes = []
    for node in range(14):
        kind = 0 if node < 7 else 1 if node < 9 else 2 if node < 11 else 3
        line = str(node) + '\tfixture\t' + str(kind)
        if kind < 3:
            line += '\t' + ('1,0,1' if kind == 0 else '1,0')
        nodes.append(line)
    (root / 'node.dat').write_text('\n'.join(nodes) + '\n')
    # Nonconsecutive raw IDs prove that semantic order is verified without invented IDs.
    rid = (8, 6, 9, 2, 15, 10)
    links = []
    for family, offset, size in ((0, 7, 2), (2, 9, 2), (4, 11, 3)):
        for movie in range(7):
            links.append(f'{movie}\t{offset + movie % size}\t{rid[family]}\t1')
        for movie in range(7):
            links.append(f'{offset + movie % size}\t{movie}\t{rid[family + 1]}\t1')
    (root / 'link.dat').write_text('\n'.join(links) + '\n')
    (root / 'label.dat').write_text('\n'.join(f'{node}\tfixture\t0\t{label}'
                                  for node, label in enumerate(('0', '1', '0,1', '0', '1', '0,1'))) + '\n')
    (root / 'label.dat.test').write_text('PROTECTED TEST PAYLOAD MUST NEVER BE OPENED\n')
    return root, spec


def expected_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected a rejected input/role')


def verify():
    checks = []
    with tempfile.TemporaryDirectory(prefix='.fixture-', dir=HERE) as temporary:
        folder = Path(temporary)
        root, spec = fixture(folder)
        unsafe_access = []
        old_builtin, old_io = builtins.open, io.open

        def guarded(original):
            def invoke(file, *args, **kwargs):
                if isinstance(file, (str, Path)) and Path(file).name == 'label.dat.test':
                    unsafe_access.append(str(file))
                    raise AssertionError('TEST access refused by fixture guard')
                return original(file, *args, **kwargs)
            return invoke

        with patch('builtins.open', guarded(old_builtin)), patch('io.open', guarded(old_io)):
            np = FakeNP()
            role = build_roles.build(np, root, 1, 2)
            role_path = folder / 'seed1.json'
            role_path.write_text(json.dumps(role))
            data = role_loader.load(root, role_path, spec)
            assert data.train_ids == (0, 1, 2, 3, 4) and data.valid_ids == (5,)
            assert data.unassigned_movie_ids == (6,) and data.features[3] is None
            assert data.report()['TEST_membership_known'] is False
            assert data.report()['TEST_labels_present'] is False
            checks.append('Explicit complete TRAIN/VALID roles; remainder unassigned; keyword identity implicit')
            assert np.calls == [('asarray', (0, 1, 2, 3, 4, 5)), ('RandomState', 1), ('shuffle', 6)]
            checks.append('One native-shaped int64 array and one RandomState(seed).shuffle call')
            source = data.training_label_source()
            assert list(source[10:14]) == [0, 0, 0, 0]  # VALID and unassigned receive no label input.
            assert data.valid_labels == ((1, 1),)
            checks.append('Only TRAIN labels enter propagation source; VALID labels remain separate')
            original_nodes = (root / 'node.dat').read_text()
            (root / 'node.dat').write_text('\n'.join(reversed(original_nodes.splitlines())) + '\n')
            reordered_role = build_roles.build(FakeNP(), root, 1, 2)
            role_path.write_text(json.dumps(reordered_role))
            reordered = role_loader.load(root, role_path, spec)
            assert reordered.features == data.features
            checks.append('Native ID-based dense feature ordering preserved for reordered text rows')
            (root / 'node.dat').write_text(original_nodes)
            role_path.write_text(json.dumps(role))
            original_labels = (root / 'label.dat').read_text()
            (root / 'label.dat').write_text(original_labels.replace('\t0,1', '\t1'))
            new_role = build_roles.build(FakeNP(), root, 1, 2)
            assert new_role['train_ids'] == role['train_ids'] and new_role['valid_ids'] == role['valid_ids']
            assert new_role['input_files'] != role['input_files']
            checks.append('Changing development label values changes provenance only, never partition choices')
            (root / 'label.dat').write_text(original_labels)
            bad_role = dict(role, test_ids=[6])
            role_path.write_text(json.dumps(bad_role))
            expected_error(lambda: role_loader.load(root, role_path, spec))
            checks.append('TEST-role fields refused')
            role_path.write_text(json.dumps(role))
            original_links = (root / 'link.dat').read_text()
            (root / 'link.dat').write_text(original_links.replace('0\t7\t8\t1', '0\t8\t8\t1'))
            role_path.write_text(json.dumps(dict(role, input_files=role_loader.bindings(role_loader.permitted_files(root)))))
            expected_error(lambda: role_loader.load(root, role_path, spec))
            checks.append('Actual complete reverse pairing required; altered direction rejected')
            (root / 'link.dat').write_text(original_links)
            (root / 'label.dat').unlink()
            (root / 'label.dat').symlink_to(root / 'label.dat.test')
            expected_error(lambda: role_loader.permitted_files(root))
            checks.append('Dataset-member aliases refused before protected TEST access')
            assert not unsafe_access
            checks.append('Poison TEST payload never opened or hashed')
    return {'status': 'passed', 'checks': checks, 'synthetic_fixtures_only': True,
            'native_NumPy_RNG_output_parity_observed': False, 'NumPy_Torch_or_model_import': False,
            'official_IMDB_archive_or_labels_access': False, 'model_fit_or_outcome_access': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2) + '\n')
