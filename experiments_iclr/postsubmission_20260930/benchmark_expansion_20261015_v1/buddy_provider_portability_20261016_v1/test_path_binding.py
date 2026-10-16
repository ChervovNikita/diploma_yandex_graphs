"""Metadata-only regression tests; no torch/model/data execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

E = Path(__file__).resolve().parent.parent
P = E.parent
PROJECT = P.parent
PROVIDER = E / 'buddy_native_provider_v3'
PAIRS = {'link_expansion_ports_BUDDY_production_v2': 'link_expansion_ports_BUDDY_production_v3',
         'interaction_BUDDY_production_v3': 'interaction_BUDDY_production_v4',
         'buddy_private_bn_production_v2': 'buddy_private_bn_production_v4'}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value


paths = module(PROVIDER / 'path_binding.py', '_test_buddy_typed_paths')


class PathBindingTests(unittest.TestCase):
    def setUp(self):
        self.static = json.loads((PROVIDER / 'STATIC_PROVIDER.json').read_text())
        self.canonical = self.static['provider_data_files']
        self.actual = copy.deepcopy(self.canonical)
        for receipt in self.actual.values():
            receipt['path'] = receipt['path'].replace(paths.CANONICAL_ROOT, paths.SERVER51_ROOT, 1)

    def test_actual_server51_failure_fixture_passes_without_mutation(self):
        fixture = json.loads((E / 'planning/all_fit_launch_20261016_v1/SERVER51_BUDDY_PATH_FAILURE_DIAGNOSIS.json').read_text())
        actual = {row['role']: row['actual'] for row in fixture['files']}
        canonical = {row['role']: row['expected'] for row in fixture['files']}
        before = copy.deepcopy((actual, canonical))
        result = paths.validate_provider_receipts(actual, canonical, paths.SERVER51_ROOT)
        self.assertTrue(fixture['config_matches'])
        self.assertFalse(result['file_bytes_changed'])
        self.assertFalse(result['native_receipts_relabelled'])
        self.assertEqual((actual, canonical), before)

    def test_original_allocation_identity_is_retained(self):
        result = paths.validate_provider_receipts(self.canonical, self.canonical, paths.CANONICAL_ROOT)
        self.assertTrue(all(row['canonical_path'] == row['actual_path'] for row in result['files']))

    def test_altered_bytes_hash_and_role_paths_fail(self):
        for field, value in (('bytes', 1), ('sha256', '0' * 64), ('path', self.actual['train']['path'])):
            altered = copy.deepcopy(self.actual); altered['graph'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                paths.validate_provider_receipts(altered, self.canonical, paths.SERVER51_ROOT)
        altered = copy.deepcopy(self.actual); altered['graph']['bytes'] = True
        with self.assertRaises(ValueError):
            paths.validate_provider_receipts(altered, self.canonical, paths.SERVER51_ROOT)

    def test_role_root_escape_and_undisclosed_fields_fail(self):
        invalid = []
        value = copy.deepcopy(self.actual); value['test'] = value['valid']; invalid.append(value)
        value = copy.deepcopy(self.actual); del value['train']; invalid.append(value)
        value = copy.deepcopy(self.actual); value['graph']['extra'] = 'ignored'; invalid.append(value)
        value = copy.deepcopy(self.actual); value['graph']['path'] = paths.SERVER51_ROOT + '/../escape'; invalid.append(value)
        for value in invalid:
            with self.assertRaises(ValueError):
                paths.validate_provider_receipts(value, self.canonical, paths.SERVER51_ROOT)
        with self.assertRaises(ValueError):
            paths.validate_provider_receipts(self.actual, self.canonical, '/unrecognized/repository')

    def test_all_native_session_code_and_static_metadata_unchanged(self):
        self.assertEqual((PROVIDER / 'STATIC_PROVIDER.json').read_bytes(), (E / 'buddy_native_provider_v2/STATIC_PROVIDER.json').read_bytes())
        for before, after in PAIRS.items():
            self.assertEqual((E / before / 'session.py').read_bytes(), (E / after / 'session.py').read_bytes())
            expected = (E / before / 'callback.py').read_text().replace(before, after).replace('buddy_native_provider_v2', 'buddy_native_provider_v3')
            self.assertEqual(expected, (E / after / 'callback.py').read_text())
        before = (E / 'buddy_native_provider_v2/provider.py').read_text()
        expected = before.replace(
            "        if base.native_binding['data_files']!=static['provider_data_files'] or n.recipe['native_fixed_config']!=static['native_fixed_config']:\n            raise ValueError('Native GRAPH/TRAIN/VALID provider or fixed settings changed')",
            "        provider_path_binding=load(HERE/'path_binding.py').validate_provider_receipts(\n            base.native_binding['data_files'],static['provider_data_files'],job['recipe']['_repository_root'])\n        if n.recipe['native_fixed_config']!=static['native_fixed_config']:\n            raise ValueError('Native fixed settings changed')")
        expected = expected.replace("'source_receipts':pins,'static_provider':static_pin,'native_binding':copy.deepcopy(base.native_binding),",
            "'source_receipts':pins,'static_provider':static_pin,'native_binding':copy.deepcopy(base.native_binding),\n            'provider_path_binding':provider_path_binding,")
        self.assertEqual(expected, (PROVIDER / 'provider.py').read_text())

    def test_successor_source_bindings_validate_without_numerical_imports(self):
        provider = module(PROVIDER / 'provider.py', '_test_successor_provider')
        sha = hashlib.sha256((PROVIDER / 'MANIFEST.json').read_bytes()).hexdigest()
        receipts = provider.bindings(PROJECT, sha)
        self.assertIn(str(PROVIDER / 'path_binding.py'), {row['path'] for row in receipts})
        for after in PAIRS.values():
            package = E / after
            session = module(package / 'session.py', '_test_' + after)
            sha = hashlib.sha256((package / 'MANIFEST.json').read_bytes()).hexdigest()
            value = session.bindings(PROJECT, sha)
            receipts = value[0] if isinstance(value, tuple) else value
            self.assertIn(str(PROVIDER / 'MANIFEST.json'), {row['path'] for row in receipts})
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
