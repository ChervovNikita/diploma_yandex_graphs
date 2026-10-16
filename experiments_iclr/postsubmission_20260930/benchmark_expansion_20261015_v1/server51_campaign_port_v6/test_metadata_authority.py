"""Exercise the actual built source manifests; no numerical imports or jobs."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent.parent
ROOT = PHASE.parent
sys.path.insert(0, str(HERE))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pin(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


class ActualManifestGuards(unittest.TestCase):
    def test_observed_predecessor_failure_and_actual_successor_producer(self):
        predecessor = load(HERE.parent / 'server51_campaign_port_v3/composed/runtime_bridge.py', '_failed_port3')
        with self.assertRaisesRegex(ValueError, 'Wrong producer source schema'):
            predecessor.verify_sources(ROOT)
        bridge = load(HERE / 'composed/runtime_bridge.py', '_actual_port6')
        self.assertEqual(bridge.verify_sources(ROOT), pin(HERE / 'composed/MANIFEST.json'))
        self.assertEqual(bridge.SCHEMA, 'server51-composed-normalization-genuine-runtime-binding-v2')

    def test_actual_reporting_and_scoring_manifests(self):
        reporting = load(HERE / 'reporting/composed_authority.py', '_actual_port6_reporting')
        manifest = pin(HERE / 'composed/MANIFEST.json')
        reporting.verify_sources({'project_root': str(ROOT), 'producer_manifest': manifest,
                                  'source_files': [manifest]})
        from scoring.backend import source_authority
        result = source_authority(ROOT, pin(HERE / 'scoring/MANIFEST.json'))
        self.assertEqual(result['schema'], 'server51-composed-final-scoring-source-v6')

    def test_reporting_launch_hook_uses_actual_package_and_scorer_parent(self):
        from run_gpu import reporting_source_authority
        reporting = {key: pin(HERE / relative) for key, relative in {
            'bridge': 'reporting/backend_bridge.py', 'entrypoint': 'reporting/runtime_entry.py',
            'manifest': 'reporting/MANIFEST.json'}.items()}
        scorer = pin(HERE / 'scoring/MANIFEST.json')
        registry = {'selected_serving_source': scorer}
        root_manifest = json.loads((HERE / 'MANIFEST.json').read_text())
        closure = {str(HERE / row['path']): row['sha256'] for row in root_manifest['files']}
        result = reporting_source_authority(ROOT, str(PHASE.relative_to(ROOT)), reporting, registry, closure)
        self.assertEqual(result, scorer)
        bad = copy.deepcopy(reporting)
        bad['bridge'] = pin(HERE.parent / 'server51_campaign_port_v1/reporting/backend_bridge.py')
        with self.assertRaisesRegex(ValueError, 'Fixed successor reporter'):
            reporting_source_authority(ROOT, str(PHASE.relative_to(ROOT)), bad, registry, closure)
        parent = json.loads((HERE / 'scoring/MANIFEST.json').read_text())['parent_source_manifest']
        self.assertEqual(parent['sha256'], pin(HERE.parent / parent['path_relative_to_expansion'])['sha256'])

    def test_all_own_literal_schema_and_package_checks_are_coherent(self):
        allowed_sources = {'server51-composed-normalization-runtime-source-v6',
                           'server51-composed-final-scoring-source-v6',
                           'allocation-server51-reporting-source-v6'}
        for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
            path = HERE / row['path']
            if path.suffix != '.py' or path.name.startswith('test_'):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value
                    if value.startswith(('server51-composed-', 'allocation-server51-reporting-source-')) and '-source-v' in value:
                        self.assertIn(value, allowed_sources, str(path))
                    if 'benchmark_expansion_20261015_v1/server51_campaign_port_v' in value:
                        self.assertIn('server51_campaign_port_v6', value, str(path))
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('numpy', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
