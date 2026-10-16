"""Actual audit and projector regressions; no numerical imports or runtime jobs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_rng_binding as rng
from descriptor_binding import derive, digest


class SourceRNGBindingChecks(unittest.TestCase):
    def job(self):
        prefix = rng.ALLOCATION_ROOT + '/' + rng.PHASE_RELATIVE + '/'
        job = {'schema': 'native-full-epoch-fit-v1', 'id': 'source-rng-metadata-only',
               'family': 'ogb', 'procedure': 'native_single', 'TEST_access': False,
               'policy': {'track': 'source_reference', 'evaluate_TRAIN_each_epoch': True},
               'source_rng_emulation': {'mode': rng.MODE, 'audit': prefix + rng.ORIGINAL_RELATIVE,
                                        'audit_sha256': rng.ORIGINAL_SHA},
               'recipe': {'_repository_root': rng.ALLOCATION_ROOT,
                          '_redesign_root': prefix + 'reference_wrapper_redesign_20261013_v1'},
               'expanded_contract': {'fixed_science': True}}
        job['expanded_contract_hash'] = digest(job['expanded_contract'])
        return job

    def test_actual_audit_semantics_versions_and_all_24_content_pins_equal(self):
        current = rng.verify_audits()
        original = json.loads((HERE.parent / 'qualification/runtime_v2_source_rng_v1.json').read_text())
        comparison = json.loads((HERE / 'source_rng_audits/SOURCE_COMPARISON.json').read_text())
        self.assertEqual(current['semantic_check'], original['semantic_check'])
        self.assertEqual(current['assumptions'], original['assumptions'])
        self.assertEqual(current['dependency_profile'], rng.expected_profile(original['dependency_profile']))
        self.assertNotEqual(original['dependency_profile'], current['dependency_profile'])
        self.assertEqual(len(comparison['rows']), 24)
        for row in comparison['rows']:
            self.assertTrue(row['same_bytes'])
            profile = current['dependency_profile']
            actual = profile['native_manifest'] if row['group'] == 'native_manifest' else profile[row['group']][row['name']]
            self.assertEqual(actual, row['actual'])
            self.assertEqual({k: actual[k] for k in ('bytes', 'sha256')},
                             {k: row['expected'][k] for k in ('bytes', 'sha256')})
        self.assertEqual(comparison['versions'], {'torch': '2.1.2+cu118', 'torch_geometric': '2.7.0', 'ogb': '1.3.6'})
        self.assertFalse(comparison['numeric_modules_imported'])

    def test_actual_plain_projector_changes_only_execution_audit_path_and_hash(self):
        original = self.job(); before = copy.deepcopy(original)
        spec = importlib.util.spec_from_file_location('_source_rng_frozen_port5', HERE.parent / 'server51_campaign_port_v5/descriptor_binding.py')
        predecessor = importlib.util.module_from_spec(spec); spec.loader.exec_module(predecessor)
        old, _, _ = predecessor.derive(original, rng.ALLOCATION_ROOT, rng.PHASE_RELATIVE,
                                       rng.SERVER51_ROOT, rng.PHASE_RELATIVE)
        bound, raw, receipt = derive(original, rng.ALLOCATION_ROOT, rng.PHASE_RELATIVE,
                                     rng.SERVER51_ROOT, rng.PHASE_RELATIVE)
        self.assertEqual(original, before); self.assertIsNone(raw)
        self.assertEqual({k: v for k, v in bound.items() if k != 'source_rng_emulation'},
                         {k: v for k, v in old.items() if k != 'source_rng_emulation'})
        self.assertEqual(bound['source_rng_emulation'], {'mode': rng.MODE, 'audit_sha256': rng.AUDIT_SHA,
            'audit': rng.SERVER51_ROOT + '/' + rng.PHASE_RELATIVE + '/' + rng.AUDIT_RELATIVE})
        self.assertEqual(receipt['field_changes'][-2:], [
            {'pointer': '/source_rng_emulation/audit', 'from': old['source_rng_emulation']['audit'],
             'to': bound['source_rng_emulation']['audit']},
            {'pointer': '/source_rng_emulation/audit_sha256', 'from': rng.ORIGINAL_SHA, 'to': rng.AUDIT_SHA}])
        self.assertEqual(receipt['canonical_runtime_job_hash'], digest(original))
        self.assertFalse(receipt['scientific_settings_changed'])

    def test_allocation_and_undeclared_jobs_unchanged(self):
        original = self.job()
        bound, _, _ = derive(original, rng.ALLOCATION_ROOT, rng.PHASE_RELATIVE,
                              rng.ALLOCATION_ROOT, rng.PHASE_RELATIVE)
        self.assertEqual(bound, original)
        original.pop('source_rng_emulation')
        self.assertEqual(rng.rebind_source_rng(original, rng.SERVER51_ROOT, rng.PHASE_RELATIVE), (original, []))

    def test_rejects_changed_missing_unknown_or_wrong_scope_declarations(self):
        for field, value in [('audit_sha256', '0' * 64), ('mode', 'other'), ('audit', '/outside/audit.json'), ('extra', True)]:
            original = self.job(); original['source_rng_emulation'][field] = value
            with self.assertRaises(ValueError):
                derive(original, rng.ALLOCATION_ROOT, rng.PHASE_RELATIVE, rng.SERVER51_ROOT, rng.PHASE_RELATIVE)
        for field in ('audit', 'audit_sha256', 'mode'):
            original = self.job(); del original['source_rng_emulation'][field]
            with self.assertRaises(ValueError):
                derive(original, rng.ALLOCATION_ROOT, rng.PHASE_RELATIVE, rng.SERVER51_ROOT, rng.PHASE_RELATIVE)
        for field, value in [('family', 'link'), ('procedure', 'wrapper'), ('TEST_access', True), ('policy', {'track': 'matched_extended'})]:
            original = self.job(); original[field] = value
            with self.assertRaises(ValueError):
                derive(original, rng.ALLOCATION_ROOT, rng.PHASE_RELATIVE, rng.SERVER51_ROOT, rng.PHASE_RELATIVE)

    def test_rejects_changed_semantics_and_audit_bytes(self):
        original = json.loads((HERE.parent / 'qualification/runtime_v2_source_rng_v1.json').read_text())
        current = json.loads((HERE / 'source_rng_audits/server51.json').read_text())
        rng.verify_audits.cache_clear()
        changed = copy.deepcopy(current); changed['semantic_check']['cases'][0]['exact_cpu_rng_state_match'] = False
        with patch.object(rng, 'read_pinned', side_effect=[original, changed]):
            with self.assertRaisesRegex(ValueError, 'preserve every assumption'):
                rng.verify_audits()
        with self.assertRaises(ValueError):
            rng.read_pinned(HERE / 'source_rng_audits/server51.json', '0' * 64)
        rng.verify_audits.cache_clear(); rng.verify_audits()

    def test_explicit_dependency_closure_covers_restored_objectives_and_audits(self):
        phase = HERE.parent.parent
        expected = {'benchmark_expansion_20261015_v1/interaction_BUDDY_production_v5/objectives.py',
                    rng.AUDIT_RELATIVE, rng.ORIGINAL_RELATIVE,
                    'benchmark_expansion_20261015_v1/server51_campaign_port_v6/source_rng_binding.py'}
        for name in ('composed/SOURCE_BINDINGS.json', 'scoring/SOURCE_BINDINGS.json'):
            rows = json.loads((HERE / name).read_text())['files']
            by_path = {row['path_relative_to_phase']: row for row in rows}
            self.assertTrue(expected <= by_path.keys())
            for key in expected:
                raw = (phase / key).read_bytes(); row = by_path[key]
                self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (row['bytes'], row['sha256']))
        for name in ('runtime_node_molecular/source_rng.py', 'runtime_link/source_rng.py', 'composed/runtime/source_rng.py'):
            self.assertEqual((HERE / name).read_bytes(), (HERE.parent / 'server51_campaign_port_v5' / name).read_bytes())
        self.assertNotIn('torch', sys.modules); self.assertNotIn('numpy', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
