"""Source-only BUDDY descriptor, import and reporting integration checks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from descriptor_binding import derive, digest, json_bytes, byte_receipt, composition_contract, NORMALIZATION_SHA, BANKS_SHA
from host_binding import ALLOCATION_ROOT, SERVER51_ROOT, PHASE_RELATIVE
from repair_bindings import repair_metadata, repair_sources, rebind_buddy
from reporting_projection import project_profile


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


class RepairChecks(unittest.TestCase):
    def job(self, row):
        delta, _, _ = repair_metadata()
        prefix = ALLOCATION_ROOT + '/' + PHASE_RELATIVE + '/'
        return {'schema': 'native-full-epoch-fit-v1', 'id': 'engineering-source-only',
                'family': 'link', 'recipe_id': 'collab-buddy', 'scientific_method_id': 'aggregation_mean_globalJ_v1',
                'procedure': 'aggregation_mean_globalJ_v1', 'base_procedure': 'gnnm_boundary_native',
                'seed': 8, 'split': 0, 'body': None, 'TEST_access': False, 'device': 'cuda:0',
                'callback_module': prefix + row['old_callback']['path_relative_to_phase'],
                'callback_sha256': row['old_callback']['sha256'], 'data_root': ALLOCATION_ROOT + '/data',
                'policy': {'track': 'matched_extended', 'maximum_epochs': 2000},
                'recipe': {'architecture': 'BUDDY', 'learning_rate': .001, '_repository_root': ALLOCATION_ROOT,
                           '_redesign_root': prefix + 'reference_wrapper_redesign_20261013_v1',
                           '_native_callback': prefix + 'reference_wrapper_redesign_20261013_v1/smoke_20261014/link_port/callbacks.py'},
                'method_settings': {'manifest_sha256': row['old_manifest']['sha256'],
                                    'provider_manifest_sha256': delta['provider']['old_manifest']['sha256'],
                                    'static_provider_sha256': delta['provider']['static_provider']['sha256'],
                                    'provider_mode': 'production_native'},
                'expanded_contract': {'id': 'engineering-source-only', 'scientific_method_id': 'aggregation_mean_globalJ_v1',
                                      'native_settings': {'learning_rate': .001}}}

    def test_exact_replacements_preserve_scientific_fields_and_unrelated_jobs(self):
        delta, _, _ = repair_metadata()
        for row in delta['replacements']:
            original = self.job(row); original['expanded_contract_hash'] = digest(original['expanded_contract'])
            before = copy.deepcopy(original)
            for target in (ALLOCATION_ROOT, SERVER51_ROOT):
                bound, raw, binding = derive(original, ALLOCATION_ROOT, PHASE_RELATIVE, target, PHASE_RELATIVE)
                self.assertIsNone(raw)
                self.assertEqual(original, before)
                self.assertEqual(bound['expanded_contract'], original['expanded_contract'])
                self.assertEqual(bound['expanded_contract_hash'], original['expanded_contract_hash'])
                self.assertEqual(bound['callback_sha256'], row['new_callback']['sha256'])
                self.assertEqual(bound['method_settings']['manifest_sha256'], row['new_manifest']['sha256'])
                self.assertEqual(bound['method_settings']['static_provider_sha256'], original['method_settings']['static_provider_sha256'])
                self.assertEqual({c['pointer'] for c in binding['source_binding_repair']['field_changes']},
                                 {'/callback_module', '/callback_sha256', '/method_settings/manifest_sha256', '/method_settings/provider_manifest_sha256'})
                for key in ('id', 'seed', 'split', 'body', 'policy', 'procedure', 'base_procedure', 'scientific_method_id'):
                    self.assertEqual(bound[key], original[key])
            bad = copy.deepcopy(original); bad['method_settings']['provider_manifest_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'Canonical BUDDY'):
                rebind_buddy(bad, ALLOCATION_ROOT, PHASE_RELATIVE)
        untouched = self.job(delta['replacements'][0]); untouched['callback_module'] = ALLOCATION_ROOT + '/other.py'
        self.assertEqual(rebind_buddy(untouched, ALLOCATION_ROOT, PHASE_RELATIVE), (untouched, None))

    def test_composed_parent_rederived_without_changing_normalization_law(self):
        row = repair_metadata()[0]['replacements'][0]
        parent = self.job(row); parent['expanded_contract_hash'] = digest(parent['expanded_contract'])
        raw = json_bytes(parent)
        declaration = {'schema': 'native-route-BN-composition-v1', 'policy': 'private_native_running_state_shared_affine_v1',
                       'source_manifest_sha256': NORMALIZATION_SHA, 'banks_sha256': BANKS_SHA,
                       'parent_descriptor': byte_receipt(ALLOCATION_ROOT + '/original.json', raw),
                       'parent_job_sha256': digest(parent), 'matched_control_method_id': 'gnnm_private_bn_state_v3'}
        roster = json.loads((HERE.parent / 'expanded_study_design_v1/PROPOSED_ROSTER.json').read_text())
        contracts = composition_contract(HERE.parent)
        original = contracts.migrated_job(parent, declaration, roster)
        bound, rebound_raw, binding = derive(original, ALLOCATION_ROOT, PHASE_RELATIVE, SERVER51_ROOT, PHASE_RELATIVE,
            parent=parent, parent_raw=raw, execution_parent_path=SERVER51_ROOT + '/parent.json', expansion=HERE.parent, roster=roster)
        rebound = json.loads(rebound_raw)
        self.assertEqual(bound, contracts.migrated_job(rebound, bound['normalization_composition'], roster))
        self.assertEqual(bound['id'], original['id'])
        self.assertEqual(bound['normalization_composition']['source_manifest_sha256'], NORMALIZATION_SHA)
        self.assertEqual(bound['normalization_composition']['banks_sha256'], BANKS_SHA)
        self.assertEqual(binding['canonical_runtime_job_hash'], digest(original))
        self.assertEqual(binding['execution_parent_descriptor'], byte_receipt(SERVER51_ROOT + '/parent.json', rebound_raw))
        self.assertNotEqual(rebound['callback_sha256'], parent['callback_sha256'])
        self.assertFalse(binding['scientific_settings_changed'])

    def test_frozen_helper_imports_actual_new_composed_authenticator(self):
        bridge = load(HERE / 'composed/runtime_bridge.py', '_repair_test_bridge')
        names = ('common', 'controllers', 'bookkeeping', 'source_rng', 'adapters', 'link_authority',
                 'host_binding', 'descriptor_binding', 'runtime_bridge', '_composed_pinned_runtime_link')
        with patch.dict(sys.modules), patch.object(sys, 'path', list(sys.path)):
            for name in names:
                sys.modules.pop(name, None)
            bridge._fresh_runtime_imports()
            authority = bridge.load(bridge.RUNTIME / 'link_authority.py', 'link_authority')
            helper = bridge.load(HERE.parent / 'portable_dispatcher_v3/runtime_link/adapters.py', '_composed_pinned_runtime_link')
            self.assertEqual(Path(helper.create_adapter.__code__.co_filename), HERE.parent / 'portable_dispatcher_v3/runtime_link/adapters.py')
            self.assertEqual(Path(authority.authenticate_extension.__code__.co_filename), HERE / 'composed/runtime/link_authority.py')
            self.assertEqual(pin(authority.__file__)['sha256'], '31b2e3759c6989c2811602a842181d1333550ac2462a03f254939146a18e493e')
            row = repair_metadata()[0]['replacements'][1]
            phase = HERE.parent.parent
            job = self.job(row)
            job['callback_module'] = str(phase / row['new_callback']['path_relative_to_phase'])
            job['callback_sha256'] = row['new_callback']['sha256']
            job['method_settings']['manifest_sha256'] = row['new_manifest']['sha256']
            extension = pin(phase / row['new_session']['path_relative_to_phase'])
            manifest = pin(phase / row['new_manifest']['path_relative_to_phase'])
            native_manifest = pin(HERE.parent / 'link_native_integration_v1/MANIFEST.json')
            declaration = {'schema': 'explicit-BUDDY-interaction-production-v2', 'method_id': job['procedure'],
                           'TEST_accessed': False, 'source_receipts': [extension, manifest]}
            meta = {'native_integration': {'integration_sources': [], 'finalized_helper_sources': [],
                    'native_binding': {'adapter_source_receipts': {},
                                       'adapter_manifest_receipt': dict(native_manifest, adapter_hashes_verified=True)}}}
            adapter = SimpleNamespace(extension_receipt=extension, session=SimpleNamespace(interaction_metadata=declaration),
                                      metadata=lambda: meta)
            result = helper._link_source_dependencies(adapter, pin(job['callback_module']), phase.parent, job)
            self.assertEqual(result.extension_authority['callback']['sha256'], row['new_callback']['sha256'])
            self.assertEqual(result.extension_authority['mean_protocol'], 'provider')
            self.assertEqual(sys.modules['link_authority'].__file__, str(HERE / 'composed/runtime/link_authority.py'))
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('numpy', sys.modules)

    def test_reporting_binds_actual_callback_session_and_preserves_comparisons(self):
        row = repair_metadata()[0]['replacements'][0]
        original = self.job(row); original['expanded_contract_hash'] = digest(original['expanded_contract'])
        job, _, binding = derive(original, ALLOCATION_ROOT, PHASE_RELATIVE, SERVER51_ROOT, PHASE_RELATIVE)
        identity = {'fit_id': job['id'], 'bindings': {'native_source': 'fixed', 'readout': 'canonical'}}
        profile = {'fit_specs': {job['id']: {'descriptor': {'path': 'canonical'}, 'job_hash': digest(original),
                   'identity': identity, 'bindings': identity['bindings'],
                   'wrapper_authority': {'descriptor': {'path': 'canonical'}, 'callback': {'path': original['callback_module']}, 'extension': None},
                   'observed_input_files': []}}, 'I4_specs': {}, 'registry': {'fixed': True}, 'contrast_plan': {'fixed': True}}
        tasks = {job['id']: {'runtime_job_hash': digest(job), 'runtime_binding': binding}}
        candidates = {job['id']: {'_job': job, 'job_descriptor': {'path': 'execution'},
                                'runtime_entrypoint': {'path': 'runtime'}}}
        def source(path, expected=None):
            intended = Path(path)
            local = HERE.parent.parent / intended.relative_to(Path(SERVER51_ROOT) / PHASE_RELATIVE)
            actual = pin(local)
            if expected:
                self.assertTrue(all(actual[k] == expected[k] for k in ('bytes', 'sha256') if k in expected))
            return dict(actual, path=str(intended))
        result = project_profile(profile, {'path': 'canonical-profile'}, tasks, candidates,
                                 {'host': 'server51', 'project_root': SERVER51_ROOT, 'phase_relative': PHASE_RELATIVE}, source)
        spec = result['fit_specs'][job['id']]
        self.assertEqual(spec['identity'], identity)
        self.assertEqual(spec['bindings'], identity['bindings'])
        self.assertEqual(result['registry'], profile['registry'])
        self.assertEqual(result['contrast_plan'], profile['contrast_plan'])
        self.assertEqual(spec['wrapper_authority']['callback']['sha256'], row['new_callback']['sha256'])
        self.assertEqual(spec['wrapper_authority']['extension']['sha256'], row['new_session']['sha256'])
        self.assertEqual(spec['canonical_wrapper_authority'], profile['fit_specs'][job['id']]['wrapper_authority'])
        self.assertTrue(set(r['sha256'] for r in repair_sources()).issubset({r['sha256'] for r in spec['observed_input_files']}))

    def test_private_bn_scoring_authenticates_actual_v5_class_and_owned_snapshot(self):
        from scoring import buddy_inference as inference
        row = repair_metadata()[0]['replacements'][2]
        job = self.job(row)
        job['procedure'] = 'gnnm_private_bn_state_v3'
        phase = HERE.parent.parent
        job['callback_module'] = str(phase / row['new_callback']['path_relative_to_phase'])
        job['callback_sha256'] = row['new_callback']['sha256']
        job['method_settings']['manifest_sha256'] = row['new_manifest']['sha256']
        name = '_source_test_actual_BUDDY_v5'
        parent = load(phase / row['new_session']['path_relative_to_phase'], name)
        with patch.dict(sys.modules, {name: parent}):
            session = parent.BUDDYPrivateBN.__new__(parent.BUDDYPrivateBN)
            session.job = copy.deepcopy(job)
            session.method_metadata = {'schema': 'native-BUDDY-private-BN-production-source-v1',
                                      'method_id': 'gnnm_private_bn_state_v3', 'port': 'buddy_private_bn_production_v2',
                                      'source_receipts': parent.bindings(phase.parent, row['new_manifest']['sha256'])}
            before = copy.deepcopy(session.method_metadata)
            actual, _ = inference.authenticate_producer(session, job, phase.parent)
            self.assertIs(actual, parent)
            wrong = copy.deepcopy(job); wrong['callback_module'] = str(HERE.parent / 'buddy_private_bn_production_v2/callback.py')
            session.job = wrong
            with self.assertRaisesRegex(ValueError, 'Actual v5 callback'):
                inference.authenticate_producer(session, wrong, phase.parent)
            session.job = copy.deepcopy(job)
            session.inference_metadata = {'schema': 'bound-BUDDY-private-BN-inference-api-v2',
                                          'inference_port': inference.INFERENCE_PORT,
                                          'producer_package': 'buddy_private_bn_production_v5'}
            calls = []
            def owned_state():
                calls.append('owned state export')
                return {'schema': 'native-BUDDY-private-BN-production-state-v1', 'method': session.method_metadata,
                        'private_BN_banks': {'bank': 'private-buffer'}, 'logical_branch_calls': {'mutable': 10},
                        'native_integration': {'schema': 'BUDDY-production-native-provider-state-v1', 'ledger': {'mutable': 20},
                            'native_integration': {'schema': 'actual-native-link-extension-state-v1', 'global_rng': 'mutable',
                                                  'native': {'model': {'weight': 'parameter', 'extra': {'typed': True}},
                                                             'optimizer': {'mutable': 30}}}}}
            session.state = owned_state
            result = inference.owned_model_state(session, phase.parent)
            self.assertEqual(calls, ['owned state export'])
            self.assertEqual(result, {'model': {'weight': 'parameter', 'extra': {'typed': True}},
                                      'private_BN_banks': {'bank': 'private-buffer'}})
            self.assertEqual(session.method_metadata, before)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('numpy', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
