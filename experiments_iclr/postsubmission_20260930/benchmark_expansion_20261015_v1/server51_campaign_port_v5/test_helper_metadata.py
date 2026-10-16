"""Actual helper classes/factory with session fixtures; no native/model work."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
ROOT = EXPANSION.parent.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ActualHelperInterfaces(unittest.TestCase):
    def setUp(self):
        modules = patch.dict(sys.modules)
        modules.start(); self.addCleanup(modules.stop)
        paths = patch.object(sys, 'path', list(sys.path))
        paths.start(); self.addCleanup(paths.stop)
        for name in ('common', 'controllers', 'bookkeeping', 'source_rng', 'adapters', 'link_authority',
                     'host_binding', 'descriptor_binding', 'runtime_bridge', '_composed_pinned_runtime_link'):
            sys.modules.pop(name, None)
        self.bridge = load(HERE / 'composed/runtime_bridge.py', '_port5_helper_test_bridge')
        self.bridge._fresh_runtime_imports()
        self.authority = self.bridge.load(HERE / 'composed/runtime/link_authority.py', 'link_authority')
        self.helper = self.bridge.load(EXPANSION / 'portable_dispatcher_v3/runtime_link/adapters.py',
                                       '_composed_pinned_runtime_link')
        self.native = load(EXPANSION / 'link_native_integration_v1/session.py', '_port5_actual_epoch_class')
        self.normalization = load(EXPANSION / 'composable_normalization_v2/adapter.py', '_port5_actual_normalization')

    def native_fixture(self):
        pin = self.bridge.receipt(EXPANSION / 'link_native_integration_v1/MANIFEST.json')
        metadata = {'native_integration': {'integration_sources': [], 'finalized_helper_sources': [],
                    'native_binding': {'adapter_source_receipts': {},
                                       'adapter_manifest_receipt': dict(pin, adapter_hashes_verified=True)}}}
        session = SimpleNamespace(model=object(), optimizers=[object()], native=SimpleNamespace(),
                                  metadata=lambda: copy.deepcopy(metadata))
        return self.native.LinkEpochAdapter(session)

    def job(self, callback, procedure='gnnm_boundary_native', manifest=None):
        policy = self.bridge.receipt(HERE / 'runtime_link/engineering_link_policy.py')
        return {'family': 'link', 'procedure': procedure, 'callback_module': str(callback),
                'callback_sha256': self.bridge.receipt(callback)['sha256'],
                'link_policy_module': policy['path'], 'link_policy_sha256': policy['sha256'],
                'method_settings': {} if manifest is None else {'manifest_sha256': manifest['sha256']}}

    def through_factory(self, inner, job):
        # Load and verify the real callback. Substitute only native session
        # construction; the factory, dependency authenticator and wrappers run.
        callback, _ = self.helper.load_module(job['callback_module'], ROOT, job['callback_sha256'])
        with patch.object(callback, 'create_adapter', return_value=inner):
            return self.helper.create_adapter(None, job, SimpleNamespace(family='link'), ROOT)

    def normalized(self, inner):
        # Exercise the real delegation interface without native BN/model work.
        normalized = self.normalization.NormalizedAdapter.__new__(self.normalization.NormalizedAdapter)
        normalized.inner = inner
        normalized.binding = {}
        return normalized

    def plain(self):
        job = self.job(EXPANSION / 'link_native_integration_v1/session.py')
        return self.through_factory(self.native_fixture(), job), job

    def test_observed_plain_class_failure_and_successor_binding(self):
        inner, job = self.plain()
        normalized = self.normalized(inner)
        self.assertIs(type(inner), self.native.LinkEpochAdapter)
        self.assertFalse(hasattr(inner, 'policy_receipt'))
        old = load(EXPANSION / 'server51_campaign_port_v4/composed/runtime_bridge.py', '_failed_port4_helper')
        with self.assertRaisesRegex(AttributeError, 'policy_receipt'):
            old._helper_binding(normalized, 'link')
        before = dict(inner.__dict__)
        actual = self.bridge._helper_binding(normalized, job, ROOT)
        self.assertEqual(actual['policy_sha256'], job['link_policy_sha256'])
        self.assertEqual(actual['callback_sha256'], job['callback_sha256'])
        self.assertIsNone(actual['extension_sha256'])
        self.assertEqual(actual['dependency_sha256'], {p['path']: p['sha256'] for p in inner.dependency_receipts})
        self.assertEqual(inner.__dict__, before)
        self.assertFalse(hasattr(inner, 'policy_receipt'))

    def test_all_registered_extension_protocols_use_actual_return_classes(self):
        seen = set()
        for package, entry in self.authority.REGISTRY.items():
            methods = ('aggregation_mean_globalJ_v1', 'fixture_unchanged_mean') if entry['mean'] == 'C6' else ('fixture_method',)
            for method in methods:
                with self.subTest(package=package, method=method):
                    folder = EXPANSION / package
                    extension = self.bridge.receipt(folder / entry['extension_name'])
                    manifest = self.bridge.receipt(folder / 'MANIFEST.json')
                    job = self.job(folder / entry['callback_name'], method, manifest)
                    inner = self.native_fixture()
                    inner.extension_receipt = extension
                    declaration = {'schema': entry['schema'], 'method_id': method, 'TEST_accessed': False,
                                   'source_receipts': [extension, manifest]}
                    if entry['mean'] == 'C6' and method in self.authority.C6:
                        declaration['aggregation'] = {'fixture': True}
                    target = inner.session
                    for attribute in entry['declaration_path'][:-1]:
                        target = getattr(target, attribute)
                    setattr(target, entry['declaration_path'][-1], declaration)
                    actual = self.through_factory(inner, job)
                    expected = self.helper.LinkSameCallMeanAdapter if entry['mean'] == 'provider' else self.helper.LinkDeclaredExtensionAdapter
                    self.assertIs(type(actual), expected)
                    seen.add(type(actual).__name__)
                    bound = self.bridge._helper_binding(self.normalized(actual), job, ROOT)
                    self.assertEqual(bound['policy_sha256'], job['link_policy_sha256'])
                    self.assertEqual(bound['callback_sha256'], job['callback_sha256'])
                    self.assertEqual(bound['extension_sha256'], extension['sha256'])
                    self.assertEqual(bound['dependency_sha256'], {p['path']: p['sha256'] for p in actual.dependency_receipts})
        self.assertEqual(seen, {'LinkSameCallMeanAdapter', 'LinkDeclaredExtensionAdapter'})
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('numpy', sys.modules)

    def test_policy_and_other_source_contract_guards(self):
        inner, job = self.plain()
        for field in ('link_policy_module', 'link_policy_sha256'):
            bad = copy.deepcopy(job); del bad[field]
            with self.assertRaisesRegex(ValueError, 'Explicit pinned job link policy'):
                self.bridge._helper_binding(inner, bad, ROOT)
        bad = copy.deepcopy(job); bad['link_policy_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Job link policy source changed'):
            self.bridge._helper_binding(inner, bad, ROOT)
        bad = copy.deepcopy(job); bad['link_policy_module'] = str(ROOT.parent / 'outside.py')
        with self.assertRaisesRegex(ValueError, 'escapes actual project'):
            self.bridge._helper_binding(inner, bad, ROOT)
        inner.policy_receipt = dict(self.bridge.receipt(job['link_policy_module']), sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'Helper policy receipt differs'):
            self.bridge._helper_binding(inner, job, ROOT)
        del inner.policy_receipt
        for field in ('callback_receipt', 'extension_receipt', 'dependency_receipts'):
            with self.subTest(field=field):
                original = copy.deepcopy(getattr(inner, field))
                corrupt = dict(inner.callback_receipt, sha256='0' * 64)
                setattr(inner, field, [corrupt] if field == 'dependency_receipts' else corrupt)
                with self.assertRaisesRegex(ValueError, 'Helper source receipt changed'):
                    self.bridge._helper_binding(inner, job, ROOT)
                setattr(inner, field, original)
        bad = copy.deepcopy(job); bad['callback_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Helper callback differs'):
            self.bridge._helper_binding(inner, bad, ROOT)

    def test_producer_constructor_and_recheck_preserve_real_helper_interface(self):
        inner, job = self.plain()
        normalized = self.normalized(inner)
        source = self.bridge.verify_sources(ROOT)
        producer = self.bridge.ProducerAdapter(normalized, job, ROOT, source)
        result = self.bridge.build_runtime_binding(producer, ROOT)
        self.assertEqual(result['policy_sha256'], job['link_policy_sha256'])
        self.assertEqual(result['composition_producer']['unchanged_helper_runtime'], producer.helper_runtime)
        self.assertFalse(hasattr(inner, 'policy_receipt'))
        producer.job['link_policy_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Job link policy source changed'):
            self.bridge.build_runtime_binding(producer, ROOT)


if __name__ == '__main__':
    unittest.main(verbosity=2)
