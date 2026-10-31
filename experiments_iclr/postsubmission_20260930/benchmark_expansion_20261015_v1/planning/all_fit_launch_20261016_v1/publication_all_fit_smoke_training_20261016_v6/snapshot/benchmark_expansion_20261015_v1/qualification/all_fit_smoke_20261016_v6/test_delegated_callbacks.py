"""Exercise actual native epoch/operation methods with scalar optimizer state."""
import ast
from contextlib import contextmanager
import copy
import importlib.util
import math
from pathlib import Path
import unittest

from smoke_control import install_step_cap, train_two_updates
from support import E, HERE
from test_source import Adapter, Optimizer, Session, observer_class


def actual_method(path, klass, method):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == klass)
    function = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    namespace = {'math': math, 'copy': copy, 'contextmanager': contextmanager}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[method]


def delegated_native_chain():
    integration = E / 'link_native_integration_v1/session.py'
    provider = E / 'buddy_native_provider_v4/provider.py'
    private = E / 'buddy_private_bn_production_v5/session.py'
    runtime = E / 'server51_campaign_port_v6/runtime_link/adapters.py'
    optimizer = Optimizer()

    class Link:
        train_epoch = actual_method(integration, 'LinkSession', 'train_epoch')
        completed_epochs = 0

        def __init__(self):
            self.updates = 0

        def train_update(self):
            self.updates += 1
            optimizer.step()
            return {'loss': 1.0, 'work': {}}

        def epoch_progress(self):
            return {'maps_to_next_completed_epoch': 1585}

    class Provider:
        train_epoch = actual_method(provider, 'NativeProviderSession', 'train_epoch')
        train_update = actual_method(provider, 'NativeProviderSession', 'train_update')
        operation = actual_method(provider, 'NativeProviderSession', 'operation')

        def __init__(self, base):
            self.base, self.native = base, object()
            self._closed, self._role, self._query_traversal, self._pending = False, None, False, None

        def assert_identity(self):
            pass

    class Private:
        train_epoch = actual_method(private, 'BUDDYPrivateBN', 'train_epoch')
        operation = actual_method(private, 'BUDDYPrivateBN', 'operation')

        def __init__(self, base):
            self.base = base
            self._operation = self._state_operation = None
            self.manager = type('Manager', (), {'_active': None})()
            self._calls = {'TRAIN': {}}

        def train_update(self):
            return self.base.train_update()

        def _counts(self):
            return {}

        def assert_identity(self):
            pass

        def _audit_TRAIN(self, *args):
            raise AssertionError('A partial smoke may not claim a complete native epoch')

    class Inner:
        train_epoch = actual_method(integration, 'LinkEpochAdapter', 'train_epoch')

        def __init__(self, session):
            self.session = session

    class Outer:
        train_epoch = actual_method(runtime, 'LinkSameCallMeanAdapter', 'train_epoch')
        completed_epochs = 0

        def __init__(self, inner):
            self.inner, self.session = inner, inner.session

        def _invalidate(self):
            pass

    link = Link(); production = Provider(link); session = Private(production)
    return Outer(Inner(session)), optimizer, (session, production, link)


class DelegationTests(unittest.TestCase):
    def test_exact_native_delegation_reproduces_old_failure(self):
        path = HERE.parent / 'all_fit_smoke_20261016_v5/smoke_control.py'
        spec = importlib.util.spec_from_file_location('_old_smoke_control_regression', path)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        adapter, optimizer, sessions = delegated_native_chain()
        observer = observer_class('runtime_link')(None, [optimizer])
        old.install_step_cap(observer, [optimizer])
        try:
            with self.assertRaisesRegex(ValueError, 'Third optimizer transition'):
                old.train_two_updates(adapter, observer, {})
            self.assertEqual(optimizer.calls, 2)
            self.assertEqual(sessions[-1].updates, 3)
            self.assertIsNone(sessions[0]._operation)
            self.assertIsNone(sessions[1]._role)
        finally:
            observer.close()

    def test_actual_delegated_epoch_stops_before_third_native_update(self):
        adapter, optimizer, sessions = delegated_native_chain()
        before = [dict(vars(session)) for session in sessions]
        observer = observer_class('runtime_link')(None, [optimizer])
        install_step_cap(observer, [optimizer]); report = {}
        try:
            train_two_updates(adapter, observer, report)
            self.assertEqual((optimizer.calls, sessions[-1].updates), (2, 2))
            self.assertEqual(observer.state(), {'attempts': [2], 'completed': [2]})
            self.assertEqual(report['TRAIN_losses'], [1.0, 1.0])
            self.assertTrue(report['TRAIN_finite'])
            self.assertEqual(report['TRAIN_boundary_probes'], 1)
            self.assertIsNone(sessions[0]._operation)
            self.assertIsNone(sessions[1]._role)
            self.assertEqual([('train_update' in vars(s)) for s in sessions],
                             [('train_update' in values) for values in before])
            with self.assertRaisesRegex(ValueError, 'Third optimizer transition'):
                optimizer.step()
        finally:
            observer.close()

    def test_nested_wrappers_count_each_real_update_once_and_restore(self):
        optimizer = Optimizer(); base = Session(optimizer)

        class Wrapper:
            def __init__(self, base):
                self.base = base

            def train_update(self):
                return self.base.train_update()

        session = Wrapper(Wrapper(base)); adapter = Adapter(session, 38)
        observer = observer_class('runtime_link')(None, [optimizer])
        install_step_cap(observer, [optimizer]); report = {}
        try:
            train_two_updates(adapter, observer, report)
            self.assertEqual(optimizer.calls, 2)
            self.assertEqual(report['TRAIN_callback_calls'], 2)
            self.assertEqual(report['TRAIN_nested_callback_calls'], 4)
            self.assertEqual(report['TRAIN_losses'], [1.0, 1.0])
            self.assertTrue(all('train_update' not in vars(s) for s in (session, session.base, base)))
        finally:
            observer.close()

    def test_cycle_and_failed_callback_restore_without_more_work(self):
        optimizer = Optimizer(); session = Session(optimizer); session.base = session
        observer = observer_class('runtime_link')(None, [optimizer])
        install_step_cap(observer, [optimizer])
        try:
            with self.assertRaisesRegex(ValueError, 'Cyclic'):
                train_two_updates(Adapter(session, 38), observer, {})
            self.assertEqual(optimizer.calls, 0)
            self.assertNotIn('train_update', vars(session))
            del session.base
            original = lambda: (_ for _ in ()).throw(RuntimeError('native callback failed'))
            session.train_update = original
            with self.assertRaisesRegex(RuntimeError, 'native callback failed'):
                train_two_updates(Adapter(session, 38), observer, {})
            self.assertIs(session.train_update, original)
            self.assertEqual(optimizer.calls, 0)
        finally:
            observer.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
