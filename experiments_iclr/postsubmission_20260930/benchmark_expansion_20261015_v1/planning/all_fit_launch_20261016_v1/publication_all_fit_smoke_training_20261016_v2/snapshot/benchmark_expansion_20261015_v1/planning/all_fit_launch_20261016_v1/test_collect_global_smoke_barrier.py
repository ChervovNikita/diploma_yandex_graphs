"""Bounded metadata/transport tests; no SSH, model, dataset or scientific PASS."""
import contextlib
import copy
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('coordinator_unit_only', HERE / 'collect_global_smoke_barrier.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


class CoordinatorTests(unittest.TestCase):
    def test_pending_never_fetches_validates_or_delivers(self):
        def remote(host, config, operation, **kwargs):
            self.assertEqual(operation, 'probe')
            return {'workers': [{'worker': worker, 'status': 'pending'}
                                for worker in c.WORKERS if worker.split('/')[0] == host]}
        with tempfile.TemporaryDirectory(dir=HERE, prefix='coordinator-unit-') as directory:
            proof = Path(directory)
            with patch.object(c, 'remote', side_effect=remote), patch.object(c, 'fetch', side_effect=AssertionError('fetch')), \
                    patch.object(c, 'deliver', side_effect=AssertionError('delivery')), \
                    patch.object(c.subprocess, 'run', side_effect=AssertionError('validator')):
                result = c.collect({}, b'{"unit_test_only":true}\n', proof)
            self.assertEqual(result['status'], 'pending')
            self.assertEqual(result['large_files_transferred'], 0)
            self.assertEqual({p.name for p in proof.iterdir()}, {'BINDINGS.json'})

    def test_mixed_worker_inventory_is_refused(self):
        rows = [{'worker': worker, 'status': 'PASS', 'inventory': {'sha256': 'a'}} for worker in c.WORKERS]
        rows[-1]['inventory'] = {'sha256': 'b'}
        probes = {host: {'workers': [r for r in rows if r['worker'].split('/')[0] == host]} for host in c.ROOTS}
        with self.assertRaisesRegex(ValueError, 'different inventories'):
            c.ready_workers(probes)
        probes['server51']['workers'][-1] = copy.deepcopy(probes['server51']['workers'][0])
        with self.assertRaisesRegex(ValueError, 'worker set'):
            c.ready_workers(probes)

    def test_existing_local_evidence_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix='coordinator-unit-') as directory:
            path = Path(directory) / 'proof.json'
            c.exact_write(path, b'original')
            c.exact_write(path, b'original')
            with self.assertRaises(ValueError):
                c.exact_write(path, b'changed')
            self.assertEqual(path.read_bytes(), b'original')

    def test_actual_upload_fragment_retry_preserves_bytes(self):
        # Execute only the upload branch against an isolated directory in L.
        # The real remote() wrapper and scientific host transports are never called.
        with tempfile.TemporaryDirectory(dir=HERE, prefix='coordinator-unit-') as directory:
            raw = b'unit-test transport bytes only'
            packet = c.identity(raw)
            request = {'root': directory, 'expansion_relative': 'unit_only', 'packages': {'execution': 'queue'},
                       'config': {}, 'operation': 'upload', 'packet': packet, 'offset': 0,
                       'data': base64.b64encode(raw).decode()}
            (Path(directory) / 'unit_only/queue').mkdir(parents=True)
            for _ in range(2):
                with contextlib.redirect_stdout(io.StringIO()) as stream:
                    exec(c.REMOTE, {'REQUEST': copy.deepcopy(request)})
                self.assertEqual(json.loads(stream.getvalue())['next_offset'], len(raw))
            request['data'] = base64.b64encode(b'x' * len(raw)).decode()
            with self.assertRaisesRegex(ValueError, 'Existing upload bytes differ'):
                exec(c.REMOTE, {'REQUEST': request})
            path = Path(directory) / 'unit_only/queue/.global_smoke_barrier_transfer' / (packet['sha256'] + '.upload')
            self.assertEqual(path.read_bytes(), raw)

    def test_packed_engine_and_request_use_small_commands(self):
        # Execute the real stage/loader with an offline local transport, inside L.
        # No host connector, source manifest or scientific validator is invoked.
        class Offline:
            commands = []
            def call(self, code, timeout):
                self.commands.append(code)
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    exec(code, {})
                return output.getvalue().strip()
        with tempfile.TemporaryDirectory(dir=HERE, prefix='coordinator-unit-') as directory:
            transport = Offline()
            root = Path(directory)
            (root / c.EXPANSION_REL / c.PACKAGES['execution']).mkdir(parents=True)
            raw = b''.join(hashlib.sha256(str(i).encode()).digest() for i in range(32))
            config = {'sources': {kind: c.identity(kind.encode()) for kind in c.PACKAGES},
                      'workers': [{'worker': worker, 'plan': {'path': 'x' * 240, **c.identity(worker.encode())}}
                                  for worker in c.WORKERS]}
            request = {'root': directory, 'expansion_relative': c.EXPANSION_REL, 'packages': c.PACKAGES,
                       'config': config, 'operation': 'upload', 'packet': c.identity(raw), 'offset': 0,
                       'data': base64.b64encode(raw).decode()}
            with patch.dict(c.ROOTS, {'server51': directory}):
                c.stage_engine(transport, 'server51')
                code = c.invocation_code('server51', request)
                value = json.loads(transport.call(code, timeout=60))
            self.assertEqual(value['next_offset'], len(raw))
            self.assertTrue(all(len(code.encode()) <= c.COMMAND_LIMIT for code in transport.commands))
            self.assertLess(max(map(len, transport.commands)), 5000)


if __name__ == '__main__':
    unittest.main()
