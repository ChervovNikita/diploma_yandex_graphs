"""Local metadata and transport fixtures only; no host connections or native fits."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import coordinator as c
import readable_transport as transport
from remote_backend import Backend, PACKAGES, WORKERS, exact_write, identity, json_bytes, receipt

COUNTS = {'allocation/gpu0': 2154, 'server51/gpu0': 2153, 'server51/gpu1': 2153}
CANONICAL = 'be7bfe7c3beb5e5c9e639b53404455ee8b92b20f4dc038b05224fe141782ff64'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def save(path, value):
    exact_write(path, json_bytes(value))
    return receipt(path)


def coverage_fixture(folder, config):
    """Synthetic canonical-sized metadata; never a research or smoke receipt."""
    ids = ['synthetic-local-test-%04d' % i for i in range(6460)]
    source = {'path': '/synthetic/MANIFEST.json', **config['sources']['smoke']}
    fits, owned = [], {}
    offset = 0
    for worker, count in COUNTS.items():
        owned[worker] = ids[offset:offset + count]
        offset += count
        for fit_id in owned[worker]:
            fits.append({'id': fit_id, 'worker': worker, 'runtime_job_hash': digest(fit_id),
                         'canonical_runtime_job_hash': digest(fit_id), 'science_contract_hash': digest(fit_id),
                         'descriptor': {'path': '/synthetic/' + fit_id, 'bytes': 1, 'sha256': digest(fit_id)},
                         'runtime_entrypoint': source})
    inventory = {'schema': 'all-fit-smoke-inventory-v1', 'source_manifest': source,
                 'canonical_plan': {'sha256': CANONICAL}, 'class_substitution': False,
                 'canonical_fit_ids': ids, 'canonical_fit_ids_sha256': digest(ids), 'fits': fits}
    inventories = {host: save(folder / ('inventory-' + host + '.json'), inventory) for host in c.ROOTS}
    rows, values = [], {}
    for worker in WORKERS:
        inv = inventories[worker.split('/')[0]]
        results = []
        for row in (row for row in fits if row['worker'] == worker):
            result = {**row, 'schema': 'all-fit-two-update-smoke-result-v1', 'status': 'PASS', 'inventory': inv,
                      'source_manifest': source, 'optimizer_calls': {'attempts': [2], 'completed': [2]}, 'TRAIN_updates': 2,
                      'TRAIN_finite': True, 'TRAIN_losses': [1.0, 0.9], 'VALID_finite': True, 'VALID_calls': 1,
                      'VALID': {'role': 'VALID', 'complete_source_VALID': True, 'records': 3, 'loss': 0.5,
                                'metrics': {'scalar': 0.5}},
                      'checkpoint_reload': {'status': 'PASS', 'exact_adapter_state_restored': True,
                           'parameter_perturbation_restored': True, 'optimizer_state_reloaded': True,
                           'optimizer_moment_perturbation_restored': True, 'TRAIN_updates_after_reload': 0},
                      'TEST_accessed': False, 'scientific_checkpoint_reused': False,
                      'scientific_fit_policy_changed': False, 'automatic_retries': 0}
            results.append({'id': row['id'], 'runtime_job_hash': row['runtime_job_hash'], 'attempt': 1,
                            'result': result, 'result_receipt': {'path': '/synthetic/' + row['id'] + '/RESULT.json',
                                                               **identity(json_bytes(result))}})
        summary = {'schema': 'all-fit-worker-smoke-summary-v1', 'status': 'PASS', 'worker': worker,
                   'inventory': inv, 'expected_fit_ids': owned[worker], 'results': results,
                   'counts': {'expected': COUNTS[worker], 'passed': COUNTS[worker]}}
        summary_pin = save(folder / ('summary-' + worker.replace('/', '_') + '.json'), summary)
        rows.append({'worker': worker, 'status': 'PASS', 'plan': {'path': '/synthetic/plan', 'bytes': 1, 'sha256': digest(worker)},
                     'inventory': inv, 'summary': summary_pin})
        values[(worker, 'summary')] = Path(summary_pin['path']).read_bytes()
        values[(worker, 'inventory')] = Path(inv['path']).read_bytes()
    return rows, values


class NoTransport:
    def call(self, *args, **kwargs):
        raise AssertionError('No real remote call is allowed in fixtures')

    def stage_plaintext(self, *args, **kwargs):
        raise AssertionError('Verification must never stage remote files')


class CoordinatorTests(unittest.TestCase):
    def test_owned_read_only_staging_temporary_is_removed_without_pty_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'backend.py'
            source.write_text('print("fixture")\n')
            obj = transport.Transport(Path(temp) / 'audit')
            target = Path(c.ROOTS['allocation']) / 'fixture/backend.py'
            commands = []
            def call(host, command, timeout=180):
                commands.append(command)
                return source.read_bytes() if command.startswith('cat -- ') else b''
            with patch.object(obj, 'call', call), patch.object(obj, 'run', return_value=b''):
                obj.stage_plaintext('allocation', source, target)
            removals = [shlex.split(line) for command in commands for line in command.splitlines() if line.startswith('rm ')]
            self.assertEqual(len(removals), 1)
            self.assertEqual(removals[0][:3], ['rm', '-f', '--'])
            self.assertTrue(removals[0][3].startswith(str(target) + '.incoming-'))
            self.assertNotEqual(removals[0][3], str(target))

    def test_default_is_verify_and_backend_default_is_probe(self):
        args = c.parser().parse_args(['--bindings-sha256', 'a' * 64, '--source-sha256', 'b' * 64])
        self.assertEqual(args.operation, 'verify')
        import remote_backend
        args = remote_backend.parser().parse_args(['--host', 'allocation', '--config', '/unused', '--config-sha256', 'a' * 64])
        self.assertEqual(args.operation, 'probe')

    def test_pending_never_fetches_validates_stages_or_releases(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = c.BINDINGS.read_bytes()
            obj = c.Coordinator(json.loads(raw), raw, Path(temp), NoTransport())
            calls = []
            def invoke(host, operation='probe', **options):
                calls.append(operation)
                self.assertEqual(operation, 'probe')
                return {'workers': [{'worker': worker, 'status': 'pending'} for worker in WORKERS if worker.split('/')[0] == host]}
            with patch.object(obj, 'invoke', invoke), patch.object(c.subprocess, 'run', side_effect=AssertionError('validator called')):
                result = obj.verify()
            self.assertEqual(result['markers_deployed'], 0)
            self.assertEqual(calls, ['probe', 'probe'])
            self.assertFalse((Path(temp) / 'CANDIDATE_GLOBAL_SMOKE_PASS.json').exists())

    def test_exact_frozen_validator_all_6460_proof_and_no_implicit_release(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            raw = c.BINDINGS.read_bytes()
            config, loaded = c.load_bindings(c.BINDINGS, identity(raw)['sha256'])
            self.assertEqual(loaded, raw)
            rows, values = coverage_fixture(folder / 'remote-fixture', config)
            obj = c.Coordinator(config, raw, folder / 'proof', NoTransport())
            calls = []
            def invoke(host, operation='probe', **options):
                calls.append(operation)
                if operation == 'probe':
                    return {'workers': [row for row in rows if row['worker'].split('/')[0] == host], 'existing_marker': None}
                self.assertEqual(operation, 'read-json')
                return values[(options['worker'], options['kind'])]
            with patch.object(obj, 'invoke', invoke):
                result = obj.verify()
            self.assertEqual(result['fit_count'], 6460)
            self.assertEqual(result['status'], 'VERIFIED')
            self.assertEqual(result['markers_deployed'], 0)
            self.assertNotIn('release', calls)
            candidate = json.loads((obj.proof / 'CANDIDATE_GLOBAL_SMOKE_PASS.json').read_text())
            self.assertEqual(candidate['TRAIN_updates_completed_total'], 12920)
            self.assertEqual(len(candidate['coverage']), 6460)
            self.assertFalse((obj.proof / 'GLOBAL_SMOKE_PASS.json').exists())
            # A changed summary cannot reuse a previously validated proof.
            changed = copy.deepcopy(rows)
            changed[0]['summary']['sha256'] = '0' * 64
            with patch.object(obj, 'probe', return_value={host: {'workers': [r for r in changed if r['worker'].split('/')[0] == host]}
                                                        for host in c.ROOTS}):
                with self.assertRaisesRegex(ValueError, 'Existing evidence differs'):
                    obj.verify()
            # The frozen validator must reject an internally rehashed one-update receipt.
            bundle = json.loads((obj.proof / 'BUNDLE.json').read_text())
            summary_path = Path(bundle['workers'][0]['summary']['path'])
            summary = json.loads(summary_path.read_text())
            entry = summary['results'][0]
            entry['result']['TRAIN_updates'] = 1
            entry['result_receipt'].update(identity(json_bytes(entry['result'])))
            bad_summary = save(folder / 'invalid-summary.json', summary)
            bundle['workers'][0]['summary'] = bad_summary
            invalid_bundle = folder / 'invalid-bundle.json'
            save(invalid_bundle, bundle)
            child = c.subprocess.run([sys.executable, '-B', str(c.E / PACKAGES['smoke'] / 'validate.py'),
                                      '--bundle', str(invalid_bundle), '--output', str(folder / 'invalid-marker.json')],
                                     capture_output=True, text=True)
            self.assertNotEqual(child.returncode, 0)
            self.assertIn('Exactly two completed optimizer transitions required', child.stderr)
            self.assertFalse((folder / 'invalid-marker.json').exists())

    def test_release_requires_exact_confirmation_before_any_remote_call(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = c.BINDINGS.read_bytes()
            obj = c.Coordinator(json.loads(raw), raw, Path(temp), NoTransport())
            for confirmation in (None, '0' * 64):
                with self.assertRaises(ValueError):
                    obj.release(confirmation)

    def test_existing_evidence_and_mixed_inventory_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'evidence.json'
            exact_write(path, b'{"status":"first"}\n')
            with self.assertRaises(ValueError):
                exact_write(path, b'{"status":"second"}\n')
        probes = {host: {'workers': [{'worker': worker, 'status': 'PASS', 'inventory': {'sha256': worker}}
                                     for worker in WORKERS if worker.split('/')[0] == host]} for host in c.ROOTS}
        with self.assertRaisesRegex(ValueError, 'different inventories'):
            c.ready_workers(probes)

    def test_remote_release_uses_execution7_and_preserves_canonical_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            raw = c.BINDINGS.read_bytes()
            config = json.loads(raw)
            backend = Backend('allocation', config, root, identity(raw))
            queue = backend.queue
            queue.mkdir(parents=True)
            # Ordinary import of this reviewed, unchanged metadata checker only.
            (queue / 'common.py').write_bytes((c.E / PACKAGES['execution'] / 'common.py').read_bytes())
            ids = ['fixture-%04d' % i for i in range(6460)]
            inventory = save(root / 'inventory.json', {'canonical_fit_ids': ids})
            smoke = save(root / 'smoke-source.json', {'fixture': True})
            plan = {'worker': 'allocation/gpu0', 'inventory': inventory, 'project_root': str(root),
                    'canonical_fit_ids_sha256': digest(ids), 'smoke_source_manifest': smoke}
            plan_pin = save(root / 'plan.json', plan)
            summary = save(root / 'summary.json', {'fixture': True})
            rows = [{'worker': 'allocation/gpu0', 'status': 'PASS', 'inventory': inventory,
                     'plan': plan_pin, 'summary': summary}]
            marker_value = {'schema': 'all-fit-global-smoke-pass-v1', 'status': 'PASS', 'fit_count': 6460,
                            'canonical_fit_ids': ids, 'canonical_fit_ids_sha256': digest(ids),
                            'coverage': [{'id': fit_id} for fit_id in ids], 'TEST_accessed': False,
                            'class_substitution': False, 'source_manifest': smoke,
                            'inventory_sha256s': dict.fromkeys(WORKERS, inventory['sha256']),
                            'worker_summary_sha256s': dict.fromkeys(WORKERS, summary['sha256'])}
            candidate = save(root / 'CANDIDATE_GLOBAL_SMOKE_PASS.json', marker_value)
            request = save(root / 'request.json', {'schema': 'all-fit-readable-release-request-v1',
                           'bindings': identity(raw), 'ready_workers': rows, 'marker': candidate})
            marker = queue / 'GLOBAL_SMOKE_PASS.json'
            with patch.object(backend, 'probe', return_value=rows):
                with self.assertRaises(ValueError):
                    backend.release(request['path'], request['sha256'], '0' * 64)
                self.assertFalse(marker.exists())
                previous = sys.modules.pop('common', None)
                try:
                    result = backend.release(request['path'], request['sha256'], candidate['sha256'])
                    self.assertFalse(result['existing_identical'])
                    self.assertEqual(marker.read_bytes(), Path(candidate['path']).read_bytes())
                    self.assertTrue(backend.release(request['path'], request['sha256'], candidate['sha256'])['existing_identical'])
                finally:
                    sys.modules.pop('common', None)
                    if previous is not None:
                        sys.modules['common'] = previous
            with patch.object(backend, 'probe', return_value=[{**rows[0], 'status': 'pending'}]):
                with self.assertRaisesRegex(ValueError, 'changed before release'):
                    backend.release(request['path'], request['sha256'], candidate['sha256'])
            self.assertEqual(marker.read_bytes(), Path(candidate['path']).read_bytes())

    def test_commands_use_literal_authorized_routes_and_readable_source(self):
        a = transport.remote_argv('allocation', 'python3 -B /approved/source.py --operation probe\n')
        self.assertIn('-tt', a)
        self.assertIn(transport.LOGIN, a)
        self.assertIn('anogena-2-0', a[-1])
        b = transport.remote_argv('server51', 'python3 -B /approved/source.py --operation probe\n')
        self.assertEqual(b[:len(transport.MACLINK)], transport.MACLINK)
        self.assertIn('shmelev@192.168.18.51', b)
        self.assertIn('GPU-30283973-140d-9e7b-e780-0fcc487dec54', b[-1])
        self.assertNotIn('shell', b)
        for name in ('coordinator.py', 'remote_backend.py', 'readable_transport.py'):
            tree = ast.parse((c.HERE / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {'exec', 'eval', 'compile'})
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
                    self.assertFalse({'base64', 'zlib', 'gzip', 'marshal'} & set(modules))


if __name__ == '__main__':
    unittest.main()
