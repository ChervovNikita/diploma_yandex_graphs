"""Independent source/custody audit and fabricated controller cancellation checks."""
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PACKET = PHASE / 'graph_heterogeneous_dblp_parallel_cpu_preparation_20261003_v1'
EXPECTED_MANIFEST = '4ccd3b3469893e078a545eb8797daea197047df4303d84f63fc9be2dbd3e5551'
EXPECTED_SOURCE = '923d107b03b8411466a691649e9585a84c2998b890d9c23aa279a5e6e531671f'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(row, path):
    data = path.read_bytes()
    assert digest(data) == row['sha256'], str(path)
    if 'bytes' in row:
        assert len(data) == row['bytes'], str(path)
    return dict(path=str(path), sha256=digest(data), bytes=len(data), status='PASS')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    manifest_bytes = (PACKET / 'MANIFEST.json').read_bytes()
    assert digest(manifest_bytes) == EXPECTED_MANIFEST
    manifest = json.loads(manifest_bytes)
    seal = json.loads((PACKET / 'SEAL.json').read_text())
    assert seal['manifest_sha256'] == EXPECTED_MANIFEST
    payloads = [checked(row, PACKET / row['path']) for row in manifest['payload']]
    assert len(payloads) == 11
    assert digest((PACKET / 'parallel_cpu.py').read_bytes()) == EXPECTED_SOURCE
    binding = json.loads((PACKET / 'BINDINGS.json').read_text())
    def local(row):
        suffix = row['path'].split('/postsubmission_20260930/', 1)[1]
        return PHASE / suffix
    sources = [checked(row, local(row)) for row in binding['source_records']]
    assert len(sources) == 17
    freeze = checked(binding['freeze'], local(binding['freeze']))
    frozen = json.loads(local(binding['freeze']).read_text())
    assert frozen['seeds'] == [131,137,139,149,151]
    assert frozen['arms'] == ['native_HGT','global_BE','shared_relation','CP','unrestricted','untied_HGT','wider_BE']
    wrapper_tree = ast.parse(local(binding['resource_transport']['wrapper']).read_text())
    remote = next(ast.literal_eval(node.value) for node in wrapper_tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REMOTE' for t in node.targets))
    assert remote.encode() == local(binding['resource_transport']['remote_code']).read_bytes()
    transport = json.loads(local(binding['resource_transport']['receipt']).read_text())
    actual = transport['remote_receipt']
    assert transport['exit_code'] == 0 and actual['exit_code'] == 0 and actual['status'] == 'completed'
    assert transport['helper_sha256'] == binding['resource_transport']['wrapper']['sha256']
    assert actual['preflight']['memory_limit_bytes'] == 8*2**30
    qualified = actual['qualification_result']
    assert qualified['status'] == 'qualified' and qualified['originals_preserved'] is True
    assert [row['arm'] for row in qualified['rows']] == frozen['arms']
    assert all(row['status'] == 'qualified' for row in qualified['rows'])
    reconstructed = (json.dumps(qualified, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    assert digest(reconstructed) == binding['resource_qualification']['sha256']
    assert len(reconstructed) == binding['resource_qualification']['bytes']
    resource = dict(status='PASS', reconstruction='canonical JSON bytes from bound transport receipt',
                    sha256=digest(reconstructed), bytes=len(reconstructed),
                    preimport_address_space_limit_bytes=actual['preflight']['memory_limit_bytes'])
    scheduler = module('independent_final_scheduler_review', PACKET / 'parallel_cpu.py')
    class ClosedDriver:
        def paired_development(self, *args):
            raise AssertionError('An interrupted study must not be scored')
    old_handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        root = Path(directory)
        packet = root / 'synthetic_packet'
        packet.mkdir()
        original = root / 'synthetic_original'
        original.mkdir()
        synthetic_binding = dict(source_records=[], freeze={}, resource_qualification={}, canonical_phase=str(original))
        synthetic_frozen = dict(archive={}, development_labels={}, splits=[])
        release = root / 'release.json'
        release.write_text(json.dumps(dict(required_free_host_bytes=40*2**30,worker_wall_budget_seconds=1)))
        def interrupt(commands, out, *args):
            seed_out = out / 'seed131'
            seed_out.mkdir()
            (seed_out / 'ARM_STARTED.jsonl').write_text(json.dumps(dict(seed=131,arm='native_HGT'))+'\n')
            (out / 'worker131.exit.json').write_text(json.dumps(dict(exit_code=-15,status='controller_aborted'))+'\n')
            raise KeyboardInterrupt('fabricated independent cancellation after arm start')
        with patch.object(scheduler,'PACKET',packet), \
             patch.object(scheduler,'packet_guard',return_value=(synthetic_binding,synthetic_frozen,ClosedDriver(),'synthetic-manifest')), \
             patch.object(scheduler,'admission_guard',return_value=dict(scheduler_manifest_sha256='synthetic-manifest')), \
             patch.object(scheduler,'verify'), patch.object(scheduler,'scheduler_preservation'), \
             patch.object(scheduler,'resource_screen',return_value=dict(status='passed')), \
             patch.object(scheduler,'run_workers',side_effect=interrupt):
            code = scheduler.main(['--admission',str(release),'--run-name','fabricated_cancellation'])
        out = packet / 'runs/fabricated_cancellation'
        parallel = json.loads((out/'PARALLEL_STUDY.json').read_text())
        canonical = json.loads((out/'STUDY.json').read_text())
        assert code == 1 and parallel['status'] == 'controller_failed'
        assert parallel['error_type'] == 'KeyboardInterrupt'
        assert len(canonical['rows']) == 35 and canonical['rows'] == parallel['rows']
        assert [(r['seed'],r['arm']) for r in canonical['rows']] == [(s,a) for s in scheduler.SEEDS for a in scheduler.ARMS]
        assert canonical['summary'] == dict(status='incomplete',all_frozen_terminals=True,successful_subset_scored=False)
        assert canonical['rows'][0]['attempted'] is True
        assert all(r['attempted'] is False for r in canonical['rows'][1:])
        assert parallel['closure'] is None
        assert parallel['worker_exits']['131']['status'] == 'controller_aborted'
        assert canonical['original_inputs_verified_unchanged'] is False
        interruption = dict(status='PASS',source_sha256=EXPECTED_SOURCE,KeyboardInterrupt_escaped=False,
                            return_code=code,canonical_terminal_count=35,ordered_frozen_slots=True,
                            attempted_arm_count=1,worker_exit_receipt_recovered=True,
                            summary=canonical['summary'],handler_restoration=True)
    assert all(signal.getsignal(number) == handler for number,handler in old_handlers.items())
    # Re-run only the final author's real-signal fixture, using synthetic subprocesses.
    fixtures = module('independent_selected_signal_fixture', PACKET / 'scheduler_fixtures.py')
    suite = unittest.TestSuite([fixtures.SchedulerChecks('test_actual_SIGTERM_reaps_owned_child_and_keeps_exit_receipt')])
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    assert result.wasSuccessful(), log.getvalue()
    assert 'torch' not in sys.modules
    receipt = dict(schema='independent_DBLP_parallel_CPU_source_audit_v1',status='PASS',
                   audited_manifest_sha256=EXPECTED_MANIFEST,audited_seal_sha256=digest((PACKET/'SEAL.json').read_bytes()),
                   audited_scheduler_sha256=EXPECTED_SOURCE,payloads=payloads,source_records=sources,freeze=freeze,
                   resource_transport=resource,independent_keyboard_interrupt=interruption,
                   selected_author_SIGTERM_fixture=dict(status='PASS',tests_run=result.testsRun,owned_child_reaped=True),
                   Torch_imported=False,real_labels_models_or_checkpoints_used=False,remote_or_GPU_execution=False)
    (HERE/'FINAL_SOURCE_CHECKS.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    (HERE/'SELECTED_SIGNAL_FIXTURE.txt').write_text(log.getvalue())
    print(json.dumps(dict(status='PASS',payloads=len(payloads),source_records=len(sources),
                         canonical35_interrupt_recovery=True,SIGTERM_fixture_passed=True,Torch_imported=False)))


if __name__ == '__main__':
    main()
