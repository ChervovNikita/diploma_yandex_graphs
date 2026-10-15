"""New shared-host race/claim/STOP checks only; no model, tensor, GPU or network I/O."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
from custody import digest, read, write
from engine import Engine, pin
from host_ledger import Ledger, admissible
from stop_transport import issue, deliver
from backend import memory_available

HERE = Path(__file__).resolve().parent
CHECKS = []


def check(name, condition):
    assert condition, name
    CHECKS.append(name)


def reject(callback, text):
    try:
        callback()
    except (ValueError, KeyError) as exc:
        assert text in str(exc), str(exc)
        return True
    raise AssertionError('Expected rejection: ' + text)


def fixture(root, count=4):
    rows = [{'id': f'metadata-reservation-{i}', 'worker': f'server77/gpu{i % 2}',
             'canonical_runtime_job_hash': digest(i), 'science_contract_hash': digest(['metadata-only', i]),
             'dependencies': [], 'output_relative': f'metadata/{i}', 'cohort': f'fixture-{i}'} for i in range(count)]
    return {'schema': 'metadata-only-ledger-test-owner', 'stop_issuer': 'metadata-test-operator',
            'assignments': rows, 'hosts': {'server77': {'project_root': str(root)}}}


def resources():
    return {'cpu_slots': 2, 'thread_count': 2, 'vram_bytes': 10, 'ram_bytes': 30, 'disk_bytes': 20,
            'profile': 'metadata-compatible', 'qualified_peer_profiles': ['metadata-compatible'],
            'max_concurrent_owned_children': 2}


def budget(cpu=8):
    return {'host_cpu_slots': cpu, 'gpu_headroom_bytes': 0, 'host_ram_headroom_bytes': 0, 'disk_headroom_bytes': 0}


def snapshot(active):
    return {'free_vram_by_uuid': {'metadata-gpu0': 1000, 'metadata-gpu1': 1000},
            'available_ram_bytes': 1000, 'free_disk_bytes': 1000, 'allocated_vram_by_pid': {},
            'rss_by_pid': {}, 'untracked_own_gpu_pids': []}


def reserve(ledger, row, root, *, limits=None, observed=snapshot):
    return ledger.reserve(row, row['canonical_runtime_job_hash'], root / row['output_relative'],
        'metadata-gpu' + row['worker'][-1], resources(), limits or budget(), observed,
        lambda identity: 'same', {'pid': 123, 'fixture': True})


def contender(path, owner, row, gate, result, same_gpu, cpu):
    root = Path(path)
    ledger = Ledger(root / 'ledger', owner, 'server77')
    ledger.establish()
    gate.wait(timeout=15)
    ok, reason = reserve(ledger, row, root, limits=budget(cpu))
    if ok:
        ledger.commit({'pid': 1000 + int(row['id'].rsplit('-', 1)[1]), 'fixture': True})
    result.put((ok, reason))


def crashed_startup(path, owner):
    root = Path(path)
    ledger = Ledger(root / 'ledger', owner, 'server77')
    ledger.establish()
    reserve(ledger, owner['assignments'][0], root)
    # This process does no child spawn; exit intentionally leaves durable startup.


def race(root, same_gpu, cpu):
    owner = fixture(root)
    if same_gpu:
        for row in owner['assignments']:
            row['worker'] = 'server77/gpu0'
    ctx = multiprocessing.get_context('spawn')
    gate, results = ctx.Barrier(4), ctx.Queue()
    processes = [ctx.Process(target=contender, args=(str(root), owner, row, gate, results, same_gpu, cpu)) for row in owner['assignments']]
    for process in processes:
        process.start()
    answers = [results.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    state = Ledger(root / 'ledger', owner, 'server77').load()
    return answers, state


def functions(path):
    tree = ast.parse(path.read_text())
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main():
    with tempfile.TemporaryDirectory(prefix='portable-ledger-metadata-') as folder:
        root = Path(folder).resolve()
        answers, state = race(root / 'cpu-race', False, 2)
        check('real_multiprocess_cross_GPU_CPU_race_admits_one', sum(ok for ok, _ in answers) == 1 and len(state['active']) == 1)
        answers, state = race(root / 'gpu-race', True, 16)
        check('real_multiprocess_same_GPU_race_never_exceeds_two', sum(ok for ok, _ in answers) == 2 and len(state['active']) == 2)
        active = [{'gpu_uuid': 'metadata-gpu0', 'resources': resources(), 'child_identity': {'pid': 1}}]
        p = snapshot(active)
        for field, limit, reason in (('available_ram_bytes', 59, 'host_shared_RAM_reservation'), ('free_disk_bytes', 39, 'host_shared_disk_reservation')):
            observed = dict(p, **{field: limit})
            check(reason + '_across_GPUs', admissible(resources(), 'metadata-gpu1', active, observed, budget()) == (False, reason))
        observed = dict(p, available_ram_bytes=40, rss_by_pid={1: 25})
        check('observed_RSS_subtracted_once_from_remaining_reservation', admissible(resources(), 'metadata-gpu1', active, observed, budget())[0])
        observed = dict(p, untracked_own_gpu_pids=[999])
        check('unknown_same_UID_GPU_process_blocks_without_signaling', admissible(resources(), 'metadata-gpu1', active, observed, budget())[1] == 'untracked_owned_GPU_process')
        groups = [{'path': 'metadata-cgroup', 'maximum': '100', 'current': 90, 'stats': {'inactive_file': 30, 'shmem': 20}}]
        check('finite_cgroup_headroom_constrains_host_RAM_without_counting_shmem_as_reclaimable',
              memory_available(1000, groups)['available_ram_bytes'] == 20
              and memory_available(15, groups)['available_ram_bytes'] == 15
              and memory_available(1000, [dict(groups[0], maximum='max')])['available_ram_bytes'] == 1000)
        crash_root = root / 'crashed-startup'
        owner = fixture(crash_root)
        ctx = multiprocessing.get_context('spawn')
        child = ctx.Process(target=crashed_startup, args=(str(crash_root), owner))
        child.start(); child.join(timeout=20)
        check('startup_reservation_survives_process_exit', child.exitcode == 0)
        ledger = Ledger(crash_root / 'ledger', owner, 'server77')
        ledger.establish()
        check('startup_cannot_be_stolen_after_custodian_exit', reject(lambda: reserve(ledger, owner['assignments'][1], crash_root), 'Uncertain startup'))
        changed = copy.deepcopy(owner)
        changed['assignments'][0]['worker'] = 'server77/gpu1'
        check('immutable_host_owner_rejects_reassignment', reject(lambda: Ledger(crash_root / 'ledger', changed, 'server77').establish(), 'Permanent authority conflict'))
        # Separate no-spawn transaction permits safe cancellation; its claim remains permanent.
        clean_root = root / 'clean'
        owner = fixture(clean_root)
        ledger = Ledger(clean_root / 'ledger', owner, 'server77'); ledger.establish()
        row = owner['assignments'][0]
        ok, token = reserve(ledger, row, clean_root)
        ledger.cancel_unspawned('metadata test; no child created')
        check('definite_no_spawn_preserves_claim_and_archive', ok and not ledger.load()['active'] and (ledger.root / 'archive' / (token + '.json')).exists())
        check('permanent_claim_rejects_new_output_namespace', reject(lambda: reserve(ledger, row, clean_root / 'another'), 'Permanent authority conflict'))
        ok, token = reserve(ledger, row, clean_root)
        ledger.commit({'pid': 777, 'fixture': True})
        check('duplicate_claim_while_active_rejected', reject(lambda: reserve(ledger, row, clean_root), 'Task already reserved'))
        terminal = clean_root / 'METADATA_REAP_RECEIPT.json'
        write(terminal, {'metadata_only': True, 'no_scientific_child': True})
        ledger.reaped(token, pin(terminal))
        check('only_actual_custodian_release_path_archives_terminal_custody', not ledger.load()['active'] and read(ledger.root / 'archive' / (token + '.json'))['events'][-1]['terminal'] == pin(terminal))
        # STOP command creation/delivery are ordinary local metadata files.
        authority = clean_root / 'STOP_TEST_AUTHORITY.json'
        write(authority, {'metadata_only': True})
        first = issue(clean_root / 'outbox', owner, owner['stop_issuer'], ['server77'], 'metadata test STOP', pin(authority))
        ack = deliver(first['path'], first['sha256'], ledger.root, owner, 'server77')
        check('STOP_ACK_means_received_not_drained', ack['ack']['received_and_durable'] and ack['ack']['all_children_drained'] == 'not_claimed')
        check('identical_STOP_redelivery_idempotent', deliver(first['path'], first['sha256'], ledger.root, owner, 'server77') == ack)
        check('delivered_STOP_blocks_next_admission', reject(lambda: reserve(ledger, owner['assignments'][1], clean_root), 'Host STOP received'))
        conflict = read(first['path']); conflict['reason'] = 'conflicting replay'
        conflict_path = clean_root / 'CONFLICT.json'; write(conflict_path, conflict)
        check('conflicting_same_sequence_STOP_rejected', reject(lambda: deliver(conflict_path, pin(conflict_path)['sha256'], ledger.root, owner, 'server77'), 'Conflicting STOP'))
        second = issue(clean_root / 'outbox', owner, owner['stop_issuer'], ['server77'], 'second metadata STOP', pin(authority))
        deliver(second['path'], second['sha256'], ledger.root, owner, 'server77')
        check('STOP_sequence_monotonically_increases', read(second['path'])['sequence'] == read(first['path'])['sequence'] + 1)
        check('wrong_STOP_pin_rejected', reject(lambda: deliver(second['path'], '0' * 64, ledger.root, owner, 'server77'), 'Pinned input bytes'))
        envelope = read(ledger.root / 'STATE.json'); envelope['state']['generation'] += 1
        write(ledger.root / 'STATE.json', envelope, replace=True)
        check('ledger_corruption_fails_closed', reject(ledger.load, 'Host ledger corruption'))
        # Pure engine policy checks: no Engine.run, child, native result, scoring or tensor fixture.
        engine = object.__new__(Engine)
        engine.states = {str(i): {'status': 'complete'} for i in range(4)}
        engine.states['2']['status'] = 'failed'
        task = {'dependencies': [str(i) for i in range(4)]}
        check('ordinary_failed_fit_blocks_I4_instead_of_waiting', engine.dependency(task) == 'failed')
        engine.states['2']['status'] = 'no_selected_state'
        check('no_selected_I4_policy_preserved', engine.dependency(task) == 'no_selected_state')
    # Narrow source inheritance checks reuse prior engine verification, no broad fake-worker suite.
    old = functions(HERE.parent / 'expanded_dispatcher_v2/engine.py')
    new = functions(HERE / 'engine.py')
    inherited = ('output', 'folder', 'verify', 'save_state', 'immutable_claim', 'recover_terminal', 'resume_for', 'validate_terminal', 'launch')
    check('unchanged_durable_engine_methods_reuse_existing38check_evidence', all(old[name] == new[name] for name in inherited))
    source = (HERE / 'engine.py').read_text()
    check('ordinary_failed_fit_does_not_emit_child_failed_STOP', "reason':'child_failed'" not in source and 'retain_failure_without_retry' in source)
    check('resource_block_waits_for_other_GPU_custodian', 'not self.backend.peer_activity()' in source)
    check('sole_production_spawn_is_reservation_owner', sum(p.read_text().count('subprocess.Popen(') for p in (HERE / 'backend.py', HERE / 'run_gpu.py', HERE / 'engine.py')) == 1)
    for worker in ('server77/gpu0', 'server77/gpu1'):
        result = subprocess.run([sys.executable, '-B', str(HERE / 'run_gpu.py'), '--worker', worker, '--run-spec', '/not-read.json', '--run-spec-sha256', '0' * 64], text=True, capture_output=True)
        check(worker + '_disabled_before_project_or_hardware_access', result.returncode == 2 and 'Host disabled before probes' in result.stdout)
    for path in HERE.rglob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    check('all_successor_Python_sources_parse_without_importing_science', True)
    result = {'schema': 'portable-dispatcher-v2-local-metadata-checks', 'status': 'PASS', 'checks': CHECKS,
              'count': len(CHECKS), 'scientific_jobs_started': 0, 'tensor_reads': 0, 'hardware_or_network_probes': 0,
              'server77_accessed': False, 'native_runtime_or_resource_qualification_claim': False,
              'predecessor38checks_reused_not_rerun': True}
    (HERE / 'CHECKS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
