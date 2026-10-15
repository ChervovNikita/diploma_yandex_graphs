"""Repo-local JSON/filesystem fault point; never calls a launcher or model."""
import copy
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from final_scoring_integration_v1.common import digest, pin
from final_scoring_integration_v1.offline_seal import role_semantics, snapshot
from portable_dispatcher_v3.custody import write
from combined_campaign_orchestration_v2.local_ready import ensure_local_ready


def run_faultpoint_check():
    with tempfile.TemporaryDirectory(prefix='metadata_faultpoint_', dir=HERE) as temporary:
        root = Path(temporary)
        def save(path, body):
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path, body)
            return pin(path, root)
        task_id, job_hash = 'metadata_fixture/fit0', 'a'*64
        train = root / 'training'
        raw_path = train / 'fit0/RESULT.json'
        # Selected tensor receipts are deliberately metadata-only/nonexistent.
        # This check has no checkpoint, logits, labels or query payload.
        selected = {'path': str(raw_path.parent / 'selected.pt'), 'bytes': 1, 'sha256': 'b'*64}
        valid = {'path': str(raw_path.parent / 'selected_VALID.pt'), 'bytes': 1, 'sha256': 'c'*64}
        raw = {'status': 'complete', 'job_id': task_id, 'job_hash': job_hash, 'TEST_accessed': False,
               'selected_checkpoint': selected, 'selected_epoch': 2, 'selected_VALID_export': valid, 'runtime_binding': {}}
        raw_pin = save(raw_path, raw)
        control = train / 'control/tasks' / digest(task_id)
        terminal_pin = save(control / 'attempts/0001/TERMINAL.json', {'id': task_id, 'status': 'complete', 'live_native_result_receipt': raw_pin})
        state_pin = save(control / 'STATE.json', {'status': 'complete', 'attempt_terminal': terminal_pin})
        release = {'host': 'allocation', 'output_root': str(train), 'tasks': [
            {'id': task_id, 'kind': 'fit', 'runtime_job_hash': job_hash, 'output_relative': 'fit0'}]}
        release_pin = save(root / 'TRAIN_RELEASE.json', release)
        role = {'schema': 'metadata-faultpoint-role-fixture', 'family': 'node', 'role': 'TEST', 'preset_id': 'fixture',
                'populations': {'0': {'records': 1, 'ids_sha256': 'd'*64, 'literal_split_seed': 0}}}
        role_pin = save(root / 'ROLE.json', role)
        roster_pin = save(root / 'ROSTER.json', {'tasks': [{'preset_id': 'fixture', 'role_contract': role_pin}]})
        outcome = {'status': 'complete', 'state': state_pin, 'terminal': terminal_pin, 'training_result': raw_pin,
            'selected_state': {'checkpoint': selected, 'epoch': 2, 'VALID_export': valid, 'runtime_binding': {}},
            'scientific_settings_reselected': False}
        shard = {'schema': 'offline-terminal-host-shard-v1', 'host': 'allocation', 'training_ownership_sha256': 'e'*64,
            'training_release': snapshot(release_pin, root), 'scoring_roster': snapshot(roster_pin, root),
            'role_semantics': {'fixture': role_semantics(role)}, 'outcomes': {task_id: outcome},
            'metadata_by_sha256': {value['sha256']: snapshot(value, root) for value in (state_pin, terminal_pin, raw_pin)},
            'all_local_tasks_terminal': True, 'TEST_accessed': False, 'retry_or_arm_changes_after_seal': False}
        expected = {'host': 'allocation', 'training_release': release_pin, 'scoring_roster': roster_pin,
            'state_pins': {task_id: state_pin}, 'training_ownership_sha256': 'e'*64}
        shard_path, ready_path = root / 'HOST_TERMINAL_SHARD.json', root / 'LOCAL_READY.json'
        # Exact requested crash point: shard durably committed, readiness absent.
        save(shard_path, shard)
        original_shard_pin = pin(shard_path, root)
        assert not ready_path.exists()
        calls = []
        def forbidden_builder():
            calls.append('rebuild')
            raise AssertionError('A committed shard must never be rebuilt on this restart')
        ack = {'allocation/gpu0': save(root / 'ACK.json', {'fixture': True})}
        contract_pin = save(root / 'CONTRACT.json', {'fixture': True})
        repaired = ensure_local_ready(shard_path, ready_path, expected, ack, contract_pin, root, forbidden_builder)
        assert calls == [] and repaired['existing_shard_reused'] is True
        assert repaired['host_shard'] == original_shard_pin and ready_path.is_file()
        original_ready_pin = pin(ready_path, root)
        again = ensure_local_ready(shard_path, ready_path, expected, ack, contract_pin, root, forbidden_builder)
        assert calls == [] and again['local_ready'] == original_ready_pin and pin(shard_path, root) == original_shard_pin
        changed = copy.deepcopy(shard)
        changed['outcomes'][task_id]['status'] = 'failed'
        write(shard_path, changed, replace=True)
        try:
            ensure_local_ready(shard_path, ready_path, expected, ack, contract_pin, root, forbidden_builder)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed existing shard must be refused')
        assert calls == [] and pin(ready_path, root) == original_ready_pin
        assert not Path(selected['path']).exists() and not Path(valid['path']).exists()
        return {'status': 'PASS', 'fault_point': 'shard committed before LOCAL_READY write',
                'builder_calls_on_recovery': 0, 'readiness_recreated': True, 'shard_bytes_unchanged_on_recovery': True,
                'second_recovery_idempotent': True, 'changed_shard_refused': True,
                'tensor_files_created_or_read': 0, 'native_work': 0}


if __name__ == '__main__':
    import json
    print(json.dumps(run_faultpoint_check(), indent=2))
