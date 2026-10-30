"""Preserve host-local checks and current canonical training progress."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'allocation_six_gpu_20261031_v1'
sys.path.insert(0, str(PACKAGE))
from host_extension import bind, pin
from partition import original_plan, validate_partition, source


def main():
    bind('allocation/gpu0')
    plan = original_plan()
    partition = validate_partition(json.loads((PACKAGE / 'PARTITION.json').read_text()), plan)
    snapshots = [HERE / 'LIVE-anogena-2-0-1793401984620793873.json', HERE / 'LIVE-anogena-0-1793401984221854610.json']
    checked = []
    for path in snapshots:
        value = json.loads(path.read_text())
        assert value['source'] == source() and value['partition'] == pin(PACKAGE / 'PARTITION.json')
        assert time.time() - value['observed'] < 1800 and not value['new_running_duplicate_attempts']
        for worker, row in value['workers'].items():
            if 'active_children' not in row:
                continue
            assert row['owner_alive'] and row['status_age_seconds'] < 30
            for child in row['active_children']:
                assert child['alive'] and child['process_title'] == 'AI Auto Research Pipeline'
                assert len(child['actual_cuda_devices']) == 1
                assert child['actual_cuda_devices'][0]['gpu_uuid'] == child['physical_mask']
                checked.append({'worker': worker, 'task_id': child['task_id'], 'pid': child['identity']['pid'],
                                'physical_gpu_uuid': child['physical_mask'], 'snapshot': pin(path)})
    workers = {}; combined = Counter()
    canonical = Path(plan['state_root']) / 'full'
    for worker, ids in partition['assignment'].items():
        counts = Counter(); progress = []; new_completed = []
        for task_id in ids:
            folder = canonical / hashlib.sha256(task_id.encode()).hexdigest()
            state = json.loads((folder / 'STATE.json').read_text()) if (folder / 'STATE.json').exists() else {'status': 'pending'}
            counts[state['status']] += 1
            if state['status'] == 'running':
                curve = Path(state['output']) / 'CURVE.jsonl'
                rows = [json.loads(line) for line in curve.read_text().splitlines() if line.strip()] if curve.exists() else []
                progress.append({'task_id': task_id, 'completed_epoch_records': len(rows),
                                 'last_completed_epoch': rows[-1] if rows else None,
                                 'curve_age_seconds': time.time() - curve.stat().st_mtime if curve.exists() else None})
            if state['status'] == 'complete' and partition['original_task_states'][task_id]['status'] == 'pending':
                result = json.loads((Path(state['output']) / 'RESULT.json').read_text())
                new_completed.append({'task_id': task_id, 'completed_epochs': result.get('completed_epochs'),
                                      'status': result['status'], 'state': pin(folder / 'STATE.json')})
        assert counts.get('failed', 0) == 0
        combined.update(counts)
        workers[worker] = {'counts': dict(counts), 'training_progress': progress, 'completed_since_expansion': new_completed}
    result = {'status': 'PASS', 'observed': time.time(), 'source': source(), 'partition': pin(PACKAGE / 'PARTITION.json'),
              'host_local_snapshots': [pin(path) for path in snapshots], 'verified_live_children_at_snapshot': checked,
              'counts': dict(combined), 'workers': workers, 'server51_touched': False,
              'scientific_sources_changed': False, 'prior_fits_interrupted': False,
              'diagnostic_correction': 'Initial read-only verifier expected CUDA mask in /proc/environ. setproctitle relocates this block. The replacement verifies actual NVIDIA PID-to-UUID mapping and the persisted invocation mask; all diagnostic failures are retained.'}
    target = PACKAGE / 'LIVE_VERIFICATION.json'
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'receipt': pin(target), 'counts': dict(combined),
                      'workers': {key: {'counts': row['counts'], 'epoch_records': [item['completed_epoch_records'] for item in row['training_progress']],
                                        'new_completed': row['completed_since_expansion']} for key,row in workers.items()}}))


if __name__ == '__main__':
    main()
