"""Read-only, host-local check of queue custody, physical masks and progress."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'allocation_six_gpu_20261031_v1'
sys.path.insert(0, str(PACKAGE))
from host_extension import bind, config, pin, ROOT
from partition import original_plan, validate_partition, live, source, stop_pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    identity = bind(args.worker)
    plan = original_plan()
    partition = validate_partition(json.loads((PACKAGE / 'PARTITION.json').read_text()), plan)
    assert source() == partition['source'] and stop_pins() == partition['permanent_stops']
    canonical = Path(plan['state_root']) / 'full'
    workers = {}; combined = Counter(); duplicate = []
    for worker, ids in partition['assignment'].items():
        states = {}; counts = Counter()
        for task_id in ids:
            folder = canonical / hashlib.sha256(task_id.encode()).hexdigest()
            value = json.loads((folder / 'STATE.json').read_text()) if (folder / 'STATE.json').exists() else {'status': 'pending'}
            counts[value['status']] += 1
            states[task_id] = value
        combined.update(counts)
        if config(worker)['expected_hostname'] != socket.gethostname():
            workers[worker] = {'counts': dict(counts), 'processes_not_interpreted_on_foreign_host': True}
            continue
        root = PACKAGE / 'runs' / worker.replace('/', '_')
        status = json.loads((root / 'STATUS.json').read_text())
        owned = []
        for task_id, state in states.items():
            if state['status'] != 'running':
                continue
            directory = Path(state['directory'])
            assert directory.is_relative_to(ROOT)
            request = json.loads((directory / 'INVOCATION.json').read_text())
            child = json.loads((directory / 'CHILD.json').read_text())
            assert request['physical_shard'] == worker
            assert request['environment']['CUDA_VISIBLE_DEVICES'] == config(worker)['physical_gpu_uuid']
            alive = live(child['identity'])
            record = {'task_id': task_id, 'attempt_directory': str(directory), 'identity': child['identity'],
                      'alive': alive, 'physical_mask': request['environment']['CUDA_VISIBLE_DEVICES']}
            if alive:
                process = Path('/proc') / str(child['identity']['pid'])
                environment = process.joinpath('environ').read_bytes().split(b'\0')
                mask = [value.split(b'=', 1)[1].decode() for value in environment if value.startswith(b'CUDA_VISIBLE_DEVICES=')]
                assert mask == [config(worker)['physical_gpu_uuid']]
                record['verified_live_process_mask'] = mask[0]
                record['process_title'] = process.joinpath('cmdline').read_bytes().split(b'\0')[0].decode()
            events = Path(request['output']) / 'EVENTS.jsonl'
            if events.exists():
                lines = events.read_text().splitlines()
                records = [json.loads(line) for line in lines if line.strip()]
                record['event_count'] = len(records)
                record['last_event'] = records[-1] if records else None
                record['events_age_seconds'] = time.time() - events.stat().st_mtime
            attempts = directory.parent
            record['attempt_count'] = len([path for path in attempts.iterdir() if path.is_dir()])
            if record['attempt_count'] != 1:
                duplicate.append(task_id)
            owned.append(record)
        workers[worker] = {'counts': dict(counts), 'phase': status['phase'],
                           'owner_alive': live(status['owner']), 'status_age_seconds': time.time() - status['updated'],
                           'active_children': owned}
    allowed = '1,2,3,4,6' if socket.gethostname() == 'anogena-0' else '0'
    gpu = subprocess.check_output(['nvidia-smi', '-i', allowed, '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
    result = {'observed': time.time(), 'hostname': socket.gethostname(), 'source': source(),
              'partition': pin(PACKAGE / 'PARTITION.json'), 'partition_summary': partition['summary'],
              'counts': dict(combined), 'workers': workers, 'authorized_gpu_readings': gpu.splitlines(),
              'new_running_duplicate_attempts': duplicate, 'server51_touched': False}
    destination = HERE / ('LIVE-' + socket.gethostname() + '-' + str(time.time_ns()) + '.json')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'receipt': pin(destination), 'counts': dict(combined), 'hostname': socket.gethostname(),
                      'workers': {worker: {key: value for key, value in row.items() if key != 'active_children'}
                                  for worker, row in workers.items()}, 'authorized_gpu_readings': gpu.splitlines()}))


if __name__ == '__main__':
    main()
