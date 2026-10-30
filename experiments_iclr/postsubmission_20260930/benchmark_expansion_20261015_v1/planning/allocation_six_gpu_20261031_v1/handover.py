"""Request graceful stop and prove admission quiescence in the owner's host."""
import argparse
import json
import os
from pathlib import Path
import time

from host_extension import bind, config, pin
from partition import HERE, PREVIOUS, PREVIOUS_SHARDS, PREVIOUS_SOURCE_SHA, c, live, previous_partition, source


def admission_boundary(owner, phase):
    """Check locally, while holding the prior owner's admission ledger lock."""
    if not live(owner):
        return 'prior_owner_exited_no_new_admission'
    c.require(phase in ('full', 'waiting_for_physical_custody'), 'Unexpected live prior owner phase')
    try:
        channel = (Path('/proc') / str(owner['pid']) / 'wchan').read_text().strip()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if channel == 'hrtimer_nanosleep' and live(owner):
        return 'prior_owner_asleep_outside_admission_ledger'
    return None


def stop_and_ack(worker, source_sha):
    bind(worker); source(source_sha); previous_partition()
    c.require(worker in PREVIOUS_SHARDS, 'Only existing allocation shard owners may be stopped')
    lease = c.lock(HERE / ('HANDOVER-' + worker.replace('/', '_') + '.lock'))
    c.require(lease is not None, 'Another handover request already owns this worker')
    try:
        return stop_and_ack_locked(worker)
    finally:
        lease.close()


def stop_and_ack_locked(worker):
    root = PREVIOUS / 'runs' / worker.replace('/', '_')
    receipt_path = HERE / ('ACK-' + worker.replace('/', '_') + '.json')
    if receipt_path.exists():
        result = c.read(receipt_path)
        c.require(result['stop'] == pin(root / 'STOP'), 'Existing handover marker changed')
        c.require(result['worker'] == worker and result['status'] == 'PASS'
                  and result['source'] == source()
                  and result['prior_source_sha256'] == PREVIOUS_SOURCE_SHA,
                  'Existing handover receipt authority changed')
        return result
    before = c.read(root / 'STATUS.json')
    owner = before['owner']
    c.require(owner['boot_id'] == c.identity()['boot_id'], 'Do not inspect another host PID namespace')
    request_path = HERE / ('STOP_REQUEST-' + worker.replace('/', '_') + '.json')
    if request_path.exists():
        request = c.read(request_path)
        c.require(request == c.read(root / 'STOP'), 'Unrelated stop marker')
        owner = request['owner']
    else:
        c.require(not (root / 'STOP').exists(), 'Pre-existing STOP requires separate inspection')
        c.write(HERE / ('PRE_HANDOVER_STATUS-' + worker.replace('/', '_') + '.json'), before)
        request = {'schema': 'six-device-graceful-shard-handover-v1', 'owner': owner,
                   'worker': worker, 'requested': time.time(), 'source': source(),
                   'reason': 'User authorized allocation queue sharing on additional GPUs 2, 3 and 4',
                   'mode': 'finish current children, stop new launches'}
        c.write(request_path, request)
        c.write(root / 'STOP', request)
    ledger = PREVIOUS / 'shared_disk_and_host_ledgers'
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline:
        handle = c.lock(ledger / 'LEDGER.lock')
        if handle is None:
            time.sleep(0.5); continue
        try:
            c.require(c.read(root / 'STOP') == request, 'Preserved STOP changed')
            state = c.read(root / 'STATUS.json')
            c.require(state['owner'] == owner, 'Prior owner changed unexpectedly')
            mode = admission_boundary(owner, state['phase'])
            # start() owns this ledger throughout its probe/launch. In full
            # phase a sleep is between scans. A waiting owner reaches run_phase
            # STOP before launching even if its locks become available next.
            if mode is not None:
                result = {'status': 'PASS', 'worker': worker, 'owner': owner,
                      'mode': mode, 'observed_phase': state['phase'], 'observed': time.time(),
                      'source': source(), 'prior_source_sha256': PREVIOUS_SOURCE_SHA,
                      'stop': pin(root / 'STOP'), 'status_at_ack': state,
                      'ledger_lock_held': str(ledger / 'LEDGER.lock'),
                      'physical_hostname': config(worker)['expected_hostname'],
                      'native_fits_interrupted': False}
                c.write(receipt_path, result)
                return result
        finally:
            handle.close()
        time.sleep(0.5)
    raise TimeoutError('Original owner has not reached an acknowledged STOP boundary; preserved request remains pending')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=PREVIOUS_SHARDS, required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(stop_and_ack(args.worker, args.source_sha256)))


if __name__ == '__main__':
    main()
