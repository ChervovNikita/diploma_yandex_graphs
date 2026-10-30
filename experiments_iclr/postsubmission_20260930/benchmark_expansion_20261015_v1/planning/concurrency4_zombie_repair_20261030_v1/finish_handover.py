"""Finish a verified drained handover when an exited owner remains a zombie."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent.parent
sys.path.insert(0, str(EXPANSION / 'campaign_concurrency_20261029_v1'))
import handover

c = handover.c
SOURCE_SHA = '82b056580848c7074d6d9e2a3aa9a13f298c2123256d0d9f07c9139ba6bbe912'


def process_state(identity):
    """PID identity is distinct from liveness: a matching zombie has exited."""
    if not c.same_process(identity):
        return 'absent_or_reused'
    try:
        raw = (Path('/proc') / str(identity['pid']) / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return 'absent_or_reused'
    fields = raw.rsplit(')', 1)[1].split()
    c.require(len(fields) >= 20, 'Malformed process status')
    c.require(int(fields[19]) == identity['start_ticks'], 'Identity changed during process read')
    return fields[0]


def original_exited(identity):
    return process_state(identity) in ('Z', 'X', 'absent_or_reused')


def require_drained(request, status, current_pane):
    c.require(original_exited(request['old_owner']), 'Original scheduler is still alive')
    c.require(status['owner'] == request['old_owner'] and status['phase'] == 'stopped'
              and not status['active_fit_ids'], 'Queue has not drained normally')
    c.require(not status['counts'].get('running', 0) and not status['counts'].get('failed', 0)
              and not status.get('error'), 'Queue has active or failed work')
    c.require(current_pane['id'] == request['pane']['id'] and current_pane['dead'], 'Original pane is not dead')


def stop_exact_watcher(identity, request_path):
    if original_exited(identity):
        return False
    fd = os.pidfd_open(identity['pid'])
    try:
        c.require(c.same_process(identity), 'Watcher identity changed before signalling')
        argv = (Path('/proc') / str(identity['pid']) / 'cmdline').read_bytes().split(b'\0')
        c.require(str(handover.HERE / 'handover.py').encode() in argv
                  and b'watch' in argv and str(request_path).encode() in argv,
                  'Target is not the exact campaign handover watcher')
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    deadline = time.monotonic() + 10
    while not original_exited(identity) and time.monotonic() < deadline:
        time.sleep(0.2)
    c.require(original_exited(identity), 'Stuck watcher did not exit after TERM')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'apply'])
    parser.add_argument('--worker', choices=c.WORKERS, required=True)
    args = parser.parse_args()
    handover.four.recovery.guard()
    handover.four.own_source(SOURCE_SHA)
    host_config, _ = c.environment(args.worker)
    project = Path(host_config['project_root'])
    c.require(EXPANSION == project / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1', 'Wrong repository')
    journal = EXPANSION / 'planning/concurrency4_20261029_v1' / args.worker.replace('/', '_')
    request_path = journal / 'REQUEST.json'
    request = c.read(request_path)
    c.require(request['worker'] == args.worker and request['source'] == c.pin(handover.HERE / 'MANIFEST.json'), 'Request/source differs')
    root = Path(request['state_root'])
    c.require(root == EXPANSION / 'campaign_execution_20261016_v7/runs' / args.worker.replace('/', '_'), 'Unexpected queue root')
    status = c.read(root / 'STATUS.json')
    producer = status.get('actual_scheduler_producer', {})
    if producer.get('actual_outer_source_manifest') == request['source'] and status['phase'] == 'full':
        c.require(not original_exited(status['owner']), 'Four-slot owner is not running')
        print(json.dumps({'worker': args.worker, 'status': 'already_active', 'counts': status['counts']}))
        return
    current = handover.pane(request['session'])
    require_drained(request, status, current)
    plan = c.read(c.verify(request['plan'], project))
    c.require(request['plan']['sha256'] == handover.PLAN_SHAS[args.worker], 'Frozen plan changed')
    c.verify(request['recovery_authority'], project)
    c.require(c.global_barrier(plan['global_smoke_pass'], plan, c.pin(root / 'SMOKE_SUMMARY.json')), 'All-fit smoke release changed')
    handover.four.recovery.named.verify_dependency(str(EXPANSION / 'campaign_process_title_dependency_20261021_v1'))
    marker = json.dumps({'reason': 'graceful transition to four concurrent tasks per GPU',
                         'request': c.pin(request_path)}, sort_keys=True) + '\n'
    stop = root / 'STOP'
    c.require(stop.read_text() == marker, 'STOP does not belong to this handover')
    watcher = c.read(journal / 'WATCHER.json')
    c.require(watcher['request'] == c.pin(request_path), 'Watcher request changed')
    report = {'worker': args.worker, 'old_owner_state': process_state(request['old_owner']),
              'queue_phase': status['phase'], 'counts': status['counts'],
              'watcher_state': process_state(watcher['identity']), 'maximum_concurrency': 4,
              'training_processes_signalled': False, 'source': c.pin(__file__)}
    if args.action == 'inspect':
        print(json.dumps(report, indent=2))
        return
    audit = HERE / args.worker.replace('/', '_')
    audit.mkdir(exist_ok=False)
    for source, name in [(root / 'STATUS.json', 'queue-before.json'),
                         (journal / 'STATUS.json', 'watcher-status-before.json'),
                         (journal / 'WATCHER.json', 'watcher-before.json'),
                         (stop, 'STOP-before.json')]:
        shutil.copy2(source, audit / name)
    c.write(audit / 'REQUEST.json', {**report, 'original_request': c.pin(request_path)})
    handles = []
    try:
        for path in [root / 'SUPERVISOR.lock', *[plan[k] for k in ('slot_lock', 'physical_lock', 'worker_lock')]]:
            lease = c.lock(path)
            c.require(lease is not None, 'An owner or training child still holds a queue lock')
            handles.append(lease)
        report['stuck_watcher_signalled'] = stop_exact_watcher(watcher['identity'], request_path)
        watcher_lease = c.lock(journal / 'WATCHER.lock')
        c.require(watcher_lease is not None, 'Watcher custody remains active')
        handles.append(watcher_lease)
        require_drained(request, c.read(root / 'STATUS.json'), handover.pane(request['session']))
        c.require(stop.read_text() == marker, 'STOP changed before archival')
        c.require(not (audit / 'CONSUMED_STOP.json').exists(), 'STOP archive already exists')
        stop.rename(audit / 'CONSUMED_STOP.json')
    finally:
        for lease in handles:
            lease.close()
    command = 'exec ' + shlex.join(request['argv']) + ' >> ' + shlex.quote(str(audit / 'new-owner.log')) + ' 2>&1'
    c.write(audit / 'ACTIVATION.json', {'argv': request['argv'], 'shell_command': command, 'pane': current,
                                      'original_request': c.pin(request_path), 'updated': time.time()})
    subprocess.check_call(['tmux', 'respawn-pane', '-t', current['id'], '-c', str(project), command])
    report.update(status='four_slot_owner_dispatched', updated=time.time())
    c.write(audit / 'RESULT.json', report)
    c.write(journal / 'STATUS.json', {'phase': 'four_slot_owner_dispatched_after_zombie_fix',
             'target_concurrency': 4, 'repair_receipt': c.pin(audit / 'RESULT.json'), 'updated': time.time()})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
