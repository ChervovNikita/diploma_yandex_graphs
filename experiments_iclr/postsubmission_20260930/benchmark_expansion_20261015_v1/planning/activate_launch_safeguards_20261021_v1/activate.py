"""Inspect or activate a verified queue owner; global training release is separate."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

WORKERS = ('allocation/gpu0', 'server51/gpu0', 'server51/gpu1')
ALLOCATION = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
SERVER51 = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
RELATIVE = 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
PLAN_SHAS = dict(zip(WORKERS, (
    '19fa1a8b96bcf3383965c266c4fa900e0ac09c62ca040553e7669acb659ffaad',
    'b23301b8674faf5dfad2000efeeb067d0c1a2fb97524277c2185d174a307a345',
    'bc025b3209c0a7f5c2f10fd011fd54ca072c78269693b035ca3e53bbf243ad3c')))
SOURCE_SHAS = {
    'campaign_execution_20261016_v7': '52a996e6effb77748339c57a716159a80a06272d88914d37beaa2dd680e7e890',
    'campaign_memory_admission_20261019_v1': '6905ecec8b840925d2c3e03fb9dd43f06083372a05077579082f468a93aa4c31',
    'campaign_launch_safeguards_20261021_v1': 'adf6710ccf68808d21d8f4aeb6a67b665f17b9ee06de18051886caf11fd7bfdc',
    'campaign_named_processes_20261021_v1': 'b7da088e468530267e0fd2afbaafa927c82fb6b39b5ee5008f0a1222168e9824'}
TARGET_PACKAGE = 'campaign_named_processes_20261021_v1'
TARGET_SCRIPT = 'named_worker.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def guard(worker):
    # No project file is read, resolved or written until both checks succeed.
    allocation = worker.startswith('allocation/')
    expected_host = 'anogena-2-0' if allocation else 'olig'
    uuids = ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'] if allocation else [
        'GPU-9c6c20fa-a620-479d-f72f-163ac74deef2', 'GPU-30283973-140d-9e7b-e780-0fcc487dec54']
    require(socket.gethostname() == expected_host, 'Wrong host; no project metadata accessed')
    observed = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                       text=True, timeout=10).splitlines()
    require(observed == uuids, 'Wrong full GPU identity table; no project metadata accessed')
    return Path(ALLOCATION if allocation else SERVER51)


def pin(path):
    raw = Path(path).read_bytes()
    return {'path': str(Path(path)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def manifest(folder, expected):
    value = pin(folder / 'MANIFEST.json')
    require(value['sha256'] == expected, 'Frozen source manifest differs: ' + str(folder))
    for row in read(folder / 'MANIFEST.json')['files']:
        path = (folder / row['path']).resolve()
        require(path.is_relative_to(folder.resolve()), 'Manifest file escapes package')
        actual = pin(path)
        require(all(actual[key] == row[key] for key in ('bytes', 'sha256')), 'Frozen source changed: ' + str(path))
    return value


def pane(session):
    result = subprocess.run(['tmux', 'has-session', '-t', session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode:
        return None
    rows = subprocess.check_output(['tmux', 'list-panes', '-t', session, '-F', '#{pane_id}\t#{pane_dead}'], text=True).splitlines()
    require(len(rows) == 1, 'Exactly one original queue pane required')
    pane_id, dead = rows[0].split('\t')
    return {'id': pane_id, 'dead': dead == '1'}


def lease_busy(path):
    # Only existing supervisor custody is observed; no new supervisor is made.
    with Path(path).open('r+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def preserve(path, history):
    if not path.exists():
        return None
    raw = path.read_bytes()
    target = history / (path.name + '-' + hashlib.sha256(raw).hexdigest())
    if target.exists():
        require(target.read_bytes() == raw, 'Historical bytes differ')
    else:
        with target.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(history, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return pin(target)


def require_waiting(status, owner, worker, count, plan_pin):
    require(status['worker'] == worker and status['plan'] == plan_pin, 'Status identity differs')
    require(status['phase'] == 'waiting_for_global_all_fit_PASS', 'Old owner is not waiting at the global gate')
    require(status['active_fit_ids'] == [] and status['counts'] == {'PASS': count}, 'Old owner has active or non-PASS fits')
    require(status['owner'] == owner['identity'], 'Status and OWNER identities differ')
    require(owner['plan'] == plan_pin, 'OWNER plan differs')


def current_producer(owner):
    return owner.get('actual_scheduler_producer', {}).get('actual_outer_source_manifest', {}).get('sha256')


def ready(owner, status, expected):
    return (current_producer(owner) == expected and current_producer(status) == expected
            and status.get('owner') == owner['identity']
            and status.get('phase') == 'waiting_for_global_all_fit_PASS'
            and status.get('active_fit_ids') == [])


def verify_smokes(q, plan, root):
    c = q.c
    inventory = c.read(c.verify(plan['inventory'], plan['project_root']))
    tasks = [task for task in plan['tasks'] if task['kind'] == 'fit']
    expected = sorted(task['id'] for task in tasks)
    summary = c.read(root / 'SMOKE_SUMMARY.json')
    require(summary['status'] == 'PASS' and summary['worker'] == plan['worker'], 'Wrong smoke summary')
    require(summary['inventory'] == plan['inventory'] and summary['expected_fit_ids'] == expected,
            'Smoke summary inventory or IDs differ')
    require(summary['fit_count'] == len(expected) == c.COUNTS[plan['worker']], 'Wrong assigned smoke count')
    require(sorted(row['id'] for row in summary['results']) == expected, 'Missing or duplicate smoke result')
    instance = q.LaunchSafeguardsWorker.__new__(q.LaunchSafeguardsWorker)
    instance.plan = plan
    instance.smokes = {row['id']: row for row in inventory['fits'] if row['worker'] == plan['worker']}
    index = {task['id']: task for task in tasks}
    for row in summary['results']:
        raw = c.read(c.verify(row['result_receipt'], plan['project_root']))
        require(raw == row['result'] and row['status'] == 'PASS', 'Retained smoke summary differs')
        attempt = c.read(c.verify(row['attempt_result'], plan['project_root']))
        require(attempt['status'] == 'PASS' and attempt['native_result'] == row['result_receipt'], 'Smoke attempt receipt differs')
        instance.validate_result(index[row['id']], 'smoke', raw, Path(attempt['output']))
        folder = root / 'smokes' / c.token(row['id'])
        latest = sorted((folder / 'attempts').glob('[0-9][0-9][0-9][0-9]'))[-1]
        require(latest / 'ATTEMPT_RESULT.json' == Path(row['attempt_result']['path']), 'Summary is not the latest attempt')
        state = c.read(folder / 'STATE.json')
        require(state['status'] == 'PASS' and state['native_result'] == row['result_receipt'], 'Current smoke state differs')
    return c.pin(root / 'SMOKE_SUMMARY.json')


def command(project, expansion, worker, plan_path):
    return [str(project / '.venv/bin/python'), '-B', '-u', str(expansion / TARGET_PACKAGE / TARGET_SCRIPT),
            '--worker', worker, '--plan', str(plan_path), '--plan-sha256', PLAN_SHAS[worker],
            '--source-sha256', SOURCE_SHAS[TARGET_PACKAGE],
            '--process-title-dependency-dir', str(expansion / 'campaign_process_title_dependency_20261021_v1')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inspect', 'activate'))
    parser.add_argument('--worker', choices=WORKERS, required=True)
    parser.add_argument('--helper-sha256', required=True)
    parser.add_argument('--wait-seconds', type=int, default=45)
    args = parser.parse_args()
    require(1 <= args.wait_seconds <= 55, 'Bounded wait must be 1..55 seconds')
    project = guard(args.worker)
    expansion = project / RELATIVE
    source = expansion / 'planning/activate_launch_safeguards_20261021_v1/activate.py'
    require(Path(__file__).resolve() == source and pin(source)['sha256'] == args.helper_sha256, 'Exact helper source required')
    sources = {name: manifest(expansion / name, sha) for name, sha in SOURCE_SHAS.items()}
    plan_path = expansion / 'campaign_execution_20261016_v7' / ('PLAN-' + args.worker.replace('/', '_') + '.json')
    plan_pin = pin(plan_path)
    require(plan_pin['sha256'] == PLAN_SHAS[args.worker], 'Original plan hash differs')
    plan = read(plan_path)
    require(plan['worker'] == args.worker and plan['project_root'] == str(project), 'Plan host differs')
    root = Path(plan['state_root'])
    require(root == expansion / 'campaign_execution_20261016_v7/runs' / args.worker.replace('/', '_'), 'Original state root required')
    require(not Path(plan['global_smoke_pass']).exists(), 'Global marker present; activation is closed')
    sys.path.insert(0, str(expansion / 'campaign_launch_safeguards_20261021_v1'))
    import queue_worker as q
    c = q.c
    sys.path.insert(0, str(expansion / TARGET_PACKAGE))
    import named_worker
    dependency = named_worker.verify_dependency(str(expansion / 'campaign_process_title_dependency_20261021_v1'))
    require(plan['source_manifest'] == c.source_manifest(), 'Plan source receipt differs')
    c.verify(plan['host_config'], str(project))
    summary = verify_smokes(q, plan, root)
    for reservation in (Path(plan['host_ledger']) / 'reservations').glob('*.json'):
        record = read(reservation)
        if record.get('worker') == args.worker:
            attempt_lease = Path(record['directory']) / 'LEASE.lock'
            require(not attempt_lease.exists() or not lease_busy(attempt_lease), 'An assigned attempt is still active')
    session = 'gnnm-allfit-' + args.worker.replace('/', '-') + '-20261016'
    argv = command(project, expansion, args.worker, plan_path)
    owner = read(root / 'OWNER.json')
    status = read(root / 'STATUS.json')
    snapshot = {'worker': args.worker, 'session': session, 'owner': owner, 'status': status,
                'pane': pane(session), 'supervisor_lease_busy': lease_busy(root / 'SUPERVISOR.lock'),
                'summary': summary, 'sources': sources, 'process_title_dependency': dependency,
                'argv': argv, 'global_marker_present': False}
    if args.action == 'inspect':
        print(json.dumps(snapshot, indent=2))
        return 0
    deadline = time.monotonic() + args.wait_seconds
    with (root / 'LAUNCH_SAFEGUARDS_HANDOFF.lock').open('a+') as handoff:
        fcntl.flock(handoff, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = root / 'LAUNCH_SAFEGUARDS_HANDOFF.json'
        journal = read(journal_path) if journal_path.exists() else None
        owner = read(root / 'OWNER.json')
        if ready(owner, read(root / 'STATUS.json'), SOURCE_SHAS[TARGET_PACKAGE]) and c.same_process(owner['identity']):
            print(json.dumps({'status': 'already_active', 'owner': owner, 'status_receipt': pin(root / 'STATUS.json')}))
            return 0
        if journal is None:
            require_waiting(read(root / 'STATUS.json'), owner, args.worker, c.COUNTS[args.worker], plan_pin)
            require(c.same_process(owner['identity']) and lease_busy(root / 'SUPERVISOR.lock'), 'Old waiting owner is not live')
            require(snapshot['pane'] is not None and not snapshot['pane']['dead'], 'Original live tmux pane required')
            history = root / 'launch_safeguards_history_20261021' / pin(root / 'OWNER.json')['sha256']
            history.mkdir(parents=True, exist_ok=True)
            preserved = {name: preserve(root / name, history) for name in ('OWNER.json', 'STATUS.json', 'TMUX_LAUNCH.json', 'STOP')}
            journal = {'worker': args.worker, 'old_owner': owner, 'sources': sources, 'plan': plan_pin,
                       'history': str(history), 'preserved': preserved, 'argv': argv,
                       'process_title_dependency': dependency,
                       'state': 'stop_requested', 'created': time.time(), 'global_marker_written': False}
            c.write(journal_path, journal)
        require(journal['plan'] == plan_pin and journal['sources'] == sources and journal['argv'] == argv, 'Handoff journal differs')
        if journal['state'] == 'stop_requested':
            require(read(root / 'OWNER.json') == journal['old_owner'], 'Unexpected owner; inspect instead of replacing')
            require_waiting(read(root / 'STATUS.json'), journal['old_owner'], args.worker, c.COUNTS[args.worker], plan_pin)
            if not (root / 'STOP').exists():
                require(not Path(plan['global_smoke_pass']).exists(), 'Global release raced with STOP')
                c.write(root / 'STOP', {'requested': time.time(), 'mode': 'finish current children, stop new launches',
                                        'requester': pin(source), 'reason': 'Replace waiting owner with verified named safeguards'})
        else:
            require(journal['state'] in ('launch_requested', 'active'), 'Unknown handoff state')
        while time.monotonic() < deadline:
            owner = read(root / 'OWNER.json')
            if ready(owner, read(root / 'STATUS.json'), SOURCE_SHAS[TARGET_PACKAGE]) and c.same_process(owner['identity']):
                journal.update(state='active', new_owner=owner, completed=time.time())
                c.write(journal_path, journal)
                print(json.dumps({'status': 'active', 'journal': pin(journal_path), 'owner': owner}))
                return 0
            require(not Path(plan['global_smoke_pass']).exists(), 'Global release raced with handoff')
            view = pane(session)
            drained = not c.same_process(journal['old_owner']['identity']) and not lease_busy(root / 'SUPERVISOR.lock')
            if drained and (view is None or view['dead']) and journal['state'] == 'stop_requested':
                history = Path(journal['history'])
                stop = root / 'STOP'
                if stop.exists():
                    target = history / ('STOP-acknowledged-' + str(time.time_ns()) + '.json')
                    stop.rename(target)
                    history_fd = os.open(history, os.O_RDONLY)
                    try:
                        os.fsync(history_fd)
                    finally:
                        os.close(history_fd)
                    journal['acknowledged_stop'] = pin(target)
                log = root / ('OWNER-named-safeguards-' + str(time.time_ns()) + '.log')
                script = shlex.join(argv) + ' >> ' + shlex.quote(str(log)) + ' 2>&1'
                journal.update(state='launch_requested', owner_log=str(log), exact_tmux_command=script, requested=time.time())
                c.write(journal_path, journal)
                # Record intent before dispatch. An uncertain return is inspected,
                # never automatically resent. No kill-session, signal or -k flag.
                if view is None:
                    subprocess.check_call(['tmux', 'new-session', '-d', '-s', session, '-c', str(project), script])
                else:
                    subprocess.check_call(['tmux', 'respawn-pane', '-t', view['id'], '-c', str(project), script])
                subprocess.check_call(['tmux', 'set-option', '-t', session, 'remain-on-exit', 'on'])
            time.sleep(1)
        print(json.dumps({'status': 'pending_inspection', 'journal': pin(journal_path),
                          'state': journal['state'], 'automatic_resubmit': False}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
