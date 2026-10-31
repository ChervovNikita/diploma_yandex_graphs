"""Wait for one owned server51 queue to drain; audit known reporting failures once."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
EXPANSION = PROJECT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
REPAIR = EXPANSION / 'campaign_reporting_repair_20261024_v2'
WORKER = 'server51/gpu1'
UUIDS = ['GPU-9c6c20fa-a620-479d-f72f-163ac74deef2', 'GPU-30283973-140d-9e7b-e780-0fcc487dec54']
FAILURE_DIRECTORY = None


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def verify_package(folder, expected):
    require(pin(folder / 'MANIFEST.json')['sha256'] == expected, 'Package manifest changed')
    for row in json.loads((folder / 'MANIFEST.json').read_text())['files']:
        path = (folder / row['path']).resolve()
        require(path.is_relative_to(folder), 'Package source escaped its folder')
        actual = pin(path)
        require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Package source bytes changed')


def eligible(status, expected_worker, expected_failures):
    require(status['worker'] == expected_worker, 'Queue worker changed')
    phase = status['phase']
    if phase == 'full':
        return False
    require(phase == 'full_failed', 'Queue entered an unexpected state: ' + phase)
    require(not status['active_fit_ids'], 'Failed queue still has active work')
    require(set(status['failed_fit_ids']) == set(expected_failures), 'Unexpected failed job set; manual investigation required')
    return True


def save(path, value):
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    global FAILURE_DIRECTORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    require(socket.gethostname() == 'olig', 'Wrong host')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == UUIDS,
            'Wrong physical GPU identities')
    require(HERE == EXPANSION / 'campaign_reporting_drain_waiter_20261024_v1', 'Wrong waiter source location')
    verify_package(HERE, args.source_sha256)
    config_path = Path(args.config).resolve()
    require(config_path.is_relative_to(EXPANSION / 'planning/server51_recovery_20261024_v1'), 'Wait config outside evidence folder')
    require(pin(config_path)['sha256'] == args.config_sha256, 'Wait config changed')
    config = json.loads(config_path.read_text())
    require(config['worker'] == WORKER, 'Only the audited GPU1 queue may drain here')
    verify_package(REPAIR, config['repair_source_sha256'])
    plan_path = Path(config['plan']['path'])
    require(pin(plan_path) == config['plan'], 'Frozen worker plan changed')
    plan = json.loads(plan_path.read_text())
    root = Path(plan['state_root'])
    require(root == EXPANSION / 'campaign_execution_20261016_v7/runs/server51_gpu1', 'Unexpected queue root')
    require(plan['worker'] == WORKER and plan['project_root'] == str(PROJECT), 'Unexpected queue host or project')
    host_path = Path(plan['host_config']['path'])
    require(pin(host_path) == plan['host_config'], 'Qualified host configuration changed')
    # Match the frozen queue interpreter; host_config binds devices, not Python.
    python = str(PROJECT / '.venv/bin/python')
    require(Path(python).is_file(), 'Qualified project Python is missing')
    sys.path.insert(0, str(REPAIR))
    from repair_native import source
    source(config['repair_source_sha256'])
    from recovery_worker import named
    named.own_source('b7da088e468530267e0fd2afbaafa927c82fb6b39b5ee5008f0a1222168e9824')
    named.set_process_title(str(EXPANSION / 'campaign_process_title_dependency_20261021_v1'))
    evidence = config_path.parent
    FAILURE_DIRECTORY = evidence
    recovery = evidence / 'RECOVERY_CONFIG-server51_gpu1.json'
    require(not recovery.exists(), 'Recovery authority already exists; inspect prior attempt')
    started = time.time()
    with (evidence / 'WAIT_RECOVERY.lock').open('a+') as waiter_lease:
        fcntl.flock(waiter_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            require(time.time() - started < 86400, 'Drain did not finish within 24 hours; inspect manually')
            owner = json.loads((root / 'OWNER.json').read_text())
            require(owner['identity'] == config['expected_owner_identity'], 'Existing queue owner changed; stop without action')
            status = json.loads((root / 'STATUS.json').read_text())
            ready = eligible(status, WORKER, config['expected_failed_fit_ids'])
            snapshot = {'phase': 'waiting_for_owned_training_to_finish', 'utc_epoch': time.time(),
                        'worker': WORKER, 'queue_phase': status['phase'], 'queue_counts': status['counts'],
                        'active_fit_ids': status['active_fit_ids'], 'no_processes_signalled': True,
                        'config': pin(config_path), 'source_sha256': args.source_sha256}
            if ready:
                panes = subprocess.check_output(['tmux', 'list-panes', '-t', config['queue_session'],
                                                 '-F', '#{pane_id}\t#{pane_dead}'], text=True).splitlines()
                require(len(panes) == 1 and panes[0].split('\t')[0] == config['queue_pane'], 'Queue pane identity changed')
                if panes[0].split('\t')[1] == '1':
                    try:
                        with (root / 'SUPERVISOR.lock').open('r+') as queue_lease:
                            fcntl.flock(queue_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        break
            save(evidence / 'WAIT_STATUS.json', snapshot)
            time.sleep(30)
        audit_argv = [python, '-B', str(REPAIR / 'audit_recovery.py'), '--plan', str(plan_path), '--output', str(recovery)]
        save(evidence / 'DRAIN_AUDIT_COMMAND.json', {'argv': audit_argv, 'config': pin(config_path)})
        with (evidence / 'DRAIN_AUDIT.log').open('x') as log:
            subprocess.run(audit_argv, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600, cwd=PROJECT)
        audit = json.loads(recovery.read_text())
        require(set(audit['repairs']) == set(config['expected_failed_fit_ids']), 'Audit produced unexpected repair set')
        argv = [python, '-B', str(REPAIR / 'activate_recovery.py'), '--source-sha256', config['repair_source_sha256'],
                '--config', str(recovery), '--config-sha256', pin(recovery)['sha256']]
        save(evidence / 'DRAIN_ACTIVATION_COMMAND.json', {'argv': argv, 'recovery': pin(recovery)})
        with (evidence / 'DRAIN_ACTIVATION.log').open('x') as log:
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120, cwd=PROJECT)
        save(evidence / 'WAIT_STATUS.json', {'phase': 'reporting_recovery_dispatched', 'utc_epoch': time.time(),
             'worker': WORKER, 'recovery': pin(recovery), 'no_processes_signalled': True})
        print(json.dumps({'status': 'DISPATCHED', 'worker': WORKER, 'recovery': pin(recovery)}))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if FAILURE_DIRECTORY is not None:
            save(FAILURE_DIRECTORY / 'WAIT_STATUS.json', {
                'phase': 'failed_requires_investigation', 'utc_epoch': time.time(),
                'error': type(exc).__name__ + ': ' + str(exc),
                'no_processes_signalled': True, 'automatic_retry': False})
        raise
