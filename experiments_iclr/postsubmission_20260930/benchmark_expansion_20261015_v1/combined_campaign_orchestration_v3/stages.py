"""Durable sequencing of existing supervisors; no native work or scheduler.

This module starts only the existing TRAIN/final supervisor commands. Their
original backends remain the sole owners of native child admission/Popen.
An unknown previous supervisor attempt is never repeated automatically.
"""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
from final_scoring_integration_v2.common import digest, pin, require, verify
from portable_dispatcher_v3.custody import write


def exclusive(path, value):
    write(path, value)  # actual portable fsync + atomic create-if-absent


def same_or_create(path, value, root):
    path = Path(path)
    if path.exists():
        require(verify(pin(path, root), root, json_metadata=True) == value, 'Permanent sequencer authority changed: ' + str(path))
    else:
        exclusive(path, value)
    return pin(path, root)


def supervisor_argv(executable, entry_pin, worker, spec_pin):
    return [executable, '-B', '-u', entry_pin['path'], '--worker', worker,
            '--run-spec', spec_pin['path'], '--run-spec-sha256', spec_pin['sha256']]


def run_stage(name, folder, executable, entry_pin, worker, spec_pin, root, identity):
    folder = Path(folder) / name
    folder.mkdir(parents=True, exist_ok=True)
    start, returned = folder / 'START.json', folder / 'RETURN.json'
    argv = supervisor_argv(executable, entry_pin, worker, spec_pin)
    authority = {'schema': 'combined-existing-supervisor-authority-v1', 'stage': name, 'worker': worker,
                 'entry': entry_pin, 'run_spec': spec_pin, 'lexical_executable': executable, 'argv': argv}
    verify(entry_pin, root); verify(spec_pin, root, json_metadata=True)
    if start.exists():
        require(verify(pin(start, root), root, json_metadata=True)['authority'] == authority, 'Previous supervisor stage has changed authority')
        require(returned.exists(), 'Previous supervisor START has no durable return; inspect retained job/state, never duplicate it')
        previous = verify(pin(returned, root), root, json_metadata=True)
        require(previous['authority'] == authority and previous['exit_code'] == 0, 'Retained supervisor failure is not an automatic retry')
        return pin(returned, root)
    require(not returned.exists(), 'Return without a start is corrupt stage custody')
    exclusive(start, {'schema': 'combined-existing-supervisor-start-v1', 'authority': authority,
                      'UTC_unix': time.time(), 'automatic_retries': 0})
    log = (folder / 'STDOUT.log').open('x')
    child, started = None, time.monotonic()
    try:
        child = subprocess.Popen(argv, cwd=root, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            observed = identity(child.pid)
        except Exception as error:
            observed = {'pid': child.pid, 'identity_unavailable': str(error)}
        exclusive(folder / 'PID.json', {'supervisor_identity': observed,
            'native_child_custody': 'unchanged actual portable backend; this is only its supervisor'})
        code = child.wait()
    except KeyboardInterrupt:
        # Signal only this actual child supervisor. Its unchanged Engine owns
        # STOP/drain for its native children; never signal other users/jobs.
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            child.wait()
        raise
    finally:
        if child is not None and child.poll() is None:
            child.wait()  # preserve/reap this actual supervisor; never launch another
        log.close()
    exclusive(returned, {'schema': 'combined-existing-supervisor-return-v1', 'authority': authority,
                        'exit_code': code, 'elapsed_wall_seconds': time.monotonic() - started,
                        'automatic_retries': 0, 'previous_attempts_preserved': True})
    require(code == 0, 'Existing ' + name + ' supervisor did not finish successfully; native outcomes/costs retained')
    return pin(returned, root)


def confirm_worker_terminal(training_release_pin, training_release, worker, root):
    terminal = {'complete', 'no_selected_state', 'failed', 'launch_failed',
                'blocked_dependency_no_selected_state', 'blocked_dependency_failed'}
    states, counts = {}, {}
    for task in training_release['tasks']:
        if task['worker'] != worker:
            continue
        path = Path(training_release['output_root']) / 'control/tasks' / digest(task['id']) / 'STATE.json'
        receipt = pin(path, root)
        state = verify(receipt, root, json_metadata=True)
        require(state['status'] in terminal, 'Own TRAIN task is not terminal; no final serving or silent exclusion')
        outcome = verify(state['attempt_terminal'], root, json_metadata=True)
        require(outcome['id'] == task['id'] and outcome['status'] == state['status'], 'Own terminal chain changed')
        states[task['id']] = receipt
        counts[state['status']] = counts.get(state['status'], 0) + 1
    require(states, 'Worker has no owned TRAIN tasks')
    return {'schema': 'combined-training-worker-terminal-ack-v1', 'worker': worker,
            'training_release': training_release_pin, 'states': states, 'status_counts': counts,
            'TEST_accessed': False, 'new_training_fits': 0}


def wait_for_files(paths, check_stop, *, interval_seconds=30, observe=None):
    """Filesystem existence polling only; full validation happens after return."""
    require(1 <= interval_seconds <= 60, 'Bounded quiet wait interval required')
    while True:
        check_stop()
        missing = [str(path) for path in paths if not Path(path).is_file()]
        if not missing:
            return
        if observe is not None:
            observe(missing)
        time.sleep(interval_seconds)


def lock_worker(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle
