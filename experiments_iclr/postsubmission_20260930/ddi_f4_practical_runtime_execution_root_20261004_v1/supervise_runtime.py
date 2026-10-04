"""Run three fresh, complete DDI TRAIN epochs under owned process supervision.

This is a co-resident runtime measurement. It cannot produce VALID or TEST
results, donor checkpoints, full-family comparisons or scientific acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
HELD_CHILDREN = []


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def pin(path, row):
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']


def gpu_window(uuid):
    q = subprocess.run(['nvidia-smi', '-i', uuid,
                        '--query-gpu=uuid,memory.total,memory.free', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True, timeout=15)
    values = [v.strip() for v in q.stdout.strip().split(',')]
    assert len(values) == 3 and values[0] == uuid
    return dict(UTC=datetime.now(timezone.utc).isoformat(), GPU_UUID=uuid,
                memory_total_MiB=int(values[1]), memory_free_MiB=int(values[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release_path = HERE / 'ROOT_RELEASE.json'
    assert sha(release_path) == args.release_sha256
    release = json.loads(release_path.read_text())
    assert release['execution_authorized'] is True
    assert os.uname().nodename == 'peptide' and str(Path.cwd()) == release['repository']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == release['GPU_UUID']
    assert signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL
    for row in release['source_inventory']:
        path = PHASE / row['path']
        assert path.resolve().is_relative_to(PHASE)
        pin(path, row)
    assert sha(Path(__file__)) == release['supervisor_sha256']
    assert release['arms'] == ['target_only', 'joint', 'separate']
    assert release['TRAIN_epochs_per_arm'] == 1 and release['fresh_seed'] == 0
    assert release['cuda_memory_fraction'] == 0.30
    assert not release['VALID_scoring'] and not release['TEST_access'] and not release['checkpoint_donor']
    assert release['co_resident_timing'] is True and release['automatic_retry'] is False
    assert sha(Path(release['interpreter'])) == release['interpreter_sha256']

    # Reuse exact reviewed exit-aware ownership primitives; no numerical import.
    owned_source = PHASE / release['owned_supervisor_source']
    assert sha(owned_source) == release['owned_supervisor_sha256']
    sys.path.insert(0, str(owned_source.parent))
    spec = importlib.util.spec_from_file_location('reviewed_owned_process_primitives', owned_source)
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    received = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda s, f: received.append(s))

    caps = release['caps']
    terminals = []
    failure = None
    for arm in release['arms']:
        window = gpu_window(release['GPU_UUID'])
        if window['memory_free_MiB'] < release['minimum_free_MiB_before_arm']:
            failure = dict(arm=arm, reason='INSUFFICIENT_SHARED_WINDOW', window=window,
                           scientific_infeasibility_inferred=False)
            break
        slot = HERE / arm
        slot.mkdir(exist_ok=False)
        command = [release['interpreter'], '-B', str(PHASE / release['runtime_entrypoint']),
                   '--arm', arm, '--output-dir', str(slot / 'run01'), '--device', 'cuda:0',
                   '--cuda-memory-fraction', '0.30']
        child = identity = None
        started = time.monotonic()
        peak_rss, stop, errors, actions = 0, None, [], []
        session_closed = reaped = False
        last_window, next_window = window, started
        with (slot / 'STDOUT.txt').open('xb') as out, (slot / 'STDERR.txt').open('xb') as err:
            try:
                child = subprocess.Popen(command, cwd=release['repository'], stdin=subprocess.DEVNULL,
                                         stdout=out, stderr=err, start_new_session=True)
                HELD_CHILDREN.append(child)
                identity = owned.process_identity(child.pid)
                assert identity is not None and identity['session'] == identity['group'] == child.pid
                save(slot / 'OWNED_CHILD.json', dict(identity=identity, command=command,
                     root_release_sha256=args.release_sha256, window_before_launch=window))
                while True:
                    owned.held_identity(child, identity)
                    members = owned.members_of_session(child.pid)
                    peak_rss = max(peak_rss, sum(row['RSS_bytes'] for row in members))
                    live = [row for row in members if row['state'] != 'Z']
                    exit_info = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    if received:
                        stop = 'SUPERVISOR_SIGNAL'
                    elif time.monotonic() - started >= caps['wall_seconds_per_arm']:
                        stop = 'OWNED_WALL_CAP'
                    elif peak_rss > caps['host_RSS_bytes']:
                        stop = 'OWNED_RSS_CAP'
                    if stop is None and time.monotonic() >= next_window and live:
                        last_window = gpu_window(release['GPU_UUID'])
                        next_window = time.monotonic() + 5
                        if last_window['memory_free_MiB'] < release['minimum_global_headroom_MiB']:
                            stop = 'PROTECT_EXISTING_SHARED_GPU_WINDOW'
                    if stop is not None:
                        owned.kill_owned(child, identity, actions, stop)
                        break
                    if exit_info is not None and not live:
                        child.wait(timeout=0)
                        reaped = session_closed = True
                        break
                    time.sleep(0.25)
            except BaseException as error:
                stop = stop or 'SUPERVISION_EXCEPTION'
                errors.append(dict(type=type(error).__name__, message=str(error)))
            finally:
                if child is not None and not reaped:
                    try:
                        owned.kill_owned(child, identity, actions, stop or 'FINAL_OWNED_CLEANUP')
                        cleanup_deadline = time.monotonic() + 30
                        while time.monotonic() < cleanup_deadline:
                            owned.held_identity(child, identity)
                            live = [row for row in owned.members_of_session(child.pid) if row['state'] != 'Z']
                            info = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                            if info is not None and not live:
                                child.wait(timeout=0)
                                reaped = session_closed = True
                                break
                            time.sleep(0.25)
                    except BaseException as error:
                        errors.append(dict(type=type(error).__name__, message=str(error)))
        terminal = dict(UTC=datetime.now(timezone.utc).isoformat(), arm=arm, identity=identity,
             physical_exit_code=None if child is None else child.returncode,
             direct_child_reaped=reaped, physical_session_closed=session_closed,
             wall_seconds=time.monotonic() - started, sampled_peak_session_RSS_bytes=peak_rss,
             stop=stop, errors=errors, termination_actions=actions, last_GPU_window=last_window,
             root_release_sha256=args.release_sha256, automatic_retry=False,
             VALID_scoring=False, TEST_access=False, co_resident_timing=True)
        save(slot / 'PHYSICAL_TERMINAL.json', terminal)
        terminals.append(terminal)
        if not (child is not None and child.returncode == 0 and reaped and session_closed
                and stop is None and not errors):
            failure = dict(arm=arm, reason='FAILED_OR_INCOMPLETE_PHYSICAL_RUNTIME', terminal=terminal)
            break
    complete = failure is None and len(terminals) == 3
    save(HERE / 'QUEUE_TERMINAL.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
         status='COMPLETE_PHYSICAL_RUNTIME_FAMILY' if complete else 'FAILED_OR_INCOMPLETE',
         terminals=terminals, failure=failure, unattempted_arms=release['arms'][len(terminals):],
         source_manifest_sha256=release['candidate_manifest_sha256'], root_release_sha256=args.release_sha256,
         full_scientific_family_qualified=False, VALID_scoring=False, TEST_access=False,
         co_resident_timing=True, automatic_retry=False))
    # Do not let Popen destructors reap an unresolved held identity.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if complete else 1)


if __name__ == '__main__':
    main()
