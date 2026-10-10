"""Finite same-host CPU owner; needs a separately approved enabled release."""
import time
START = time.perf_counter()
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys

from common import HERE, admitted, require, write_json


def own_handle(pid):
    stat = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'parent_pid': int(stat[1]), 'group': int(stat[2]), 'session': int(stat[3]),
            'start_ticks': int(stat[19]), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def group_absent(group):
    try:
        os.killpg(group, 0)
        return False
    except ProcessLookupError:
        return True


def main(release_path, release_sha256):
    release, _, output = admitted(release_path, release_sha256)
    require(not output.exists(), 'fresh_output_required')
    output.mkdir(exist_ok=False)
    argv = [sys.executable, str(HERE / 'reader.py'), '--release', str(Path(release_path).resolve()),
            '--release-sha256', release_sha256]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    write_json(output / 'OWNER_START.json', {'complete': False, 'argv': argv, 'release_sha256': release_sha256,
               'source_manifest_sha256': release['source_manifest_sha256'], 'CPU_only': True, 'automatic_retry': False,
               'wall_budget_seconds': release['wall_budget_seconds'], 'budget_includes_owner_admission': True,
               'owner_handle': own_handle(os.getpid())})
    deadline = START + release['wall_budget_seconds']
    child = None
    timed_out = False
    error_type = None
    child_handle = None
    try:
        require(time.perf_counter() < deadline, 'owner_admission_exhausted_budget')
        with (output / 'CHILD_STDOUT.txt').open('x') as stdout, (output / 'CHILD_STDERR.txt').open('x') as stderr:
            child = subprocess.Popen(argv, env=env, cwd=str(HERE), stdout=stdout, stderr=stderr, start_new_session=True)
            child_handle = own_handle(child.pid)
            require(child_handle['group'] == child.pid and child_handle['session'] == child.pid, 'child_session_not_owned')
            write_json(output / 'CHILD_HANDLE.json', child_handle)
            remaining = max(0.001, deadline - time.perf_counter())
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
    except BaseException as error:
        error_type = type(error).__name__
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
    terminal = output / 'readout' / 'TERMINAL.json'
    reader_terminal = json.loads(terminal.read_text()) if terminal.exists() else None
    child_group_absent = child is not None and group_absent(child.pid)
    complete = bool(not timed_out and error_type is None and child is not None and child.returncode == 0
                    and child_group_absent and reader_terminal is not None and reader_terminal.get('complete') is True)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    write_json(output / 'OWNER_TERMINAL.json', {'complete': complete, 'status': 'complete' if complete else 'failed',
               'timeout': timed_out, 'exception_type': error_type, 'child_returncode': None if child is None else child.returncode,
               'child_handle': child_handle, 'child_group_absent_after_wait': child_group_absent,
               'reader_terminal': reader_terminal, 'seconds_from_before_remaining_imports': time.perf_counter() - START,
               'owner_CPU_user_seconds': usage.ru_utime, 'owner_CPU_system_seconds': usage.ru_stime,
               'child_CPU_user_seconds': children.ru_utime, 'child_CPU_system_seconds': children.ru_stime,
               'owner_lifetime_RSS_peak_bytes': usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
               'child_lifetime_RSS_peak_bytes': children.ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
               'nested_wall_times_not_summed': True, 'post_deadline_reap_allowance_seconds': 5,
               'automatic_retry': False, 'original_jobs_untouched': True})
    return 0 if complete else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='One finite resident mask readout; separate enabled release required.')
    parser.add_argument('--release', required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    try:
        exit_code = main(args.release, args.release_sha256)
    except BaseException as error:
        print(json.dumps({'complete': False, 'exception_type': type(error).__name__, 'no_retry': True}))
        exit_code = 1
    sys.exit(exit_code)
