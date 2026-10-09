"""Normal-host, once-only supervision of disposable full-input engineering work."""
from pathlib import Path
import hashlib
import json
import os
import resource
import signal
import subprocess
import time

A = Path(__file__).resolve().parent
P = A.parent
R = P.parents[1]
LIMIT = 1800


def write(name, value):
    with (A / name).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def identity(pid):
    try:
        raw = Path('/proc', str(pid), 'stat').read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), group=int(fields[2]),
                session=int(fields[3]), state=fields[0])


def main():
    started = time.monotonic()
    runtime = json.loads((P / 'staged_label_posterior_native_metadata_root_20261008_v1/RUNTIME.json').read_text())
    program = P / 'private_sheaf_full_input_qualifier_20261009_v1/qualifier.py'
    assert hashlib.sha256(program.read_bytes()).hexdigest() == '01969b86dd51c978ded2c77c0a07be73f39d1bc5698127a2225bf2c55944f283'
    release = A / 'RELEASE.json'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=runtime['GPU_uuid'],
               PYTHONPATH=os.pathsep.join(runtime['PYTHONPATH']), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    env.pop('PYTHONHOME', None)
    child = saved = None
    error = None
    actions = []
    try:
        with (A / 'WORKER.log').open('xb') as log:
            child = subprocess.Popen([runtime['python'], '-B', str(program), '--execute', '--release', str(release),
                '--roles', str(P / 'private_sheaf_train_valid_roles_allocation_root_20261009_v1/roles.npz'),
                '--output', str(P / 'private_sheaf_full_input_qualification_execution_root_20261009_v1'),
                '--device', 'cuda:0', '--dependency-overlay', str(P / 'private_sheaf_dependency_overlay_20261009_v1'),
                '--direct-index-check', 'full'],
                cwd=R, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        saved = identity(child.pid)
        assert saved and saved['group'] == saved['session'] == child.pid
        write('WORKER_OWNER.json', dict(child=saved, parent=identity(os.getpid()),
            wall_limit_seconds=LIMIT, scientific_fit=False, automatic_retry=False))
        child.wait(timeout=LIMIT)
        assert child.returncode == 0, 'Full-input engineering qualification failed; retain outputs'
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        if child is not None and child.poll() is None:
            actual = identity(child.pid)
            assert actual and all(actual[k] == saved[k] for k in ('pid', 'start_ticks', 'group', 'session'))
            os.killpg(saved['group'], signal.SIGTERM)
            actions.append('SIGTERM exact owned worker group')
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                actual = identity(child.pid)
                assert actual and all(actual[k] == saved[k] for k in ('pid', 'start_ticks', 'group', 'session'))
                os.killpg(saved['group'], signal.SIGKILL)
                actions.append('SIGKILL exact owned worker group')
                child.wait(timeout=5)
    finally:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        write('TERMINAL.json', dict(complete=error is None, error=error, child=saved,
            exit_code=child.returncode if child is not None else None,
            actual_worker_absent=saved is not None and identity(saved['pid']) is None,
            reaped=child is not None and child.returncode is not None,
            inclusive_seconds=time.monotonic() - started, wall_limit_seconds=LIMIT,
            child_CPU_user_seconds=usage.ru_utime, child_CPU_system_seconds=usage.ru_stime,
            child_peak_RSS_bytes=usage.ru_maxrss * 1024, signals=actions,
            scientific_fit=False, VALID_scores_computed=False, automatic_retry=False,
            partial_files_and_costs_retained=True, other_processes_changed=False))


if __name__ == '__main__':
    main()
