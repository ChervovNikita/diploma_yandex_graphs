"""Once-only normal-host owner for one authentic full-input BSNN engineering attempt."""
from pathlib import Path
import hashlib
import json
import os
import resource
import signal
import socket
import subprocess
import time

A = Path(__file__).resolve().parent
P = A.parent
R = P.parents[1]
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
LIMIT = 1800
GPU_CAP = 24 * 1024**3
RSS_CAP = 16 * 1024**3


def write(name, value):
    with (A / name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def identity(pid):
    try:
        raw = Path('/proc', str(pid), 'stat').read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), group=int(fields[2]),
                session=int(fields[3]), state=fields[0],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def same(actual, saved):
    return actual is not None and all(actual[k] == saved[k]
        for k in ('pid', 'start_ticks', 'group', 'session', 'boot_id'))


def query(fields):
    return subprocess.check_output(['nvidia-smi', fields, '--format=csv,noheader,nounits'],
        text=True, timeout=10).splitlines()


def main():
    started = time.monotonic()
    assert socket.gethostname() == 'anogena-2-0' and Path.cwd() == R
    assert query('--query-gpu=uuid') == [GPU]
    runtime = json.loads((P / 'staged_label_posterior_native_metadata_root_20261008_v1/RUNTIME.json').read_text())
    program = P / 'bsnn_cayley_d2_full_input_engineering_qualifier_source_20261009_v2/qualifier.py'
    assert hashlib.sha256(program.read_bytes()).hexdigest() == 'e9d5c7a6e5c1ede2ce6f26bab3a2ed8f2ece7385bf801e0add3aad4ab47c07a7'
    release = A / 'RELEASE.json'
    cfg = json.loads(release.read_text())
    assert cfg['source_seal_sha256'] == '9aee42829c3afef712a3b8c7b705666e9eb7cc4ad7cbad6ff5858eb9e690124d'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU,
        PYTHONPATH=os.pathsep.join([str(P / 'private_sheaf_dependency_overlay_20261009_v1'), *runtime['PYTHONPATH']]),
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    env.pop('PYTHONHOME', None)
    child = saved = None
    error = None
    actions = []
    max_gpu = max_rss = 0
    try:
        free = query('--query-gpu=uuid,memory.free')
        assert len(free) == 1 and free[0].split(',')[0].strip() == GPU
        free_bytes = int(free[0].split(',')[1].strip()) * 1024**2
        assert free_bytes >= GPU_CAP, 'Insufficient current free memory; no numerical child or retry'
        write('RESOURCE_ADMISSION.json', dict(fresh_free_GPU_bytes=free_bytes,
            own_GPU_cap_bytes=GPU_CAP, own_RSS_cap_bytes=RSS_CAP,
            runtime_evidence='bsnn_cayley_runtime_provider_probe_root_20261009_v1/RUNTIME.json',
            other_processes_changed=False))
        with (A / 'WORKER.log').open('xb') as log:
            child = subprocess.Popen([runtime['python'], '-B', str(program),
                '--execute', '--release', str(release),
                '--roles', str(P / 'private_sheaf_train_valid_roles_allocation_root_20261009_v1/roles.npz'),
                '--output', cfg['output_directory']],
                cwd=R, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        saved = identity(child.pid)
        assert saved and saved['group'] == saved['session'] == child.pid
        write('WORKER_OWNER.json', dict(child=saved, parent=identity(os.getpid()),
            wall_limit_seconds=LIMIT, engineering_updates=1, automatic_retry=False,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip(),
            root_release_sha256=hashlib.sha256(release.read_bytes()).hexdigest()))
        while child.poll() is None:
            assert time.monotonic() - started < LIMIT, 'Full-input engineering wall limit exceeded'
            current = identity(child.pid)
            if current is not None:
                assert same(current, saved), 'Owned process identity changed'
                try:
                    status = Path('/proc', str(child.pid), 'status').read_text()
                except FileNotFoundError:
                    assert child.poll() is not None, 'Owned process observation disappeared while active'
                    continue
                rss = next((int(line.split()[1]) * 1024 for line in status.splitlines()
                            if line.startswith('VmRSS:')), 0)
                max_rss = max(max_rss, rss)
                assert rss <= RSS_CAP, 'Owned RSS cap exceeded'
            gpu_bytes = 0
            for row in query('--query-compute-apps=gpu_uuid,pid,used_memory'):
                parts = [v.strip() for v in row.split(',')]
                if len(parts) == 3 and parts[0] == GPU and parts[1] == str(child.pid):
                    assert parts[2].isdigit(), 'Owned GPU memory observation unavailable'
                    gpu_bytes += int(parts[2]) * 1024**2
            max_gpu = max(max_gpu, gpu_bytes)
            assert gpu_bytes <= GPU_CAP, 'Owned GPU cap exceeded'
            assert (A / 'WORKER.log').stat().st_size < 8 * 1024**2, 'Owned log cap exceeded'
            time.sleep(5)
        child.wait(timeout=15)
        assert child.returncode == 0, 'Full-input engineering failed; preserve the attempt and numerical evidence'
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        if child is not None and child.poll() is None:
            actual = identity(child.pid)
            assert same(actual, saved), 'Refuse signals without exact owned identity'
            os.killpg(saved['group'], signal.SIGTERM)
            actions.append('SIGTERM exact owned worker group')
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                assert same(identity(child.pid), saved)
                os.killpg(saved['group'], signal.SIGKILL)
                actions.append('SIGKILL exact owned worker group')
                child.wait(timeout=5)
    finally:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        cuda_absent = saved is not None and not any(
            len(parts) > 1 and parts[1].strip() == str(saved['pid'])
            for parts in (row.split(',') for row in query('--query-compute-apps=gpu_uuid,pid,used_memory')))
        write('TERMINAL.json', dict(complete=error is None, error=error, child=saved,
            exit_code=child.returncode if child is not None else None,
            actual_worker_absent=saved is not None and identity(saved['pid']) is None,
            actual_worker_CUDA_absent=cuda_absent,
            reaped=child is not None and child.returncode is not None,
            inclusive_seconds=time.monotonic() - started, wall_limit_seconds=LIMIT,
            child_CPU_user_seconds=usage.ru_utime, child_CPU_system_seconds=usage.ru_stime,
            child_peak_RSS_bytes=usage.ru_maxrss * 1024,
            max_sampled_own_GPU_bytes=max_gpu, max_sampled_own_RSS_bytes=max_rss,
            signals=actions, engineering_updates=1, comparative_outcomes_opened=False,
            TEST_truth_accessed=False, automatic_retry=False,
            partial_files_and_costs_retained=True, other_processes_changed=False))


if __name__ == '__main__':
    main()
