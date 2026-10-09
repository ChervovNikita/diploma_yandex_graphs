"""Inactive adaptation of the existing BSNN normal-host subprocess/resource owner."""
from pathlib import Path
import hashlib
import argparse
import math
import json
import os
import resource
import signal
import socket
import subprocess
import time

HERE = Path(__file__).resolve().parent
P = HERE.parent
R = P.parents[1]
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
A = None
LIMIT = GPU_CAP = RSS_CAP = None


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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inside(relative):
    assert isinstance(relative, str) and not Path(relative).is_absolute()
    value = (P / relative).resolve()
    assert value.is_relative_to(P) and value != P and not value.is_relative_to(HERE)
    return value


def bound(row):
    value = inside(row['path']).resolve(strict=True)
    assert value.stat().st_size == row['bytes'] and sha(value) == row['sha256']
    return value


def sealed(root, expected):
    seal = read(root / 'SEAL.json')
    assert seal['source_only'] is True and seal['execution_enabled'] is False
    assert sha(root / 'MANIFEST.json') == seal['manifest_sha256'] == expected
    for row in read(root / 'MANIFEST.json')['files']:
        file = (root / row['path']).resolve(strict=True)
        assert file.is_relative_to(root) and file.stat().st_size == row['bytes'] and sha(file) == row['sha256']


def gate(release, release_sha256, execute=False):
    assert execute is True and sha(release) == release_sha256
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    sealed(HERE, read(HERE / 'SEAL.json')['manifest_sha256'])
    for row in pins['files']:
        file = (P / row['path']).resolve(strict=True)
        assert file.is_relative_to(P) and file.stat().st_size == row['bytes'] and sha(file) == row['sha256']
    cfg = read(release)
    template = read(HERE / 'ROOT_RELEASE_TEMPLATE_DISABLED.json')
    assert set(cfg) == set(template) and cfg['schema'] == template['schema']
    assert cfg['enabled'] is True and cfg['release_owner'] == 'root' and cfg['mode'] in ('qualify', 'full_three_seed')
    assert cfg['owner_manifest_sha256'] == read(HERE / 'SEAL.json')['manifest_sha256']
    assert cfg['host'] == 'anogena-2-0' and cfg['GPU_uuid'] == GPU
    assert socket.gethostname() == cfg['host'] and Path.cwd() == R and query('--query-gpu=uuid') == [GPU]
    activation = inside(cfg['activation_relative'])
    assert activation.is_dir() and Path(release).resolve(strict=True).parent == activation
    output = inside(cfg['output_relative'])
    assert not output.exists() and not output.is_relative_to(activation)
    forbidden = [P / pins[k]['directory'] for k in ('qualifier','richer_wrapper')]
    forbidden += [P / row for row in pins['immutable_directories']]
    assert not any(output.is_relative_to(v) or activation.is_relative_to(v) for v in forbidden)
    commit = cfg['execution_source_commit']
    assert isinstance(commit, str) and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True,timeout=10).strip() == commit
    for directory, digest in [(HERE.name,cfg['owner_manifest_sha256'])]+[(pins[k]['directory'],pins[k]['manifest_sha256']) for k in ('qualifier','richer_wrapper')]:
        blob = subprocess.check_output(['git','show',commit+':experiments_iclr/postsubmission_20260930/'+directory+'/MANIFEST.json'],cwd=R,timeout=10)
        assert hashlib.sha256(blob).hexdigest() == digest
    assert set(cfg['resources']) == set(template['resources'])
    assert all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in cfg['resources'].values())
    assert cfg['resources']['minimum_fresh_GPU_bytes'] >= cfg['resources']['own_GPU_cap_bytes']
    runtime_path = bound(cfg['runtime_metadata'])
    assert cfg['role_archive_relative'] == template['role_archive_relative'] and inside(cfg['role_archive_relative']).is_file()
    worker_path = bound(cfg['worker_release'])
    worker = read(worker_path)
    target = pins['qualifier' if cfg['mode']=='qualify' else 'richer_wrapper']
    root = P / target['directory']
    sealed(root,target['manifest_sha256'])
    assert worker['enabled'] is True and worker['source_seal_sha256'] == target['seal_file_sha256']
    assert worker['execution_source_commit'] == commit and worker['output_directory'] == str(output)
    assert worker['roles_sha256'] == pins['roles_sha256'] and worker['role_metadata_sha256'] == pins['role_metadata_sha256']
    assert worker['expected_runtime_versions'] == pins['runtime_versions']
    receipt = bound(cfg['root_owner_receipt'])
    owner = read(receipt)
    assert owner['released'] is True and owner['mode'] == cfg['mode'] and owner['resources'] == cfg['resources']
    assert owner['predecessor_groups_and_cuda_absent'] is True
    assert owner['execution_source_commit'] == commit and owner['output_directory'] == str(output)
    if cfg['mode'] == 'qualify':
        assert cfg['qualification_adoption'] is None
    else:
        assert worker['owner_release_receipt'] == dict(path=str(receipt),sha256=cfg['root_owner_receipt']['sha256'])
        adoption = read(bound(cfg['qualification_adoption']))
        assert adoption['adopted_by_root'] is True and adoption['richer_source_seal_sha256'] == pins['richer_wrapper']['seal_file_sha256']
        assert adoption['qualifier_source_seal_sha256'] == pins['qualifier']['seal_file_sha256']
        report = read(bound(adoption['result']))
        assert report['status'] == 'complete' and report['qualification_passed'] is True
        assert report['model_parameter_buffer_state_exact'] is report['optimizer_state_exact'] is report['selected_streams_exact'] is True
        assert report['protocol']['seed'] == 7409 and report['native_args'] == read(root / 'PROTOCOL.json')['native_args']
        counts = report['counters']
        assert counts['TRAIN_forwards_completed'] == counts['backwards_completed'] == counts['Adam_steps_completed'] == 1
        assert counts['evaluation_forwards_completed'] == counts['reconstruction_forwards_completed'] == 4
        assert counts['native_SO_sampler_calls_completed'] == counts['native_sampler_array_transfers_completed'] == 36
        assert report['peak_CUDA_reserved_bytes'] <= cfg['resources']['own_GPU_cap_bytes']
        assert adoption['actual_parent_worker_groups_and_cuda_absent'] is True
        assert adoption['actual_parameters_samples_peaks_imported_providers_and_failures_reviewed'] is True
    program = root / target['program']
    assert sha(program) == target['program_sha256']
    return cfg, activation, runtime_path, program, worker_path


def main():
    global A, LIMIT, GPU_CAP, RSS_CAP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--release-sha256')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True, numerical_host_or_role_access=False)))
        return
    assert args.release and args.release_sha256
    started = time.monotonic()
    owner_cfg, A, runtime_path, program, release = gate(args.release,args.release_sha256,True)
    runtime = read(runtime_path)
    cfg = read(release)
    limits = owner_cfg['resources']
    LIMIT, GPU_CAP, RSS_CAP = limits['wall_limit_seconds'], limits['own_GPU_cap_bytes'], limits['own_RSS_cap_bytes']
    workload = dict(engineering_updates=1,full_model_eval_calls=8) if owner_cfg['mode']=='qualify' else dict(full_task_fits=3)
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
        assert free_bytes >= limits['minimum_fresh_GPU_bytes'], 'Insufficient current free memory; no numerical child or retry'
        write('RESOURCE_ADMISSION.json', dict(fresh_free_GPU_bytes=free_bytes,
            own_GPU_cap_bytes=GPU_CAP, own_RSS_cap_bytes=RSS_CAP,
            runtime_metadata=owner_cfg['runtime_metadata'], root_owner_receipt=owner_cfg['root_owner_receipt'],
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
            wall_limit_seconds=LIMIT, mode=owner_cfg['mode'], workload=workload, automatic_retry=False,
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
            signals=actions, mode=owner_cfg['mode'], workload=workload, comparative_outcomes_opened=False,
            TEST_truth_accessed=False, automatic_retry=False,
            partial_files_and_costs_retained=True, other_processes_changed=False))


if __name__ == '__main__':
    main()
