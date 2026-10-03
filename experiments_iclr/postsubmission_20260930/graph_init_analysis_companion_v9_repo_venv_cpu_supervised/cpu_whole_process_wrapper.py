"""Source-only v9 CPU supervisor; wrapper imports stdlib only.

Run only through the exact SSH route documented in PLAN.md, after a separate
root release and fresh source review. This wrapper neither transfers records
nor creates the canonical audit output. A separate preflight imports NumPy only
for runtime metadata; the numerical audit is one directly owned wait4 child.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import select
import signal
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
REMOTE = REPO / 'experiments_iclr/postsubmission_20260930'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
GPU_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
OUTPUT = PHASE / 'graph_init_analysis_companion_v9_source_valid_execution_root_v1/run_v1'
WRAPPER_OUTPUT = PHASE / 'graph_init_analysis_companion_v9_cpu_supervision_root_v1/run_v1'
CPU_PYTHON = REPO / '.venv/bin/python'
EXPECTED_RUNTIME = {'python': '3.11.14', 'numpy': '1.26.4'}
PREFLIGHT_CAP = 30
INVENTORY_SHA = 'c1292ae389e4f8294334d70d848530f20c01a89bf49b93f233cb180d777976aa'
WALL_CAP = 1200
FORECAST = {'wall_cap_seconds': WALL_CAP, 'cpu_threads': 1, 'working_memory_bytes': 1 << 30,
            'output_free_bytes': 64 << 20, 'GPU_computation': False, 'measured': False}


class DeadlineExceeded(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def confined(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.is_relative_to(PHASE),
            'Absolute exact-phase path required')
    require(not any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(PHASE)),
            'Symlink in exact-phase path')
    return path


def sha(path):
    digest = hashlib.sha256()
    with confined(path).open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    return {'path': str(confined(path)), 'sha256': sha(path)}


def read(path):
    path = confined(path)
    require(path.suffix == '.json', 'Only JSON metadata is parsed')
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def bound(record):
    require(isinstance(record, dict) and set(record) == {'path', 'sha256'}, 'Exact metadata descriptor required')
    path = confined(record['path'])
    require(path.suffix == '.json' and sha(path) == record['sha256'], 'Metadata descriptor differs')
    return path


def write(path, value):
    with confined(path).open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())


def source_guard(manifest_record, seal_record):
    manifest_path, seal_path = bound(manifest_record), bound(seal_record)
    require(manifest_path == HERE / 'MANIFEST.json' and seal_path == HERE / 'SEAL.json', 'Exact v9 source packet required')
    manifest, seal = read(manifest_path), read(seal_path)
    require(seal['manifest_sha256'] == sha(manifest_path) and seal['source_only'] is True
            and seal['execution_authorized'] is False, 'Source-only seal differs')
    names = []
    for record in manifest['payload']:
        relative = Path(record['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source payload')
        path = confined(HERE / relative)
        require(sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], 'Source packet changed')
        names.append(str(relative))
    require(len(names) == len(set(names)) and
            {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
             and p.name not in ('MANIFEST.json', 'SEAL.json')} == set(names), 'Source inventory differs')


def inventory_guard():
    inventory_path = HERE / 'CLIENT_METADATA_UPLOAD_INVENTORY.json'
    require(sha(inventory_path) == INVENTORY_SHA, 'Original v5 inventory bytes changed')
    inventory = read(inventory_path)
    client, existing = inventory['client_upload_files'], inventory['required_existing_remote_dependencies']
    require(inventory['client_upload_count'] == len(client) == 149 and len(existing) == 726
            and inventory['execution_or_upload_authorized'] is False and inventory['client_origin_explicit'] is True
            and inventory['remote_phase_root'] == str(REMOTE), 'Exact original 149+726 custody inventory required')
    bindings = []
    for record in client + existing:
        relative = Path(record['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe inventory path')
        path = confined(PHASE / relative)
        require(sha(path) == record['sha256'], 'Original inventory hash differs: ' + str(path))
        if 'bytes' in record:
            require(type(record['bytes']) is int and path.stat().st_size == record['bytes'], 'Inventory byte length differs')
        bindings.append({'path': str(path), 'sha256': record['sha256'], 'origin': record['origin']})
    require(len({row['path'] for row in bindings}) == 875, 'Duplicate immutable inventory path')
    return bindings


def request_guard(path, record):
    require(descriptor(path) == record, 'Root wrapper request changed')
    request = read(path)
    fields = {'schema', 'execution_authorized', 'authorized_phase', 'ssh_destination', 'ssh_port', 'repository',
              'gpu_uuid', 'companion_admission', 'source_manifest', 'source_seal', 'audit_output', 'wrapper_output',
              'resource_forecast', 'cpu_python', 'runtime_preflight', 'automatic_retry', 'filesystem_isolation'}
    require(set(request) == fields and request['schema'] == 'graph-init-companion-v9-cpu-wrapper-request-v1'
            and request['execution_authorized'] is True and request['authorized_phase'] == 'source_VALID_audit',
            'Separate exact root wrapper release required')
    require(request['ssh_destination'] == LOGIN and type(request['ssh_port']) is int and request['ssh_port'] == 2222
            and request['repository'] == str(REPO) and request['gpu_uuid'] == GPU_UUID
            and request['audit_output'] == str(OUTPUT) and request['wrapper_output'] == str(WRAPPER_OUTPUT)
            and request['resource_forecast'] == FORECAST and request['cpu_python'] == str(CPU_PYTHON)
            and request['runtime_preflight'] == EXPECTED_RUNTIME and request['automatic_retry'] is False
            and request['filesystem_isolation'] is False, 'Route, resources, scope or once-only contract differs')
    source_guard(request['source_manifest'], request['source_seal'])
    admission = read(bound(request['companion_admission']))
    require(admission['schema'] == 'graph-init-companion-v9-source-valid-admission-v1'
            and admission['execution_authorized'] is True and admission['authorized_phase'] == 'source_VALID_audit'
            and admission['source_manifest'] == request['source_manifest'] and admission['source_seal'] == request['source_seal']
            and admission['output_root'] == str(OUTPUT) and admission['cpu_only'] is True
            and all(admission[key] is False for key in ('Tensor_GPU_execution_authorized', 'heldout_access_authorized',
                'final_report_access_authorized', 'serialized_checkpoint_replay_authorized')), 'Companion admission differs')
    review = read(bound(admission['independent_source_review']))
    require(set(review) == {'schema', 'approved', 'source_manifest', 'source_seal', 'source_VALID_only',
                           'serialized_checkpoint_replay_missing_disclosed'}
            and review['schema'] == 'graph-init-companion-v9-independent-source-review-v1'
            and review['approved'] is True and review['source_manifest'] == request['source_manifest']
            and review['source_seal'] == request['source_seal'] and review['source_VALID_only'] is True
            and review['serialized_checkpoint_replay_missing_disclosed'] is True, 'Fresh v9 source review required')
    return request, admission, inventory_guard()


def route_guard(out, label, deadline):
    # The root request and exact reviewed SSH argv bind the gateway route.
    # SSH_CONNECTION records the server-side forwarded endpoint, not gateway port 2222.
    require(os.environ.get('ANOGENA_WRAPPER_SSH_DESTINATION') == LOGIN
            and os.environ.get('ANOGENA_WRAPPER_SSH_PORT') == '2222'
            and len(os.environ.get('SSH_CONNECTION', '').split()) == 4, 'Exact reviewed live SSH launch required')
    stdout, stderr = out / (label + '_gpu_uuid.stdout'), out / (label + '_gpu_uuid.stderr')
    with stdout.open('xb') as so, stderr.open('xb') as se:
        result = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                stdin=subprocess.DEVNULL, stdout=so, stderr=se,
                                timeout=max(0.001, min(10, deadline - time.monotonic())), check=False)
    require(result.returncode == 0 and stdout.read_text().splitlines() == [GPU_UUID], 'Wrong sole-GPU account inventory')
    return {'ssh_destination': LOGIN, 'ssh_port': 2222, 'SSH_CONNECTION': os.environ['SSH_CONNECTION'],
            'repository': str(REPO), 'gpu_uuid': GPU_UUID, 'query_only_no_GPU_computation': True,
            'stdout': descriptor(stdout), 'stderr': descriptor(stderr)}


def runtime_preflight_guard(out, env, deadline):
    """Import/version/path checks only in the exact repository CPU interpreter."""
    require(CPU_PYTHON.is_file() and os.access(CPU_PYTHON, os.X_OK),
            'Authorized repository venv interpreter unavailable')
    code = """import json, os, platform, sys
import numpy as np
value = {'python': platform.python_version(), 'numpy': np.__version__,
         'executable': os.path.abspath(sys.executable), 'prefix': sys.prefix,
         'numpy_file': np.__file__,
         'Torch_imported': any(name in sys.modules for name in ('torch', 'torch_geometric')),
         'scientific_payloads_opened': False}
print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)
"""
    stdout = out / 'runtime_preflight.stdout.json'
    stderr = out / 'runtime_preflight.stderr.log'
    argv = [str(CPU_PYTHON), '-B', '-c', code]
    started = time.monotonic()
    with stdout.open('xb') as so, stderr.open('xb') as se:
        require(stat.S_ISREG(os.fstat(so.fileno()).st_mode) and stat.S_ISREG(os.fstat(se.fileno()).st_mode),
                'Regular-file runtime preflight stdout/stderr required')
        process = subprocess.run(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                 stdout=so, stderr=se, check=False,
                                 timeout=max(0.001, min(PREFLIGHT_CAP, deadline - time.monotonic())))
    process_path = out / 'RUNTIME_PREFLIGHT_PROCESS.json'
    write(process_path, {'schema': 'graph-init-companion-v9-runtime-preflight-process-v1', 'UTC': utc(),
          'argv': argv, 'cwd': str(REPO), 'exit_code': process.returncode,
          'wall_seconds': time.monotonic() - started, 'wall_cap_seconds': PREFLIGHT_CAP,
          'expected_runtime': EXPECTED_RUNTIME, 'stdout': descriptor(stdout), 'stderr': descriptor(stderr),
          'scientific_payloads_opened': False, 'GPU_computation': False})
    require(process.returncode == 0, 'Repository venv NumPy preflight exited unsuccessfully')
    metadata = read(stdout)
    require(set(metadata) == {'python', 'numpy', 'executable', 'prefix', 'numpy_file',
                             'Torch_imported', 'scientific_payloads_opened'}
            and {name: metadata[name] for name in ('python', 'numpy')} == EXPECTED_RUNTIME
            and metadata['executable'] == str(CPU_PYTHON) and metadata['prefix'] == str(REPO / '.venv')
            and isinstance(metadata['numpy_file'], str) and metadata['numpy_file']
            and metadata['Torch_imported'] is False and metadata['scientific_payloads_opened'] is False,
            'Exact frozen repository CPU NumPy runtime preflight required')
    return {'process': descriptor(process_path), 'metadata': metadata,
            'stdout': descriptor(stdout), 'stderr': descriptor(stderr)}


def deadline_alarm(signum, frame):
    raise DeadlineExceeded('Prospective 1200-second whole-process wall deadline reached')


def reap_owned(child, block):
    # Blocking reap is used only after pidfd signals exit. Mask the alarm while
    # recording that reap, so deadline delivery cannot lose the exit identity.
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        pid, status, usage = os.wait4(child.pid, 0 if block else os.WNOHANG)
        if not pid:
            return None
        require(pid == child.pid, 'Unexpected owned child reap')
        child.returncode = os.waitstatus_to_exitcode(status)
        child.wrapper_usage = {'exit_code': child.returncode, 'user_seconds': usage.ru_utime,
                               'system_seconds': usage.ru_stime, 'peak_resident_bytes': int(usage.ru_maxrss) * 1024,
                               'linux_ru_maxrss_unit': 'KiB'}
        return child.wrapper_usage
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


def terminate_owned(child, pidfd):
    if child is None or child.returncode is not None:
        return {'signal_sent': False, 'reaped': child is not None and child.returncode is not None}, getattr(child, 'wrapper_usage', None)
    # Until this wrapper reaps its direct child, that PID cannot be recycled.
    # pidfd is preferred; the fallback handles a pidfd_open failure after Popen.
    sent = False
    try:
        if pidfd is not None:
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        else:
            os.kill(child.pid, signal.SIGKILL)
        sent = True
    except ProcessLookupError:
        pass
    cleanup_deadline = time.monotonic() + 5
    usage = None
    while time.monotonic() < cleanup_deadline:
        usage = reap_owned(child, False)
        if usage is not None:
            break
        time.sleep(0.02)
    return {'signal_sent': sent, 'signal': 'SIGKILL', 'target': 'exact unreaped direct child only',
            'reaped': usage is not None, 'cleanup_cap_seconds': 5}, usage


def successful_audit_guard(request):
    require(not (OUTPUT / 'AUDIT_FAILED.json').exists(), 'Audit failure marker blocks success')
    freeze_path = OUTPUT / 'AUDIT_FREEZE.json'
    freeze = read(freeze_path)
    require(freeze['schema'] == 'graph-init-companion-v9-source-valid-freeze-v1' and freeze['completed'] is True
            and freeze['admission'] == request['companion_admission'] and freeze['source_cells'] == 30
            and freeze['heldout_labels_opened'] is False and freeze['serialized_checkpoint_inference_replayed'] is False,
            'Successful canonical audit freeze required')
    require(bound(freeze['audit']) == OUTPUT / 'SOURCE_VALID_AUDIT.json', 'Canonical numerical receipt differs')
    require({str(bound(row)) for row in freeze['payload']} ==
            {str(OUTPUT / 'AUDIT_STARTED.json'), str(OUTPUT / 'SOURCE_VALID_AUDIT.json')}
            and len(freeze['payload']) == 2, 'Audit freeze payload differs')
    return descriptor(freeze_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    require(PHASE == REMOTE and Path.cwd() == REPO and platform.system() == 'Linux', 'Exact remote Linux repository required')
    out = confined(WRAPPER_OUTPUT)
    require(not out.exists(), 'Once-only separate wrapper receipt directory required')
    out.mkdir(parents=True, exist_ok=False)
    started, started_utc = time.monotonic(), utc()
    deadline = started + WALL_CAP
    child = pidfd = usage = request_record = None
    child_identity = route_before = route_after = audit_freeze = runtime_preflight = None
    payload = None
    completed, error, termination = False, None, None
    signal.signal(signal.SIGALRM, deadline_alarm)
    signal.setitimer(signal.ITIMER_REAL, WALL_CAP)
    try:
        request_record = descriptor(args.request)
        write(out / 'START.json', {'schema': 'graph-init-companion-v9-cpu-wrapper-start-v1', 'UTC': started_utc,
              'request': request_record, 'wall_cap_seconds': WALL_CAP, 'resource_forecast': FORECAST,
              'canonical_audit_output_created_by_wrapper': False, 'automatic_retry': False})
        require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal') and hasattr(os, 'wait4'),
                'Linux owned-child identity and accounting APIs required')
        request, admission, inventory_bindings = request_guard(args.request, request_record)
        require(not OUTPUT.exists(), 'Canonical audit output already occupied; no retry')
        route_before = route_guard(out, 'before', deadline)
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))) * 1024
        disk = os.statvfs(REPO)
        resources = {'available_memory_bytes': available, 'logical_cpu_count': os.cpu_count(),
                     'available_filesystem_bytes': disk.f_bavail * disk.f_frsize,
                     'memory_forecast_is_not_an_address_space_limit': True}
        require(available >= FORECAST['working_memory_bytes'] and resources['logical_cpu_count'] >= 1
                and resources['available_filesystem_bytes'] >= FORECAST['output_free_bytes'], 'Prospective CPU resources insufficient')
        env = os.environ.copy()
        env.update({'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
        runtime_preflight = runtime_preflight_guard(out, env, deadline)
        argv = [str(CPU_PYTHON), '-B', str(HERE / 'audit_source_valid.py'),
                '--admission', str(bound(request['companion_admission'])), '--output', str(OUTPUT)]
        write(out / 'COMMAND.json', {'argv': argv, 'cwd': str(REPO), 'request': request_record,
              'source_manifest': request['source_manifest'], 'source_seal': request['source_seal'],
              'resources': resources, 'route': route_before, 'runtime_preflight': runtime_preflight,
              'inventory_bindings': inventory_bindings,
              'CPU_environment_overrides': {key: env[key] for key in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                   'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')},
              'filesystem_isolation': False, 'process_group_termination': False})
        with (out / 'stdout.log').open('xb') as so, (out / 'stderr.log').open('xb') as se:
            require(stat.S_ISREG(os.fstat(so.fileno()).st_mode) and stat.S_ISREG(os.fstat(se.fileno()).st_mode),
                    'Regular-file child stdout/stderr required')
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            try:
                child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=so, stderr=se,
                                         preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask))
                pidfd = os.pidfd_open(child.pid, 0)
                fields = Path('/proc/' + str(child.pid) + '/stat').read_text().rsplit(')', 1)[1].split()
                require(int(fields[1]) == os.getpid(), 'New child parent identity differs')
                child_identity = {'pid': child.pid, 'parent_pid': os.getpid(), 'start_ticks': int(fields[19]),
                                  'identity_handle': 'pidfd retained until own wait4 reap'}
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            write(out / 'CHILD_STARTED.json', {'UTC': utc(), **child_identity})
            ready, _, _ = select.select([pidfd], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                raise DeadlineExceeded('Owned child reached whole-process deadline')
            usage = reap_owned(child, True)
        require(usage['exit_code'] == 0, 'Owned child exited unsuccessfully')
        require(usage['peak_resident_bytes'] <= FORECAST['working_memory_bytes'], 'Observed child RSS exceeds reviewed forecast')
        require(request_guard(args.request, request_record) == (request, admission, inventory_bindings),
                'Root/source/review/original inventory custody changed during process')
        route_after = route_guard(out, 'after', deadline)
        audit_freeze = successful_audit_guard(request)
        payload = [descriptor(path) for path in sorted(out.iterdir()) if path.is_file()]
        require(time.monotonic() <= deadline, 'Post-process custody checks exceeded whole-process cap')
        completed = True
    except BaseException as failure:
        error = {'type': type(failure).__name__, 'message': str(failure)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if not completed:
            termination, cleanup_usage = terminate_owned(child, pidfd)
            if cleanup_usage is not None:
                usage = cleanup_usage
        if pidfd is not None:
            os.close(pidfd)
    if payload is None:
        payload = [descriptor(path) for path in sorted(out.iterdir()) if path.is_file()]
    seconds = time.monotonic() - started
    if completed and seconds > WALL_CAP:
        completed = False
        error = {'type': 'DeadlineExceeded', 'message': 'Final wrapper accounting exceeded whole-process deadline'}
    write(out / '.TERMINAL.pending.json', {'schema': 'graph-init-companion-v9-cpu-wrapper-terminal-v1', 'UTC': utc(),
          'completed': completed, 'request': request_record, 'child_identity': child_identity, 'child_usage': usage,
          'child_exit_code': None if usage is None else usage['exit_code'], 'timed_out': error is not None
              and error['type'] == 'DeadlineExceeded', 'error': error, 'termination': termination,
          'wall_cap_seconds': WALL_CAP, 'whole_supervised_seconds': seconds,
          'within_whole_cap': seconds <= WALL_CAP, 'cost_scope': 'Wrapper START through admission/source/inventory/route/resource checks and repository NumPy import/version preflight, entire owned audit child and final custody checks; failure cleanup up to five seconds is included; terminal serialization is excluded.',
          'resource_forecast': FORECAST, 'runtime_preflight': runtime_preflight,
          'route_before': route_before, 'route_after': route_after,
          'audit_freeze': audit_freeze, 'canonical_audit_output_exists': OUTPUT.exists(),
          'canonical_audit_output_created_by_wrapper': False, 'filesystem_isolation': False,
          'Torch_imported_or_GPU_computation': False, 'automatic_retry': False, 'payload': payload})
    os.link(out / '.TERMINAL.pending.json', out / 'TERMINAL.json')
    if not completed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
