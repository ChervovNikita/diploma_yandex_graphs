#!/usr/bin/env python3
"""One ordinary host engineering child, with exact root release and source pins."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
EXECUTION = PHASE / 'pubmed_heart_native_numerical_qualification_execution_root_20261004_v1'


def helper_bindings():
    """Use reviewed normal helper bodies only; never import the old supervisor."""
    binding = json.loads((HERE / 'SOURCE_BINDING.json').read_text())
    row = binding['normal_supervision_helpers']
    path = PHASE / row['path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row['sha256']:
        raise RuntimeError('Qualified normal helper source changed')
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in row['functions']]
    if len(functions) != len(row['functions']):
        raise RuntimeError('Normal helper binding incomplete')
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), globals())


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--stage', required=True, choices=('cpu_bookkeeping', 'real_graph'))
    args = parser.parse_args()
    # The child gate is stdlib-only. Loading it here launches no scientific code.
    import importlib.util
    spec = importlib.util.spec_from_file_location('reviewed_pubmed_qualification_gate', HERE / 'qualify.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    release, plan = source.gate(args.root_release, args.release_sha256, args.stage)
    helper_bindings()
    output = EXECUTION / 'supervision' / args.stage / 'run01'
    require(not output.exists(), 'Fresh supervision output required; no retry')
    caps = plan['stages'][args.stage]['caps']
    profile = plan['execution_profile']
    gpu_rows = None
    if args.stage == 'real_graph':
        query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        gpu_rows = [[part.strip() for part in line.split(',')] for line in query.stdout.splitlines() if line.strip()]
        selected = [row for row in gpu_rows if row[0] == profile['CUDA_VISIBLE_DEVICES']]
        require(len(selected) == 1 and selected[0][1] == profile['gpu_name'] and int(selected[0][2]) == 81920 and int(selected[0][3]) >= 73728, 'Selected GPU identity/headroom differs')
        mem = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        require(mem['MemAvailable'] >= 40 * 1024**3, 'Fresh host memory headroom insufficient')
    output.mkdir(parents=True)
    supervisor = physical(os.getpid())
    write(output / 'SUPERVISOR_STARTED.json', {'physical_identity': supervisor, 'hostname': socket.gethostname(), 'UTC': utc(),
          'release_sha256': args.release_sha256, 'stage': args.stage, 'caps': caps, 'GPU_inventory': gpu_rows,
          'ordinary_host_execution': True, 'namespaces_created': False, 'other_jobs_mutated': False})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='' if args.stage == 'cpu_bookkeeping' else profile['CUDA_VISIBLE_DEVICES'], PYTHONPATH=profile['project_PYTHONPATH'],
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONHASHSEED='0')
    argv = [profile['interpreter_path'], '-B', str(HERE / 'qualify.py'), '--root-release', str(args.root_release.resolve()),
            '--release-sha256', args.release_sha256, '--stage', args.stage]
    child = identity = usage = None
    cap_violation = None
    observed_peak = 0
    exit_code = None
    with (output / 'CHILD.stdout.log').open('x') as stdout, (output / 'CHILD.stderr.log').open('x') as stderr:
        try:
            child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
            identity = physical(child.pid)
            require(identity['parent_PID'] == os.getpid() and identity['session'] == child.pid and identity['process_group'] == child.pid, 'Own child process identity differs')
            require(identity['namespaces'] == supervisor['namespaces'] and identity['exe'] == str(Path(profile['interpreter_path']).resolve()) and identity['cwd'] == str(REPO), 'Normal host child profile differs')
            write(output / 'CHILD_STARTED.json', {'physical_identity': identity, 'requested_argv': argv, 'stage': args.stage})
            last_status = 0.0
            while True:
                rss, members = session_rss(child.pid)
                observed_peak = max(observed_peak, rss)
                elapsed = time.monotonic() - started
                if cap_violation is None and (elapsed > caps['wall_seconds'] or rss > caps['host_RSS_bytes']):
                    cap_violation = {'kind': 'wall' if elapsed > caps['wall_seconds'] else 'host_RSS', 'elapsed_seconds': elapsed, 'RSS_bytes': rss}
                    require(physical(child.pid)['start_time_ticks'] == identity['start_time_ticks'], 'PID reuse; refusing signal')
                    os.killpg(child.pid, signal.SIGTERM)
                    termination_started = time.monotonic()
                if cap_violation is not None and time.monotonic() - termination_started > 5:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                pid, status, ru = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    usage = ru
                    exit_code = os.waitstatus_to_exitcode(status)
                    child.returncode = exit_code
                    break
                if elapsed - last_status >= 1:
                    write(output / 'SUPERVISOR_STATUS.json', {'child_identity': identity, 'elapsed_seconds': elapsed, 'RSS_bytes': rss,
                          'peak_observed_session_RSS_bytes': observed_peak, 'session_members': members, 'cap_violation': cap_violation})
                    last_status = elapsed
                time.sleep(.25)
        except BaseException as error:
            if child is not None and child.returncode is None:
                try:
                    require(identity is not None and physical(child.pid)['start_time_ticks'] == identity['start_time_ticks'], 'PID guard unavailable')
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if physical(child.pid)['start_time_ticks'] == identity['start_time_ticks']:
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            write(output / 'SUPERVISOR_FAILURE.json', {'type': type(error).__name__, 'condition': str(error), 'child_identity': identity,
                  'child_exit_code': child.returncode if child else None, 'inclusive_wall_seconds': time.monotonic() - started,
                  'peak_observed_session_RSS_bytes': observed_peak, 'automatic_retry': False, 'kernel_usage_available': False})
            raise
    run = EXECUTION / args.stage / 'run01'
    peak = json.loads((run / 'CUDA_PEAKS.json').read_text()) if (run / 'CUDA_PEAKS.json').exists() else None
    result_path = run / 'QUALIFICATION.json'
    custody_path = run / 'FINAL_CUSTODY.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    kernel_peak = int(usage.ru_maxrss * 1024) if usage else None
    if kernel_peak is not None and kernel_peak > caps['host_RSS_bytes']:
        cap_violation = {'kind': 'kernel_child_peak_RSS', 'bytes': kernel_peak}
    if peak and args.stage == 'real_graph' and any(peak[key] > caps[key] for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')):
        cap_violation = {'kind': 'CUDA_peak', 'values': peak}
    qualification = {'sha256': sha(result_path), 'bytes': result_path.stat().st_size} if result_path.exists() else None
    final_custody = {'sha256': sha(custody_path), 'bytes': custody_path.stat().st_size} if custody_path.exists() else None
    # Include descriptor reads/hashes in the one final classification sample.
    final_elapsed = time.monotonic() - started
    if cap_violation is None and final_elapsed > caps['wall_seconds']:
        cap_violation = {'kind': 'wall', 'elapsed_seconds': final_elapsed}
    success = bool(exit_code == 0 and cap_violation is None and result and result.get('status') == 'PASS' and final_custody is not None
                   and peak and peak['CUDA_observed'] == (args.stage == 'real_graph'))
    terminal = {'status': 'COMPLETE' if success else 'FAILED', 'stage': args.stage, 'supervisor_identity': supervisor, 'child_identity': identity,
                'exit_code': exit_code, 'signal': -exit_code if exit_code < 0 else None, 'cap_violation': cap_violation,
                'inclusive_supervisor_wall_seconds': final_elapsed, 'peak_observed_session_RSS_bytes': observed_peak,
                'kernel_child_peak_RSS_bytes': kernel_peak, 'kernel_child_user_CPU_seconds': usage.ru_utime, 'kernel_child_system_CPU_seconds': usage.ru_stime,
                'CUDA_peaks': peak, 'source_manifest_sha256': release['source_manifest_sha256'], 'release_sha256': args.release_sha256,
                'qualification': qualification,
                'final_custody': final_custody,
                'ordinary_host_execution': True, 'automatic_retry': False, 'scientific_fit_admitted': False, 'terminal_write_tail_measured': False}
    write(output / 'SUPERVISOR_TERMINAL.json', terminal)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
