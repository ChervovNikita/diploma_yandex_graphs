"""Existing owned CPU wait/TERM/KILL pattern; 300 active + 10 cleanup seconds."""
import argparse, hashlib, json, os, resource, signal, socket, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTIVE, CLEANUP, HARD = 300, 10, 310

def ticks(pid):
    return int(Path('/proc/' + str(pid) + '/stat').read_text().rsplit(') ', 1)[1].split()[19])

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'); os.replace(temporary, path)

def interrupted(signum, frame): raise TimeoutError('Owned CPU parent received signal ' + str(signum))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True, help='Actual worker interpreter set by root')
    parser.add_argument('--dependency-path', type=Path, action='append', default=[], help='Explicit root-set PYTHONPATH component; repeat if needed')
    for name in ('public-root', 'adapter-root', 'interface-root', 'allocation-root', 'train', 'valid', 'geometry', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic(); os.umask(0o077)
    if sys.platform != 'linux': raise RuntimeError('Linux owned-process identities/RSS units required')
    args.output = args.output.absolute(); args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    child = None; identity = None; code = None; status = 'setup_failed'
    before = resource.getrusage(resource.RUSAGE_CHILDREN)

    def stop_owned():
        if child is None or child.poll() is not None: return
        if identity is None or ticks(child.pid) != identity['worker_start_ticks'] or os.getpgid(child.pid) != child.pid:
            raise RuntimeError('Owned child identity/group differs; no signal sent')
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=max(0, min(5, HARD - (time.monotonic() - started))))
        except subprocess.TimeoutExpired:
            if ticks(child.pid) != identity['worker_start_ticks'] or os.getpgid(child.pid) != child.pid:
                raise RuntimeError('Owned child identity/group changed; no further signal sent')
            os.killpg(child.pid, signal.SIGKILL)
            try: child.wait(timeout=max(0, HARD - (time.monotonic() - started)))
            except subprocess.TimeoutExpired: pass

    try:
        manifest = json.loads((HERE / 'MANIFEST.json').read_text())
        for row in manifest['files']:
            path = (HERE / row['path']).resolve(strict=True)
            if not path.is_relative_to(HERE) or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('Worker source changed: ' + row['path'])
        if not args.python.is_file(): raise FileNotFoundError(args.python)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=os.pathsep.join(str(path.absolute()) for path in args.dependency_path),
            OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
        env.pop('PYTHONHOME', None)
        command = [str(args.python.absolute()), '-B', str(HERE / 'check.py')]
        for name in ('public_root', 'adapter_root', 'interface_root', 'allocation_root', 'train', 'valid', 'geometry'):
            command.extend(['--' + name.replace('_', '-'), str(getattr(args, name).absolute())])
        command.extend(['--output', str(args.output / 'work')])
        write(args.output / 'INVOCATION.json', {'command': command, 'dependency_paths': [str(path.absolute()) for path in args.dependency_path],
            'worker_source_manifest_sha256': sha(HERE / 'MANIFEST.json'), 'hostname': socket.gethostname(), 'CUDA_VISIBLE_DEVICES': '',
            'threads': 2, 'active_seconds': ACTIVE, 'cleanup_seconds': CLEANUP, 'hard_seconds': HARD,
            'automatic_retry': False, 'GPU_execution': False, 'resource_supervisor_or_detector_interaction': False})
        with (args.output / 'check.log').open('x') as log:
            child = subprocess.Popen(command, cwd=HERE, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
            identity = {'parent_PID': os.getpid(), 'parent_start_ticks': ticks(os.getpid()), 'worker_PID': child.pid,
                'worker_start_ticks': ticks(child.pid), 'active_seconds': ACTIVE, 'cleanup_seconds': CLEANUP, 'hard_seconds': HARD}
            write(args.output / 'OWNER.json', identity)
            try:
                code = child.wait(timeout=max(0, ACTIVE - (time.monotonic() - started)))
                status = 'complete' if code == 0 else 'worker_failed'
            except subprocess.TimeoutExpired:
                status = 'active_timeout'; stop_owned(); code = child.poll()
    except BaseException as error:
        status = 'parent_failure' if child is not None else 'setup_failed'
        write(args.output / 'PARENT_FAILURE.json', {'complete': False, 'error_type': type(error).__name__, 'error': str(error), 'automatic_retry': False})
        try: stop_owned()
        except BaseException as cleanup_error:
            write(args.output / 'CLEANUP_FAILURE.json', {'error_type': type(cleanup_error).__name__, 'error': str(cleanup_error), 'automatic_retry': False})
        code = child.poll() if child is not None else None
    after = resource.getrusage(resource.RUSAGE_CHILDREN); elapsed = time.monotonic() - started
    candidate_path = args.output / 'work/CANDIDATE.json'
    candidate = json.loads(candidate_path.read_text()) if status == 'complete' and candidate_path.is_file() else None
    complete = status == 'complete' and code == 0 and elapsed <= ACTIVE and candidate is not None and candidate['complete']
    failure_path = args.output / 'work/FAILURE.json'
    terminal = {'complete': complete, 'status': status if elapsed <= HARD else 'cap_exceeded', 'exit_code': code,
        'reaped': child is not None and code is not None, 'identity': identity, 'inclusive_parent_seconds': elapsed,
        'active_seconds': ACTIVE, 'cleanup_seconds': CLEANUP, 'hard_seconds': HARD, 'cap_exceeded': elapsed > HARD,
        'CPU_user_seconds': after.ru_utime - before.ru_utime, 'CPU_system_seconds': after.ru_stime - before.ru_stime,
        'cumulative_owned_process_peak_RSS_bytes': after.ru_maxrss * 1024, 'RSS_units': 'Linux RUSAGE_CHILDREN cumulative peak; bytes',
        'worker_failure_sha256': sha(failure_path) if failure_path.is_file() else None,
        'worker_candidate_sha256': sha(candidate_path) if candidate_path.is_file() else None,
        'quality_values_closed': True, 'TEST_scoring': False, 'automatic_retry': False, 'GPU_execution': False,
        'resource_supervisor_or_detector_interaction': False, 'other_jobs_modified': False}
    write(args.output / 'TERMINAL.json', terminal)
    if complete:
        receipt = {**candidate, 'schema': 'allocation-OI-MolHIV-CPU-representative-work-v1', 'exit_code': 0, 'reaped': True,
            'worker_source_manifest_sha256': sha(HERE / 'MANIFEST.json'), 'identity': identity,
            'parent_terminal_sha256': sha(args.output / 'TERMINAL.json'), 'parent_costs': terminal,
            'inclusive_parent_seconds': time.monotonic() - started}
        if receipt['inclusive_parent_seconds'] > ACTIVE: receipt['complete'] = False; complete = False
        write(args.output / 'WORK_RECEIPT.json', receipt)
    print(json.dumps({'complete': complete, 'status': terminal['status'], 'reaped': terminal['reaped'], 'quality_values_closed': True}))
    raise SystemExit(0 if complete else 1)

if __name__ == '__main__': main()
