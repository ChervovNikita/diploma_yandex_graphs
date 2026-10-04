"""One ordinary owned child, wait4 accounting, sampled wall/RSS/GPU0 bounds."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HERE = PHASE / 'ncnc_fixed_fabricated_f4_repeatability_diagnostic_execution_root_20261004_v1'
SOURCE = PHASE / 'graph_ncNC_fixed_fabricated_f4_repeatability_diagnostic_preparation_20261004_v1'
RELEASE = HERE / 'ROOT_DIAGNOSTIC_RELEASE.json'
CAPS = HERE / 'CAPS.json'
OUTPUT = HERE / 'diagnostic/run01'
GPU = 'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'
EXPECTED_RELEASE_SHA = '4d34feef2c2b414f198e32f3d60cef06f43706706cc3427493e6a68f02c75f44'
EXPECTED_CAPS_SHA = '3f1c7032f35816f402dfbf6137bd5570f754def261be4e2dd0d1d235c01f7623'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def identity(pid):
    text = Path(f'/proc/{pid}/stat').read_text()
    fields = text[text.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), pgid=int(fields[2]), sid=int(fields[3]),
                RSS_bytes=int(fields[21]) * os.sysconf('SC_PAGE_SIZE'))


def owned_processes(handle):
    rows = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            row = identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if row['pgid'] == row['sid'] == handle['pid']:
            rows.append(row)
    return rows


def GPU0_owned_memory(rows, timeout):
    result = subprocess.run(['nvidia-smi', '--id=' + GPU,
        '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=timeout)
    pids = {row['pid'] for row in rows}
    total = 0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        values = [part.strip() for part in line.split(',')]
        assert len(values) == 2, 'GPU0 memory observation malformed'
        if int(values[0]) in pids:
            total += int(values[1]) * 1024**2
    return total


def kill_owned(handle, reason, events):
    try:
        actual = identity(handle['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return
    assert actual['start_ticks'] == handle['start_ticks'] and actual['pgid'] == actual['sid'] == handle['pid'], 'Owned identity changed; no signal sent'
    try:
        os.killpg(handle['pid'], signal.SIGKILL)
    except ProcessLookupError:
        return
    events.append(dict(UTC=datetime.now(timezone.utc).isoformat(), signal='SIGKILL',
        own_pgid=handle['pid'], own_start_ticks=handle['start_ticks'], reason=reason))


def main():
    start = time.monotonic(); began = datetime.now(timezone.utc).isoformat()
    child = handle = usage = wait_status = code = child_started = None
    reason = error = None
    events = []; peak_RSS = peak_GPU = 0; GPU_samples = 0; argv = []; last_GPU_sample = 0.
    try:
        assert Path.cwd().resolve() == REPO and Path(__file__).resolve() == HERE / 'run_fixed_diagnostic_once.py'
        assert sha(RELEASE) == EXPECTED_RELEASE_SHA and sha(CAPS) == EXPECTED_CAPS_SHA
        release = json.loads(RELEASE.read_text()); caps = json.loads(CAPS.read_text())
        assert caps['maximum_wrapper_wall_seconds'] == 600 and caps['maximum_sampled_owned_session_RSS_bytes'] == 16 * 1024**3 and caps['maximum_sampled_GPU0_owned_process_memory_bytes'] == 8 * 1024**3
        assert release['execution_enabled'] is True and release['authorized_invocations'] == [dict(stage='fixed_fabricated_f4_repeatability_diagnostic', output_directory=str(OUTPUT), cuda_visible_devices=GPU)]
        assert not OUTPUT.exists()
        pin = release['runtime_authority']; runtime_path = Path(pin['path'])
        assert runtime_path.stat().st_size == pin['bytes'] and sha(runtime_path) == pin['sha256']
        runtime = json.loads(runtime_path.read_text())
        assert sha(Path(runtime['interpreter_path'])) == runtime['interpreter_sha256'] == '14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
        assert sha(SOURCE / 'MANIFEST.json') == release['diagnostic_manifest_sha256'] == 'a5a256b7448371253c0db5d6489c7724b668e3ccc9931904c20d104cacd99a58'
        env = os.environ.copy()
        env.update(release['original_process_environment'])
        env.update(CUDA_VISIBLE_DEVICES=GPU, CUBLAS_WORKSPACE_CONFIG=release['common_CUBLAS_WORKSPACE_CONFIG'], PYTHONDONTWRITEBYTECODE='1')
        assert env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and env['PYTHONPATH'] == runtime['project_PYTHONPATH'] and env['OMP_NUM_THREADS'] == env['MKL_NUM_THREADS'] == '2'
        argv = [runtime['interpreter_path'], '-B', str(SOURCE / 'fixed_diagnostic.py'),
                '--execute', '--root-release', str(RELEASE), '--output', str(OUTPUT)]
        with (HERE / 'CHILD.stdout.log').open('x') as stdout, (HERE / 'CHILD.stderr.log').open('x') as stderr:
            child_started = time.monotonic()
            child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            handle = identity(child.pid); assert handle['pgid'] == handle['sid'] == child.pid
            save('CHILD_STARTED.json', dict(UTC=datetime.now(timezone.utc).isoformat(), identity=handle,
                runner_identity=identity(os.getpid()), argv=argv, root_release_sha256=EXPECTED_RELEASE_SHA,
                caps_sha256=EXPECTED_CAPS_SHA, process_environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG')}, no_retry=True))
            while True:
                pid, status, observed = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    wait_status, usage = status, observed; code = os.waitstatus_to_exitcode(status)
                    child.returncode = code
                    break
                rows = owned_processes(handle); current_RSS = sum(row['RSS_bytes'] for row in rows)
                peak_RSS = max(peak_RSS, current_RSS)
                now = time.monotonic()
                if reason is None:
                    if now - start >= 600:
                        reason = 'OWNED_WRAPPER_WALL_600_SECONDS'
                    elif current_RSS > 16 * 1024**3:
                        reason = 'OWNED_SESSION_SAMPLED_RSS_EXCEEDED_16_GiB'
                    elif now - last_GPU_sample >= .5:
                        current_GPU = GPU0_owned_memory(rows, min(3., max(.1, 600 - (now - start))))
                        last_GPU_sample = time.monotonic(); GPU_samples += 1; peak_GPU = max(peak_GPU, current_GPU)
                        if current_GPU > 8 * 1024**3:
                            reason = 'GPU0_OWNED_PROCESS_SAMPLED_MEMORY_EXCEEDED_8_GiB'
                        elif last_GPU_sample - start >= 600:
                            reason = 'OWNED_WRAPPER_WALL_600_SECONDS'
                    if reason:
                        kill_owned(handle, reason, events)
                time.sleep(.25)
    except BaseException as exception:
        error = dict(type=type(exception).__name__, condition=str(exception))
        if child is not None and usage is None:
            if handle is None:
                handle = identity(child.pid)
            kill_owned(handle, 'OWNED_SUPERVISOR_FAILURE', events)
            _, wait_status, usage = os.wait4(child.pid, 0)
            code = os.waitstatus_to_exitcode(wait_status); child.returncode = code
    terminal = dict(schema='ncnc-fixed-fabricated-f4-owned-physical-terminal-v1', start_UTC=began,
        terminal_UTC=datetime.now(timezone.utc).isoformat(),
        status='PHYSICALLY_COMPLETE' if code == 0 and not error and reason is None else 'PHYSICALLY_FAILED_OR_BOUNDED_STOP',
        exit_code=code, raw_wait_status=wait_status, runner_identity=identity(os.getpid()), child_identity=handle,
        argv=argv, root_release_sha256=EXPECTED_RELEASE_SHA, caps_sha256=EXPECTED_CAPS_SHA,
        physical_wrapper_wall_seconds=time.monotonic() - start,
        child_spawn_through_wait4_wall_seconds=time.monotonic() - child_started if child_started is not None else None,
        owned_child_wait4_user_seconds=usage.ru_utime if usage is not None else None,
        owned_child_wait4_system_seconds=usage.ru_stime if usage is not None else None,
        owned_child_wait4_peak_RSS_bytes=usage.ru_maxrss * 1024 if usage is not None else None,
        sampled_peak_owned_session_RSS_bytes=peak_RSS, sampled_peak_GPU0_owned_process_memory_bytes=peak_GPU,
        GPU0_memory_samples=GPU_samples, maximum_wrapper_wall_seconds=600,
        maximum_sampled_owned_session_RSS_bytes=16 * 1024**3, maximum_sampled_GPU0_owned_process_memory_bytes=8 * 1024**3,
        bounded_stop_reason=reason, own_session_signal_events=events, error=error,
        terminal_write_tail_measured=False, overlapping_diagnostic_intervals_not_added=True,
        fabricated_inputs_only=True, scientific_execution=False, training_updates=0, old_qualification_status_preserved='FAILED',
        no_retry=True, GPU1_signaled=False, other_jobs_signaled=False, numerical_runtime_hooks_added=False)
    save('PHYSICAL_TERMINAL.json', terminal)
    return 0 if terminal['status'] == 'PHYSICALLY_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
