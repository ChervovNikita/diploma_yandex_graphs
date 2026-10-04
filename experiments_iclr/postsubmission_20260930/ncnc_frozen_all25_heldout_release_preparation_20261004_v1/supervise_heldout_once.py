"""One frozen all25 TEST invocation with inherited owned-session bounds."""
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
PREPARATION = PHASE / 'ncnc_frozen_all25_heldout_release_preparation_20261004_v1'
HERE = PREPARATION / 'supervision_run01'
SOURCE = PHASE / 'ncnc_frozen_all25_heldout_source_preparation_20261004_v1'
OUTPUT = PREPARATION / 'heldout/run01'
EXPECTED_SOURCE_SHA = '0a1a49f3959afb5471562802d91a2358408cf7f499a77eeccf02ec5bbc6acf3a'
EXPECTED_IDENTITY = {'data_authority_sha256': 'df6980064b3ee31377e6d96215146e1d954abf5464182464582801eddf17077d', 'design_manifest_sha256': 'e97aeb2655d54f261acca1874c8e03047608625aac1b5e1e980853f7d44e0004', 'driver_manifest_sha256': 'a59c669356e1a1437ce90107a7c76eb8f0b5579dc48d67b4378a7e237e956b7e', 'family_id': 'ncnc-collab-predictive-pilot-20261003-v1', 'family_lock_output_directory': '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930/graph_ncNC_predictive_family_execution_root_20261003_v1/family_lock', 'prototype_manifest_sha256': 'a99b0e3b0e8ec597b03e2c390ac176597b5b6422d365fc20624ab98a113d42f9', 'resource_manifest_sha256': '031c6a1fc36514a7dd1b7b520e5ec251b2c3f385bcab917ca101f6184712790a', 'runtime_authority_sha256': 'e63f602baa8a91e129fb4a0c0debd6ec282e7cd132be87eb66cd8c5cf0b7c882'}
MAX_SECONDS = 1800
MAX_OWNED_RSS_BYTES = 32 * 1024**3


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0], ppid=int(fields[1]),
                pgid=int(fields[2]), sid=int(fields[3]), RSS_bytes=int(fields[21]) * os.sysconf('SC_PAGE_SIZE'))


def session_RSS(handle):
    rows = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            row = identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if row['pgid'] == handle['pid'] and row['sid'] == handle['pid']:
            rows.append(row)
    return sum(row['RSS_bytes'] for row in rows)


def kill_owned_session(handle, reason, events):
    try:
        actual = identity(handle['pid'])
    except FileNotFoundError:
        return
    if actual['start_ticks'] != handle['start_ticks'] or actual['pgid'] != handle['pid'] or actual['sid'] != handle['pid']:
        raise RuntimeError('Owned child identity changed; no signal sent')
    try:
        os.killpg(handle['pid'], signal.SIGKILL)
    except ProcessLookupError:
        return
    events.append(dict(UTC=datetime.now(timezone.utc).isoformat(), signal='SIGKILL',
                       own_pgid=handle['pid'], own_start_ticks=handle['start_ticks'], reason=reason))


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    RELEASE = Path(args.root_release)
    EXPECTED_RELEASE_SHA = args.release_sha256
    # Reject disabled or borrowed releases before creating run receipts.
    assert RELEASE.is_absolute() and RELEASE.resolve() == RELEASE and RELEASE.parent == PREPARATION
    assert sha(RELEASE) == EXPECTED_RELEASE_SHA
    release = json.loads(RELEASE.read_text())
    assert release['schema'] == 'ncnc-frozen-all25-one-time-heldout-root-release-v1'
    assert release['execution_enabled'] is True and release['root_authorization_reference']
    assert release['identity'] == EXPECTED_IDENTITY and release['TEST_access_authorized'] is True and release['evaluation_once'] is True
    assert release['authorized_stages'] == ['frozen_all25_heldout_confirmation']
    assert release['authorized_invocations'] == [dict(stage='frozen_all25_heldout_confirmation', output_directory=str(OUTPUT), cuda_visible_devices='GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998')]
    assert not HERE.exists() and not OUTPUT.exists() and not (PREPARATION / 'ONE_TIME_TEST_CLAIM.json').exists()
    HERE.mkdir(mode=0o700)
    start = time.monotonic()
    began = datetime.now(timezone.utc).isoformat()
    child = handle = usage = None
    child_started = None
    wait_status = code = None
    reason = None
    events = []
    cleanup_errors = []
    unreaped_child = None
    peak = 0
    argv = []
    error = None
    try:
        assert Path.cwd().resolve() == REPO and Path(__file__).resolve() == PREPARATION / 'supervise_heldout_once.py'
        assert sha(RELEASE) == EXPECTED_RELEASE_SHA and sha(SOURCE / 'MANIFEST.json') == EXPECTED_SOURCE_SHA
        release = json.loads(RELEASE.read_text())
        runtime = json.loads(Path(release['runtime_authority']['path']).read_text())
        assert release['execution_enabled'] is True and release['authorized_stages'] == ['frozen_all25_heldout_confirmation']
        assert release['authorized_invocations'] == [dict(stage='frozen_all25_heldout_confirmation',
            output_directory=str(OUTPUT), cuda_visible_devices='GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998')]
        assert not OUTPUT.exists()
        assert sha(Path(runtime['interpreter_path'])) == runtime['interpreter_sha256']
        for pin in [release['runtime_authority'], *runtime['runtime_source_pins'], *runtime['runtime_binary_files'], runtime['negative_sampler']]:
            path = Path(pin['path']); assert path.resolve() == path and sha(path) == pin['sha256']
            assert 'bytes' not in pin or path.stat().st_size == pin['bytes']
        import sys
        sys.path.insert(0, str(SOURCE))
        # The sidecar is authenticated before importing its stdlib admission.
        manifest = json.loads((SOURCE / 'MANIFEST.json').read_text())
        for pin in manifest['files']:
            path = SOURCE / pin['path']; assert path.resolve() == path and path.is_relative_to(SOURCE)
            assert path.stat().st_size == pin.get('bytes', pin.get('size')) and sha(path) == pin['sha256']
        os.environ['CUDA_VISIBLE_DEVICES'] = release['cuda_visible_devices']
        from heldout_gate import metadata_admission
        context, family = metadata_admission(RELEASE, OUTPUT, expected_sha=EXPECTED_RELEASE_SHA)
        assert context['identity'] == EXPECTED_IDENTITY
        admission = json.loads(Path(release['runtime_resource_admission']['path']).read_text())
        assert admission['minimum_GPU_free_MiB'] == 24576 and admission['minimum_host_MemAvailable_bytes'] == 16 * 1024**3
        gpu = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.free,memory.total', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15).stdout.strip().split(',')
        assert len(gpu) == 3 and gpu[0].strip() == release['cuda_visible_devices']
        free = int(gpu[1]); total = int(gpu[2])
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        assert free >= admission['minimum_GPU_free_MiB'] and available >= admission['minimum_host_MemAvailable_bytes']
        assert sha(RELEASE) == EXPECTED_RELEASE_SHA and not OUTPUT.exists()
        save('RESOURCE_DISPATCH_RECHECK.json', dict(UTC=datetime.now(timezone.utc).isoformat(), physical_gpu_index=0, GPU_UUID=gpu[0].strip(), GPU_free_MiB=free, GPU_total_MiB=total, host_MemAvailable_bytes=available, release_sha256=EXPECTED_RELEASE_SHA, sidecar_manifest_sha256=EXPECTED_SOURCE_SHA, only_owned_child_start_authorized=True))
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=release['cuda_visible_devices'],
                   PYTHONPATH=runtime['project_PYTHONPATH'], PYTHONDONTWRITEBYTECODE='1',
                   PYTHONNOUSERSITE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        argv = [runtime['interpreter_path'], '-B', str(SOURCE / 'heldout_run.py'),
                '--root-release', str(RELEASE), '--release-sha256', EXPECTED_RELEASE_SHA]
        with (PREPARATION / 'ONE_TIME_TEST_CLAIM.json').open('x') as claim:
            json.dump(dict(schema='ncnc-frozen-all25-one-time-TEST-claim-v1', UTC=datetime.now(timezone.utc).isoformat(), root_release_sha256=EXPECTED_RELEASE_SHA, heldout_source_manifest_sha256=EXPECTED_SOURCE_SHA, output_directory=str(OUTPUT), supervisor_identity=identity(os.getpid()), TEST_access_once=True, no_retry=True), claim, indent=2, allow_nan=False)
            claim.write('\n'); claim.flush(); os.fsync(claim.fileno())
        with (HERE / 'CHILD.stdout.log').open('x') as stdout, (HERE / 'CHILD.stderr.log').open('x') as stderr:
            child_started = time.monotonic()
            child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            handle = identity(child.pid)
            assert handle['pgid'] == handle['sid'] == child.pid
            save('CHILD_STARTED.json', dict(UTC=datetime.now(timezone.utc).isoformat(), identity=handle,
                runner_identity=identity(os.getpid()), argv=argv, root_release_sha256=EXPECTED_RELEASE_SHA,
                sidecar_manifest_sha256=EXPECTED_SOURCE_SHA, timeout_seconds=MAX_SECONDS,
                maximum_sampled_owned_session_RSS_bytes=MAX_OWNED_RSS_BYTES, no_retry=True,
                fabricated_inputs_only=False, original_TRAIN_VALID_lock_selected_checkpoint_access_authorized=True, TEST_access_authorized=True))
            while True:
                pid, status, observed = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    wait_status, usage = status, observed
                    code = os.waitstatus_to_exitcode(status)
                    child.returncode = code
                    break
                observed_RSS = session_RSS(handle)
                peak = max(peak, observed_RSS)
                if reason is None:
                    if time.monotonic() - child_started >= MAX_SECONDS:
                        reason = 'OWNED_CHILD_TIMEOUT_1800_SECONDS'
                    elif observed_RSS > MAX_OWNED_RSS_BYTES:
                        reason = 'OWNED_SESSION_SAMPLED_RSS_EXCEEDED_32_GiB'
                    if reason:
                        kill_owned_session(handle, reason, events)
                time.sleep(.25)
    except BaseException as exception:
        error = dict(type=type(exception).__name__, condition=str(exception))
        if child is not None and usage is None:
            if handle is None:
                try:
                    handle = identity(child.pid)
                except BaseException as cleanup_error:
                    cleanup_errors.append(dict(phase='acquire_owned_child_identity', type=type(cleanup_error).__name__, condition=str(cleanup_error)))
            # A failed identity probe never authorizes a signal. Reap is still
            # attempted independently, including for an already absent child.
            if handle is not None:
                try:
                    kill_owned_session(handle, 'OWNED_SUPERVISOR_FAILURE', events)
                except BaseException as cleanup_error:
                    cleanup_errors.append(dict(phase='signal_identified_owned_child', type=type(cleanup_error).__name__, condition=str(cleanup_error)))
            reap_deadline = time.monotonic() + 5
            while True:
                try:
                    waited_pid, status, observed = os.wait4(child.pid, os.WNOHANG)
                except BaseException as cleanup_error:
                    cleanup_errors.append(dict(phase='reap_owned_child', type=type(cleanup_error).__name__, condition=str(cleanup_error)))
                    unreaped_child = dict(pid=child.pid, reason='wait4_failed')
                    break
                if waited_pid:
                    wait_status, usage = status, observed
                    code = os.waitstatus_to_exitcode(status)
                    child.returncode = code
                    break
                if time.monotonic() >= reap_deadline:
                    unreaped_child = dict(pid=child.pid, reason='bounded_cleanup_reap_deadline', owned_identity_acquired=handle is not None)
                    break
                time.sleep(.05)
    try:
        runner_identity = identity(os.getpid())
    except BaseException as cleanup_error:
        runner_identity = dict(pid=os.getpid(), identity_unavailable=True)
        cleanup_errors.append(dict(phase='terminal_runner_identity', type=type(cleanup_error).__name__, condition=str(cleanup_error)))
    terminal = dict(schema='ncnc-frozen-all25-heldout-owned-physical-terminal-v1',
        start_UTC=began, terminal_UTC=datetime.now(timezone.utc).isoformat(),
        status='PHYSICALLY_COMPLETE' if code == 0 and not error and reason is None and not cleanup_errors and unreaped_child is None else 'PHYSICALLY_FAILED_OR_BOUNDED_STOP',
        exit_code=code, raw_wait_status=wait_status, runner_identity=runner_identity,
        child_identity=handle, argv=argv, root_release_sha256=EXPECTED_RELEASE_SHA,
        sidecar_manifest_sha256=EXPECTED_SOURCE_SHA,
        physical_wrapper_wall_seconds=time.monotonic() - start,
        child_spawn_through_wait4_wall_seconds=time.monotonic() - child_started if child_started is not None else None,
        owned_child_wait4_user_seconds=usage.ru_utime if usage is not None else None,
        owned_child_wait4_system_seconds=usage.ru_stime if usage is not None else None,
        owned_child_wait4_peak_RSS_bytes=usage.ru_maxrss * 1024 if usage is not None else None,
        peak_RSS_scope='Linux wait4 ru_maxrss for the owned child and its waited descendants; not an aggregate sum of simultaneous RSS.',
        sampled_peak_owned_session_RSS_bytes=peak,
        sampled_RSS_scope='Resident-page sum across only the child-owned PGID/SID, sampled every .25 s; shared pages may be counted for each process.',
        timeout_seconds=MAX_SECONDS, maximum_sampled_owned_session_RSS_bytes=MAX_OWNED_RSS_BYTES,
        bounded_stop_reason=reason, own_session_signal_events=events, error=error,
        cleanup_errors=cleanup_errors, unreaped_owned_child=unreaped_child,
        terminal_write_tail_measured=False, overlapping_harness_intervals_not_added=True,
        fabricated_inputs_only=False, scientific_fit_updates_authorized=0,
        original_TRAIN_VALID_lock_selected_checkpoint_access_authorized=True, TEST_access_authorized=True, no_retry=True,
        other_jobs_signaled=False, namespace_isolation_used=False, runtime_hooks_added=False)
    save('PHYSICAL_TERMINAL.json', terminal)
    return 0 if terminal['status'] == 'PHYSICALLY_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
