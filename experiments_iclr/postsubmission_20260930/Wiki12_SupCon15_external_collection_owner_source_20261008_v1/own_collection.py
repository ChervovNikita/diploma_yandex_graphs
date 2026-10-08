"""Disabled minimal normal77 owner; reuse successful Context9 collection supervision."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def inside(relative):
    value = Path(relative)
    require(not value.is_absolute() and '..' not in value.parts, 'Phase-relative custody')
    path = (PHASE / value).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Owner path leaves project phase')
    return path


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(PHASE)), sha256=sha(path), bytes=path.stat().st_size)


def bound(row):
    path = inside(row['path'])
    require(path.is_file() and sha(path) == row['sha256']
        and ('bytes' not in row or path.stat().st_size == row['bytes']), 'Bound bytes changed: ' + row['path'])
    return path


def seal(row):
    path = bound(row)
    for item in read(path)['files']:
        file = (path.parent / item['path']).resolve()
        require(file.is_relative_to(path.parent), 'Sealed payload leaves source directory')
        bound(dict(item, path=str(file.relative_to(PHASE))))


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')


def proc(pid):
    try:
        text = Path('/proc', str(pid), 'stat').read_text(); fields = text[text.rfind(')') + 2:].split()
        return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0], group=int(fields[2]), session=int(fields[3]))
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--launch-started-monotonic', type=float, required=True)
    args = parser.parse_args(); os.umask(0o077); owner_began = args.launch_started_monotonic
    require(math.isfinite(owner_began) and 0 <= time.monotonic() - owner_began < 30, 'Finite launcher/admission origin')
    require(sha(args.release) == args.release_sha256, 'Exact separate root owner release')
    cfg = read(args.release)
    require(cfg['schema'] == 'Wiki12-SupCon15-external-owner-root-release-v1'
        and all(cfg.get(key) is True for key in ('enabled', 'root_collection_execution_authorized', 'source_review_approved',
            'actual_union_all15_terminal_bindings_supplied', 'actual_runtime_resource_bindings_supplied', 'cost_and_external_supervision_authorized')),
        'Owner remains disabled pending actual root custody/resource release')
    require(all(cfg.get(key) is False for key in ('TEST_access', 'training', 'reselection', 'calibration', 'automatic_retry'))
        and cfg['maximum_member_forwards'] == 60, 'No training/TEST/reselection/calibration/retry')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    own_manifest = dict(path=str((HERE / 'MANIFEST.json').relative_to(PHASE)), sha256=cfg['owner_manifest_sha256'])
    seal(own_manifest)
    seal(pins['collector_manifest']); bound(pins['collector_entry']); bound(pins['reused_successful_collection_owner'])
    runtime = pins['runtime']; gpu = pins['physical_gpu_uuid']
    require(Path.cwd().resolve() == Path(runtime['repository']).resolve() and socket.gethostname() == runtime['hostname']
        and str(Path(sys.executable).absolute()) == runtime['python'] and os.environ.get('PYTHONPATH', '') == '', 'Original normal77 host/runtime')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
        == runtime['physical_gpu_inventory'], 'Original physical GPU inventory')
    activation = inside(pins['root_activation_directory'])
    require(args.release.resolve() == activation / pins['owner_release_name'], 'Exact separate root owner activation')
    release_path = bound(cfg['collection_release']); collection = read(release_path)
    require(release_path == activation / pins['collection_release_name'], 'Exact separate root collection activation')
    require(collection['schema'] == 'Wiki12-SupCon15-union-collection-release-v1'
        and collection['enabled'] is True and collection['collection_manifest_sha256'] == pins['collector_manifest']['sha256']
        and collection['external_active_seconds'] == 300 and collection['external_cleanup_seconds'] == 10
        and collection['external_hard_seconds'] == 310 and collection['owned_GPU_cap_bytes'] == 8 * 1024**3
        and collection['minimum_fresh_GPU_free_bytes'] == 10 * 1024**3 and collection['physical_gpu_uuid'] == gpu
        and collection['maximum_member_forwards'] == 60
        and all(collection.get(key) is True for key in ('root_execution_authorized', 'source_review_approved',
            'entire_Wiki12_union_complete', 'entire_SupCon3_complete', 'owners_and_all15_children_terminal',
            'trusted_checkpoint_deserialization_authorized', 'runtime_resource_readiness_confirmed',
            'collection_and_analysis_cost_charged', 'external_owned_bound_confirmed'))
        and all(collection.get(key) is False for key in ('TEST_access', 'training', 'reselection', 'calibration', 'automatic_retry')),
        'Exact reviewed combined15 source/scope and300+10/8GiB/10GiB collection release')
    # Root supplies these real evidence files. This owner creates no terminal/runtime/resource proof.
    for key in ('Wiki12_union_closure', 'SupCon3_closure', 'terminal_evidence', 'runtime_evidence', 'resource_readiness', 'external_supervision'):
        bound(collection[key])
    require(collection['external_supervision'] == binding(HERE / 'MANIFEST.json'),
        'External supervision binds immutable owner source; separate releases have no hash cycle')
    free = int(subprocess.check_output(['nvidia-smi', '--id=' + gpu, '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=5).strip()) * 1024**2
    require(free >= 10 * 1024**3, 'Fresh10GiB GPU0 admission; no indefinite resource wait')
    output = inside(cfg['owner_output_directory'])
    require(cfg['owner_output_directory'] == pins['owner_output_directory']
        and collection['output_directory'] == pins['collector_output_directory']
        and not output.exists() and output.parent.is_dir() and not output.is_relative_to(HERE)
        and not inside(collection['output_directory']).exists(), 'Fresh once-only owner and server-only collection outputs')
    require(time.monotonic() - owner_began < pins['launch_admission_reserve_seconds'], 'Finite30s launch/admission reserve exhausted')
    output.mkdir(mode=0o700)
    environment = dict(os.environ, PYTHONPATH='', CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    environment.pop('PYTHONHOME', None)
    argv = [runtime['python'], '-B', str(bound(pins['collector_entry'])), '--release', str(release_path), '--release-sha256', cfg['collection_release']['sha256']]
    actions = []; refusals = []; max_rss = max_gpu = 0; reason = None; interrupted = [False]; stopped_at = None
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda number, frame: interrupted.__setitem__(0, True))
    with (output / 'collection.stdout.log').open('xb') as stdout, (output / 'collection.stderr.log').open('xb') as stderr:
        require(time.monotonic() - owner_began < pins['launch_admission_reserve_seconds'], 'Finite30s immediate prelaunch reserve')
        began = time.monotonic()
        child = subprocess.Popen(argv, cwd=runtime['repository'], env=environment, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=True)
        saved = proc(child.pid)
        def stop(signum):
            actual = proc(child.pid)
            if actual is None:
                return
            if saved is not None and all(actual[key] == saved[key] for key in ('pid', 'start_ticks', 'group', 'session')) and saved['group'] == saved['session'] == child.pid:
                try:
                    os.killpg(saved['group'], signum); actions.append(int(signum))
                except ProcessLookupError:
                    pass
            else:
                refusals.append(dict(requested_signal=int(signum), actual_identity=actual, reason='Exact owned identity changed; no signal'))
        def finish_cleanup():
            deadline = min(began + 310, stopped_at + 10)
            try:
                child.wait(timeout=max(.01, deadline - time.monotonic() - 1))
            except subprocess.TimeoutExpired:
                stop(signal.SIGKILL)
                child.wait(timeout=max(.01, deadline - time.monotonic()))
        try:
            require(saved is not None and saved['group'] == saved['session'] == child.pid, 'Exact actual owned collection child session')
            write(output / 'COLLECTION_OWNER.json', dict(owner=saved, collector_child_identity=saved,
                supervisor_identity=proc(os.getpid()), argv=argv, owner_source_manifest=binding(HERE / 'MANIFEST.json'),
                owner_release=binding(args.release),
                collection_release=cfg['collection_release'], collector_manifest=pins['collector_manifest'],
                active_seconds=300, cleanup_seconds=10, terminal_reserve_seconds=20, owned_GPU_and_RSS_cap_bytes=8 * 1024**3,
                admission_seconds=began - owner_began, fresh_GPU_free_bytes=free, training_updates=0, TEST_access=False))
            while child.poll() is None:
                if interrupted[0]:
                    reason = 'owner_stop_requested'; break
                if time.monotonic() - began >= 300:
                    reason = 'external_active_deadline'; break
                try:
                    text = Path('/proc', str(child.pid), 'status').read_text()
                except FileNotFoundError:
                    if child.poll() is not None:
                        break
                    raise
                rss = next((int(row.split()[1]) * 1024 for row in text.splitlines() if row.startswith('VmRSS:')), 0)
                max_rss = max(max_rss, rss)
                rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                    text=True, timeout=min(5, max(.01, 300 - (time.monotonic() - began)))).splitlines()
                owned = [parts[1].strip() for parts in (row.split(',') for row in rows)
                    if len(parts) == 2 and parts[0].strip() == str(child.pid)]
                require(all(value.isdigit() for value in owned), 'Owned CUDA memory telemetry is numeric')
                memory = sum(int(value) * 1024**2 for value in owned)
                max_gpu = max(max_gpu, memory)
                if rss > 8 * 1024**3 or memory > 8 * 1024**3:
                    reason = 'owned_resource_cap'; break
                try:
                    child.wait(timeout=min(2, max(.01, 300 - (time.monotonic() - began))))
                except subprocess.TimeoutExpired:
                    pass
            if reason:
                stopped_at = time.monotonic()
                stop(signal.SIGTERM)
                finish_cleanup()
            else:
                child.wait(timeout=.01)
        except BaseException as error:
            reason = reason or ('owned_supervision_failure: ' + type(error).__name__)
            stopped_at = stopped_at or time.monotonic()
            stop(signal.SIGTERM)
            try:
                finish_cleanup()
            except subprocess.TimeoutExpired:
                pass
            raise
        finally:
            reaped = child.poll() is not None
            reaped_elapsed = time.monotonic() - began
            active_elapsed = (stopped_at - began) if stopped_at is not None else reaped_elapsed
            cleanup_elapsed = (time.monotonic() - stopped_at) if stopped_at is not None else 0
            terminal_error = None; cuda_absent = False
            try:
                rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
                cuda_absent = str(child.pid) not in {row.strip() for row in rows}
            except Exception as error:
                terminal_error = type(error).__name__ + ': ' + str(error)
            after = proc(child.pid); elapsed = time.monotonic() - began
            value = dict(exit_code=child.returncode, reaped=reaped, exit_authority='subprocess.Popen.wait/poll',
                owned_after=after, child_no_CUDA_rows=cuda_absent, terminal_telemetry_error=terminal_error,
                reason=reason, signals=actions, signal_refusals=refusals, inclusive_child_seconds=elapsed,
                active_seconds_used=active_elapsed, cleanup_seconds_used=cleanup_elapsed,
                child_exit_or_reap_seconds=reaped_elapsed, terminal_telemetry_seconds=elapsed - reaped_elapsed,
                max_sampled_RSS_bytes=max_rss, max_sampled_owned_GPU_bytes=max_gpu,
                active_cap_exceeded=active_elapsed > 300, cleanup_cap_exceeded=cleanup_elapsed > 10,
                active_plus_cleanup_cap_exceeded=reaped_elapsed > 310, terminal_reserve_exceeded=elapsed - reaped_elapsed > 20,
                terminal_envelope_exceeded=elapsed > 330,
                inclusive_owner_seconds=time.monotonic() - owner_began, total_owner_envelope_exceeded=time.monotonic() - owner_began > 360,
                source_manifest=pins['collector_manifest'], owner_source_manifest=binding(HERE / 'MANIFEST.json'),
                collection_release=cfg['collection_release'], owner_release=binding(args.release),
                maximum_member_forwards=60, training_updates=0, TEST_access=False, reselection=False, automatic_retry=False,
                partial_results_and_costs_preserved=True, raw_arrays_server_only=True, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write(output / 'COLLECTION_EXIT.json', value)
    print(json.dumps(dict(collection_child_terminal=True, exit_code=value['exit_code'], reaped=value['reaped'], scores_read=False)))
    return 0 if value['exit_code'] == 0 and reason is None and reaped and after is None and cuda_absent and not refusals \
        and not any(value[key] for key in ('active_cap_exceeded', 'cleanup_cap_exceeded', 'active_plus_cleanup_cap_exceeded',
            'terminal_reserve_exceeded', 'terminal_envelope_exceeded', 'total_owner_envelope_exceeded')) else 1


if __name__ == '__main__':
    raise SystemExit(main())
