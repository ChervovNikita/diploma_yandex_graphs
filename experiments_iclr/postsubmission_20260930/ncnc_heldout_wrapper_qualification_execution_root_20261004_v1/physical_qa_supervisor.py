"""Bound one fabricated QA child and record its physical outcome.

Use exact reviewed heldout supervision helpers. Never open study arrays/states.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def descriptor(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    assert sha(args.release) == args.release_sha256
    release = json.loads(args.release.read_text())
    assert release['execution_enabled'] is True and release['fabricated_inputs_only'] is True
    assert release['TEST_access_authorized'] is False and release['study_checkpoint_access_authorized'] is False
    root = args.release.parent
    phase = root.parent
    repo = phase.parent.parent
    assert Path.cwd() == repo and os.uname().nodename == 'peptide'
    pin = release['supervisor_source']
    path = Path(pin['path'])
    assert path.resolve().is_relative_to(phase) and not path.is_symlink()
    assert sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes']
    spec = importlib.util.spec_from_file_location('original_heldout_supervisor', path)
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    output = root / 'physical/run01'
    assert not output.exists()
    output.mkdir(parents=True)
    source = phase / 'ncnc_frozen_all25_heldout_source_preparation_20261004_v2'
    entry = source / 'heldout_qualification.py'
    assert sha(source / 'MANIFEST.json') == release['heldout_source_manifest_sha256']
    cmd = [sys.executable, '-B', str(entry), '--root-release', str(args.release),
           '--release-sha256', args.release_sha256]
    own = supervisor.identity(os.getpid())
    save(output / 'SUPERVISOR_STARTED.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
         physical_identity=own, release=descriptor(args.release), entry=descriptor(entry),
         wrapper=descriptor(Path(__file__)), command=cmd, scope='FABRICATED_ONLY'))
    started = time.monotonic()
    child = handle = usage = None
    events = []
    reason = error = exit_code = None
    peak = 0
    try:
        with (output / 'CHILD.stdout.log').open('xb') as out, (output / 'CHILD.stderr.log').open('xb') as err:
            child = subprocess.Popen(cmd, cwd=repo, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=err, start_new_session=True)
            handle = supervisor.identity(child.pid)
            assert handle['pid'] == child.pid
            save(output / 'CHILD_STARTED.json', dict(physical_identity=handle, command=cmd))
            status, usage, exit_code, reason, peak = supervisor.monitor_owned_child(
                child, handle, started, events,
                max_seconds=release['maximum_child_wall_seconds'],
                max_rss_bytes=release['maximum_sampled_owned_session_RSS_bytes'])
    except BaseException as exc:
        error = dict(type=type(exc).__name__, condition=str(exc))
    finally:
        if child is not None and child.returncode is None:
            if handle is not None:
                supervisor.kill_owned_session(handle, 'ROOT_QA_EXCEPTION_CLEANUP', events)
            child.wait(timeout=30)
            exit_code = child.returncode
        logical_path = Path(release['output_directory']) / 'QUALIFICATION.json'
        logical = json.loads(logical_path.read_text()) if logical_path.exists() else None
        after = sha(args.release) == args.release_sha256 and sha(path) == pin['sha256']
        complete = bool(child is not None and exit_code == 0 and reason is None and error is None
                        and after and logical and logical['status'] == 'PASS'
                        and logical['root_release_sha256'] == args.release_sha256
                        and logical['heldout_source_manifest_sha256'] == release['heldout_source_manifest_sha256'])
        terminal = dict(UTC=datetime.now(timezone.utc).isoformat(),
                        status='PHYSICALLY_COMPLETE' if complete else 'PHYSICALLY_FAILED_OR_BOUNDED_STOP',
                        root_release_sha256=args.release_sha256,
                        heldout_source_manifest_sha256=release['heldout_source_manifest_sha256'],
                        invocation=release['authorized_invocations'][0],
                        supervisor_identity=own, child_identity=handle, exit_code=exit_code,
                        bounded_stop_reason=reason, error=error, signal_events=events,
                        observed_session_RSS_peak_bytes=peak,
                        kernel_child_RSS_peak_bytes=int(usage.ru_maxrss * 1024) if usage else None,
                        inclusive_wall_seconds=time.monotonic() - started,
                        logical_result=descriptor(logical_path) if logical_path.exists() else None,
                        release_and_supervisor_after_sha256_match=after,
                        fabricated_inputs_only=True, TEST_access=False, study_checkpoint_access=False,
                        source_entry=descriptor(entry), physical_supervisor=descriptor(Path(__file__)))
        save(output / 'SUPERVISOR_TERMINAL.json', terminal)
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
