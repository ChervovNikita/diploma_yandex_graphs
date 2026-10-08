"""Normal-host, one-child finite engineering ownership; no remote operations."""
import argparse
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
PHASE_RELATIVE = 'experiments_iclr/postsubmission_20260930'
OWNER_ID = 'label-four-bank-full-WikiCS-engineering-20261008-v1'
ACTIVE_SECONDS = 900
CLEANUP_SECONDS = 15


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def birth(pid):
    # Linux /proc field22, with the command-name parentheses parsed as one field.
    text = Path('/proc/' + str(pid) + '/stat').read_text()
    return int(text[text.rfind(')') + 2:].split()[19])


def validate_release(release, *, later_execution_authorized=False):
    require(later_execution_authorized is True and release.get('enabled') is True
            and release.get('engineering_execution_authorized') is True,
            'Disabled source: a separate enabled root engineering release is required')
    fixed = json.loads((HERE / 'ROOT_RELEASE_TEMPLATE_DISABLED.json').read_text())
    variable = {'enabled', 'engineering_execution_authorized', 'source_manifest_sha256'}
    require(set(release) == set(fixed) and all(release[key] == fixed[key]
            for key in fixed if key not in variable), 'Only the declared root release fields may change')
    require(sha(HERE / 'MANIFEST.json') == release['source_manifest_sha256'],
            'Exact reviewed qualifier manifest required')
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes']
                and sha(path) == row['sha256'], 'Changed qualifier source: ' + row['path'])
    phase = Path(release['phase_root']).resolve(strict=True)
    require(HERE.parent == phase, 'Installed source must be under the declared allocation phase')
    require(set(release['roles']) == {'train', 'valid'}, 'Only complete original TRAIN/VALID roles')
    for role, row in release['roles'].items():
        require(set(row) == {'relative_path', 'sha256', 'bytes'}
                and row['relative_path'] == fixed['roles'][role]['relative_path']
                and isinstance(row['sha256'], str) and len(row['sha256']) == 64
                and all(c in '0123456789abcdef' for c in row['sha256']), 'Exact root role binding')
    for row in [release['safe_payload'], release['polynormer'], *release['roles'].values()]:
        path = (phase / row['relative_path']).resolve(strict=True)
        require(path.is_relative_to(phase) and sha(path) == row['sha256'], 'Root input hash mismatch')
        if 'bytes' in row:
            require(path.stat().st_size == row['bytes'], 'Root SAFE payload byte count')
    # Metadata formatting is not an admission gate. Validate its actual role,
    # schema and source-custody fields against the exact numeric payload pins.
    projection_path = (phase / release['projection_manifest']['relative_path']).resolve(strict=True)
    require(projection_path.is_relative_to(phase), 'Projection metadata remains inside the phase')
    projection = json.loads(projection_path.read_text())
    require(projection['schema'] == 'internal-be-official-role-projection-v2'
            and projection['task'] == 'wikics' and projection['format'] == 'NPZ_numeric_only'
            and projection['official_split_preserved'] is True and projection['split_index'] == 0
            and projection['train_count'] == 580 and projection['valid_count'] == 5274
            and projection['TEST_values_in_payload'] is False, 'Official complete split0 role metadata')
    for role, row in release['roles'].items():
        require(projection['payloads'][role] == dict(path=row['relative_path'],
                sha256=row['sha256'], bytes=row['bytes']), 'Numeric role/source-custody join')
    safe = release['safe_payload']; custody = projection['source_custody']
    require(custody['safe_payload'] == dict(path=safe['relative_path'], sha256=safe['sha256'], bytes=safe['bytes'])
            and custody['TEST_values_excluded_from_safe_payload'] is True
            and custody['official_manifest']['path'] == release['available_manifest']['relative_path'],
            'Exact admitted original SAFE source-custody fields')
    available_path = (phase / release['available_manifest']['relative_path']).resolve(strict=True)
    require(available_path.is_relative_to(phase) and isinstance(json.loads(available_path.read_text()), dict),
            'Authoritative original source manifest remains accessible JSON')
    require(Path(release['runtime_python']).is_file()
            and all(Path(path).is_dir() for path in release['dependency_overlay']),
            'Existing authorized runtime and overlay are required; no installation')
    return phase


def run_owned(*, release, later_execution_authorized=False):
    """Own exactly one fresh discarded worker; source defaults refuse execution."""
    phase = validate_release(release, later_execution_authorized=later_execution_authorized)
    output = phase / release['output_relative']
    require(output.resolve().is_relative_to(phase) and not output.exists(),
            'One fresh owned output; no overwrite, resume or automatic retry')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); worker = None; outcome = 'failed'; reason = None
    owner = dict(schema='label-four-bank-cuda-engineering-owner-v1', owner_id=OWNER_ID,
        parent_PID=os.getpid(), parent_start_ticks=birth(os.getpid()),
        allocation_host=release['allocation_host'], allocation_gpu=release['allocation_gpu'],
        observed_hostname=socket.gethostname(), coexecution_authorized=release['coexecution_authorized'],
        output=str(output), active_seconds=ACTIVE_SECONDS, cleanup_seconds=CLEANUP_SECONDS,
        hard_seconds=ACTIVE_SECONDS + CLEANUP_SECONDS, automatic_retry=False,
        scope='Only the newly created worker process group and this fresh output',
        release=release)
    write(output / 'OWNER.json', owner)
    try:
        environment = os.environ.copy()
        environment['PYTHONPATH'] = os.pathsep.join(release['dependency_overlay'])
        with (output / 'WORKER.log').open('xb') as log:
            worker = subprocess.Popen([release['runtime_python'], str(HERE / 'qualify.py'),
                '--worker', str(output / 'OWNER.json')], cwd=str(phase), env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            owner.update(worker_PID=worker.pid, worker_start_ticks=birth(worker.pid),
                         worker_process_group=worker.pid)
            write(output / 'OWNER.json', owner)
            try:
                return_code = worker.wait(timeout=ACTIVE_SECONDS)
            except subprocess.TimeoutExpired:
                reason = 'Fixed active wall deadline reached'
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    return_code = worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    return_code = worker.wait(timeout=10)
            result_path = output / 'ENGINEERING_RESULT.json'
            result = json.loads(result_path.read_text()) if result_path.is_file() else None
            if reason is None and return_code == 0 and result and result.get('engineering_passed') is True:
                outcome = 'engineering_passed'
            else:
                reason = reason or 'Worker failed; inspect retained finite work and worker log'
    except BaseException as error:
        reason = type(error).__name__ + ': ' + str(error)
        if worker is not None and worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=10)
        raise
    finally:
        write(output / 'TERMINAL.json', dict(schema='label-four-bank-cuda-engineering-terminal-v1',
            owner_id=OWNER_ID, outcome=outcome, reason=reason,
            owner=owner, return_code=worker.returncode if worker is not None else None,
            worker_reaped=worker is None or worker.poll() is not None,
            seconds=time.monotonic() - started, fixed_native_complete_update_limit=4,
            results_path='ENGINEERING_RESULT.json', logs_path='WORKER.log',
            discarded_engineering_work=True, automatic_retry=False,
            VALID_truth_scoring=False, scientific_accuracy_endpoints=False,
            scientific_launch_admitted=False, complete_1100_schedule_qualified=False))
    require(outcome == 'engineering_passed', reason)
    return dict(output=str(output), outcome=outcome, scientific_launch_admitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--later-execution-authorized', action='store_true')
    arguments = parser.parse_args()
    release = json.loads(arguments.release.read_text())
    run_owned(release=release, later_execution_authorized=arguments.later_execution_authorized)


if __name__ == '__main__':
    main()
