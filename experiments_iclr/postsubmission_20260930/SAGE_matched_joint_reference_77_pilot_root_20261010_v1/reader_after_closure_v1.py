"""Wait for both frozen F owners, then directly own one reviewed CPU reader.

Source preparation does not execute this file. Root deploys and starts it only
after actual qualification and the two fixed seed-block owners are launched.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
ROOT_NAME = 'SAGE_matched_joint_reference_77_pilot_root_20261010_v1'
READER_NAME = 'SAGE_matched_joint_reference_77_complete_reader_source_20261010_v1'
HOST = 'peptide'
GPU_UUIDS = ['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998',
             'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
SEEDS = [7301, 7403, 7507]
BLOCKS = {'shard0': [7301, 7403], 'shard1': [7507]}
PYTHON = REPO / '.gnnm_runtime/private_transfer_cp311_cu118_20261005_v1/bin/python'
READER_SHA = 'b2d4deb6e7a1b9c6da5597e851a42caf11ece728f13b95ebd4a3871b01941ef0'
PROTOCOL_SHA = '747dd472737b546b5b11d4a697e3a3555a1b0a5feecd95c7f72b763d02d4789a'
READY_SHA = '8d1b7f1f1638d9cc814d358cab38662bd59001bfdf26ccac88a880e0218199b4'
CONFIG_SHA = '69a5f7dc8e1c38997266dd113161ac1d3a90ac34f46b09ee071b2a7d51001fbb'
FINITE_SECONDS = 3600
CLEANUP_RESERVE_SECONDS = 60
POLL_SECONDS = 20


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def descriptor(path, expected_sha=None):
    path = path.resolve()
    require(path.is_file(), 'Required artifact missing: ' + str(path))
    result = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
    if expected_sha is not None:
        require(result['sha256'] == expected_sha, 'Frozen artifact changed: ' + str(path))
    return result


def read(path):
    return json.loads(path.read_text())


def write_once(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def stopped(signum, frame):
    raise RuntimeError('Watcher interrupted by signal ' + str(signum))


def main():
    began = time.monotonic()
    work_deadline = began + FINITE_SECONDS - CLEANUP_RESERVE_SECONDS
    require(socket.gethostname() == HOST, 'Wrong host; no project operation admitted')
    observed = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, timeout=10).splitlines()
    require(observed == GPU_UUIDS, 'Wrong physical GPU roster')
    root = Path(__file__).resolve().parent
    require(root == PHASE / ROOT_NAME and Path.cwd() == REPO, 'Wrong project root or cwd')
    reader = PHASE / READER_NAME / 'analysis.py'
    protocol = reader.with_name('PROTOCOL.json')
    reader_pin = descriptor(reader, READER_SHA)
    protocol_pin = descriptor(protocol, PROTOCOL_SHA)
    ready_pin = descriptor(root / 'ACTUAL_READY_V1.json', READY_SHA)
    config_pin = descriptor(root / 'CONFIG.json', CONFIG_SHA)
    ready, config = read(Path(ready_pin['path'])), read(Path(config_pin['path']))
    freeze_pin = descriptor(root / 'FREEZE.json')
    freeze = read(Path(freeze_pin['path']))
    require(freeze['study_id'] == root.name, 'Frozen study identity differs')
    require(set(freeze['shards']) == set(BLOCKS), 'Exactly two frozen shards required')
    require(config['seeds'] == SEEDS and config['device'] == 'cuda:0'
            and config['stage'] == 'F' and config['K'] is None
            and config['k_selection_receipt'] is None, 'Frozen F configuration differs')
    require(ready['hostname'] == HOST and ready['gpu_uuids'] == GPU_UUIDS
            and ready['TEST_access'] is False and ready['no_quality_scored'] is True,
            'Actual readiness scope differs')
    require(ready['runtime']['executable'] == str(PYTHON), 'Normal admitted provider differs')
    fingerprint = hashlib.sha256(json.dumps(ready['runtime'], sort_keys=True).encode()).hexdigest()
    require(fingerprint == ready['runtime_fingerprint_sha256'], 'Runtime fingerprint differs')
    for index, (name, block) in enumerate(BLOCKS.items()):
        frozen = freeze['shards'][name]
        require(frozen['seed_block'] == block and frozen['gpu_uuid'] == GPU_UUIDS[index],
                'Frozen shard assignment differs: ' + name)
    output = root / 'complete_analysis_v1'
    input_path = root / 'reader_INPUT_SPEC_V1.json'
    require(not output.exists() and not input_path.exists(), 'Reader artifacts already exist; no rerun')
    # The exclusive permanent claim also prevents concurrent or failed-run retries.
    write_once(root / 'READER_JOB_START_V1.json', {
        'UTC': stamp(), 'PID': os.getpid(), 'study_id': root.name,
        'watcher': descriptor(Path(__file__)), 'finite_seconds': FINITE_SECONDS,
        'cleanup_reserve_seconds': CLEANUP_RESERVE_SECONDS,
        'reader_source': reader_pin, 'protocol': protocol_pin,
        'actual_ready': ready_pin, 'submitted_root_config': config_pin, 'freeze': freeze_pin,
        'waiting_for': [str(root / name / 'OWNER_END.json') for name in BLOCKS],
        'CPU_reader_only': True, 'TEST_access': False, 'retries': 0})
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    child = None
    result = {'success': False, 'status': 'FAILED_RETAINED', 'study_id': root.name,
              'TEST_access': False, 'reader_launches': 0, 'direct_child_wait': False}
    try:
        owners = {}
        while len(owners) != 2:
            require(time.monotonic() < work_deadline, 'Both-owner closure wait timed out')
            for name in BLOCKS:
                path = root / name / 'OWNER_END.json'
                if name in owners or not path.is_file():
                    continue
                # Root writes these normalized terminal records by atomic rename.
                owner = read(path)
                require(owner.get('status') == 'COMPLETE_EXIT_0'
                        and type(owner.get('exit_code')) is int and owner['exit_code'] == 0,
                        'Shard owner failed; no partial readout: ' + name)
                require(all(owner.get(k) is True for k in (
                    'scientific_success', 'complete_family', 'direct_child_wait',
                    'child_pid_absent', 'owned_cuda_pid_absent'))
                    and owner.get('timed_out') is False, 'Real owner closure flags differ: ' + name)
                owners[name] = owner
            if len(owners) != 2:
                time.sleep(min(POLL_SECONDS, max(0, work_deadline - time.monotonic())))

        # No source/data/complete-family descriptor is refreshed until both owners succeed.
        descriptor(root / 'ACTUAL_READY_V1.json', ready_pin['sha256'])
        descriptor(root / 'CONFIG.json', config_pin['sha256'])
        descriptor(root / 'FREEZE.json', freeze_pin['sha256'])
        source_names = {
            'training_family': 'SAGE_matched_joint_reference_77_source_20261010_v1/run_family.py',
            'common_routes': 'shared_fast_graph_model_interface_20261010_v1/common_routes.py',
            'factors': 'portable_internal_be_public_interface_20261007_v2/core/factors.py',
            'models': 'models.py', 'run_base': 'run_base.py', 'run_common': 'run_common.py'}
        sources = {}
        for key, name in source_names.items():
            admitted = ready['sources'][name]
            current = descriptor(Path(admitted['path']), admitted['sha256'])
            require(current == admitted, 'Admitted source metadata differs: ' + key)
            sources[key] = current
        roles = {}
        for role in ('train', 'valid'):
            admitted = ready['roles'][role]['file']
            current = descriptor(Path(admitted['path']), admitted['sha256'])
            require(current == admitted and current['path'] == config[role + '_npz'],
                    'Current safe role differs: ' + role)
            roles[role] = current
        shards = []
        for index, (name, block) in enumerate(BLOCKS.items()):
            here = root / name
            submitted = descriptor(here / 'CONFIG.json', CONFIG_SHA)
            require(read(here / 'CONFIG.json') == config, 'Submitted full-roster config differs: ' + name)
            execution_pin = descriptor(here / 'EXECUTION_RECEIPT.json')
            owner_pin = descriptor(here / 'OWNER_END.json')
            execution, owner = read(Path(execution_pin['path'])), read(Path(owner_pin['path']))
            require(owner == owners[name], 'Successful owner record changed: ' + name)
            expected = {
                'schema_version': 1, 'study_id': root.name, 'stage': 'F', 'K': None,
                'hostname': HOST, 'runtime_fingerprint_sha256': fingerprint,
                'gpu_uuid': GPU_UUIDS[index], 'logical_device': config['device'],
                'seed_block': block, 'submitted_config_sha256': submitted['sha256'],
                'training_family_sha256': sources['training_family']['sha256'],
                'safe_role_hashes': {k: v['sha256'] for k, v in roles.items()}, 'TEST_access': False}
            require(all(execution.get(k) == v and owner.get(k) == v for k, v in expected.items()),
                    'Normalized execution/owner custody differs: ' + name)
            require(execution.get('status') == 'HOST_GPU_SOURCE_DATA_ADMITTED'
                    and execution.get('freeze_sha256') == freeze_pin['sha256'],
                    'Actual frozen admission differs: ' + name)
            family_root = here / 'actual_family_v1'
            # Complete-family bytes are opaque here; the reviewed reader owns its full gate.
            require(isinstance(owner.get('complete_sha256'), str)
                    and len(owner['complete_sha256']) == 64, 'Owner complete-file digest missing: ' + name)
            complete = descriptor(family_root / 'COMPLETE_FAMILY.json', owner['complete_sha256'])
            shards.append({'name': name, 'root': str(family_root), 'submitted_config': submitted,
                           'execution_receipt': execution_pin, 'owner_end_receipt': owner_pin,
                           'complete_family': complete})
        spec = {
            'schema_version': 1, 'study_id': root.name,
            'reader_source': descriptor(reader, READER_SHA),
            'protocol': descriptor(protocol, PROTOCOL_SHA), 'fixed_seed_roster': SEEDS,
            'runtime': {'hostname': HOST, 'fingerprint_sha256': fingerprint,
                        'allowed_gpu_uuids': GPU_UUIDS},
            'expected_nondevice_config': {k: v for k, v in config.items() if k != 'device'},
            'sources': sources, 'safe_roles': roles, 'shards': shards}
        write_once(input_path, spec)
        input_pin = descriptor(input_path)
        require(time.monotonic() < work_deadline, 'No finite reader time remains')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        command = [str(PYTHON), '-I', '-B', str(reader),
                   '--input-spec', str(input_path), '--output', str(output)]
        with (root / 'reader.stdout.log').open('x') as out, (root / 'reader.stderr.log').open('x') as err:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=err, start_new_session=True)
            result.update(reader_launches=1, reader_child_PID=child.pid, input_spec=input_pin,
                          both_successful_owner_closures_before_reader=True)
            write_once(root / 'READER_EXECUTION_START_V1.json', {
                'UTC': stamp(), 'child_PID': child.pid, 'command': command, 'input_spec': input_pin,
                'owner_end_receipts': [s['owner_end_receipt'] for s in shards],
                'both_successful_owner_closures_before_reader': True,
                'CPU_reader_only': True, 'TEST_access': False, 'retries': 0})
            code = child.wait(timeout=max(0.001, work_deadline - time.monotonic()))
            result.update(exit_code=code, direct_child_wait=True)
        require(code == 0, 'Reviewed reader exited unsuccessfully; partial artifacts retained')
        summary_path = output / 'COMPLETE_ANALYSIS_SUMMARY.json'
        summary = read(summary_path)
        closure = {'complete': True, 'study_id': root.name, 'stage': 'F', 'K': None,
                   'TEST_access': False, 'banks': 15, 'optimizer_acquisition_bundles': 24,
                   'native_body_fit_records': 33, 'fixed_seed_roster': SEEDS,
                   'entire_same_runtime_roster_admitted_before_arrays': True,
                   'all75_calibration_attempts_terminal_before_interpretation': True,
                   'input_spec_sha256': input_pin['sha256'], 'reader_source_sha256': READER_SHA,
                   'raw_authority_preserved': True}
        require(all(summary.get(k) == v for k, v in closure.items()), 'Reader summary closure differs')
        require(summary['full_report'] == descriptor(output / 'COMPLETE_REPORT.json'),
                'Complete report custody differs')
        # Scientific gate verdicts remain in the reader output; they do not choose job success.
        result.update(success=True, status='COMPLETE_EXIT_0', complete_summary=descriptor(summary_path))
    except BaseException as exc:
        # Stop only this watcher-owned reader process group, never either fit owner.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        result.update(error_type=type(exc).__name__, error=str(exc))
        if child is not None:
            try:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait(timeout=15)
                else:
                    child.wait()
                result.update(exit_code=child.returncode, direct_child_wait=True)
            except BaseException as cleanup_error:
                result.update(cleanup_error_type=type(cleanup_error).__name__,
                              cleanup_error=str(cleanup_error), exit_code=child.returncode)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    result.update(UTC=stamp(), seconds=time.monotonic() - began,
                  reader_child_PID=child.pid if child is not None else None,
                  reader_child_reaped=child is not None and child.returncode is not None,
                  CPU_reader_only=True, finite_seconds=FINITE_SECONDS, retries=0)
    write_once(root / 'READER_JOB_END_V1.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'success', 'reader_launches', 'seconds', 'TEST_access')},
                     allow_nan=False), flush=True)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
