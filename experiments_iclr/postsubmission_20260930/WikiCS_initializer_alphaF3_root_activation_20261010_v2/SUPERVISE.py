"""F-only invocation over the existing reviewed finite run_fit helper."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
GPUS = ('GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998',
        'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def bound(row):
    path = (PHASE / row['path']).resolve(strict=True)
    assert path.is_relative_to(PHASE) and path.is_file()
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    return path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--supervision-output', required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'peptide'
    assert tuple(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                 '--format=csv,noheader'], text=True, timeout=10).splitlines()) == GPUS
    assert Path.cwd().resolve() == REPO
    release = args.release.resolve(strict=True)
    assert release.is_relative_to(PHASE) and sha(release) == args.release_sha256
    cfg = read(release)
    assert cfg['enabled'] is True and cfg['root_execution_authorized'] is True
    supervision = read(bound(cfg['external_supervision']))
    assert supervision['enabled'] is True and supervision['finite_owned_bound_confirmed'] is True
    assert supervision['release_argument_path'] == str(release)
    entry_program = bound(supervision['owned_entry_program'])
    assert supervision['argv_prefix'][1:] == ['-B', str(entry_program)]
    assert supervision['physical_gpu_uuid'] == cfg['physical_gpu_uuid'] and cfg['physical_gpu_uuid'] in GPUS
    owner_path = bound(supervision['existing_run_fit_helper'])
    helper_path = bound(supervision['existing_ownership_helper'])
    output = (PHASE / cfg['output_directory']).resolve()
    root = (PHASE / args.supervision_output).resolve()
    assert root.is_relative_to(PHASE) and root.parent == PHASE and not root.exists()
    assert output.parent == PHASE and output != root and not output.exists()
    assert root != release.parent and root != entry_program.parent
    root.mkdir(mode=0o700)
    (root / 'logs').mkdir(mode=0o700)
    owner = load(owner_path, '_relation18_existing_run_fit')
    helper, context = owner.lane(cfg['physical_gpu_uuid'])
    assert owner.HELPER == helper_path
    context.SOURCE = entry_program.parent
    context.SOURCE_SHA = cfg['readout_manifest_sha256']
    identity = helper.identity(os.getpid())
    write(root / 'PARENT_OWNER.json', dict(identity=identity, release=str(release),
          release_sha256=args.release_sha256, detached_parent_OS_exit=None,
          existing_run_fit=supervision['existing_run_fit_helper'],
          existing_ownership_helper=supervision['existing_ownership_helper'],
          shim_sha256=sha(__file__), numerical_execution_by_parent=False))
    limits = dict(minimum_fresh_GPU_free_bytes=cfg['minimum_fresh_GPU_free_bytes'],
          resource_wait_seconds=0, telemetry_timeout_seconds=10, poll_interval_seconds=5,
          owned_tree_RSS_cap_bytes=supervision['owned_RSS_cap_bytes'],
          owned_tree_GPU_memory_cap_bytes=cfg['owned_GPU_cap_bytes'],
          own_fit_output_cap_bytes=cfg['maximum_output_bytes'],
          combined_child_log_cap_bytes=supervision['combined_child_log_cap_bytes'])
    argv = supervision['argv_prefix'] + ['--release', str(release),
                                       '--release-sha256', args.release_sha256]
    entry = dict(cell_id='WikiCS_initializer_alphaF3_readout', job_relative=str(release.relative_to(PHASE)),
                 job_sha256=args.release_sha256, argv=argv,
                 hard_seconds=cfg['external_active_seconds'])
    # The existing helper kills only its verified child group at the active
    # bound and directly waits up to five seconds, within the released cleanup
    # allowance. No new polling, signal, or ownership implementation is added.
    assert cfg['external_cleanup_seconds'] >= 5
    assert cfg['external_hard_seconds'] >= cfg['external_active_seconds'] + 5
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=cfg['physical_gpu_uuid'], PYTHONPATH='')
    environment.pop('PYTHONHOME', None)
    receipt = None
    failure = None
    try:
        receipt = owner.run_fit(helper, root, entry, {'resource_limits': limits},
                                environment, output, context)
        child = receipt['raw_identity_observation']
        assert child is not None
        current = helper.identity(child['PID'])
        absent = current is None or current['start_ticks'] != child['start_ticks']
        cuda = helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory',
                             '--format=csv,noheader,nounits'], 10)
        no_cuda = not any(len(parts) >= 2 and parts[1].strip() == str(child['PID'])
                         for parts in (row.split(',') for row in cuda))
        write(root / 'CHILD_TERMINAL_OBSERVATION.json', dict(identity=child,
              child_absent=absent, child_no_CUDA_rows=no_cuda,
              actual_exit_code=receipt['exit_code'],
              actual_direct_wait=receipt['terminal_wait_observed']))
        assert receipt['exit_code'] == 0 and receipt['terminal_wait_observed'] is True
        assert receipt['reason'] is None and not receipt['signals_sent']
        assert receipt['signal_refusal'] is None and absent and no_cuda
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        write(root / 'SUPERVISION_RESULT.json', dict(complete=failure is None and receipt is not None,
              child_exit_receipt=receipt, failure=failure, release=str(release),
              release_sha256=args.release_sha256, original_experiment_criteria_changed=False,
              active_seconds=cfg['external_active_seconds'],
              cleanup_allowance_seconds=cfg['external_cleanup_seconds'],
              hard_ceiling_seconds=cfg['external_hard_seconds'],
              detached_parent_direct_wait_observed=False, detached_parent_OS_exit=None,
              automatic_retry=False, numerical_outputs_server_only=True))
    print(json.dumps(dict(complete=True, supervision_output=str(root),
                         child_exit_code=receipt['exit_code'])))


if __name__ == '__main__':
    main()
