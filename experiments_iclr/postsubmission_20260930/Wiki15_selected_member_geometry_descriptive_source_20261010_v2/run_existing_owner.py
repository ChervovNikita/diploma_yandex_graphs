"""One CPU Wiki15 aggregate through the existing reviewed run_fit helper.

No new supervision loop, termination primitive, GPU allocation or retry.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
EXEC = PHASE / 'Wiki15_selected_member_geometry_descriptive_execution_root_20261010_v2'
OWNER = PHASE / 'citeseer_known_ranking_control_matched_reference_owned_preparation_20261006_v2/owner.py'
OWNER_SHA = '1b1edf43a895600515bbf340fbc813b5dbc48e325ef73603fc5118146d4d5fdf'
HELPER = PHASE / 'shared_private_transfer_gpu77_qualification_preparation_20261005_v3/ownership_helpers.py'
HELPER_SHA = 'e71503c87865546319cddbbf7a4f9f15d13cdf9e4875e406d65de21e64e047fd'
GPU = 'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'
INVENTORY = [GPU, 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
LIMITS = dict(combined_child_log_cap_bytes=262144, minimum_fresh_GPU_free_bytes=0,
              own_fit_output_cap_bytes=1048576, owned_tree_GPU_memory_cap_bytes=0,
              owned_tree_RSS_cap_bytes=2147483648, poll_interval_seconds=1,
              resource_wait_seconds=0, telemetry_timeout_seconds=5)
ENV = dict(CUDA_VISIBLE_DEVICES='', PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(row):
    path = (PHASE / row['path']).resolve(strict=True)
    assert path.is_relative_to(PHASE) and path.is_file()
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-adopted', action='store_true')
    parser.add_argument('--adoption', type=Path, required=True)
    parser.add_argument('--adoption-sha256', required=True)
    args = parser.parse_args()
    if not args.root_adopted:
        parser.error('Disabled until explicit root adoption; no child started')
    assert socket.gethostname() == 'peptide'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True, timeout=10).splitlines() == INVENTORY
    assert Path.cwd().resolve() == REPO
    assert args.adoption.resolve(strict=True) == EXEC / 'ROOT_ADOPTION.json'
    assert sha(args.adoption) == args.adoption_sha256
    adoption = json.loads(args.adoption.read_text())
    assert adoption['execution_enabled'] is adoption['source_review_approved'] is True
    assert adoption['CPU_only'] is True and adoption['retry'] is False
    assert adoption['TEST_access'] is False and adoption['maximum_member_forwards'] == adoption['maximum_fits'] == 0
    assert adoption['external_hard_seconds'] == 100 and adoption['resource_limits'] == LIMITS and adoption['env'] == ENV
    assert bound(adoption['owner_adapter']) == Path(__file__).resolve()
    manifest = bound(adoption['source_manifest'])
    for row in json.loads(manifest.read_text())['files']:
        bound(dict(row, path=str(manifest.parent.relative_to(PHASE) / row['path'])))
    worker, config = bound(adoption['aggregate']), bound(adoption['config'])
    assert adoption['python'] == json.loads(config.read_text())['python']
    assert bound(adoption['existing_owner']) == OWNER and sha(OWNER) == OWNER_SHA
    assert bound(adoption['ownership_helper']) == HELPER and sha(HELPER) == HELPER_SHA
    module = importlib.util.spec_from_file_location('_Wiki15_existing_finite_owner', OWNER)
    old = importlib.util.module_from_spec(module)
    module.loader.exec_module(old)
    old.physical_host()
    helper, _ = old.lane(GPU)
    owner_root, output = EXEC / 'owner', EXEC / 'result'
    owner_root.mkdir(); (owner_root / 'logs').mkdir(); output.mkdir()
    old.write(owner_root / 'ROOT_ADOPTION.json', adoption)
    context = SimpleNamespace(REPO=REPO, SOURCE=manifest.parent, SOURCE_SHA=adoption['source_manifest']['sha256'],
                              GPU_UUID=GPU, GPU_UUIDS=old.GPU_UUIDS, phase_file=old.phase_file,
                              physical_host=old.physical_host, sha=old.sha, write=old.write)
    argv = [adoption['python'], '-B', str(worker), '--root-adopted', '--config', str(config),
            '--config-sha256', adoption['config']['sha256'], '--output', str(output / 'AGGREGATE.json')]
    entry = dict(cell_id='Wiki15_geometry_CPU', argv=argv, job_relative=str(args.adoption.relative_to(PHASE)),
                 job_sha256=args.adoption_sha256, hard_seconds=100)
    environment = dict(os.environ, **ENV); environment.pop('PYTHONHOME', None)
    terminal = old.run_fit(helper, owner_root, entry, {'resource_limits': LIMITS}, environment, output, context)
    identity = terminal['raw_identity_observation']
    absent = identity is not None and helper.identity(identity['PID']) is None
    cuda = helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], 5)
    no_cuda = identity is not None and not any(len(parts) > 1 and parts[1].strip() == str(identity['PID'])
                                              for parts in (row.split(',') for row in cuda))
    old.write(owner_root / 'PHYSICAL_TERMINAL.json', dict(owned_PID_absent=absent,
              owned_PID_no_CUDA_rows=no_cuda, terminal_wait_observed=terminal['terminal_wait_observed'],
              child_PID=identity['PID'] if identity else None, partial_progress_preserved=True, retry=False))
    assert terminal['exit_code'] == 0 and terminal['reason'] is None and not terminal['signals_sent']
    assert terminal['terminal_wait_observed'] and absent and no_cuda
    result = json.loads((output / 'AGGREGATE.json').read_text())
    progress = json.loads((output / 'AGGREGATE.json.progress.json').read_text())
    assert result['status'] == progress['status'] == 'complete' and len(result['cells']) == progress['completed_cell_count'] == 15
    assert result['source'] == progress['source'] == adoption['aggregate'] and result['config'] == progress['config'] == adoption['config']
    old.write(owner_root / 'COMPLETE.json', dict(passed=True, aggregate_sha256=sha(output / 'AGGREGATE.json'),
              progress_sha256=sha(output / 'AGGREGATE.json.progress.json'), terminal=terminal,
              model_forwards=0, fits=0, scores_recomputed=False, retry=False))


if __name__ == '__main__':
    main()
