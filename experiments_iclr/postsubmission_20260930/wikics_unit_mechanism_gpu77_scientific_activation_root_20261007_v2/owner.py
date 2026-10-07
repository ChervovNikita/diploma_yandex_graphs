"""Run the fixed twelve attribution fits in two detached, bounded GPU lanes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import datetime
import hashlib
import importlib.util
import json
import os
import socket
import subprocess

R = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
P = R / 'experiments_iclr/postsubmission_20260930'
A = P / 'wikics_unit_mechanism_gpu77_scientific_activation_root_20261007_v2'
S = P / 'internal_BE_WikiCS_unit_mechanism_ablation_source_20261007_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def phase_file(relative):
    path = (P / relative).resolve()
    assert path.is_relative_to(P) and path.is_file()
    return path


def module(name, row):
    path = phase_file(row['path'])
    assert sha(path) == row['sha256']
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def physical():
    assert socket.gethostname() == 'peptide' and Path.cwd() == R
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
        '--format=csv,noheader'], text=True).splitlines() == pins['physical_gpu_inventory']


def lane(index):
    uuid = pins['physical_gpu_inventory'][index]
    helper = module('unit_mechanism_owned_' + str(index), pins['reviewed_ownership_helper'])
    helper.GPU = uuid
    supervisor = module('unit_mechanism_runfit_' + str(index), pins['reviewed_owned_fit_helper'])
    context = SimpleNamespace(REPO=R, SOURCE=S, SOURCE_SHA=bundle['source_manifest_sha256'],
        GPU_UUID=uuid, GPU_UUIDS=tuple(pins['physical_gpu_inventory']),
        phase_file=phase_file, physical_host=physical, sha=sha, write=write)
    limits = dict(owned_tree_GPU_memory_cap_bytes=34359738368,
        owned_tree_RSS_cap_bytes=34359738368, combined_child_log_cap_bytes=8388608,
        own_fit_output_cap_bytes=4294967296, minimum_fresh_GPU_free_bytes=38654705664,
        resource_wait_seconds=1800, poll_interval_seconds=5, telemetry_timeout_seconds=10)
    completed = []
    try:
        for item in bundle['lanes'][str(index)]:
            physical()
            path = phase_file(item['path'])
            assert sha(path) == item['sha256'] and sha(S / 'MANIFEST.json') == bundle['source_manifest_sha256']
            cfg = json.loads(path.read_text())
            assert cfg['physical_gpu_uuid'] == uuid
            output = P / cfg['output']
            assert output.resolve().is_relative_to(P) and not output.exists()
            entry = dict(cell_id=item['cell_id'], job_relative=item['path'],
                job_sha256=item['sha256'], hard_seconds=cfg['external_active_seconds'],
                argv=[pins['python'], '-B', str(S / 'run_cell.py'),
                    '--release', str(path), '--release-sha256', item['sha256']])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONPATH='',
                PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
            env.pop('PYTHONHOME', None)
            receipt = supervisor.run_fit(helper, A, entry, {'resource_limits': limits}, env, output, context)
            child = receipt['raw_identity_observation']
            absent = child is not None and helper.identity(child['PID']) is None
            cuda = helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory',
                '--format=csv,noheader,nounits'], 10)
            no_cuda = child is not None and not any(len(parts) > 1 and parts[1].strip() == str(child['PID'])
                for parts in (row.split(',') for row in cuda))
            assert receipt['exit_code'] == 0 and receipt['reason'] is None and not receipt['signals_sent']
            assert receipt['terminal_wait_observed'] and absent and no_cuda
            assert not (output / 'FAILURE.json').exists()
            endpoint = json.loads((output / 'COMPLETE.json').read_text())
            assert endpoint['complete'] and endpoint['epochs'] == endpoint['steps'] == 1100
            assert endpoint['seed'] == cfg['seed'] and endpoint['mechanism_ablation']['condition'] == cfg['condition']
            assert endpoint['execution_accounting'] == dict(shadow_member_forwards=8800,
                replay_member_forwards=8800, output_cotangent_collections=1100,
                member_reverse_collections=8800, optimizer_bank_updates=1100,
                exact_member_RNG_endpoint_checks=1100)
            completed.append(dict(cell_id=item['cell_id'], complete=True,
                completion_sha256=sha(output / 'COMPLETE.json'), exit_receipt=receipt,
                child_absent=absent, child_no_CUDA_rows=no_cuda))
            write(A / ('LANE_' + str(index) + '_PROGRESS.json'), dict(completed=completed,
                quality_scores_read=False, TEST_access=False))
        write(A / ('LANE_' + str(index) + '_COMPLETE.json'), dict(complete=True,
            completed=completed, quality_scores_read=False, TEST_access=False))
        return True
    except Exception as error:
        write(A / ('LANE_' + str(index) + '_FAILURE.json'), dict(complete=False,
            completed=completed, error=type(error).__name__ + ': ' + str(error), automatic_retry=False))
        return False


if __name__ == '__main__':
    physical_pins = json.loads((S / 'SOURCE_BINDINGS.json').read_text())
    pins = physical_pins
    physical()
    bundle = json.loads((A / 'BUNDLE.json').read_text())
    assert bundle['enabled'] and bundle['fixed12_protocol_adopted']
    assert sha(A / 'owner.py') == bundle['owner_sha256']
    assert sha(S / 'MANIFEST.json') == bundle['source_manifest_sha256']
    assert [item['cell_id'] for key in ('0', '1') for item in bundle['lanes'][key]] == [
        str(seed) + '_' + condition for seed in (6101, 6307, 6203)
        for condition in ('plain', 'alignment_only', 'residual_only', 'combined')]
    qualifier = phase_file(bundle['qualification']['path'])
    assert sha(qualifier) == bundle['qualification']['sha256']
    assert json.loads(qualifier.read_text())['complete']
    (A / 'logs').mkdir()
    owner_helper = module('unit_mechanism_parent_identity', pins['reviewed_ownership_helper'])
    write(A / 'PARENT_OWNER.json', owner_helper.identity(os.getpid()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(lane, (0, 1)))
    write(A / 'FAMILY_CLOSURE.json', dict(complete=all(result), lane_results=result,
        fixed_scientific_cells=12, quality_scores_read=False, TEST_access=False,
        automatic_retry=False, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
