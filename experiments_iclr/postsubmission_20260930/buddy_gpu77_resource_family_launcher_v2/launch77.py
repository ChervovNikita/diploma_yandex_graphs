"""Root-only staged resource/family launcher; delegates unchanged sealed BUDDY v5."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
SOURCE = PHASE / 'buddy_shared_cache_execution_v5'
EXTRA = REPO / '.gnnm_runtime/buddy_extra_v1/site'
PYTHON = Path('/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python')
UUIDS = ('GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998', 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced')
SOURCE_SHA = 'c479cedbd244ff645c7ee822625120b9f809d5082f4d9d4ace265237b5712e0f'
QUAL_DIR = PHASE / 'gpu77_buddy_numerical_qualification_v2/root_run_v1'
QUAL = QUAL_DIR / 'CPU_QUALIFICATION.json'
DATA_DIR = PHASE / 'buddy_complete_data_cache_preparation_v3/root_run_77_v1'
CACHE = DATA_DIR / 'cache'
RESOURCE = PHASE / 'buddy_gpu77_resource_family_launcher_v2/root_resource_v1'
FAMILY = PHASE / 'buddy_gpu77_resource_family_launcher_v2/root_family_v1'
ARMS = ('native1024', 'single256', 'factorized4', 'independent4', 'matched_single')
SEEDS = (0, 1, 2)
PARAMETERS = dict(native1024=1185091, single256=99907, factorized4=106349, independent4=399628, matched_single=106457)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject(value):
        raise RuntimeError(f'Nonfinite JSON constant: {value}')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique)


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def confined(path):
    absolute = Path(path).absolute()
    if absolute != absolute.resolve() or not absolute.is_relative_to(REPO) or absolute == REPO:
        raise RuntimeError(f'Repository-confined canonical path required: {path}')
    return absolute


def verify_manifest(folder, expected=None):
    folder = confined(folder)
    actual = sha(folder / 'SOURCE_MANIFEST.json')
    if (expected is not None and actual != expected) or read_json(folder / 'SEAL.json')['source_manifest_sha256'] != actual:
        raise RuntimeError('Source seal/manifest differs')
    for item in read_json(folder / 'SOURCE_MANIFEST.json')['files']:
        path = confined(folder / item['path'])
        if sha(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise RuntimeError(f'Source byte differs: {path}')
    return actual


def environment():
    env = dict(os.environ, PYTHONPATH=str(EXTRA), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               NUMEXPR_NUM_THREADS='4', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    env.pop('TORCH_FORCE_WEIGHTS_ONLY_LOAD', None)
    return env


def context():
    started = time.monotonic()
    if Path.cwd().resolve() != REPO or not HERE.is_relative_to(REPO):
        raise RuntimeError('Execute this synced packet from the actual authorized checkout')
    top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()
    if Path(top).resolve() != REPO or Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError('Actual repository/base interpreter differs')
    actual = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, capture_output=True, check=True).stdout.splitlines()
    if len(actual) != 2 or set(actual) != set(UUIDS):
        raise RuntimeError('Actual physical GPU UUID inventory differs')
    wrapper_sha = verify_manifest(HERE)
    verify_manifest(SOURCE, SOURCE_SHA)
    contract = read_json(HERE / 'CONTRACT.json')
    verify_manifest(DATA_DIR.parent, contract['data_wrapper_source_manifest_sha256'])
    sources = {row['path']: row['sha256'] for row in read_json(SOURCE / 'SOURCE_MANIFEST.json')['files']}
    expected_implementation = {name: digest for name, digest in sources.items()
                               if (name.endswith('.py') and '/' not in name) or name.startswith('vendor/')
                               or name in {'CONFIG.json', 'SOURCE_PINS.json', 'requirements-qualification-extra.txt'}}
    certificate = read_json(QUAL)
    receipt = read_json(QUAL_DIR / 'CPU_RUN_RECEIPT.json')
    if (certificate.get('status') != 'synthetic_cpu_pass' or certificate.get('test_count') != 7
            or str(certificate.get('torch_version', '')).split('+')[0] != '2.7.1'
            or certificate.get('test_script_sha256') != sources['test_cpu.py']
            or certificate.get('implementation_hashes') != expected_implementation
            or receipt.get('status') != 'SEVEN_CPU_CHECKS_PASSED' or receipt.get('exit_code') != 0
            or receipt.get('certificate') != certificate or receipt.get('actual_git_root') != str(REPO)
            or set(receipt.get('physical_GPU_UUIDs', [])) != set(UUIDS)
            or receipt.get('source_manifest_sha256') != SOURCE_SHA or receipt.get('other_jobs_stopped') is not False
            or receipt.get('GPU_compute') is not False or receipt.get('dataset_access') is not False):
        raise RuntimeError('Actual 77 exact-v5 seven-test qualification is required')
    setup = read_json(EXTRA.parent / 'SETUP_RECEIPT.json')
    if setup.get('exit_code') != 0:
        raise RuntimeError('Isolated runtime setup failed')
    for name, digest in setup['installed_RECORD_sha256'].items():
        if sha(confined(EXTRA / name)) != digest:
            raise RuntimeError('Isolated dependency RECORD differs')
    cache = read_json(CACHE / 'manifest.json')
    data = read_json(DATA_DIR / 'QUALIFICATION.json')
    if (data.get('status') != 'complete_official_train_valid_cache_qualified'
            or data.get('builder_manifest_sha256') != SOURCE_SHA
            or data.get('wrapper_manifest_sha256') != contract['data_wrapper_source_manifest_sha256']
            or data.get('CPU_qualification_sha256') != sha(QUAL)
            or data.get('Torch_version') != certificate['torch_version']
            or data.get('cache_output') != str(CACHE) or data.get('test_members_opened') is not False
            or data.get('combined_split_accessor_called') is not False
            or data.get('cache_manifest_sha256') != sha(CACHE / 'manifest.json')
            or cache.get('test_split_opened') is not False or cache.get('graph_policy') != 'training_only_all_splits'
            or cache.get('implementation_hashes') != certificate['implementation_hashes']
            or set(cache['files']) != {'common.pt', 'train.pt', 'valid.pt', 'official_split_loader.py.txt'}):
        raise RuntimeError('Complete common train/valid cache qualification is required')
    for name, digest in cache['files'].items():
        if sha(confined(CACHE / name)) != digest:
            raise RuntimeError('Qualified common cache byte differs')
    env = environment()
    env['CUDA_VISIBLE_DEVICES'] = ''
    runtime = subprocess.run([str(PYTHON), '-B', '-c', 'import json, torch; print(json.dumps(dict(torch_version=str(torch.__version__), cpu_threads=torch.get_num_threads())))'], env=env, text=True, capture_output=True, check=True)
    runtime = json.loads(runtime.stdout)
    if runtime != {'torch_version': certificate['torch_version'], 'cpu_threads': 4}:
        raise RuntimeError('Qualified Torch runtime or four CPU threads differs')
    return dict(wrapper_manifest_sha256=wrapper_sha, source_manifest_sha256=SOURCE_SHA,
                CPU_qualification_sha256=sha(QUAL), CPU_execution_receipt_sha256=sha(QUAL_DIR / 'CPU_RUN_RECEIPT.json'),
                data_qualification_sha256=sha(DATA_DIR / 'QUALIFICATION.json'), cache_manifest_sha256=sha(CACHE / 'manifest.json'),
                actual_git_root=str(REPO), torch_version=certificate['torch_version'], cpu_threads=4,
                physical_GPU_UUIDs=list(UUIDS), implementation_hashes=certificate['implementation_hashes'],
                preflight_seconds=time.monotonic() - started,
                concurrent_host_cost_limitation='Other authorized jobs remain active; observed timing/memory is contention-dependent and does not establish isolated speedup.')


def command(stage, arm, output):
    return [str(PYTHON), '-B', str(SOURCE / 'run.py'), stage, '--cache', str(CACHE),
            '--arm', arm, '--seed', '0', '--device', 'cuda:0', '--qualification', str(QUAL), '--output', str(output)]


def family_command():
    return [str(PYTHON), '-B', str(SOURCE / 'launch_family.py'), '--cache', str(CACHE), '--runs', str(FAMILY / 'runs'),
            '--qualification', str(QUAL), '--gpus', ','.join(UUIDS), '--execute']


def resource_evidence(ctx):
    result = []
    for arm in ARMS:
        folder = RESOURCE / arm
        completion = read_json(folder / 'completion.json')
        expected = dict(arm=arm, seed=0, status='resource_only_complete', epochs_completed=1,
                        cache_manifest_sha256=ctx['cache_manifest_sha256'], torch_version=ctx['torch_version'],
                        parameters=PARAMETERS[arm], optimizer_fits=4 if arm == 'independent4' else 1,
                        implementation_hashes=ctx['implementation_hashes'], test_loaded_or_scored=False)
        if any(completion.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f'Complete fixed resource epoch required: {arm}')
        lines = (folder / 'epochs.jsonl').read_text().splitlines()
        if len(lines) != 1:
            raise RuntimeError('Resource ledger must contain one complete epoch')
        record = json.loads(lines[0])
        if record.get('epoch') != 1 or any(key in record for key in ('validation_hits50', 'selected_checkpoint_sha256')):
            raise RuntimeError('Resource pass must be forward-only without selection')
        for key in ('train_bce', 'train_seconds', 'validation_forward_seconds'):
            if type(record.get(key)) not in (int, float) or not math.isfinite(record[key]) or record[key] < 0:
                raise RuntimeError('Finite resource record required')
        result.append(dict(arm=arm, completion_sha256=sha(folder / 'completion.json'),
                           identity_sha256=sha(folder / 'identity.json'), epochs_sha256=sha(folder / 'epochs.jsonl'),
                           record=record, peak_cuda_allocated=completion['peak_cuda_allocated'], peak_cuda_reserved=completion['peak_cuda_reserved']))
    return result


def resources(ctx):
    RESOURCE.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(schema='buddy77-all-arm-resource-receipt-v1', UTC=datetime.now(timezone.utc).isoformat(),
                  context=ctx, status='in_progress', commands=[], other_jobs_stopped=False, test_access=False)
    try:
        for index, arm in enumerate(ARMS):
            gpu = UUIDS[index % len(UUIDS)]
            argv = command('resource', arm, RESOURCE / arm)
            env = environment(); env['CUDA_VISIBLE_DEVICES'] = gpu
            begin = time.monotonic()
            with (RESOURCE / (arm + '.log')).open('x') as log:
                run = subprocess.run(argv, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
            result['commands'].append(dict(arm=arm, GPU_UUID=gpu, argv=argv, exit_code=run.returncode, command_wall_seconds=time.monotonic() - begin))
            if run.returncode:
                raise RuntimeError(f'Resource arm failed: {arm}')
        result['resources'] = resource_evidence(ctx)
        result['forecast_train_and_validation_seconds_15_cells'] = 300 * sum(row['record']['train_seconds'] + row['record']['validation_forward_seconds'] for row in result['resources'])
        result.update(status='all_five_complete_resource_epochs', resource_optimizer_fits=8,
                      family_cells=15, family_optimizer_fits=24, family_epochs_per_cell=100,
                      family_validation_forwards=1500, prospective_root_admission_required=True,
                      unmeasured_costs=['Family checkpoint writes, validation metric/selection overhead, fixed fit setup/cache reads and later final scoring require separate cost accounting.'])
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result['total_resource_wrapper_seconds'] = time.monotonic() - started + ctx['preflight_seconds']
        write_json(RESOURCE / 'RESOURCE_RECEIPT.json', result)


def family(ctx, admission_path):
    path = confined(admission_path)
    resources_saved = read_json(RESOURCE / 'RESOURCE_RECEIPT.json')
    evidence = resource_evidence(ctx)
    if (resources_saved.get('status') != 'all_five_complete_resource_epochs'
            or resources_saved.get('context') != ctx or resources_saved.get('resources') != evidence):
        # Preflight timing changes across invocations; it is not an identity.
        previous = dict(resources_saved.get('context', {})); current = dict(ctx)
        previous.pop('preflight_seconds', None); current.pop('preflight_seconds', None)
        if resources_saved.get('status') != 'all_five_complete_resource_epochs' or previous != current or resources_saved.get('resources') != evidence:
            raise RuntimeError('Unchanged all-five resource evidence required before family')
    admission = read_json(path)
    required = dict(schema='buddy77-prospective-family-admission-v1', decision='admitted',
                    source_manifest_sha256=SOURCE_SHA, wrapper_manifest_sha256=ctx['wrapper_manifest_sha256'],
                    cache_manifest_sha256=ctx['cache_manifest_sha256'], CPU_qualification_sha256=ctx['CPU_qualification_sha256'],
                    resource_receipt_sha256=sha(RESOURCE / 'RESOURCE_RECEIPT.json'), family_cells=15, optimizer_fits=24,
                    cpu_threads=4, physical_GPU_UUIDs=list(UUIDS))
    if any(admission.get(key) != value for key, value in required.items()) or not admission.get('root_resource_cost_decision'):
        raise RuntimeError('Separate prospective root resource admission required')
    FAMILY.mkdir(parents=True, exist_ok=False)
    argv = family_command()
    started = time.monotonic()
    with (FAMILY / 'family.log').open('x') as log:
        run = subprocess.run(argv, cwd=SOURCE, env=environment(), stdout=log, stderr=subprocess.STDOUT)
    write_json(FAMILY / 'FAMILY_LAUNCH_RECEIPT.json', dict(schema='buddy77-family-launch-receipt-v1',
               UTC=datetime.now(timezone.utc).isoformat(), context=ctx, admission_sha256=sha(path),
               resource_receipt_sha256=sha(RESOURCE / 'RESOURCE_RECEIPT.json'), argv=argv, exit_code=run.returncode,
               seconds=time.monotonic() - started, family_cells=15, optimizer_fits=24, other_jobs_stopped=False, test_access=False))
    if run.returncode:
        raise RuntimeError('Family incomplete; review its logs and existing outputs before any replay')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['resource', 'family'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--admission')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(stage=args.stage, resource_commands=[command('resource', arm, RESOURCE / arm) for arm in ARMS],
                              family_command=family_command(),
                              GPU_UUIDs=list(UUIDS), cache=str(CACHE), family_cells=15, optimizer_fits=24,
                              actual_77_numerical_and_data_qualification_required=True, separate_family_admission_required=True), indent=2))
        return
    if args.stage == 'family' and not args.admission:
        parser.error('family execution requires a separately prepared --admission receipt')
    ctx = context()
    resources(ctx) if args.stage == 'resource' else family(ctx, args.admission)


if __name__ == '__main__':
    main()
