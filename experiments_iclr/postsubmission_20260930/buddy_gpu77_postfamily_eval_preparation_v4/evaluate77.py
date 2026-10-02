"""Root-only production lock audit and separately admitted, one-time final scoring.

Dry-run is the default. No SSH, download, fitting, hyperparameter choice or
automatic replay is provided. Preparation alone never authorizes test access.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from gate_contract import require, matches, terminal_gate, complete_cells_gate, evaluation_admission_gate
from gate_contract import archive_path_gate, audit_admission_gate
from runtime_paths import PATHS, path_environment, create_paths
from runtime_boundary import current_execution_boundary, historical_family_boundary


HERE = Path(__file__).resolve().parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
SOURCE = PHASE / 'buddy_shared_cache_execution_v5'
DATA = PHASE / 'buddy_complete_data_cache_preparation_v3'
LAUNCHER = PHASE / 'buddy_gpu77_resource_family_launcher_v3'
PYTHON = '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python'
UUIDS = ['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998', 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
SOURCE_SHA = 'c479cedbd244ff645c7ee822625120b9f809d5082f4d9d4ace265237b5712e0f'
DATA_SHA = '8469488ee3338b4488b0ff310fe23840a91080dc6b5939ff350d6092abeda348'
LAUNCHER_SHA = '9ac4c0c8f475b61c612c4345cee7f8ef6d133cdb6eacd2f7cd13f83ff121c410'
QUAL = PHASE / 'gpu77_buddy_numerical_qualification_v2/root_run_v1/CPU_QUALIFICATION.json'
DATA_RUN = DATA / 'root_run_77_v1'
DATASET = DATA_RUN / 'dataset'
CACHE = DATA_RUN / 'cache'
ARCHIVE = PHASE / 'buddy_official_archive_staging_v1/root_transfer_v2/collab_stream77_v1.zip'
TEST_MEMBER = 'collab/split/time/test.pt'
TEST_SPLIT = DATASET / 'ogbl_collab/split/time/test.pt'
RESOURCE = PHASE / 'buddy_gpu77_resource_family_launcher_v2/root_resource_v1/RESOURCE_RECEIPT.json'
FAMILY_ADMISSION = LAUNCHER / 'ROOT_ADMISSION.json'
TERMINAL = LAUNCHER / 'root_family_v2/FAMILY_LAUNCH_RECEIPT.json'
FAMILY_RUNTIME = LAUNCHER / 'root_family_v2/RUNTIME_ENVIRONMENT.json'
EXCLUDED_PRIOR_FAMILY = PHASE / 'buddy_gpu77_resource_family_launcher_v2/root_family_v1'
RUNS = LAUNCHER / 'root_family_v2/runs'
LOCK_DIR = HERE / 'root_lock_v1'
LOCK = LOCK_DIR / 'FAMILY_LOCK.json'
LOCK_AUDIT = LOCK_DIR / 'LOCK_AUDIT.json'
EVAL_DIR = HERE / 'root_eval_v1'
ADMISSION = HERE / 'ROOT_EVALUATION_ADMISSION.json'
AUDIT_ADMISSION = HERE / 'ROOT_LOCK_AUDIT_ADMISSION.json'
CACHE_ENVIRONMENTS = PATHS



def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_staging_module():
    # Verify the exact bootstrap module before importing even its stdlib code.
    manifest_path = DATA / 'SOURCE_MANIFEST.json'
    require(hashlib.sha256(manifest_path.read_bytes()).hexdigest() == DATA_SHA and
            json.loads((DATA / 'SEAL.json').read_text())['source_manifest_sha256'] == DATA_SHA,
            'Data bootstrap source seal differs')
    item = next(row for row in json.loads(manifest_path.read_text())['files'] if row['path'] == 'prepare_cache.py')
    path = DATA / 'prepare_cache.py'
    require(not path.is_symlink() and path.stat().st_size == item['bytes'] and
            hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Data bootstrap module changed')
    return load_module(path, 'sealed_buddy77_data_preparation')


def lock_command():
    return [PYTHON, '-B', str(SOURCE / 'run.py'), 'lock', '--cache', str(CACHE),
            '--runs', str(RUNS), '--output', str(LOCK)]


def score_command():
    return [PYTHON, '-B', str(SOURCE / 'run.py'), 'test', '--cache', str(CACHE),
            '--family-lock', str(LOCK), '--qualification', str(QUAL), '--device', 'cuda:0']


def deny_existing_test_artifacts():
    paths = [TEST_SPLIT, CACHE / 'test.pt', CACHE / 'test_manifest.json']
    paths += list(RUNS.glob('*/final_test.json'))
    require(not any(path.exists() or path.is_symlink() for path in paths),
            'Test staging/results already exist; explicit root replay/recovery review required')


def bootstrap_environment():
    # Family runtime paths already exist; no new folder or scientific output is
    # created before closure and stage admission. Current external proof is checked
    # before the first Torch preflight, and these values match the sealed launcher.
    existing = LAUNCHER / 'root_runtime_v1'
    values = path_environment(REPO, existing)
    require(all(Path(values[name]).is_dir() for name in PATHS), 'Qualified repo-local family runtime directories required')
    values.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    os.environ.update(values)


def owned_runtime_environment(launcher, output):
    values = create_paths(REPO, launcher.confined(output / 'runtime'))
    values.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    os.environ.update(values)
    environment = launcher.environment()
    environment.update(values)
    return environment, values


def base_context(stage, admission_path, gpu, allow_completed_primary_evaluation=False):
    require(HERE == PHASE / 'buddy_gpu77_postfamily_eval_preparation_v4' and Path.cwd().resolve() == REPO,
            'Execute the synced packet from the actual authorized checkout')
    # This loader is stdlib-only. Verify every packet before calling its gates.
    staging = bootstrap_staging_module()
    staging.verify_manifest(DATA, DATA_SHA)
    staging.verify_manifest(SOURCE, SOURCE_SHA)
    staging.verify_manifest(LAUNCHER, LAUNCHER_SHA)
    preparation_sha = staging.verify_manifest(HERE)
    binding = json.loads((HERE / 'ARCHIVE_PATH_BINDING.json').read_text())
    archive_path_gate(ARCHIVE.relative_to(PHASE).as_posix(), binding['actual77_archive_relative_path'])
    archive_path_gate(json.loads((HERE / 'CONTRACT.json').read_text())['archive_relative_path'],
                      binding['actual77_archive_relative_path'])
    launcher = load_module(LAUNCHER / 'launch77.py', 'sealed_buddy77_launcher')
    admission_path = launcher.confined(admission_path)
    boundary = current_execution_boundary(launcher, admission_path, stage, gpu, HERE)
    bootstrap_environment()
    context = launcher.context()  # exact CPU certificate, cache bytes, runtime/UUID/4-thread gates
    for path in [ARCHIVE, DATASET, CACHE, RUNS, RESOURCE, FAMILY_ADMISSION, TERMINAL,
                 LOCK_DIR, LOCK, LOCK_AUDIT, EVAL_DIR, ADMISSION, AUDIT_ADMISSION, TEST_SPLIT, FAMILY_RUNTIME]:
        launcher.confined(path)
    terminal = launcher.read_json(TERMINAL)
    _, family_admission = launcher.operational_admission(FAMILY_ADMISSION)
    family_boundary = historical_family_boundary(launcher, terminal, FAMILY_RUNTIME, FAMILY_ADMISSION, family_admission)
    context.update(operational_repair_admission_sha256=launcher.sha(FAMILY_ADMISSION),
                   runtime_environment_sha256=launcher.sha(FAMILY_RUNTIME))
    terminal_gate(terminal, context, launcher.family_command(), launcher.sha(RESOURCE), launcher.sha(FAMILY_ADMISSION))
    resource = launcher.read_json(RESOURCE)
    matches(resource, dict(status='all_five_complete_resource_epochs', family_cells=15,
                            family_optimizer_fits=24, family_epochs_per_cell=100, other_jobs_stopped=False,
                            test_access=False), 'Complete resource receipt')
    guards = load_module(SOURCE / 'guards.py', 'sealed_buddy77_guards')
    config = guards.read_json(SOURCE / 'CONFIG.json')
    matches(config, dict(arms=list(launcher.ARMS), seeds=[0, 1, 2], epochs=100,
                          graph_policy='training_only_all_splits'), 'Fixed scientific schedule')
    expected = {f'{arm}_seed{seed}' for arm in config['arms'] for seed in config['seeds']}
    require({path.name for path in RUNS.iterdir() if path.is_dir()} == expected, 'Training directory cohort differs')
    rows, summaries, epoch_train, epoch_valid = [], [], 0.0, 0.0
    for arm in config['arms']:
        for seed in config['seeds']:
            folder = RUNS / f'{arm}_seed{seed}'
            launcher.confined(folder)
            row, _ = guards.validate_run_metadata(folder, arm, seed, context['cache_manifest_sha256'],
                                                   config, context['implementation_hashes'])
            summary = guards.read_json(folder / 'completion.json')
            matches(summary, dict(torch_version=context['torch_version'], device='cuda:0'), 'Completed runtime')
            rows.append(row)
            summaries.append(summary)
            for line in (folder / 'epochs.jsonl').read_text().splitlines():
                epoch = guards.strict_json(line)
                epoch_train += epoch['train_seconds']
                epoch_valid += epoch['validation_forward_seconds']
    complete_cells_gate(summaries, config['arms'], config['seeds'], config['epochs'])
    if not allow_completed_primary_evaluation:
        deny_existing_test_artifacts()
    identity = dict(evaluation_preparation_manifest_sha256=preparation_sha,
                    source_manifest_sha256=SOURCE_SHA, data_wrapper_manifest_sha256=DATA_SHA,
                    family_launcher_manifest_sha256=LAUNCHER_SHA,
                    CPU_qualification_sha256=context['CPU_qualification_sha256'],
                    data_qualification_sha256=context['data_qualification_sha256'],
                    cache_manifest_sha256=context['cache_manifest_sha256'],
                    terminal_family_receipt_sha256=launcher.sha(TERMINAL),
                    resource_receipt_sha256=launcher.sha(RESOURCE), family_admission_sha256=launcher.sha(FAMILY_ADMISSION),
                    actual_git_root=str(REPO), physical_GPU_UUIDs=UUIDS, cpu_threads=4,
                    family_namespace=str(LAUNCHER / 'root_family_v2'), excluded_prior_family_namespace=str(EXCLUDED_PRIOR_FAMILY),
                    prior_partial_fits_excluded=True,
                    family_runtime_environment_sha256=launcher.sha(FAMILY_RUNTIME),
                    family_external_namespace_proof_sha256=family_boundary['proof']['sha256'],
                    prior_partial_preservation_receipt_sha256=family_admission['prior_partial_preservation_receipt_sha256'],
                    prior_termination_receipt_sha256=family_admission['prior_termination_receipt_sha256'])
    costs = dict(family_launcher_subprocess_seconds=guards.finite_number(terminal['seconds'], 'family wall time'),
                 family_preflight_seconds=guards.finite_number(terminal['context']['preflight_seconds'], 'family preflight'),
                 completed_fit_loop_seconds=sum(row['total_seconds'] for row in summaries),
                 epoch_training_seconds=epoch_train, epoch_validation_forward_seconds=epoch_valid,
                 prior_partial_cost_and_failure_evidence_preserved_sha256=family_admission['prior_partial_preservation_receipt_sha256'],
                 prior_partial_fits_excluded_from_scientific_totals=True,
                 fit_loop_scope='Completion total_seconds excludes cache/model/optimizer startup and final checkpoint envelope audit; family subprocess wall time includes queue/setup.')
    return launcher, staging, guards, context, config, rows, identity, costs, boundary


def audit_lock(admission_path, gpu):
    started = time.monotonic()
    launcher, _, guards, context, config, rows, identity, training_costs, boundary = base_context('audit-lock', admission_path, gpu)
    require(launcher.confined(admission_path) == AUDIT_ADMISSION, 'Use this packet\'s separate root lock-audit admission')
    audit_admission_gate(launcher.read_json(admission_path), identity)
    require(not LOCK_DIR.exists(), 'Production lock audit already attempted; root recovery review required')
    LOCK_DIR.mkdir(exist_ok=False)
    record = dict(schema='buddy77-production-family-lock-audit-v1', UTC=datetime.now(timezone.utc).isoformat(),
                  **identity, status='in_progress', family_cells=15, optimizer_fits=24, epochs_per_cell=100,
                  test_payload_opened=False, all_selected_checkpoints_runtime_validated=False, training_costs=training_costs,
                  audit_admission_sha256=launcher.sha(admission_path), execution_boundary=boundary)
    try:
        env, runtime_paths = owned_runtime_environment(launcher, LOCK_DIR)
        record['repository_local_runtime_paths'] = runtime_paths
        env['CUDA_VISIBLE_DEVICES'] = ''
        command = lock_command()
        begin = time.monotonic()
        with (LOCK_DIR / 'lock.log').open('x') as log:
            run = subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(argv=command, exit_code=run.returncode, production_lock_command_seconds=time.monotonic() - begin)
        require(run.returncode == 0, 'Production lock/checkpoint audit failed; test remains closed')
        value, checkpoints = guards.verify_family_metadata(LOCK, context['cache_manifest_sha256'], guards.file_sha(SOURCE / 'CONFIG.json'))
        require(value['runs'] == rows and len(checkpoints) == 15, 'Production lock cohort/bytes differ')
        record.update(status='all15_production_lock_and_selected_checkpoints_audited',
                      family_lock_sha256=launcher.sha(LOCK), all_selected_checkpoints_runtime_validated=True,
                      locked_runs=rows)
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record['total_audit_wrapper_seconds'] = time.monotonic() - started
        launcher.write_json(LOCK_AUDIT, record)


def stage_authentic_test(staging, contract):
    # Called only after terminal, all cells, production lock/audit and root evaluation admission.
    require(not TEST_SPLIT.exists() and not TEST_SPLIT.is_symlink(), 'Official test destination must be absent')
    require(ARCHIVE.stat().st_size == contract['archive_bytes'] and staging.file_sha(ARCHIVE) == contract['archive_sha256'],
            'Pinned official archive differs; no substitute/download is allowed')
    with zipfile.ZipFile(ARCHIVE) as packed:
        staging.check_archive_members(packed.infolist(), contract['archive_metadata'])
        info = packed.getinfo(TEST_MEMBER)
        require(not info.is_dir(), 'Authentic official test member missing')
        count = 0
        with packed.open(info, 'r') as source, TEST_SPLIT.open('xb') as target:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                target.write(block)
                count += len(block)
        require(count == info.file_size, 'Official test member truncated')
    return dict(archive_sha256=contract['archive_sha256'], member=TEST_MEMBER, bytes=count,
                staged_path=str(TEST_SPLIT), official_test_file_sha256=staging.file_sha(TEST_SPLIT),
                other_archive_members_extracted=False)


def qualify_official_test(builder, torch):
    dataset = builder.offline_dataset(DATASET)
    require(dataset.transform is None and dataset.pre_transform is None, 'Unexpected dataset transform')
    official, path, _ = builder.official_split_file(dataset, 'test', expected_loader_sha=builder.file_sha(CACHE / 'official_split_loader.py.txt'))
    require(path == TEST_SPLIT, 'Official test loader path changed')
    for name in ['edge', 'edge_neg']:
        value = official.get(name)
        require(isinstance(value, torch.Tensor) and value.dtype == torch.int64 and value.ndim == 2 and
                value.shape[1] == 2 and len(value) > 0 and not (value < 0).any() and not (value >= 235868).any(),
                'Official test pair shape/dtype/range differs: ' + name)
    require(len(official['edge_neg']) == 100000, 'Official 100000-negative pool must remain unchanged')
    positive, negative = official['edge'], official['edge_neg']
    return official, dict(positive_rows=len(positive), negative_rows=len(negative),
                           positive_sha256=builder.tensor_sha(positive), negative_sha256=builder.tensor_sha(negative),
                           pair_order_sha256=builder.tensor_sha(torch.cat([positive, negative])),
                           ordering='Authentic official positives, then authentic official negatives; no filtering/reordering',
                           combined_split_accessor_called=False)


def qualify_final_cache(builder, torch, official, split):
    manifest = builder.read_json(CACHE / 'test_manifest.json')
    matches(manifest, dict(cache_manifest_sha256=builder.file_sha(CACHE / 'manifest.json'),
                           family_lock_sha256=builder.file_sha(LOCK), official_test_file_sha256=builder.file_sha(TEST_SPLIT),
                           graph_policy='training_only_all_splits', positive_sha256=split['positive_sha256'],
                           negative_sha256=split['negative_sha256'], pair_order_sha256=split['pair_order_sha256'],
                           test_sha256=builder.file_sha(CACHE / 'test.pt')), 'Final cache manifest')
    data = torch.load(CACHE / 'test.pt', map_location='cpu')
    expected = torch.cat([official['edge'], official['edge_neg']])
    require(set(data) == {'links', 'sf', 'n_positive'} and type(data['n_positive']) is int and
            data['n_positive'] == len(official['edge']) and data['links'].dtype == torch.int64 and
            torch.equal(data['links'], expected) and data['sf'].shape == (len(expected), 8) and
            data['sf'].dtype == torch.float32 and torch.isfinite(data['sf']).all() and not (data['sf'][:, [4, 5]] != 0).any(),
            'Final cached official ordering/structural shape/finiteness differs')
    return dict(test_manifest_sha256=builder.file_sha(CACHE / 'test_manifest.json'),
                test_cache_sha256=manifest['test_sha256'], full_candidate_order_and_shapes_qualified=True,
                test_topology_added_to_graph=False)


def evaluate(admission_path, gpu):
    started = time.monotonic()
    launcher, staging, guards, context, _, rows, identity, training_costs, boundary = base_context('evaluate', admission_path, gpu)
    admission_path = launcher.confined(admission_path)
    require(admission_path == ADMISSION, 'Use this packet\'s separate root evaluation admission')
    audit = launcher.read_json(LOCK_AUDIT)
    matches(audit, dict(schema='buddy77-production-family-lock-audit-v1', **identity,
                        status='all15_production_lock_and_selected_checkpoints_audited', family_cells=15,
                        optimizer_fits=24, epochs_per_cell=100, test_payload_opened=False,
                        all_selected_checkpoints_runtime_validated=True, family_lock_sha256=launcher.sha(LOCK),
                        locked_runs=rows, exit_code=0, argv=lock_command()), 'Production lock audit')
    lock, checkpoints = guards.verify_family_metadata(LOCK, context['cache_manifest_sha256'], guards.file_sha(SOURCE / 'CONFIG.json'))
    require(lock['runs'] == rows and len(checkpoints) == 15, 'Audited production lock changed')
    contract = launcher.read_json(DATA / 'CONTRACT.json')
    evaluation_identity = dict(**identity, family_lock_sha256=launcher.sha(LOCK),
                               lock_audit_sha256=launcher.sha(LOCK_AUDIT), archive_sha256=contract['archive_sha256'],
                               archive_member=TEST_MEMBER, scoring_GPU_UUID=gpu)
    evaluation_admission_gate(launcher.read_json(admission_path), evaluation_identity)
    require(not EVAL_DIR.exists(), 'Evaluation already attempted; explicit root recovery review required')
    EVAL_DIR.mkdir(exist_ok=False)  # exclusive once-only claim before any test member/payload open
    record = dict(schema='buddy77-postfamily-evaluation-receipt-v2', UTC=datetime.now(timezone.utc).isoformat(),
                  **evaluation_identity, evaluation_admission_sha256=launcher.sha(admission_path), status='in_progress',
                  training_costs=training_costs, execution_boundary=boundary, test_payload_opened=False, scoring_attempted=False,
                  other_jobs_stopped=False, original_scores_changed=False, scientific_advantage_claimed=False)
    launcher.write_json(EVAL_DIR / 'EVALUATION_CLAIM.json', record)
    try:
        _, runtime_paths = owned_runtime_environment(launcher, EVAL_DIR)
        record['repository_local_runtime_paths'] = runtime_paths
        # Check all selected checkpoint envelopes again before opening test payload.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        os.environ.pop('TORCH_FORCE_WEIGHTS_ONLY_LOAD', None)
        sys.path.insert(0, str(SOURCE))
        sys.path.insert(1, str(REPO / '.gnnm_runtime/buddy_extra_v1/site'))
        import torch
        import cache_builder as builder
        from run import qualification
        require(Path(builder.__file__).resolve() == SOURCE / 'cache_builder.py', 'Unexpected builder module')
        qualification(QUAL)
        torch.set_num_threads(4)
        begin = time.monotonic()
        builder.verify_family_lock(LOCK, context['cache_manifest_sha256'], builder.file_sha(SOURCE / 'CONFIG.json'))
        record['selected_checkpoint_revalidation_seconds'] = time.monotonic() - begin
        staging.verify_installed_ogb(contract)
        deny_existing_test_artifacts()
        begin = time.monotonic()
        # Mark a possible payload-open attempt before streaming; preserve failures honestly.
        record['test_payload_opened'] = 'possible_after_staging_attempt'
        record['official_staging'] = stage_authentic_test(staging, contract)
        record['test_payload_opened'] = True
        record['archive_verification_and_test_extraction_seconds'] = time.monotonic() - begin
        begin = time.monotonic()
        official, split = qualify_official_test(builder, torch)
        record['official_test_qualification'] = split
        record['official_test_qualification_seconds'] = time.monotonic() - begin
        require(builder.file_sha(TEST_SPLIT) == record['official_staging']['official_test_file_sha256'],
                'Authentic staged test file changed during qualification')
        begin = time.monotonic()
        # Direct unchanged production function: authentic pairs, same train graph/cache, all locked fits.
        builder.finalize(argparse.Namespace(dataset_root=str(DATASET), output=str(CACHE), family_lock=str(LOCK)))
        record['production_finalize_seconds'] = time.monotonic() - begin
        begin = time.monotonic()
        record['final_cache_qualification'] = qualify_final_cache(builder, torch, official, split)
        record['final_cache_qualification_seconds'] = time.monotonic() - begin
        require(builder.file_sha(TEST_SPLIT) == record['official_staging']['official_test_file_sha256'],
                'Authentic staged test file changed during final hydration')
        require(not list(RUNS.glob('*/final_test.json')), 'Results appeared before once-only scoring; root review required')
        require(launcher.sha(LOCK) == evaluation_identity['family_lock_sha256'] and
                launcher.sha(LOCK_AUDIT) == evaluation_identity['lock_audit_sha256'] and
                launcher.sha(admission_path) == record['evaluation_admission_sha256'], 'Evaluation gate bytes changed')
        launcher.verify_manifest(SOURCE, SOURCE_SHA)
        env = launcher.environment()
        env.update(runtime_paths)
        env['CUDA_VISIBLE_DEVICES'] = gpu
        command = score_command()
        record.update(scoring_attempted=True, scoring_argv=command)
        begin = time.monotonic()
        with (EVAL_DIR / 'score.log').open('x') as log:
            run = subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(scoring_exit_code=run.returncode, scoring_subprocess_seconds=time.monotonic() - begin)
        require(run.returncode == 0, 'Scoring incomplete; no automatic replay/rescoring')
        results = []
        for row in rows:
            path = Path(row['run_directory']) / 'final_test.json'
            result = launcher.read_json(path)
            matches(result, dict(arm=row['arm'], seed=row['seed'], family_lock_sha256=launcher.sha(LOCK),
                                 test_manifest_sha256=launcher.sha(CACHE / 'test_manifest.json'),
                                 checkpoint_sha256=row['selected_checkpoint_sha256']), 'Final score binding')
            guards.finite_number(result['hits50'], 'Final Hits@50', upper=1.0)
            seconds = guards.finite_number(result['prediction_seconds'], 'Final prediction time')
            results.append(dict(arm=row['arm'], seed=row['seed'], result_sha256=launcher.sha(path), prediction_seconds=seconds))
        record.update(status='all15_locked_cells_scored_once', final_results=results,
                      summed_prediction_seconds=sum(row['prediction_seconds'] for row in results),
                      cost_scope='Wrapper wall time includes all gates, checkpoint validation, archive I/O, CPU hydration/qualification and scoring. Prediction seconds exclude cache/model loading and result writes; shared-host contention remains.')
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error),
                      replay_requires_explicit_root_review=True)
        raise
    finally:
        record['total_evaluation_wrapper_seconds'] = time.monotonic() - started
        launcher.write_json(EVAL_DIR / 'EVALUATION_RECEIPT.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit-lock', 'evaluate'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--admission')
    parser.add_argument('--gpu', choices=UUIDS, default=UUIDS[0])
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(stage=args.stage, production_lock_argv=lock_command(),
                              finalize_call=dict(function='unchanged cache_builder.finalize', dataset_root=str(DATASET),
                                                 output=str(CACHE), family_lock=str(LOCK)),
                              scoring_argv=score_command(), authentic_test_member=TEST_MEMBER,
                              authentic_archive=str(ARCHIVE),
                              separate_audit_admission=str(AUDIT_ADMISSION), separate_evaluation_admission=str(ADMISSION), scoring_GPU_UUID=args.gpu,
                              family_cells=15, optimizer_fits=24, no_test_access=True,
                              runtime_cache_and_temp_scope='qualified repo-local family runtime; owned per-stage folders after guarded closure/admission',
                              authorization='Preparation only; root must separately admit evaluation after full audited family closure.'), indent=2))
        return
    admission = args.admission or str(AUDIT_ADMISSION if args.stage == 'audit-lock' else ADMISSION)
    audit_lock(admission, args.gpu) if args.stage == 'audit-lock' else evaluate(admission, args.gpu)


if __name__ == '__main__':
    main()
