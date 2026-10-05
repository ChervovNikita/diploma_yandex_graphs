#!/usr/bin/env python3
"""Disabled prospective exact9 collector; no history/FREEZE/tensor deserialization."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
COLLECTOR_SOURCE_REVIEWED = False
SPEC = {'path': 'shared_private_transfer_row0_single_companion_preparation_20261005_v1/COMPANION_PLAN_DISABLED.json',
        'sha256': '774f45e1778ab7f0708aaa5f5a3a7aa6970968c63594218993673541856f72e4'}
ORIGINAL_PLAN = {'path': 'shared_private_transfer_paired_pilot_execution_root_20261005_v2/COHORT_PLAN.json',
                 'sha256': 'ae4b0f5c77cf48a86ccdbe51179fc95881157ac225f593c3992a5a5014d5346b'}
PROMOTION = {'path': 'shared_private_transfer_paired_pilot_execution_root_20261005_v2/ROOT_RELEASE.json',
             'sha256': 'f638ee0d768cabb5efb999befa9bc688322c6086b983497fa5c49c8fe44d34e3'}
PROGRAM_SHA = 'd2c7518bf1904aa7c7612ea5165a514ac7d5b11ddb0357937a56015cdb161ce6'
SOURCE_SHA = {'authorized_one_GPU_allocation': '67fab0016a144fdfda639e19cf1cf7feae4d10cbec04f1eaf8187917aa975ea8',
              'authorized_18.77': '2b069a09de826ef94b2bbda981929f1903c29aa533628855cbd778f137faa7f0'}
AVAILABLE_SHA = '1b9c8bb57278d91b0f6212136225afcfd6b067c6b17e0dfed7ee36dd6316efdc'
REPOSITORY = {'authorized_one_GPU_allocation': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs',
              'authorized_18.77': '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'}
SOURCE_DIRECTORY = {'authorized_one_GPU_allocation': 'shared_private_transfer_row0_single_companion_preparation_20261005_v1',
                    'authorized_18.77': 'shared_private_transfer_row0_training_source_gpu77_20261005_v1'}
PROVIDER_LABEL = {'authorized_one_GPU_allocation': 'singleton', 'authorized_18.77': 'gpu77'}
RUNTIME_EVIDENCE = {
    'authorized_one_GPU_allocation': {'path':'citeseer_ncn_native_runtime_qualification_20261005_v2/RESULT.json',
                                    'sha256':'2dfa804a4761c559b6fa2f954f98300b81dabd4ac7f8b2da97e5301b9d253496'},
    'authorized_18.77': {'path':'shared_private_transfer_gpu77_environment_execution_20261005_v1/PROVIDERS.json',
                       'sha256':'8f266cf0e27e908af087fb18092631eb7410c150e414d437b05ed3e7573b2fe4'}}
BOUNDS = {'F1_end_live':{'soft_seconds':30000,'hard_seconds':33000},
          'F1_end_detached':{'soft_seconds':30000,'hard_seconds':33000},
          'F1_end_joint':{'soft_seconds':15000,'hard_seconds':18000}}
INPUT_ROLES = {'train_pos.txt', 'valid_pos.txt', 'heart_valid_samples.npy', 'gnn_feature'}
EXTERNAL_TRAINING = {
    'citeseer_heart_ncn_trainval_runner_source_20261005_v1/SOURCE_MANIFEST.json':
        'efa95806d86e3cc261a8506042204d8d32e39506386d6faf90a625d52be12ff9',
    'endpoint_episode_geometry_preparation_20261005_v1/SOURCE_MANIFEST.json':
        'f1b97911cc13076bc2f72d57f2e7f9999245c5678c7979fad5c22dd82af55f10',
    'endpoint_episode_geometry_preparation_20261005_v1/episode_geometry.py':
        'f5562c94c8c90b999065e6b570f5a0e7c734f3f4f949274f98eeaf9859bf2321',
    'endpoint_episode_geometry_preparation_20261005_v1/native_episode_cycle.py':
        '8cd5596459a389aeaf435dcf5ecec819ae94c19a734f24ecf27c96cb4b97c0b9',
    'shared_core_private_learning_float64_dense_oracle_preparation_20261005_v1/recursive_adjoint.py':
        '14bf734b74983e91dec78ad8bba1ef7dcd910dc71aa20a9e75da466d76362293'}
SUPERVISOR = {
    'authorized_one_GPU_allocation': {
        'path':'shared_backbone_private_transfer_complete_cost_queue_preparation_20261005_v3/queue.py',
        'sha256':'7b195b600ad3cc57c4cf51a1da927164173ca2b0e80e96cf2b6c6449ce7ad468'},
    'authorized_18.77': {
        'path':'shared_private_transfer_gpu77_qualification_preparation_20261005_v3/ownership_helpers.py',
        'sha256':'e71503c87865546319cddbbf7a4f9f15d13cdf9e4875e406d65de21e64e047fd'}}
COMPANION_OPERATIONAL = {
    'authorized_one_GPU_allocation': {
        'path':'shared_private_transfer_row0_companion_launch_activation_20261005_v1/singleton/SOURCE_MANIFEST.json',
        'sha256':'c85f00952171e910fb5623f5f07cd57ef89df797324c06b22bcb08360c57eece'},
    'authorized_18.77': {
        'path':'shared_private_transfer_row0_companion_launch_activation_20261005_v1/gpu77/SOURCE_MANIFEST.json',
        'sha256':'a8d0d6c8311c3fb82f3228dfa27c332d0ec4441397f29a6a16da76743914a48e'}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_metadata(path):
    # Callers supply release/registry/queue/job/source/qualification/process metadata.
    # FREEZE, VALID history and tensor files are never passed to this helper.
    path = Path(path)
    require(path.name not in ('FREEZE.json','VALID_HISTORY.jsonl') and path.suffix == '.json',
            'Outcome-bearing FREEZE/history/tensor payload is not admitted by metadata reader')
    return json.loads(path.read_text(), parse_constant=reject_constant)


def reject_constant(value):
    raise ValueError('Nonfinite metadata constant: '+value)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative(value):
    path = Path(value)
    require(path.parts and not path.is_absolute() and '..' not in path.parts, 'Require a phase-relative path')
    return path


def phase_file(value):
    path = PHASE/relative(value)
    require(path.resolve(strict=True).is_relative_to(PHASE.resolve()), 'Input leaves the project phase')
    for parent in [path, *path.parents]:
        if parent == PHASE.parent:
            break
        require(not parent.is_symlink(), 'Symlink custody is not admitted')
    require(path.is_file(), 'Bound artifact must be a file')
    return path


def binding(ref):
    require(isinstance(ref, dict) and isinstance(ref.get('sha256'), str)
            and len(ref['sha256']) == 64, 'Unresolved exact binding')
    path = phase_file(ref['path'])
    require(sha(path) == ref['sha256'], 'Bound bytes changed: '+ref['path'])
    return path


def reference(path):
    return {'path': str(Path(path).relative_to(PHASE)), 'sha256': sha(path)}


def replica(donor, original):
    original, prefix = relative(original), relative(donor['donor_directory_relative'])
    require(original.is_relative_to(prefix), 'Artifact is outside its actual original donor directory')
    return phase_file(str(relative(donor['replica_directory_relative'])/original.relative_to(prefix)))


def donor_binding(donor, ref):
    path = replica(donor, ref['path'])
    require(sha(path) == ref['sha256'], 'Actual donor/replica binding changed')
    return path


def verify_packet(expected):
    require(sha(HERE/'MANIFEST.json') == expected, 'Reviewed collector source changed')
    verify_manifest(reference(HERE/'MANIFEST.json'))


def verify_manifest(ref):
    manifest = binding(ref)
    rows = read_metadata(manifest)['files']
    require(rows and len({r['path'] for r in rows}) == len(rows), 'Complete unique source manifest required')
    for row in rows:
        path = phase_file(str((manifest.parent/relative(row['path'])).relative_to(PHASE)))
        require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Source packet bytes changed')
    return manifest


def source_copy_entry(provenance, original, local, expected_sha, expected_bytes):
    """Retrospective copy metadata links genuine original and ordinary local bytes."""
    rows = provenance['files']
    require(rows and len({r['original_path'] for r in rows}) == len(rows)
            and len({r['local_path'] for r in rows}) == len(rows), 'Unique source-copy inventory required')
    expected = {'original_path':str(relative(original)), 'local_path':str(local.relative_to(PHASE)),
                'sha256':expected_sha, 'bytes':expected_bytes}
    require(sum(row == expected for row in rows) == 1, 'Genuine source-copy provenance lacks these exact bytes')


def source_copy(ref, original, expected_sha, donor):
    require(ref['original_path'] == original and ref['sha256'] == expected_sha,
            'Source-fixed original path/hash differs')
    path = binding(ref)
    provenance = read_metadata(binding(ref['copy_inventory']))
    require(provenance['schema'] == 'authenticated_retrospective_source_copy_inventory_v1'
            and provenance['retrospective_source_copy_only'] is True and provenance['pre_fit_authority'] is False
            and all(provenance[k] == donor[k] for k in ('provider','hostname','repository'))
            and provenance['source_copy_review_evidence'], 'Actual reviewed provider source-copy custody required')
    for review in provenance['source_copy_review_evidence']:
        binding(review)
    source_copy_entry(provenance, original, path, expected_sha, path.stat().st_size)
    return path, provenance


def source_copy_manifest(ref, original, expected_sha, donor):
    manifest, provenance = source_copy(ref, original, expected_sha, donor)
    rows = read_metadata(manifest)['files']
    require(rows and len({r['path'] for r in rows}) == len(rows), 'Complete unique genuine source manifest required')
    for row in rows:
        member = relative(row['path'])
        local = phase_file(str((manifest.parent/member).relative_to(PHASE)))
        require(sha(local) == row['sha256'] and local.stat().st_size == row['bytes'],
                'External or training source-manifest member changed')
        source_copy_entry(provenance, str(relative(original).parent/member), local, row['sha256'], row['bytes'])
    return manifest


def authenticate_source_custody(donor, custody, training_directory, training_sha, program_sha, operational):
    """Ordinary hashes only: training, external manifest members and actual supervisor."""
    training = source_copy_manifest(custody['training_manifest'], training_directory+'/SOURCE_MANIFEST.json', training_sha, donor)
    program = phase_file(str((training.parent/'run.py').relative_to(PHASE)))
    require(sha(program) == program_sha, 'Exact executed training program changed')
    require(set(custody['external_dependencies']) == set(EXTERNAL_TRAINING), 'Complete source-fixed external dependency set required')
    for original, expected_sha in EXTERNAL_TRAINING.items():
        ref = custody['external_dependencies'][original]
        if relative(original).name == 'SOURCE_MANIFEST.json':
            source_copy_manifest(ref, original, expected_sha, donor)
        else:
            source_copy(ref, original, expected_sha, donor)
    queue_manifest = source_copy_manifest(custody['operational_manifest'], operational['path'], operational['sha256'], donor)
    for name in ('run_queue.py','pilot_common.py'):
        require(name in {r['path'] for r in read_metadata(queue_manifest)['files']}, 'Actual operational queue/helper source missing')
    supervisor = SUPERVISOR[donor['provider']]
    source_copy(custody['supervisor'], supervisor['path'], supervisor['sha256'], donor)
    require(custody['operational_review_evidence'], 'Actual operational queue/supervisor source review required')
    for ref in custody['operational_review_evidence']:
        binding(ref)
    return training, program, queue_manifest


def fresh_output(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PHASE.resolve()) and not path.exists() and path.parent.is_dir(), 'Fresh phase output required')
    return path


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def identity(owner, argv, cwd, expected_cwd):
    require(isinstance(owner, dict) and type(owner.get('PID')) is int and owner['PID'] > 0
            and owner.get('pgid') == owner['PID'] == owner.get('sid')
            and type(owner.get('start_ticks')) is int and owner['start_ticks'] > 0
            and owner.get('argv') == argv and cwd == expected_cwd,
            'Saved fresh process/session/argv/cwd identity differs')


def terminal(receipt):
    require(receipt.get('terminal_wait_observed') is True
            and receipt.get('exit_code_authority') == 'subprocess.Popen.wait/poll'
            and receipt.get('exit_code') == 0 and receipt.get('reason') is None
            and receipt.get('signals_sent') == [] and receipt.get('attempts') == 1
            and receipt.get('retry') is False and receipt.get('identity_admitted_for_signals') is True
            and receipt.get('scores_read') is False and receipt.get('signal_refusal') is None,
            'Actual successful owned Popen terminal authority required')


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Explicit UTC-offset metadata timestamp required')
    return parsed


def source_packet(manifest_ref, program_ref, expected_source, expected_program=PROGRAM_SHA):
    manifest, program = verify_manifest(manifest_ref), binding(program_ref)
    require(manifest_ref['sha256'] == expected_source and program_ref['sha256'] == expected_program
            and program.parent == manifest.parent and program.name == 'run.py', 'Exact qualified companion source differs')


def qualified_runtime(provider, job):
    ref = RUNTIME_EVIDENCE[provider]
    require(ref in job['runtime_qualification']['evidence'], 'Actual provider runtime evidence changed')
    metadata = read_metadata(binding(ref))
    require(metadata['status'] == 'PASS', 'Actual provider runtime qualification failed')
    if provider == 'authorized_one_GPU_allocation':
        versions = metadata['runtime_versions']
    else:
        versions = {name:metadata['providers'][name]['version'] for name in job['runtime_versions'] if name != 'CUDA'}
        versions['CUDA'] = metadata['torch_cuda']
    require(versions == job['runtime_versions'], 'Qualified runtime differs from the actual frozen job')
    return versions


def gpu77_registration(donor, auth, queue, root_release):
    # These are genuine source-emitted pre-fit objects on this provider only.
    # Singleton b0 has different actual review/freeze/launch authority.
    if donor['provider'] != 'authorized_18.77':
        return None
    require(root_release['root_provider_block_approved'] is True, 'Actual separate GPU77 block release required')
    paths = [donor['donor_directory_relative']+'/'+name for name in ('PROVIDER_ADMISSION.json','DONOR_REGISTRATION.json')]
    files = [donor_binding(donor, {'path':path,'sha256':auth['artifact_inventory'][path]['sha256']}) for path in paths]
    admission, registration = map(read_metadata, files)
    require(admission['approved'] is True and admission['admitted_before_provider_first_fit'] is True
            and registration['registered_before_provider_first_fit'] is True and registration['fallback_activated'] is False
            and all(admission[k] == donor[k] == registration[k] for k in ('provider','hostname','GPU_UUID'))
            and admission['authorized_repository'] == donor['repository']
            and admission['provider_source_manifest']['sha256'] == donor['provider_source_manifest']['sha256']
            and admission['program']['sha256'] == PROGRAM_SHA and admission['python_executable'] == queue['python_executable']
            and admission['available_manifest_sha256'] == AVAILABLE_SHA
            and registration['queue_sha256'] == donor['queue_binding']['sha256']
            and registration['root_release_sha256'] == donor['root_release_binding']['sha256']
            and registration['block'] == donor['block']
            and registration['provider_admission'] == {'path':paths[0],'sha256':sha(files[0])}
            and all(admission[k] is False for k in ('TEST_access','original_score_recalculation','numerical_science_changes')),
            'Actual GPU77 pre-fit registration/admission differs')
    for ref in admission['qualification_evidence']+admission['root_source_review_evidence']:
        binding(ref)
    return admission


def prefit(donor, spec):
    """Authenticate actual review->launch links; no derived object predates the fit."""
    block = donor['block']
    assigned = spec['resource_assignment'][block]
    require(donor['provider'] == assigned['primary_provider'] and donor['hostname'] == assigned['hostname']
            and donor['GPU_UUID'] == assigned['GPU_UUID'] and donor['repository'] == REPOSITORY[donor['provider']],
            'Fixed whole-block provider/repository assignment changed')
    packet_path = verify_manifest(donor['launch_source_manifest'])
    operational_path = verify_manifest(donor['operational_source_manifest'])
    require(operational_path.parent == packet_path.parent/PROVIDER_LABEL[donor['provider']],
            'Operational queue source is not inside the actual reviewed launcher packet')
    require(sha(operational_path.parent/'STUDY_SPEC.json') == SPEC['sha256'], 'Operational nine-cell source spec changed')
    binding(donor['root_source_review'])
    review_path = binding(donor['root_job_review'])
    review = read_metadata(review_path)
    auth_path = binding(donor['job_freeze_authentication'])
    auth = read_metadata(auth_path)
    launch_path = binding(donor['launch_receipt'])
    launch = read_metadata(launch_path)
    order = [i for i in spec['execution_order'] if i.startswith(block+'_')]
    reviewed_order = [i for i in spec['execution_order'] if i.split('_',1)[0] in auth['blocks']]
    expected = auth['blocks'][block]
    require(auth['status'] == 'EXACT_COMPANION_BLOCKS_FROZEN_BEFORE_FITS'
            and auth['scientific_children_started'] == 0 and auth['fits'] == 0
            and auth['companion_spec_sha256'] == SPEC['sha256'] and auth['scores_read'] is False
            and expected['ordered_cell_ids'] == order and expected['physical_fits'] == 3
            and expected['execution_directory_relative'] == donor['donor_directory_relative']
            and expected['physical_GPU_UUID'] == donor['GPU_UUID'] and expected['scientific_children_started'] == 0
            and auth['provider'] == PROVIDER_LABEL[donor['provider']]
            and auth['source_manifest_sha256'] == donor['provider_source_manifest']['sha256']
            and auth['pilot_source_manifest_sha256'] == donor['operational_source_manifest']['sha256']
            and auth['root_source_review'] == donor['root_source_review'], 'Actual pre-fit full-spec/job freeze differs')
    require(review['root_provider_jobs_reviewed'] is True and review['full_nine_cell_spec_reviewed'] is True
            and review['scores_read'] is False and review['TEST_access'] is False
            and review['reviewed_cell_ids'] == reviewed_order and review['companion_spec_sha256'] == SPEC['sha256']
            and review['provider'] == PROVIDER_LABEL[donor['provider']] and review['all_generated_job_hashes_checked'] is True
            and all(review['generated_job_hashes'][cell] == expected['job_sha256_by_cell_id'][cell] for cell in order)
            and review['cohort_plan_sha256'] == auth['cohort_plan_sha256'],
            'Actual root reviewed job hashes differ')
    if donor['provider'] == 'authorized_one_GPU_allocation':
        require(review['queue_sha256'] == expected['queue_sha256'], 'Actual singleton reviewed queue differs')
    require(review['root_provider_jobs_reviewed'] is True and review['full_nine_cell_spec_reviewed'] is True
            and review['scores_read'] is False and review['TEST_access'] is False
            and review['freeze_authentication_sha256'] == donor['job_freeze_authentication']['sha256'],
            'Actual root job review did not bind these generated jobs')
    require(review['activation_manifest_sha256'] == auth['activation_manifest_sha256']
            == donor['launch_source_manifest']['sha256'], 'Exact root-reviewed launcher packet differs')
    require(launch['status'] == 'ONE_COMPLETE_ROOT_REVIEWED_COMPANION_BLOCK_QUEUE_LAUNCHED'
            and launch['block'] == block and launch['physical_fits'] == 3 and launch['full_family_fits'] == 9
            and launch['attempts'] == 1 and launch['retry'] is False and launch['scores_read'] is False
            and launch['TEST_access'] is False and launch['identity_admitted_for_signals'] is True
            and launch['provider_job_review'] == donor['root_job_review']
            and launch['hostname'] == donor['hostname'] and launch['physical_GPU_UUID'] == donor['GPU_UUID']
            and launch['queue_sha256'] == expected['queue_sha256']
            and launch['root_release_sha256'] == expected['root_release_sha256']
            and launch['pilot_source_manifest_sha256'] == donor['operational_source_manifest']['sha256']
            and launch['detached_launches'] == 1
            and launch['cohort_plan_sha256'] == auth['cohort_plan_sha256'], 'Actual one-shot reviewed launch binding differs')
    return expected, auth, launch


def authenticate_nine(release):
    """All9 metadata/source/terminal checks complete before any fit-state JSON read."""
    spec = read_metadata(binding(SPEC))
    binding(ORIGINAL_PLAN); binding(PROMOTION)
    require(release['companion_spec_binding'] == SPEC and release['original_plan_binding'] == ORIGINAL_PLAN
            and release['original_promotion_binding'] == PROMOTION, 'Immutable study/promotion identity changed')
    donors = release['donors']
    require([d['block'] for d in donors] == ['b0', 'b1', 'b2'], 'All three fixed complete donors required')
    records, common_inputs, common_plan_path = [], None, None
    for donor in donors:
        expected, auth, launch = prefit(donor, spec)
        queue_path = donor_binding(donor, donor['queue_binding'])
        root_release_path = donor_binding(donor, donor['root_release_binding'])
        plan_path = donor_binding(donor, donor['cohort_plan_binding'])
        q, root_release, plan = map(read_metadata, (queue_path, root_release_path, plan_path))
        block = donor['block']; order = expected['ordered_cell_ids']
        require(donor['queue_binding']['sha256'] == expected['queue_sha256'] == launch['queue_sha256']
                and donor['root_release_binding']['sha256'] == expected['root_release_sha256']
                and donor['cohort_plan_binding']['sha256'] == auth['cohort_plan_sha256'] == q['cohort_plan_sha256']
                and q['root_release_sha256'] == donor['root_release_binding']['sha256'], 'Pre-fit queue/release/plan custody differs')
        require(q['execution_blocks'] == root_release['execution_blocks'] == [block]
                and q['execution_directory_relative'] == donor['donor_directory_relative']
                and [r['cell_id'] for r in q['entries']] == order and len(q['entries']) == 3
                and root_release['new_fit_release'] is True and root_release['TEST_access'] is False
                and root_release['retry'] is False and q['retry'] is False and q['TEST_access'] is False
                and q['read_scores_or_change_family'] is False and q['stop_on_operational_failure'] is True,
                'Exact released three-cell whole block required')
        require(all(plan[k] == spec[k] for k in ('cells', 'blocks', 'execution_order', 'resource_assignment', 'selection'))
                and len(plan['cells']) == 9 and plan['companion_spec_sha256'] == SPEC['sha256']
                and plan['original_plan_binding'] == ORIGINAL_PLAN
                and plan['original_promotion_gate_binding'] == PROMOTION, 'Exact full9 science changed')
        if common_plan_path is None:
            common_plan_path = plan_path
        require(plan_path.read_bytes() == common_plan_path.read_bytes(), 'All companion jobs retain one exact full9 cohort')
        require(root_release['root_numeric_cohort_approved'] is True and root_release['pilot_source_review_approved'] is True
                and root_release['cost_completeness_approved'] is True and root_release['selection_budget_fairness_approved'] is True,
                'Actual root source/cost/budget approval required')
        for ref in root_release['root_review_evidence']+root_release['complete_cycle_cost_evidence']+[root_release['root_numeric_decision']]:
            binding(ref)
        require(donor['complete_cycle_costs'] == root_release['complete_cycle_cost_evidence']
                and len(donor['complete_cycle_costs']) == 3 and root_release['fit_bounds_by_cell'] == BOUNDS,
                'Actual three-rule complete cost provenance or root bounds changed')
        require(q['queue_hard_seconds'] == root_release['queue_hard_seconds'] == 95700
                and q['resource_limits'] == spec['resource_limits']
                and q['pilot_source_manifest_sha256'] == root_release['pilot_source_manifest_sha256']
                == auth['pilot_source_manifest_sha256'] == donor['operational_source_manifest']['sha256']
                and all(q[k] == root_release[k] for k in ('python_executable','environment_overrides','queue_overhead_seconds','resource_limits','resource_assignment')),
                'Reviewed source/runtime/own-resource bounds changed')
        training, program, operational = authenticate_source_custody(
            donor, donor['source_custody'], SOURCE_DIRECTORY[donor['provider']], SOURCE_SHA[donor['provider']],
            PROGRAM_SHA, COMPANION_OPERATIONAL[donor['provider']])
        require(training == binding(donor['provider_source_manifest']) and program == binding(donor['program'])
                and operational == binding(donor['operational_source_manifest']), 'Actual source-copy and existing donor identities differ')
        source_packet(donor['provider_source_manifest'], donor['program'], SOURCE_SHA[donor['provider']])
        admission = gpu77_registration(donor, auth, q, root_release)
        require(q['training_source_manifest_sha256'] == donor['provider_source_manifest']['sha256']
                == launch['training_source_manifest_sha256'], 'Actual provider source changed')
        start_path = replica(donor, donor['donor_directory_relative']+'/QUEUE_START.json')
        start = read_metadata(start_path)
        require(start['queue_sha256'] == donor['queue_binding']['sha256'] and start['execution_blocks'] == [block]
                and start['physical_fits'] == 3 and start['full_scientific_cohort_fits'] == 9
                and start['TEST_access'] is False and start['comparative_scoring_performed'] is False
                and start['retry'] is False, 'Actual queue start differs')
        queue_program = donor['repository']+'/experiments_iclr/postsubmission_20260930/'+str(
            relative(COMPANION_OPERATIONAL[donor['provider']]['path']).parent/'run_queue.py')
        queue_argv = [q['python_executable'], '-B', queue_program, '--queue', donor['repository']+
                      '/experiments_iclr/postsubmission_20260930/'+donor['queue_binding']['path']]
        identity(launch['queue_identity'], queue_argv, launch['observed_cwd'], donor['repository'])
        identity(start['supervisor_identity'], queue_argv, donor['repository'], donor['repository'])
        require(all(start['supervisor_identity'][k] == launch['queue_identity'][k] for k in ('PID','start_ticks','pgid','sid','argv')),
                'Captured queue start is not the actual launched queue')
        block_path = donor_binding(donor, donor['block_freeze_binding'])
        completed = read_metadata(block_path)
        require(completed['selected_blocks_complete'] is True and completed['execution_blocks'] == [block]
                and completed['physical_fits'] == 3 and [r['cell_id'] for r in completed['completed']] == order
                and completed['queue_sha256'] == donor['queue_binding']['sha256']
                and completed['cohort_plan_sha256'] == donor['cohort_plan_binding']['sha256']
                and completed['companion_spec_sha256'] == SPEC['sha256']
                and completed['TEST_access'] is False and completed['comparative_scoring_performed'] is False
                and completed['retry'] is False, 'Actual terminal complete whole block required')
        require(not (block_path.parent/'QUEUE_FAILURE.json').exists(), 'Operationally failed block is not complete')
        for entry, receipt in zip(q['entries'], completed['completed']):
            terminal(receipt)
            cell = next(c for c in spec['cells'] if c['cell_id'] == entry['cell_id'])
            job_path = replica(donor, entry['job_relative']); job = read_metadata(job_path)
            require(sha(job_path) == entry['job_sha256'] == receipt['job_sha256']
                    == expected['job_sha256_by_cell_id'][cell['cell_id']]
                    and all(job[k] == v for k, v in cell.items())
                    and job['cohort_plan_sha256'] == donor['cohort_plan_binding']['sha256'], 'Actual frozen scientific job differs')
            require(job['source_manifest_sha256'] == donor['provider_source_manifest']['sha256']
                    and job['program_sha256'] == PROGRAM_SHA and job['physical_gpu_uuid'] == donor['GPU_UUID']
                    and job['purpose'] == 'TRAIN_VALID_prospective_private_transfer_fit'
                    and job['fits_authorized'] is True and job['VALID_values_access'] is True
                    and job['TEST_access'] is False and job['retry'] is False, 'Per-fit source/scope differs')
            require(entry['job_relative'] == donor['donor_directory_relative']+'/jobs/'+cell['cell_id']+'.json'
                    and entry['output_relative'] == donor['donor_directory_relative']+'/runs/'+cell['cell_id']
                    and job['soft_seconds'] == BOUNDS[cell['cell']]['soft_seconds']
                    and entry['hard_seconds'] == BOUNDS[cell['cell']]['hard_seconds'], 'Exact frozen output/job/bounds differ')
            for role in ('source_review', 'runtime_qualification', 'feature_authority', 'negative_pool_authority'):
                require(job[role]['approved'] is True and job[role]['evidence'], 'Actual job authority missing')
                for ref in job[role]['evidence']:
                    binding(ref)
            gate = {'path':job['training_step_gate']['path'], 'sha256':job['training_step_gate']['sha256']}
            require(job['training_step_gate']['approved'] is True and gate == donor['training_step_gate']
                    and gate['sha256'] == launch['qualifier_sha256'] == auth['qualifier_sha256'], 'Actual exact-source gate differs')
            binding(gate)
            versions = qualified_runtime(donor['provider'], job)
            if admission is not None:
                require(admission['runtime_versions'] == versions and admission['training_step_gate'] == job['training_step_gate'],
                        'Actual GPU77 job and admission qualification differ')
            require(job['available_manifest_sha256'] == AVAILABLE_SHA, 'Immutable acquisition manifest changed')
            available = read_metadata(binding({'path':job['available_manifest_relative'], 'sha256':job['available_manifest_sha256']}))
            require(set(available['files']) == INPUT_ROLES and available['TEST_available_to_loader'] is False, 'Fixed input manifest differs')
            inputs = {name:{k:available['files'][name][k] for k in ('sha256','bytes')} for name in INPUT_ROLES}
            if common_inputs is None:
                common_inputs = inputs
            require(inputs == common_inputs, 'Actual declared fixed inputs differ across companions')
            output = replica(donor, receipt['freeze_relative']).parent
            require(receipt['freeze_relative'] == entry['output_relative']+'/FREEZE.json'
                    and sha(output/'FREEZE.json') == receipt['freeze_sha256']
                    and not (output/'FAILURE.json').exists(), 'Owned successful exit lacks its exact source FREEZE bytes')
            child_start_path = replica(donor, donor['donor_directory_relative']+'/logs/'+cell['cell_id']+'.CHILD_STARTED.json')
            exit_path = replica(donor, donor['donor_directory_relative']+'/logs/'+cell['cell_id']+'.EXIT.json')
            child_start, exit_record = read_metadata(child_start_path), read_metadata(exit_path)
            require(exit_record == {k:v for k,v in receipt.items() if k not in ('freeze_relative','freeze_sha256')},
                    'BLOCK_FREEZE is not the actual per-child owned EXIT receipt')
            argv = [q['python_executable'],'-B',donor['repository']+'/experiments_iclr/postsubmission_20260930/'+SOURCE_DIRECTORY[donor['provider']]+'/run.py',
                    '--job',donor['repository']+'/experiments_iclr/postsubmission_20260930/'+entry['job_relative'],
                    '--output',donor['repository']+'/experiments_iclr/postsubmission_20260930/'+entry['output_relative']]
            identity(receipt['child_identity'], argv, receipt['observed_cwd'], donor['repository'])
            require(child_start['identity'] == receipt['child_identity'] and child_start['argv'] == argv
                    and child_start['identity_admitted_for_signals'] is True
                    and child_start['observed_cwd'] == donor['repository']
                    and receipt['child_identity']['ppid'] == launch['queue_identity']['PID']
                    and receipt['child_identity']['start_ticks'] >= launch['queue_identity']['start_ticks']
                    and timestamp(start['UTC']) <= timestamp(child_start['UTC']) <= timestamp(receipt['UTC']),
                    'Actual queue->owned child chronology differs; no cross-host clock inference is admitted')
            checkpoint = phase_file(str((output/'selected_checkpoint.pt').relative_to(PHASE)))
            logits = phase_file(str((output/'selected_VALID_logits.pt').relative_to(PHASE)))
            # Newly observed binary hashes, not a fabricated source-selected-state
            # claim. D2 v2 cross-checks FREEZE references after complete39 custody.
            records.append({'cell':cell,'job':job,'donor':donor,'receipt':receipt,'output':output,
                            'source_custody':donor['source_custody'],
                            'qualified_runtime_versions':versions,
                            'job_path':job_path,'checkpoint':reference(checkpoint),'logits':reference(logits),
                            'queue_start':reference(start_path),'child_start':reference(child_start_path),
                            'exit_receipt':reference(exit_path),'root_release':reference(root_release_path),
                            'block_freeze':reference(block_path)})
    require(len(records) == 9 and [r['cell']['cell_id'] for r in records] == spec['execution_order'], 'Exactly all9 successful companions required')
    return records, common_inputs, common_plan_path


def prefit_proof(record):
    donor, job = record['donor'], record['job']
    return {'root_reviews':[donor['root_source_review'],donor['root_job_review']],
            'root_release':[record['root_release']], 'block_job_freeze_authentication':[donor['job_freeze_authentication']],
            'launch_receipt':[donor['launch_receipt']], 'source_manifest':[donor['provider_source_manifest']],
            'training_step_gate':[donor['training_step_gate']], 'qualified_runtime':job['runtime_qualification']['evidence'],
            'complete_cycle_costs':donor['complete_cycle_costs']}


def descriptor(record):
    donor, job = record['donor'], record['job']
    config = record['output']/'CONFIG.json'
    runtime = {'declared_runtime_versions':job['runtime_versions'],'qualified_runtime_versions':record['qualified_runtime_versions'],
               'training_runtime_observation_available':False,'training_runtime_observation':None,
               'observed_training_runtime_versions':None,'unavailable_reason':'Actual CONFIG runtime metadata unavailable to this collector'}
    if config.exists():
        config = phase_file(str(config.relative_to(PHASE)))
        observation = read_metadata(config)
        require(observation['runtime'] == job['runtime_versions'] and observation['job'] == job,
                'Actual CONFIG declaration/runtime differs from the owned fit')
        runtime.update(training_runtime_observation_available=True,training_runtime_observation=reference(config),
                       observed_training_runtime_versions=observation['runtime'],unavailable_reason=None)
    proof = prefit_proof(record)
    for refs in proof.values():
        require(refs, 'Actual pre-fit evidence role missing')
        for ref in refs:
            binding(ref)
    return {'schema':'authenticated_row0_companion_provider_custody_descriptor_v1',
            'derived_UTC':datetime.now(timezone.utc).isoformat(),'descriptor_is_pre_fit_admission_object':False,
            'collector_authenticated_pre_fit_evidence':True,'provider':donor['provider'],'hostname':donor['hostname'],
            'GPU_UUID':donor['GPU_UUID'],'provider_source_manifest':donor['provider_source_manifest'],'program':donor['program'],
            'pre_fit_evidence':proof,'runtime_provenance':runtime,
            'retrospective_executable_source_custody':record['source_custody'],
            'chronology':{'method':'Reviewed launcher validates actual review/source/job-freeze before queue Popen; bound queue validates jobs before owned child Popen. Saved queue/child PID,start_ticks,argv,cwd and same-host start/exit chronology authenticate actual execution.',
                          'queue_start':record['queue_start'],'child_start':record['child_start'],
                          'owned_exit_receipt':record['exit_receipt'],'block_freeze':record['block_freeze'],
                          'cross_host_wall_clock_order_assumed':False},
            'CONFIG_to_source_FREEZE_crosscheck_deferred_to_full39_D2':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(COLLECTOR_SOURCE_REVIEWED, 'Source preparation is disabled pending independent/root collector review and actual all9 prerequisites')
    release_path = phase_file(str(args.release.absolute().relative_to(PHASE)))
    release = read_metadata(release_path)
    require(release['root_collection_approved'] is True and release['all9_required'] is True
            and release['fits_authorized'] is False and release['TEST_access'] is False
            and release['original_score_recalculation'] is False
            and release['collection_contract_frozen_before_companion_score_analysis'] is True
            and release['source_review_evidence'], 'Separate metadata-only exact9 release required')
    for ref in release['source_review_evidence']:
        binding(ref)
    verify_packet(release['collector_source_manifest_sha256'])
    records, inputs, plan_path = authenticate_nine(release)
    # All9 source/metadata/terminal/artifact checks return before even CONFIG
    # runtime metadata observations. Histories/FREEZE/tensors are never parsed.
    descriptions = [descriptor(record) for record in records]
    output = fresh_output(args.output);output.mkdir()
    (output/'COHORT_PLAN.json').write_bytes(plan_path.read_bytes())
    (output/'COLLECTION_RELEASE.json').write_bytes(release_path.read_bytes())
    completed = []
    for record, value in zip(records, descriptions):
        cell, donor, receipt = record['cell'],record['donor'],record['receipt']
        path = output/('provider_custody_'+cell['cell_id']+'.json');write(path,value)
        completed.append({'cell_id':cell['cell_id'],'block':donor['block'],'provider':donor['provider'],
                          'hostname':donor['hostname'],'GPU_UUID':donor['GPU_UUID'],'provider_custody_binding':reference(path),
                          'source_manifest_sha256':donor['provider_source_manifest']['sha256'],'program_sha256':PROGRAM_SHA,
                          'original_donor_directory_relative':donor['donor_directory_relative'],
                          'job_relative':str(record['job_path'].relative_to(PHASE)),'job_sha256':receipt['job_sha256'],
                          'freeze_relative':str((record['output']/'FREEZE.json').relative_to(PHASE)),
                          'freeze_sha256':receipt['freeze_sha256'],
                          'selected_checkpoint_relative':record['checkpoint']['path'],'checkpoint_sha256':record['checkpoint']['sha256'],
                          'selected_VALID_logits_relative':record['logits']['path'],'VALID_logits_sha256':record['logits']['sha256'],
                          'donor_execution_receipt':receipt})
    write(output/'COLLECTION_FREEZE.json',{'schema':'authenticated_complete_row0_companion_collection_v1',
          'complete':True,'physical_fits':9,'completed':completed,'collection_release_sha256':sha(release_path),
          'collector_source_manifest_sha256':release['collector_source_manifest_sha256'],'cohort_plan_sha256':sha(plan_path),
          'companion_spec_sha256':SPEC['sha256'],'original_plan_binding':ORIGINAL_PLAN,'original_promotion_binding':PROMOTION,
          'input_identities':inputs,'comparative_scoring_performed':False,'quality_fields_accessed_or_emitted':False,
          'selected_prediction_payloads_opened':False,'TEST_access':False,'fits_authorized':False,
          'original_score_recalculation':False,'retry':False,'FREEZE_JSON_parsed':False,'history_JSON_parsed':False,
          'selected_artifact_hashes_retrospectively_observed':True,
          'source_FREEZE_semantic_crosschecks_deferred_to_full39_D2':True})
    print('Authenticated exactly all9 companion terminal/source/artifact metadata: '+str(output))


if __name__ == '__main__':
    main()
