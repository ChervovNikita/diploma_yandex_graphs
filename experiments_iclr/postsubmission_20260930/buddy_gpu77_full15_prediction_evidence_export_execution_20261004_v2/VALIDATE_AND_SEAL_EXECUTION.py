"""Validate complete small receipts and seal v2 without fetching tensor bodies."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
FETCHED = HERE / 'fetched_registered_outputs_v1/buddy_gpu77_postfamily_eval_preparation_v4'
SUP = FETCHED / 'root_predictions_supervision_20261004_v2'
EXPORT = FETCHED / 'root_predictions_v1'
PRIOR = PHASE / 'buddy_gpu77_full15_prediction_evidence_export_execution_20261004_v1'
PRIMARY = PHASE / 'buddy_gpu77_postfamily_heldout_evaluation_execution_20261004_v1'
ADMISSION_SHA = '969a60b5f61d97e005733bf9f00a85e313df232a7021edf4b9a5907e0bee4879'
RUNNER_SHA = 'ef4ec024b8f9f20ba30b6cec20fa5fe1e3062159ccd57cfe5cc8067d3d56fe5f'
GPU = 'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def descriptor(path):
    return dict(path=str(path.relative_to(PHASE)), bytes=path.stat().st_size, sha256=sha(path))


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


assert sha(HERE / 'ROOT_LAUNCHER_SOURCE_REVIEW.json') == '533b8948281be9754a38758365ef93cb4b5fbc52b659601f8abca18cb5309288'
assert read(HERE / 'ROOT_LAUNCHER_SOURCE_REVIEW.json')['decision'] == 'APPROVED_EXECUTE_ONCE'
assert sha(PRIOR / 'EXECUTION_MANIFEST.json') == 'b8fb3ab5b521120690a8ac159c935d2d7a0cea993ea40efb2d980deb1e8143c3'
for row in read(PRIOR / 'EXECUTION_MANIFEST.json')['files']:
    path = PHASE / row['path']
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']

monitor = read(HERE / 'MONITOR_RESULT_01.json')
assert monitor['status'] == 'ALL15_PHYSICALLY_COMPLETE' and monitor['all15_closure_established'] is True
assert monitor['physical_exit_code'] == 0 and monitor['completed_export_rows'] == 15
assert not monitor['missing'] and not monitor['oversized'] and not monitor['server_export_retention_issues']
assert monitor['remote_writes'] is False and monitor['export_relaunched'] is False
assert monitor['large_tensor_payloads_fetched'] is False and monitor['scientific_binary_payloads_fetched'] is False
assert len(monitor['owned_handles']) == 2 and all(row['observation'] == 'absent' for row in monitor['owned_handles'])
for row in monitor['files']:
    path = HERE / row['local_path']
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
transport = PHASE / monitor['transport']['path']
assert sha(transport) == monitor['transport']['sha256']
assert read(transport)['exit_code'] == 0

admission = read(FETCHED / 'ROOT_PREDICTION_EXPORT_ADMISSION.json')
assert sha(FETCHED / 'ROOT_PREDICTION_EXPORT_ADMISSION.json') == ADMISSION_SHA
assert sha(HERE / 'ROOT_PREDICTION_EXPORT_ADMISSION.json') == ADMISSION_SHA
assert sha(SUP / 'run_prediction_export_once.py') == RUNNER_SHA
assert sha(HERE / 'DETACHED_RUNNER_SOURCE.py') == RUNNER_SHA
preserved = read(SUP / 'PRESERVED_FAILED_ADMISSION_v1.json')
assert sha(SUP / 'PRESERVED_FAILED_ADMISSION_v1.json') == '3057f8b7cac62278ec4dfcf8a149cc41d84671cc98157c0af5ffcdbacfc88623'
assert {key: admission[key] for key in preserved} == preserved
assert set(admission) - set(preserved) == {'optimizer_fits', 'epochs_per_cell', 'closed_family_count_semantics'}
assert admission['optimizer_fits'] == 24 and admission['epochs_per_cell'] == 100

launch = read(SUP / 'DETACHED_LAUNCH.json')
launch_transport = PHASE / 'gpu77_connection_recovery_v1/commands/buddy_v4_prediction_export_launch_20261004_v2/RECEIPT.json'
assert read(launch_transport)['exit_code'] == 0
assert json.loads(read(launch_transport)['stdout'].strip()) == launch
assert launch['status'] == 'LAUNCHED_ONCE' and launch['HEAD'] == '6dad58e56e40175168f6fa2050d0855a97c3ab44'
assert launch['runner_handle']['pid'] == 3181380 and launch['runner_handle']['start_ticks'] == 1721678485
assert launch['admission']['sha256'] == ADMISSION_SHA and launch['runner_source']['sha256'] == RUNNER_SHA
assert launch['prior_failed_stage_invocations'] == 1 and launch['prior_extra_forwards'] == 0
assert launch['numerical_retry'] is False and launch['training_executed'] is False
assert launch['other_jobs_stopped'] is False and launch['normal_host_mode'] is True
runner = read(SUP / 'RUNNER_STARTED.json')
child = read(SUP / 'CHILD_STARTED.json')
physical = read(SUP / 'PHYSICAL_TERMINAL.json')
assert runner['identity']['pid'] == 3181380 and runner['identity']['start_ticks'] == 1721678485
assert child['identity']['pid'] == 3181382 and child['identity']['start_ticks'] == 1721678489
assert child['argv'] == physical['argv'] and physical['argv'][-1] == GPU
assert physical['exit_code'] == 0 and physical['ROOT_PREDICTION_EXPORT_ADMISSION_sha256'] == ADMISSION_SHA
assert physical['no_retry'] is True and physical['other_jobs_stopped'] is False and physical['namespace_isolation_used'] is False
assert sha(SUP / 'ADMISSION_RECOVERY_CUSTODY.json') == launch['metadata_recovery_custody']['sha256']
custody = read(SUP / 'ADMISSION_RECOVERY_CUSTODY.json')
assert custody['changed_fields'] == ['optimizer_fits', 'epochs_per_cell', 'closed_family_count_semantics']
assert custody['prior_extra_forwards'] == 0 and custody['scientific_source_or_primary_results_changed'] is False
assert custody['prior_supervision_files_modified'] is False and custody['exclusive_new_metadata_and_supervision'] is True
assert custody['preserved_failed_admission']['sha256'] == sha(SUP / 'PRESERVED_FAILED_ADMISSION_v1.json')
assert custody['corrected_admission']['sha256'] == ADMISSION_SHA
prior_sup = PRIOR / 'fetched_registered_outputs_v1/buddy_gpu77_postfamily_eval_preparation_v4/root_predictions_supervision_20261004_v1'
assert custody['prior_physical_terminal']['sha256'] == sha(prior_sup / 'PHYSICAL_TERMINAL.json')
assert custody['prior_stderr']['sha256'] == sha(prior_sup / 'CHILD.stderr.log')

receipt = read(EXPORT / 'PREDICTIONS_RECEIPT.json')
claim = read(EXPORT / 'EXPORT_CLAIM.json')
assert claim['status'] == 'in_progress' and claim['completed_exports'] == []
assert receipt['status'] == 'all15_prediction_evidence_replays_exported_once' and receipt['extra_forwards'] == 15
assert receipt['export_admission_sha256'] == ADMISSION_SHA and receipt['replay_GPU_UUID'] == GPU
assert receipt['primary_inference_timing_unchanged'] is True
assert receipt['no_new_training'] is True and receipt['no_new_selection'] is True
assert receipt['scientific_advantage_claimed'] is False and receipt['isolated_speedup_established'] is False
for key in receipt:
    if key in admission and key not in {'schema', 'UTC'}:
        assert receipt[key] == admission[key], key
for key in claim:
    if key not in {'status', 'completed_exports'}:
        assert claim[key] == receipt[key], key
boundary = receipt['execution_boundary']
assert boundary['mode'] == 'environment_and_explicit_repo_paths_only' and boundary['execution_guarded'] is False
assert boundary['stage_admission_sha256'] == ADMISSION_SHA and boundary['GPU_UUID'] == GPU
assert boundary['inherited_FD_audit']['outside_writable_regular_file_FDs_observed'] is False

primary_path = PRIMARY / 'fetched_registered_outputs_v1/buddy_gpu77_postfamily_eval_preparation_v4/root_eval_v1/EVALUATION_RECEIPT.json'
primary = read(primary_path)
assert sha(primary_path) == receipt['primary_evaluation_receipt_sha256'] == 'ac8d9e45ac87d8e8254fb3d1580e38d0884af7d675641932d59e9101b3447727'
assert receipt['test_manifest_sha256'] == '442d5f4a6d861da6b97431f49197412df3d4ac957cb0769df61327dbd7e14a9a'
assert receipt['test_cache_sha256'] == '011612668871657d4d3fff6dd6259c9da89d8d434cf1efc1f01cdf3748d977f1'
assert primary['official_test_qualification']['positive_rows'] == 46329
assert primary['official_test_qualification']['negative_rows'] == 100000
lock_path = PHASE / 'buddy_gpu77_postfamily_lock_audit_execution_20261004_v1/fetched_registered_outputs_v1/root_lock_v1/FAMILY_LOCK.json'
lock = read(lock_path)
assert sha(lock_path) == receipt['family_lock_sha256'] == '4d4041ad0d02a36c94bd9112f4e01029fb722eee75339a5743a0259435c33fd9'
cohort = {(row['arm'], row['seed']) for row in lock['runs']}
exports = receipt['completed_exports']
assert len(cohort) == len(exports) == 15 and {(row['arm'], row['seed']) for row in exports} == cohort
assert all(row['unchanged_primary_Hits50_agrees'] is True for row in exports)
bindings = {(row['arm'], row['seed']): row['result_sha256'] for row in receipt['primary_result_bindings']}
assert bindings == {(row['arm'], row['seed']): row['result_sha256'] for row in primary['final_results']}
scalar_paths = list((PRIMARY / 'fetched_registered_outputs_v1').rglob('final_test.json'))
assert len(scalar_paths) == 15
for path in scalar_paths:
    value = read(path)
    assert sha(path) == bindings[(value['arm'], value['seed'])]

for packet, key in [('buddy_gpu77_postfamily_eval_preparation_v4', 'evaluation_preparation_manifest_sha256'),
                    ('buddy_shared_cache_execution_v5', 'source_manifest_sha256'),
                    ('buddy_complete_data_cache_preparation_v3', 'data_wrapper_manifest_sha256'),
                    ('buddy_gpu77_resource_family_launcher_v3', 'family_launcher_manifest_sha256')]:
    manifest = PHASE / packet / 'SOURCE_MANIFEST.json'
    assert sha(manifest) == receipt[key]
    for row in read(manifest)['files']:
        path = manifest.parent / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']

descriptors = monitor['server_only_export_descriptors']
assert len(descriptors) == 16 and len({row['path'] for row in descriptors}) == 16
assert all(row['tensor_fetched'] is False for row in descriptors)
by_name = {Path(row['path']).name: row for row in descriptors}
for row in exports:
    observed = by_name[f"{row['arm']}_seed{row['seed']}.pt"]
    assert observed['bytes'] == row['bytes'] and observed['sha256'] == row['file_sha256']
assert by_name['CANDIDATE_ORDER.pt']['sha256'] == receipt['candidate_order_file_sha256']
assert not list(HERE.rglob('*.pt'))
assert receipt['summed_replay_forward_seconds'] == sum(row['evidence_replay_forward_seconds'] for row in exports)

validation = dict(
    schema='buddy77_corrected_once_prediction_export_completion_and_costs_v2',
    UTC=datetime.now(timezone.utc).isoformat(), status='ALL15_VALIDATED_ONCE_AND_PHYSICALLY_COMPLETE',
    root_approval=descriptor(HERE / 'ROOT_LAUNCHER_SOURCE_REVIEW.json'),
    launch=descriptor(SUP / 'DETACHED_LAUNCH.json'), physical=descriptor(SUP / 'PHYSICAL_TERMINAL.json'),
    custody=descriptor(SUP / 'ADMISSION_RECOVERY_CUSTODY.json'),
    prediction_receipt=descriptor(EXPORT / 'PREDICTIONS_RECEIPT.json'), export_claim=descriptor(EXPORT / 'EXPORT_CLAIM.json'),
    completed_exports=15, extra_forwards=15, unchanged_primary_Hits50_agreement_count=15,
    agreement_scope='The unchanged sealed exporter requires exact Hits@50 equality before each export; tensors were not downloaded or recomputed locally.',
    primary_result_bindings=receipt['primary_result_bindings'], primary_receipt_and_scalar_bytes_match_prior_fetched_evidence=True,
    original_failed_admission_bytes_preserved=True, earlier_v1_execution_manifest_and_all_listed_files_unchanged=True,
    corrected_admission_changes_only_closed_family_count_metadata=True,
    source_manifest_and_all_listed_source_files_unchanged=True,
    official_positive_rows=46329, official_negative_rows=100000, graph_policy='training_only_all_splits',
    primary_candidate_order_binding='d67d746e3b90affc22c5a6e2a64ec9a831ab17918ea9151d0ada14e10c47723c',
    candidate_order_file_sha256=receipt['candidate_order_file_sha256'],
    server_only_artifacts=descriptors, server_only_artifact_count=16,
    server_only_artifact_total_bytes=sum(row['bytes'] for row in descriptors),
    physical_costs=physical,
    launch_client_costs={key: launch[key] for key in launch if key.startswith('launch_client_')},
    internal_export_costs={key: receipt[key] for key in ['checkpoint_validation_cache_load_and_order_write_seconds',
                          'summed_replay_forward_seconds', 'total_export_wrapper_seconds', 'cost_scope']},
    earlier_failed_v1_physical_costs=read(prior_sup / 'PHYSICAL_TERMINAL.json'),
    cost_addition_rule='Physical, launch-client and internal timers have distinct scopes and overlap; internal timers are not added to physical wall time. V1 failure costs remain separately recorded.',
    owned_handles_at_final_observation=monitor['owned_handles'],
    no_new_training=True, no_new_selection=True, primary_inference_timing_unchanged=True,
    numerical_retry=False, large_tensor_payloads_fetched=False, other_jobs_stopped=False,
    namespace_isolation_used=False, scientific_advantage_claimed=False, isolated_speedup_established=False,
    root_independent_completion_review_performed_by_this_agent=False)
save('EXPORT_COMPLETION_VALIDATION_AND_COSTS.json', validation)

paths = sorted(path for path in HERE.rglob('*') if path.is_file() and path.name != 'EXECUTION_MANIFEST.json')
for identity in ['buddy_v4_prediction_export_launch_20261004_v2', 'buddy_v4_prediction_export_v2_monitor_20261004_v1']:
    paths += sorted((PHASE / 'gpu77_connection_recovery_v1/commands' / identity).glob('*'))
paths += [PHASE / 'gpu77_connection_recovery_v1/buddy_v4_prediction_export_launch_20261004_v2_command.txt',
          PHASE / 'gpu77_connection_recovery_v1/buddy_v4_prediction_export_v2_monitor_20261004_v1_command.txt']
files = [descriptor(path) for path in paths]
save('EXECUTION_MANIFEST.json', dict(
    schema='buddy77_corrected_once_export_complete_execution_manifest_v2', UTC=datetime.now(timezone.utc).isoformat(),
    status='ALL15_EXPORTED_ONCE_WITH_PRIMARY_AGREEMENT_AND_CUSTODY_PRESERVED', files=files,
    file_count=len(files), total_bytes=sum(row['bytes'] for row in files),
    corrected_v2_launch_count=1, prior_failed_v1_stage_count=1, prior_failed_v1_extra_forwards=0,
    numerical_export_forwards=15, numerical_retry_executed=False, original_scalar_results_and_timings_changed=False,
    large_tensor_payloads_fetched=False, server_only_tensor_files=16,
    earlier_manifest_or_publication_amended=False))
print(json.dumps(dict(status=validation['status'], manifest=descriptor(HERE / 'EXECUTION_MANIFEST.json'),
                      validation=descriptor(HERE / 'EXPORT_COMPLETION_VALIDATION_AND_COSTS.json'),
                      server_only_artifact_total_bytes=validation['server_only_artifact_total_bytes'])))
