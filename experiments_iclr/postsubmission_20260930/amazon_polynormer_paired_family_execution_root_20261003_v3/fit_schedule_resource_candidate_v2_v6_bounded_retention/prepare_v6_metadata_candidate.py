"""Stdlib source/metadata forecast only; every queue/release remains disabled.

Existing arrays, checkpoint bodies and runtime binaries are descriptor-only.
No imports from the scientific project, no remote access and no launch.
"""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

CANDIDATE = Path(__file__).resolve().parent
ROOT = CANDIDATE.parent
PHASE = ROOT.parent
SOURCE = PHASE / 'amazon_polynormer_paired_family_source_preparation_20261003_v6'
OLD = ROOT / 'fit_schedule_resource_candidate_v1'
TEXT = {'.json', '.py', '.md', '.txt', '.log', '.sh', '.html', '.patch'}


def read(path):
    assert path.suffix == '.json' and path.resolve().is_relative_to(PHASE)
    return json.loads(path.read_text())


def desc(path):
    assert path.is_file() and not path.is_symlink() and path.suffix in TEXT, path
    raw = path.read_bytes()
    raw.decode('utf8')
    return {'path': str(path.relative_to(PHASE)), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def verify(row):
    assert set(row) == {'path', 'sha256', 'bytes'}
    path = PHASE / row['path']
    assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
    assert desc(path) == row, path
    return path


def write(path, value):
    assert path.resolve().is_relative_to(CANDIDATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return desc(path)


def unique(rows):
    result = {}
    for row in rows:
        if row['path'] in result:
            assert result[row['path']] == row
        result[row['path']] = row
    return list(result.values())


def forecast(row, measurements, paid_qualifier, candidates=2700):
    m = measurements[row['kind']]
    # N-1 retirements is conservative when the last overall winner stays local;
    # with a distinct global winner the actual maximum is N-2.
    retirees = max(candidates - 1, 0)
    train = 200 * m['local_train_plus_VAL_seconds'] + 2500 * m['global_train_plus_VAL_seconds']
    io = candidates * m['snapshot_plus_write_seconds']
    # This is a verify+safe-load interval used as a hash proxy, not a hash timing.
    retirement_hash_proxy = retirees * m['safe_read_seconds']
    trace_fsync_assumption = 2700 * 0.1
    retirement_fsync_assumption = retirees * 0.25
    replays = 2 * m['complete_portable_replay_seconds_including_read_clone']
    # The measured portable interval excludes replay_scratch's outer preparation.
    outer_scratch_proxy = m['safe_read_seconds'] + m['joint_live_clone_seconds'] + m['CPU_snapshot_seconds']
    terminal_hash_proxy = 3 * 4 * m['safe_read_seconds']
    transition_final_proxy = 5 * m['safe_read_seconds'] + max(m['local_train_plus_VAL_seconds'], m['global_train_plus_VAL_seconds']) + m['snapshot_plus_write_seconds']
    components = {'all2700_train_plus_VAL_seconds': train,
        'all_selected_snapshot_plus_write_seconds': io,
        'retired_binary_hash_charge_proxy_seconds': retirement_hash_proxy,
        'trace_fsync_UNMEASURED_allowance_seconds': trace_fsync_assumption,
        'retirement_ledger_directory_fsync_unlink_UNMEASURED_allowance_seconds': retirement_fsync_assumption,
        'two_retained_portable_replays_including_inner_read_clone_seconds': replays,
        'outer_selected_local_scratch_preparation_charge_proxy_seconds': outer_scratch_proxy,
        'three_terminal_hash_passes_four_image_equivalents_charge_proxy_seconds': terminal_hash_proxy,
        'transition_final_restore_eval_and_logits_save_charge_proxy_seconds': transition_final_proxy,
        'setup_and_admission_metadata_charge_proxy_seconds': paid_qualifier}
    seconds = sum(components.values())
    without_fsync = seconds - trace_fsync_assumption - retirement_fsync_assumption
    return {'kind': row['kind'], 'assumed_strict_selected_candidates': candidates,
        'selection_count_is_scenario_not_future_quality_evidence': True,
        'scientific_optimizer_updates': 2700, 'retained_replay_invocations': 2,
        'retained_replay_optimizer_updates': 4, 'retirements_conservative_upper_bound': retirees,
        'components': components, 'rate_proxy_and_allowance_extrapolation_seconds': seconds,
        'with50percent_wall_margin_seconds': 1.5 * seconds,
        'retained_binary_upper_bound': min(candidates, 2),
        'retained_checkpoint_forecast_bytes': min(candidates, 2) * m['largest_observed_serialized_checkpoint_bytes'],
        'maximum_combined_fsync_seconds_per_update_to_fit_43200cap_with50percent_margin': (43200 / 1.5 - without_fsync) / 2700}


def main():
    now = datetime.now(timezone.utc).isoformat()
    source = {'manifest': desc(SOURCE / 'MANIFEST.json'), 'seal': desc(SOURCE / 'SEAL.json')}
    assert read(SOURCE / 'SEAL.json')['manifest'] == source['manifest']
    manifest = read(SOURCE / 'MANIFEST.json')
    for row in manifest['payload']:
        path = SOURCE / row['path']
        actual = desc(path)
        assert (actual['sha256'], actual['bytes']) == (row['sha256'], row['bytes'])
        if path.suffix == '.py':
            compile(ast.parse(path.read_text()), str(path), 'exec')
    inheritance = read(SOURCE / 'COHORT_SOURCE_BINDING.json')
    registry = read(verify(inheritance['registry']))
    assert registry['source'] == inheritance['registered_source']
    assert len(registry['physical_fits']) == 15 and len(registry['families']) == 9
    assert not registry['automatic_retry_authorized']
    assert read(verify(inheritance['master_source_claim']))['no_second_registry_or_replacement']
    assert not any((PHASE / row[key]).exists() for row in registry['physical_fits'] for key in ('output', 'claim_path', 'release_path'))
    old = read(OLD / 'RESOURCE_FORECAST_EVIDENCE.json')
    historical_qualification = read(verify(old['qualification_result']))
    assert historical_qualification['status'] == 'passed' and historical_qualification['source'] == old['source']
    assert read(verify(old['qualification_terminal']))['physical_exit_code'] == 0
    assert read(verify(old['remote_custody_and_resource']))['exit_code'] == 0
    measurements = old['measurements']
    fit_rows = []
    for row in registry['physical_fits']:
        value = forecast(row['row'], measurements, old['qualification_paid_wall_seconds'])
        value.update(fit_id=row['id'], registered_row=row['row'], registered_output=row['output'])
        fit_rows.append(value)
    scenarios = {kind: [forecast({'kind': kind}, measurements, old['qualification_paid_wall_seconds'], count)
        for count in (1, 10, 50, 100, 500, 2700)] for kind in measurements}
    raw_storage = sum(row['retained_checkpoint_forecast_bytes'] for row in fit_rows)
    storage_with_margin = math.ceil(raw_storage * 1.25)
    metadata_reserve = 8 * 2**30
    active_temporary_final_reserve = 2**30
    fresh_qualification_reserve = 3 * 2**30
    required_storage = storage_with_margin + metadata_reserve + active_temporary_final_reserve + fresh_qualification_reserve
    proposed_storage_budget = 32 * 2**30
    closure_proxy = sum(4 * measurements[row['kind']]['safe_read_seconds'] for row in fit_rows)
    fit_wall = sum(row['rate_proxy_and_allowance_extrapolation_seconds'] for row in fit_rows)
    caps = {'wall_seconds': 43200, 'rss_bytes': 32 * 2**30,
        'cuda_peak_allocated_bytes': 75 * 2**30, 'cuda_peak_reserved_bytes': 75 * 2**30}
    assert max(row['with50percent_wall_margin_seconds'] for row in fit_rows) < caps['wall_seconds']
    assert required_storage < proposed_storage_budget
    evidence = {'schema': 'amazon_polynormer_v6_bounded_retention_resource_forecast_v1', 'UTC': now,
        'status': 'DISABLED_PROVISIONAL_FORECAST_NOT_RESOURCE_ADMISSION', 'execution_authorized': False,
        'source': source, 'original_registry': inheritance['registry'],
        'historical_V5_forecast_preserved': desc(OLD / 'RESOURCE_FORECAST_EVIDENCE.json'),
        'historical_V5_denied_admission_preserved': desc(OLD / 'RESOURCE_ADMISSION_CANDIDATE.json'),
        'historical_V5_qualification_source': old['source'],
        'historical_V5_qualification_result': old['qualification_result'],
        'historical_V5_qualification_freeze': old['qualification_freeze'],
        'historical_V5_qualification_terminal': old['qualification_terminal'],
        'historical_V5_paid_wall_seconds': old['qualification_paid_wall_seconds'],
        'historical_V5_resource_observation': old['remote_custody_and_resource'],
        'measurement_origin': 'Actual successful V5 five-form bounded qualification. V6 scientific bodies unchanged; V6 retirement probe is unexecuted.',
        'measurements': measurements, 'all15_fit_forecasts': fit_rows, 'selection_count_scenarios': scenarios,
        'unmeasured_allowances': {'trace_flush_fsync_seconds_per_update': 0.1,
            'retirement_intent_completion_directory_fsync_unlink_seconds_per_retired_image': 0.25,
            'scope': 'Planning assumptions only. Fresh V6 actual retirement duration includes its hash and fsync/unlink operations; resource admission must replace these assumptions using actual same-filesystem evidence.'},
        'charge_proxy_scope': 'Safe verify+load encloses hash work; synchronized train+VAL encloses full inference. Neither is an independent hash/inference timing. Snapshot/clone intervals proxy untimed restoration. The whole successful five-form qualifier is charged once per future fit as setup/admission allowance. Portable replay measured intervals include inner read/clone only; outer selected-local scratch preparation is charged separately.',
        'storage': {'retained_checkpoint_bytes_all15_upper_bound': raw_storage, 'checkpoint_storage_margin': 1.25,
            'retained_checkpoint_bytes_with_margin': storage_with_margin, 'metadata_reserve_bytes': metadata_reserve,
            'active_temporary_and_final_reserve_bytes': active_temporary_final_reserve,
            'fresh_V6_qualification_reserve_bytes': fresh_qualification_reserve,
            'incremental_required_free_space_bytes': required_storage, 'candidate_budget_bytes': proposed_storage_budget,
            'max_persistent_checkpoint_bodies_per_fit': 2, 'max_new_checkpoint_temporaries_in_sequential_queue': 1,
            'old_full_history_snapshot_write_volume_bytes_still_charged': old['storage']['raw_immutable_checkpoint_history_bytes'],
            'historical_V5_outputs_preserved_at_existing_paths': True,
            'observed_available_bytes_at_historical_resource_receipt': old['storage']['observed_available_bytes'],
            'fits_historical_filesystem_capacity_observation': True,
            'user_quota_or32GiB_budget_certified': False, 'fresh_same_filesystem_observation_required': True},
        'wall': {'sum15_fit_extrapolations_seconds': fit_wall, 'closure_hash_charge_proxy_seconds': closure_proxy,
            'all15_and_closure_seconds': fit_wall + closure_proxy, 'with50percent_margin_seconds': (fit_wall + closure_proxy) * 1.5,
            'max_perfit_with50percent_margin_seconds': max(row['with50percent_wall_margin_seconds'] for row in fit_rows),
            'candidate_common_fit_cap_seconds': 43200, 'sum15_fit_cap_seconds': 15 * 43200,
            'allocation_lifetime_or_user_time_budget_certified': False, 'future_wall_guaranteed': False},
        'counts': {'physical_fits': 15, 'family_records': 9, 'scientific_optimizer_updates': 40500,
            'scientific_member_trajectory_updates': 64800, 'worst_selected_snapshots': 40500,
            'retained_replay_invocations': 30, 'retained_replay_optimizer_updates': 60,
            'retained_replay_member_trajectory_updates': 96, 'total_optimizer_updates': 40560,
            'total_member_trajectory_updates': 64896, 'native_single_alias_extra_fits': 0},
        'pending_admission_inputs': ['independent exact V6 source review', 'exact V6 runtime receipt and consumer release',
            'fresh exact V6 five-form numerical qualification with owned retirement and full-state/RNG isolation',
            'same-filesystem measured retirement/metadata cost and capacity/quota budget', 'allocation lifetime or user time budget'],
        'engineering_forecast_is_not_predictive_progress_or_acceptance': True,
        'read_scope': {'scientific_project_or_numpy_torch_imports': False, 'real_arrays_checkpoints_runtime_binaries_read': False,
            'V5_checkpoint_descriptors_carried_without_opening': old['read_accounting']['frozen_checkpoint_descriptors_carried_from_remote_verified_inventory'],
            'SSH_remote_or_launches': False, 'existing_source_or_outputs_modified': False}}
    evidence_row = write(CANDIDATE / 'RESOURCE_FORECAST_EVIDENCE.json', evidence)
    attempts = copy.deepcopy(read(OLD / 'ATTEMPT_REGISTRY_AFTER_QUALIFIER_CANDIDATE.json'))
    attempts.update(source=source, preserved_failures=desc(SOURCE / 'PRESERVED_FAILURES.json'),
        original_registered_cohort_source=inheritance['registered_source'],
        predecessor_attempt_registry=desc(OLD / 'ATTEMPT_REGISTRY_AFTER_QUALIFIER_CANDIDATE.json'),
        candidate_only_NOT_AUTHORIZATION=True, scientific_fits_started=0, fresh_V6_qualification_pending=True)
    attempts['prior_attempt_artifacts'] = unique([*attempts['prior_attempt_artifacts'],
        desc(OLD / 'ATTEMPT_REGISTRY_AFTER_QUALIFIER_CANDIDATE.json'),
        *inheritance['registered_source'].values(), desc(OLD / 'RESOURCE_FORECAST_EVIDENCE.json'),
        desc(OLD / 'RESOURCE_ADMISSION_CANDIDATE.json')])
    attempts_row = write(CANDIDATE / 'ATTEMPT_REGISTRY_CANDIDATE.json', attempts)
    resource = {'schema': 'amazon_polynormer_resource_admission_v3', 'UTC': now,
        'status': 'HELD_FRESH_V6_QUALIFICATION_REVIEW_AND_MEASURED_RESOURCE_ADMISSION_PENDING',
        'execution_authorized': False, 'source': source, 'registry': inheritance['registry'],
        'qualification_freeze': None, 'historical_V5_qualification_freeze_NOT_V6_GATE': old['qualification_freeze'],
        'full_15_fit_schedule_authorized': False, 'retained_local_and_final_checkpoint_replays_costed': True,
        'all_selected_checkpoint_replays_required': False, 'caps': caps, 'forecast_evidence': evidence_row,
        'forecast_to_fill_from_actual_probe': {
            'native_and_GNNM_complete_local_global_train_VAL_seconds': {k: {s: m[s] for s in ('local_train_plus_VAL_seconds', 'global_train_plus_VAL_seconds')} for k, m in measurements.items()},
            'four_member_inference_seconds': {'standalone_measurement': None, 'historical_enclosing_train_VAL_proxy_seconds': measurements['gnnm_boundary_4']['global_train_plus_VAL_seconds']},
            'CPU_snapshot_save_read_and_live_replay_clone_peak': {'historical_V5_per_form': measurements, 'fresh_V6_retirement_probe': None},
            'worst_case_selected_candidates_per_fit': 2700, 'immutable_checkpoint_storage_bytes': evidence['storage'],
            'full_process_wall_and_15_fit_cost': evidence['wall']},
        'not_a_utility_competence_gate': True, 'root_exact_measured_admission_required': True,
        'no_fit_authority_from_capacity_forecast_or_engineering_pass': True,
        'pending_inputs': evidence['pending_admission_inputs']}
    resource_row = write(CANDIDATE / 'RESOURCE_ADMISSION_CANDIDATE.json', resource)
    base = copy.deepcopy(read(OLD / 'ALL_FITS_RELEASE_CANDIDATE.json'))
    for key in ('fit_releases', 'closure_release'):
        base.pop(key, None)
    historical_bindings = {k: base[k] for k in ('source', 'source_review', 'runtime_receipt', 'consumer_release', 'qualification_freeze')}
    base.update(source=source, source_review=None, independent_source_review_passed=False,
        runtime_receipt=None, consumer_release=None, qualification_freeze=None,
        historical_V5_bindings_NOT_CURRENT_ADMISSION=historical_bindings,
        resource_admission=resource_row, attempt_registry=attempts_row, execution_authorized=False,
        caps=caps, candidate_only_NOT_RUNTIME_RELEASE=True, actual_registered_releases_not_materialized=True,
        pending_exact_V6_inputs=['source_review', 'runtime_receipt', 'consumer_release', 'qualification_freeze'])
    # Opaque original arrays/checkpoint bodies are preserved as exact descriptors.
    base['custody_inputs'] = unique([*base['custody_inputs'], *source.values(), evidence_row, resource_row,
        attempts_row, *attempts['prior_attempt_artifacts']])
    fit_candidates = []
    for row in registry['physical_fits']:
        value = copy.deepcopy(base)
        value.update(kind='fit', fit_id=row['id'], output=row['output'], self_path=row['release_path'], registered_claim_path=row['claim_path'])
        record = write(CANDIDATE / 'disabled_fit_releases' / (row['id'] + '.json'), value)
        fit_candidates.append({'fit_id': row['id'], 'candidate_descriptor': record,
            'registered_release_path': row['release_path'], 'registered_output': row['output'], 'registered_claim_path': row['claim_path']})
    close = copy.deepcopy(base)
    close.update(kind='close', device='cpu', output=str(ROOT.relative_to(PHASE)) + '/closure_v1',
        self_path=str(ROOT.relative_to(PHASE)) + '/releases/close_v1.json', caps=dict(caps, wall_seconds=21600))
    close_row = write(CANDIDATE / 'CLOSE_RELEASE_CANDIDATE.json', close)
    queue = copy.deepcopy(base)
    queue.update(kind='all', output=str(ROOT.relative_to(PHASE)) + '/full_schedule_v1',
        self_path=str(ROOT.relative_to(PHASE)) + '/releases/all_fits_v1.json',
        caps=dict(caps, wall_seconds=15 * 43200 + 21600 + 3600),
        fit_releases=[row['candidate_descriptor'] for row in fit_candidates], closure_release=close_row)
    queue['custody_inputs'] = unique([*queue['custody_inputs'], *queue['fit_releases'], close_row])
    queue_row = write(CANDIDATE / 'ALL_FITS_RELEASE_CANDIDATE.json', queue)
    plan = {'schema': 'amazon_polynormer_v6_disabled_sequential_queue_v1', 'UTC': now,
        'status': 'HELD_NO_EXECUTION_FRESH_V6_REVIEW_QUALIFICATION_AND_ADMISSION_PENDING', 'execution_authorized': False,
        'source': source, 'registry': inheritance['registry'], 'resource_candidate': resource_row,
        'all_release_candidate': queue_row, 'closure_release_candidate': close_row, 'fit_candidates': fit_candidates,
        'candidate_common_fit_caps': caps, 'candidate_storage_budget_bytes': proposed_storage_budget,
        'sequential_no_overlapping_fits': True, 'stop_on_first_physical_failure': True,
        'automatic_retry_or_replacement_cohort': False, 'TRAIN_control_heldout_TEST_launch_authorized': False,
        'real_registered_release_output_claim_paths_created': False,
        'all_candidate_bodies_execution_false_and_missing_exact_V6_admission_inputs': True,
        'pending_admission_inputs': evidence['pending_admission_inputs']}
    plan_row = write(CANDIDATE / 'SEQUENTIAL_QUEUE_CANDIDATE.json', plan)
    assert not any((PHASE / row[key]).exists() for row in registry['physical_fits'] for key in ('output', 'claim_path', 'release_path'))
    result = {'schema': 'amazon_polynormer_v6_metadata_preparation_result_v1', 'status': 'PREPARED_DISABLED_ONLY',
        'source': source, 'forecast': evidence_row, 'resource_candidate': resource_row,
        'attempt_registry_candidate': attempts_row, 'queue_candidate': plan_row,
        'all15_common_fit_caps': caps, 'forecast_all15_plus_closure_hours': (fit_wall + closure_proxy) / 3600,
        'forecast_with50percent_margin_hours': (fit_wall + closure_proxy) * 1.5 / 3600,
        'retained_checkpoint_bytes': raw_storage, 'incremental_required_free_space_bytes': required_storage,
        'candidate_storage_budget_bytes': proposed_storage_budget, 'source_and_old_candidates_unmodified': True,
        'scientific_or_numeric_imports_arrays_checkpoints_runtime_binaries_remote_or_launches': False,
        'registered_release_output_claim_paths_created': False, 'execution_authorized': False}
    write(CANDIDATE / 'PREPARATION_RESULT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
