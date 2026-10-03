"""Build a measured, fully disabled proposal; no scientific imports or launches."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE = ROOT.parent
SOURCE = PHASE / 'amazon_polynormer_paired_family_source_preparation_20261003_v6'
FORECAST = ROOT / 'fit_schedule_resource_candidate_v2_v6_bounded_retention'
AFTER = ROOT / 'v6_execution_metadata_preparation_v1/after_qualification_disabled_v1'
QUALIFIER = ROOT / 'v6_qualification_block0_cuda0_v1'
BENCH = ROOT / 'v6_resource_filesystem_benchmark_20261003_v1'
TEXT = {'.json', '.jsonl', '.py', '.md', '.txt', '.log', '.sh', '.html', '.patch'}
OPAQUE = {'.pt', '.npz', '.npy', '.pth'}
now = datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(p.read_text())


def desc(p):
    assert p.suffix in TEXT and not p.is_symlink()
    raw = p.read_bytes(); raw.decode('utf8')
    return dict(path=p.relative_to(PHASE).as_posix(), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def verify(row):
    p = PHASE / row['path']
    assert not Path(row['path']).is_absolute() and p.resolve().is_relative_to(PHASE)
    assert desc(p) == row
    return p


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as h:
        json.dump(value, h, indent=2, allow_nan=False); h.write('\n')
    return desc(p)


def unique(rows):
    answer = {}
    for row in rows:
        assert row['path'] not in answer or answer[row['path']] == row
        answer[row['path']] = row
    return list(answer.values())


def maximum(values):
    return max(values)


source = dict(manifest=desc(SOURCE/'MANIFEST.json'), seal=desc(SOURCE/'SEAL.json'))
assert source['manifest']['sha256'] == 'd4160d8aeec2b770274875f9a1fb137facc9f93efd4eca6ad91e58802521ff44'
assert read(SOURCE/'SEAL.json')['manifest'] == source['manifest']
for row in read(SOURCE/'MANIFEST.json')['payload']:
    actual = desc(SOURCE/row['path']); assert (actual['sha256'], actual['bytes']) == (row['sha256'], row['bytes'])
plan = read(AFTER/'PLAN.json')
registry = read(verify(plan['original_registry']))
cohort = read(SOURCE/'COHORT_SOURCE_BINDING.json')
assert registry['source'] == cohort['registered_source']
assert len(registry['physical_fits']) == 15 and len(registry['families']) == 9
assert plan['qualification_freeze'] == desc(QUALIFIER/'FREEZE.json')
freeze, q, terminal = [read(QUALIFIER/n) for n in ('FREEZE.json', 'RESULT.json', 'TERMINAL.json')]
assert freeze['status'] == terminal['status'] == 'success' and terminal['physical_exit_code'] == 0
assert q['source'] == source and q['status'] == 'passed' and len(q['forms']) == 5
assert q['report_eligible'] is False and q['full_study_authorized'] is False
assert all(f['local_replay']['bitwise_full_next_step'] and f['global_replay']['bitwise_full_next_step'] and
           f['retirement_probe']['live_model_Adam_grad_modes_stage_and_RNG_bitwise_unchanged'] and
           f['retirement_probe']['selected_local_and_global_probe_still_available'] for f in q['forms'])
historical = read(FORECAST/'RESOURCE_FORECAST_EVIDENCE.json')
assert historical['source'] == source and historical['original_registry'] == plan['original_registry']
review = read(verify(plan['independent_source_review_evidence']))
assert review['status'] == 'PASS_SOURCE_ONLY' and review['source'] == source
assert review['forecast']['manifest'] == desc(FORECAST/'MANIFEST.json')
assert read(FORECAST/'SEAL.json')['manifest'] == desc(FORECAST/'MANIFEST.json')
for row in read(FORECAST/'MANIFEST.json')['payload']:
    verify(row['descriptor'])
observation = read(BENCH/'RESOURCE_OBSERVATION.json')
benchmark = read(BENCH/'RESULT.json')
assert benchmark['status'] == 'success' and benchmark['same_filesystem_verified']
assert benchmark['resource_observation'] == desc(BENCH/'RESOURCE_OBSERVATION.json')
assert benchmark['cumulative_payload_bytes_written'] < 2*2**30
assert benchmark['all_owned_temporary_payloads_removed_after_recorded_exact_descriptor']
verify(benchmark['trace_descriptor']); verify(benchmark['retirement_journal_descriptor'])
events = [json.loads(line) for line in (BENCH/'OWNED_RETIREMENT.jsonl').read_text().splitlines()]
assert len(events) == 9
for i, sample in enumerate(benchmark['owned_payload_samples']):
    assert all(event['descriptor'] == sample['descriptor'] for event in events[i*3:(i+1)*3])
    assert events[i*3+1]['hash_size_verified'] and events[i*3+2]['binary_absent'] and sample['payload_absent']
measurements = {}
for kind in ('gnnm_boundary_4', 'native_independent'):
    forms = [f for f in q['forms'] if f['row']['kind'] == kind]
    ios = [io for f in forms for io in f['checkpoint_io']]
    replays = [f[k] for f in forms for k in ('local_replay', 'global_replay')]
    measurements[kind] = dict(
        measured_forms=len(forms), local_train_plus_VAL_seconds=maximum(u['seconds'] for f in forms for u in f['updates'] if u['stage']=='local'),
        global_train_plus_VAL_seconds=maximum(u['seconds'] for f in forms for u in f['updates'] if u['stage']=='global'),
        CPU_snapshot_seconds=maximum(io['CPU_snapshot_seconds'] for io in ios),
        safe_write_seconds=maximum(io['safe_write_seconds'] for io in ios),
        snapshot_plus_write_seconds=maximum(io['CPU_snapshot_seconds']+io['safe_write_seconds'] for io in ios),
        safe_read_seconds=maximum(r['safe_read_seconds'] for r in replays),
        joint_live_clone_seconds=maximum(r['joint_live_clone_seconds'] for r in replays),
        complete_portable_replay_seconds_including_read_clone=maximum(r['seconds'] for r in replays),
        largest_observed_serialized_checkpoint_bytes=maximum(io['serialized_bytes'] for io in ios),
        qualified_retirement_complete_seconds=maximum(f['retirement_probe']['receipt']['retirement_wall_seconds'] for f in forms),
        qualified_retirement_copy_write_seconds=maximum(f['retirement_probe']['audit_copy_write_seconds'] for f in forms),
        state_bytes=forms[0]['state_bytes'], raw_five_form_measurements=forms)
samples = benchmark['owned_payload_samples']
trace_fsync = benchmark['trace_append_flush_fsync_max_seconds']
cold_hash = maximum(s['initial_descriptor_hash_seconds'] for s in samples)
retirement_nonhash = maximum(s['full_retirement_including_completion_seconds']-s['retirement_recheck_hash_seconds'] for s in samples)
full_retirement = maximum(s['full_retirement_including_completion_seconds'] for s in samples)
qualification_retirement = maximum(f['retirement_probe']['receipt']['retirement_wall_seconds'] for f in q['forms'])
retirement_charge = max(full_retirement, qualification_retirement, cold_hash+retirement_nonhash)
fsync = dict(trace_append_flush_fsync_seconds_per_update=trace_fsync,
            owned_complete_retirement_max_seconds=full_retirement,
            qualified_complete_retirement_max_seconds=qualification_retirement,
            first_descriptor_hash_max_seconds=cold_hash,
            owned_complete_retirement_nonhash_max_seconds=retirement_nonhash,
            conservative_complete_retirement_charge_seconds=retirement_charge,
            scope='Complete retirement includes hash, intent flush/fsync, unlink, directory fsync and completion flush/fsync. A slower first read/hash plus the observed nonhash interval is charged to cover the measured cache disparity; no additional duplicate hash/fsync charge is added.')
paid_setup = terminal['whole_process_wall_seconds']
fits = []
for row in registry['physical_fits']:
    m = measurements[row['row']['kind']]
    components = dict(
        all2700_train_plus_VAL_seconds=200*m['local_train_plus_VAL_seconds']+2500*m['global_train_plus_VAL_seconds'],
        all2700_selected_snapshot_plus_write_seconds=2700*m['snapshot_plus_write_seconds'],
        all2699_complete_retirements_measured_charge_seconds=2699*retirement_charge,
        all2700_trace_append_flush_fsync_seconds=2700*trace_fsync,
        two_retained_stage_and_final_portable_replays_seconds=2*m['complete_portable_replay_seconds_including_read_clone'],
        outer_selected_local_scratch_preparation_charge_proxy_seconds=m['safe_read_seconds']+m['joint_live_clone_seconds']+m['CPU_snapshot_seconds'],
        three_terminal_hash_passes_four_image_equivalents_charge_seconds=3*4*max(m['safe_read_seconds'],cold_hash),
        transition_final_restore_eval_and_logits_save_charge_proxy_seconds=5*m['safe_read_seconds']+max(m['local_train_plus_VAL_seconds'],m['global_train_plus_VAL_seconds'])+m['snapshot_plus_write_seconds'],
        setup_admission_full_paid_qualifier_allowance_seconds=paid_setup)
    total = sum(components.values())
    fits.append(dict(fit_id=row['id'], registered_row=row['row'], registered_output=row['output'],
                     components=components, extrapolated_seconds=total, with50percent_margin_seconds=1.5*total,
                     worst_case_selected_snapshots=2700, conservative_retirements=2699,
                     retained_stage_and_final_replay_invocations=2, replay_optimizer_updates=4,
                     scientific_optimizer_updates=2700,
                     retained_binary_upper_bound=2, retained_checkpoint_forecast_bytes=2*m['largest_observed_serialized_checkpoint_bytes']))
caps = dict(wall_seconds=43200, rss_bytes=32*2**30, cuda_peak_allocated_bytes=75*2**30, cuda_peak_reserved_bytes=75*2**30)
assert max(f['with50percent_margin_seconds'] for f in fits) < caps['wall_seconds']
closure_hash = sum(4*max(measurements[f['registered_row']['kind']]['safe_read_seconds'],cold_hash) for f in fits)
future = sum(f['extrapolated_seconds'] for f in fits)+closure_hash
raw_storage = sum(f['retained_checkpoint_forecast_bytes'] for f in fits)
storage = dict(retained_checkpoint_bytes_all15_upper_bound=raw_storage, checkpoint_storage_margin=1.25,
               retained_checkpoint_bytes_with_margin=math.ceil(raw_storage*1.25),
               metadata_reserve_bytes=8*2**30, active_temporary_final_and_logits_reserve_bytes=2**30,
               additional_conservative_qualification_audit_reserve_bytes=3*2**30,
               candidate_operational_storage_budget_bytes=32*2**30,
               max_persistent_checkpoint_bodies_per_fit=2, max_new_checkpoint_temporaries_in_sequential_queue=1,
               full_worst_case_snapshot_write_volume_bytes=sum(2700*measurements[f['registered_row']['kind']]['largest_observed_serialized_checkpoint_bytes'] for f in fits),
               existing_all_historical_and_V6_qualification_outputs_preserved=True,
               observed_filesystem_available_bytes=observation['filesystem']['available_bytes'],
               user_quota_limit_bytes=None, user_quota_remaining_bytes=None, user_quota_certified=False,
               chosen_storage_budget_is_not_observed_user_quota=True)
storage['incremental_required_free_space_bytes'] = (storage['retained_checkpoint_bytes_with_margin']+
    storage['metadata_reserve_bytes']+storage['active_temporary_final_and_logits_reserve_bytes']+
    storage['additional_conservative_qualification_audit_reserve_bytes'])
assert storage['incremental_required_free_space_bytes'] < storage['candidate_operational_storage_budget_bytes'] < observation['filesystem']['available_bytes']
queue_cap = 15*caps['wall_seconds']+21600+3600
chosen_budget = 192*3600
assert 1.5*future < queue_cap < chosen_budget
wall = dict(all15_fit_extrapolations_seconds=sum(f['extrapolated_seconds'] for f in fits),
            closure_hash_charge_seconds=closure_hash, full15_plus_closure_extrapolated_seconds=future,
            with50percent_margin_seconds=1.5*future,
            max_single_fit_with50percent_margin_seconds=max(f['with50percent_margin_seconds'] for f in fits),
            chosen_common_fit_wall_cap_seconds=caps['wall_seconds'], closure_wall_cap_seconds=21600,
            queue_preparation_allowance_seconds=3600, sum15_fit_caps_seconds=15*caps['wall_seconds'],
            proposed_queue_wall_cap_seconds=queue_cap,
            chosen_prospective_research_queue_time_budget_seconds=chosen_budget,
            chosen_budget_hours=192, chosen_budget_source='Continuing user research authorization and explicit root permission to propose a prospective queue time budget.',
            physical_allocation_expiry_UTC=None, physical_allocation_expiry_certified=False,
            chosen_budget_is_not_observed_physical_allocation_expiry=True,
            future_runtime_guaranteed=False)
prior = read(verify(plan['attempt_registry_disabled']))
artifacts = unique([*prior['prior_attempt_artifacts'], plan['attempt_registry_disabled'],
                   *source.values(), desc(SOURCE/'PRESERVED_FAILURES.json'),
                   desc(FORECAST/'MANIFEST.json'), desc(FORECAST/'SEAL.json'), desc(FORECAST/'RESOURCE_FORECAST_EVIDENCE.json'),
                   plan['independent_source_review_evidence'], plan['source_review_for_runtime_gate'],
                   *[desc(BENCH/n) for n in ('RESOURCE_OBSERVATION.json','RESULT.json','TRACE_FSYNC.jsonl','OWNED_RETIREMENT.jsonl')],
                   desc(ROOT/'v6_runtime_execution_receipts_20261003_v1/EXECUTION_CLOSURE.json'),
                   desc(ROOT/'v6_qualification_execution_receipts_20261003_v1/EXECUTION_CLOSURE.json')])
paid = []
for row in artifacts:
    p = PHASE/row['path']
    if p.suffix in TEXT:
        verify(row)
    else:
        assert p.suffix in OPAQUE
    if p.name == 'TERMINAL.json':
        value = read(p)
        seconds = next((value[k] for k in ('whole_process_wall_seconds','wall_seconds','elapsed_seconds') if k in value), None)
        paid.append(dict(receipt=row,status=value.get('status'),physical_exit_code=value.get('physical_exit_code',value.get('exit_code')),
                         recorded_wall_seconds=seconds,
                         wall_not_recorded_must_not_be_invented=seconds is None,
                         recorded_peak_RSS_bytes=value.get('aggregate_peak_observed_rss_bytes')))
counts = dict(physical_fits=15, family_records=9, scientific_optimizer_updates=40500,
              scientific_member_trajectory_updates=64800, worst_selected_snapshots=40500,
              conservative_retirements=40485, retained_stage_final_replay_invocations=30,
              retained_replay_optimizer_updates=60, retained_replay_member_trajectory_updates=96,
              total_optimizer_updates=40560, total_member_trajectory_updates=64896,
              native_single_alias_extra_fits=0)
evidence = dict(schema='amazon_polynormer_V6_measured_full_schedule_resource_forecast_v1',UTC=now,
    status='MEASURED_DISABLED_RESOURCE_ADMISSION_PROPOSAL',execution_authorized=False,source=source,
    original_registry=plan['original_registry'],original_master_source_claim=plan['original_master_source_claim'],
    reviewed_retention_forecast=dict(manifest=desc(FORECAST/'MANIFEST.json'),seal=desc(FORECAST/'SEAL.json'),evidence=desc(FORECAST/'RESOURCE_FORECAST_EVIDENCE.json')),
    independent_source_review=plan['independent_source_review_evidence'],runtime_compatible_review=plan['source_review_for_runtime_gate'],
    new_qualification=dict(freeze=desc(QUALIFIER/'FREEZE.json'),result=desc(QUALIFIER/'RESULT.json'),terminal=desc(QUALIFIER/'TERMINAL.json')),
    runtime_receipt=q['runtime_receipt'],consumer_release=q['consumer_release'],
    filesystem_benchmark=dict(result=desc(BENCH/'RESULT.json'),observation=desc(BENCH/'RESOURCE_OBSERVATION.json')),
    fresh_V6_measurements=measurements,measured_fsync_retirement_charges=fsync,all15_fit_forecasts=fits,
    storage=storage,wall=wall,counts=counts,
    observed_qualification_peak_memory=q['cost']['memory'],
    paid_prior_physical_terminal_costs=paid,
    sum_recorded_prior_terminal_wall_seconds=sum(r['recorded_wall_seconds'] for r in paid if r['recorded_wall_seconds'] is not None),
    prior_terminal_costs_with_unknown_wall=sum(r['recorded_wall_seconds'] is None for r in paid),
    benchmark_paid_wall_seconds=benchmark['whole_benchmark_wall_seconds'],
    every_prior_failure_incomplete_superseded_attempt_disclosed=True,prior_attempt_registry=plan['attempt_registry_disabled'],
    prior_attempt_artifacts=artifacts,
    extrapolation_limits=['Five block-zero forms on the admitted single GPU; split/seed variation remains prospective.',
        'Whole train+VAL intervals enclose required inference; no isolated inference timing is claimed.',
        'Trace and synthetic retirement timings are short same-filesystem observations; future NFS contention remains possible.',
        'Two protected full portable replays plus outer selected-local preparation are charged; no replay of retired candidates is promised.',
        'Largest observed checkpoint size and a 25% retained-storage margin are used; every one of 40500 worst-case snapshot writes is still paid.',
        'Quota and physical allocation expiry remain unknown. Chosen operational budgets do not certify them.'],
    no_recipe_family_selector_training_checkpoint_or_replay_reduction=True,
    predictive_fit_TEST_control_or_scoring_launches=0,new_registration=False)
evidence_row = write(HERE/'RESOURCE_FORECAST_EVIDENCE.json',evidence)
attempt = copy.deepcopy(prior)
attempt.update(predecessor_attempt_registry=plan['attempt_registry_disabled'],fresh_V6_qualification_pending=False,
               new_source_numerically_qualified=True,execution_authorized=False,scientific_fits_started=0)
attempt['prior_attempt_artifacts'] = unique([*artifacts,evidence_row])
attempt_row = write(HERE/'ATTEMPT_REGISTRY_DISABLED.json',attempt)
correction = write(HERE/'METADATA_CORRECTION_NOTE.json',dict(
    schema='amazon_polynormer_prospective_attempt_metadata_correction_v1',predecessor=plan['attempt_registry_disabled'],
    corrected_disabled_successor=attempt_row,stale_predecessor_field={'fresh_V6_qualification_pending':True},
    corrected_field={'fresh_V6_qualification_pending':False},authority=desc(QUALIFIER/'FREEZE.json'),
    previous_helper_source_and_sealed_packet_unchanged=True,execution_authorized=False))
resource = dict(schema='amazon_polynormer_resource_admission_v3',UTC=now,
    status='MEASURED_CANDIDATE_AWAITING_ROOT_INSPECTION',execution_authorized=False,source=source,
    registry=plan['original_registry'],qualification_freeze=desc(QUALIFIER/'FREEZE.json'),
    full_15_fit_schedule_authorized=False,retained_local_and_final_checkpoint_replays_costed=True,
    all_selected_checkpoint_replays_required=False,caps=caps,forecast_evidence=evidence_row,
    forecast_to_fill_from_actual_probe=dict(
        native_and_GNNM_complete_local_global_train_VAL_seconds={k:{s:m[s] for s in ('local_train_plus_VAL_seconds','global_train_plus_VAL_seconds')} for k,m in measurements.items()},
        four_member_inference_seconds={'standalone_measurement':None,'measured_enclosing_train_VAL_proxy_seconds':measurements['gnnm_boundary_4']['global_train_plus_VAL_seconds'],'isolated_inference_not_claimed':True},
        CPU_snapshot_save_read_and_live_replay_clone_peak={'fresh_V6_per_form':measurements,'qualified_memory_peaks':q['cost']['memory'],'same_filesystem_fsync_retirement':fsync},
        worst_case_selected_candidates_per_fit=2700,immutable_checkpoint_storage_bytes=storage,
        full_process_wall_and_15_fit_cost=wall),
    current_resource_observation=desc(BENCH/'RESOURCE_OBSERVATION.json'),same_filesystem_benchmark=desc(BENCH/'RESULT.json'),
    attempt_registry=attempt_row,prospective_queue_time_budget_seconds=chosen_budget,
    prospective_storage_budget_bytes=storage['candidate_operational_storage_budget_bytes'],
    current_observed_capacity_sufficient=True,actual_observed_resource_blockers=[],
    quota_status='UNKNOWN_UNCERTIFIED',physical_allocation_expiry_status='UNKNOWN_UNCERTIFIED',
    observed_physical_allocation_expiry_UTC=None,chosen_budget_is_not_observed_physical_allocation_expiry=True,
    not_a_utility_competence_gate=True,root_inspection_and_release_required=True,
    predictive_progress_or_acceptance_established=False)
resource_row = write(HERE/'RESOURCE_ADMISSION_CANDIDATE.json',resource)
fit_candidates=[]
for row,c in zip(registry['physical_fits'],plan['original15fit_candidates']):
    value=copy.deepcopy(read(verify(c['candidate'])))
    value.update(execution_authorized=False,caps=caps,attempt_registry=attempt_row,resource_admission=resource_row)
    value['custody_inputs']=unique([*value['custody_inputs'],*attempt['prior_attempt_artifacts'],attempt_row,evidence_row,resource_row,correction])
    assert value['output']==row['output'] and value['self_path']==row['release_path'] and value['registered_claim_path']==row['claim_path']
    fit_candidates.append(dict(fit_id=row['id'],candidate=write(HERE/'disabled_fit_releases'/(row['id']+'.json'),value),
                               registered_output=row['output'],registered_release_path=row['release_path'],registered_claim_path=row['claim_path']))
close=copy.deepcopy(read(verify(plan['releases_disabled']['closure'])))
close.update(execution_authorized=False,caps=dict(caps,wall_seconds=21600),attempt_registry=attempt_row,resource_admission=resource_row)
close['custody_inputs']=unique([*close['custody_inputs'],*attempt['prior_attempt_artifacts'],attempt_row,evidence_row,resource_row,correction])
close_row=write(HERE/'CLOSURE_RELEASE_DISABLED.json',close)
queue=copy.deepcopy(read(verify(plan['releases_disabled']['all'])))
queue.update(execution_authorized=False,caps=dict(caps,wall_seconds=queue_cap),attempt_registry=attempt_row,resource_admission=resource_row,
             fit_releases=[r['candidate'] for r in fit_candidates],closure_release=close_row)
queue['custody_inputs']=unique([*queue['custody_inputs'],*attempt['prior_attempt_artifacts'],attempt_row,evidence_row,resource_row,correction,*queue['fit_releases'],close_row])
queue_row=write(HERE/'ALL_FITS_RELEASE_DISABLED.json',queue)
proposal=write(HERE/'PLAN.json',dict(schema='amazon_polynormer_measured_disabled_full15_resource_plan_v1',UTC=now,
    status='PREPARED_FOR_ROOT_INSPECTION',execution_authorized=False,source=source,original_registry=plan['original_registry'],
    original_master_source_claim=plan['original_master_source_claim'],qualification_freeze=desc(QUALIFIER/'FREEZE.json'),
    resource_candidate=resource_row,forecast_evidence=evidence_row,attempt_registry_disabled=attempt_row,
    metadata_correction_note=correction,fit_candidates=fit_candidates,closure_release_disabled=close_row,all_release_disabled=queue_row,
    forecast_hours=future/3600,forecast_with50percent_margin_hours=1.5*future/3600,
    common_fit_caps=caps,prospective_queue_budget_hours=192,proposed_queue_wall_cap_hours=queue_cap/3600,
    prospective_storage_budget_bytes=storage['candidate_operational_storage_budget_bytes'],
    incremental_required_free_space_bytes=storage['incremental_required_free_space_bytes'],
    quota_and_physical_allocation_expiry_unknown=True,no_observed_actual_capacity_blocker=True,
    sequential_no_overlapping_fits=True,stop_on_first_physical_failure=True,automatic_retry_or_replacement=False,
    pending_exact_admitted_resource_descriptor_rebinding_before_runtime_release=True,
    all_original_registered_fit_output_release_claim_paths_preserved=True,predictive_fit_TEST_control_or_scoring_launches=0))
validation=write(HERE/'VALIDATION.json',dict(schema='amazon_polynormer_measured_resource_candidate_validation_v1',UTC=now,
    status='passed',source_payloads_verified=36,physical_fit_candidates=15,family_records=9,
    all_fit_queue_closure_resource_and_attempt_execution_authorizations_false=True,
    every_prior_artifact_preserved=True,prior_artifact_count=len(attempt['prior_attempt_artifacts']),
    selected_stage_final_replays_paid=True,source_recipe_family_selection_unchanged=True,
    largest_per_fit_with50percent_margin_seconds=wall['max_single_fit_with50percent_margin_seconds'],
    common_fit_cap_seconds=caps['wall_seconds'],filesystem_capacity_observed_sufficient=True,
    quota_and_expiry_not_certified=True,arrays_checkpoints_runtime_binaries_not_opened=True,
    numerical_GPU_training_TEST_control_scoring_or_registration_launches=0,execution_authorized=False))
manifest=write(HERE/'MANIFEST.json',dict(schema='amazon_polynormer_measured_resource_candidate_manifest_v1',UTC=now,
    execution_authorized=False,payload=[dict(relative=p.relative_to(HERE).as_posix(),descriptor=desc(p))
    for p in sorted(HERE.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.json','SEAL.json')]))
seal=write(HERE/'SEAL.json',dict(schema='amazon_polynormer_measured_resource_candidate_seal_v1',UTC=now,
    manifest=manifest,execution_authorized=False))
print(json.dumps(dict(plan=proposal,resource_candidate=resource_row,forecast_evidence=evidence_row,
                     attempt_registry=attempt_row,validation=validation,manifest=manifest,seal=seal,
                     forecast_hours=future/3600,forecast_with50percent_margin_hours=1.5*future/3600,
                     wall=wall,storage=storage,measured_fsync_retirement_charges=fsync),indent=2))
