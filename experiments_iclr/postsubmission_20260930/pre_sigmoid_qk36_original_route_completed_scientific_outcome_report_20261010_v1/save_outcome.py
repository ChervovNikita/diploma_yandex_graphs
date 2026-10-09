"""Summarize saved compact statistics and metadata; never import scientific code."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
EVIDENCE = HERE / 'evidence'
COMPACT = EVIDENCE / 'pre_sigmoid_qk36_original_route_assembly_activation_root_20261010_v1/server_only_analysis_execution/compact'
SUPERVISION = EVIDENCE / 'pre_sigmoid_qk36_original_route_assembly_supervision_root_20261010_v1'
PREP_METADATA = PHASE / 'pre_sigmoid_qk36_original_route_phase1_activation_preparation_root_20261010_v1/compact_metadata'


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path.relative_to(PHASE)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def write(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(data)


def checked(path, expected):
    path = Path(path)
    actual = binding(path)
    assert all(actual[k] == expected[k] for k in ('bytes', 'sha256'))
    return read(path) if path.suffix == '.json' else path.read_bytes()


def compare_saved_gate(row, rule):
    metrics = row['metrics']
    required = ('served_accuracy','served_nll','mean_member_accuracy','worst_member_accuracy','mean_member_nll','worst_member_nll')
    available = all(metrics[name]['available_seeds'] == 3 for name in required)
    checks = dict(
        positive_every_paired_seed=available and all(value > 0 for value in metrics['served_accuracy']['values']),
        mean_accuracy_gain_pp_at_least=available and metrics['served_accuracy']['mean'] >= rule['mean_accuracy_gain_pp_at_least'],
        mean_pool_NLL_delta_at_most=available and metrics['served_nll']['mean'] <= rule['mean_pool_NLL_delta_at_most'],
        mean_over_seed_mean_member_accuracy_delta_pp_at_least=available and metrics['mean_member_accuracy']['mean'] >= rule['mean_over_seed_mean_member_accuracy_delta_pp_at_least'],
        mean_over_seed_worst_member_accuracy_delta_pp_at_least=available and metrics['worst_member_accuracy']['mean'] >= rule['mean_over_seed_worst_member_accuracy_delta_pp_at_least'],
        mean_over_seed_mean_member_NLL_delta_at_most=available and metrics['mean_member_nll']['mean'] <= rule['mean_over_seed_mean_member_NLL_delta_at_most'],
        mean_over_seed_worst_member_NLL_delta_at_most=available and metrics['worst_member_nll']['mean'] <= rule['mean_over_seed_worst_member_NLL_delta_at_most'])
    return dict(available=available, checks=checks, passed=available and all(checks.values()))


def main():
    inventory = read(HERE / 'HARVEST_INVENTORY.json')
    for row in inventory['files']:
        checked(EVIDENCE / row['path'], row)
    pins = read(PHASE / 'pre_sigmoid_qk36_original_route_selected_collection_source_20261010_v2/SOURCE_BINDINGS.json')
    protocol_binding = pins['reuse']['pilot_protocol']
    protocol = checked(PHASE / protocol_binding['path'], protocol_binding)
    rule = protocol['prospective_decisions']['quality_and_member_gate']
    decision = read(COMPACT / 'PILOT_DECISION.json')
    assert decision['original_fixed_rule'] == rule
    contrasts = read(COMPACT / 'CONTRASTS.json')['ordered_comparisons']
    collection = read(COMPACT / 'COLLECTION.json')
    cells = read(COMPACT / 'PER_CELL.json')['cells']
    paired = read(COMPACT / 'PAIRED_ERRORS.json')['paired']
    seed_summary = read(COMPACT / 'PAIRED_SEED_SUMMARY.json')['comparisons']
    limits = read(COMPACT / 'PER_CELL.json')['limits']
    labels = []
    for kind in ('be_init','single','independent4'):
        labels.extend(kind+'/pre_sigmoid_split-'+kind+'/'+op for op in ('native_tied','active_reversible_exp','full_qk'))
    for op in ('pre_sigmoid_split','native_tied','active_reversible_exp','full_qk'):
        labels.extend('be_init/'+op+'-'+kind+'/'+op for kind in ('single','independent4'))
    assert [row['comparison'] for row in contrasts] == labels and len(labels) == 17
    by_label = {row['comparison']:row for row in contrasts}
    checked_gates = []
    for gate in decision['operator_co_primary'] + decision['same_operator_sharing_comparisons']:
        actual = compare_saved_gate(by_label[gate['comparison']], rule)
        assert actual == {key:gate[key] for key in actual}
        checked_gates.append(dict(gate, failed_checks=[key for key,value in gate['checks'].items() if not value], reported_metrics=by_label[gate['comparison']]['metrics']))
    assert collection['analysis_complete'] and decision['whole36_readout_available']
    assert len(cells) == len(collection['cells']) == 36
    assert len({row['cell'] for row in cells}) == 36
    assert all(row['collection_status'] == row['family_status'] == 'complete' for row in cells)
    assert sum(row['members'] for row in cells) == 108
    assert len(paired) == 51 and all(row['available'] for row in paired)
    assert len(seed_summary) == 85
    assert all(row['full_population']['nodes'] == 5274 for row in cells)
    assert all(stat['available_seeds'] == 3 for row in contrasts for stat in row['metrics'].values())

    supervision = read(SUPERVISION / 'SUPERVISION_RESULT.json')
    terminal = read(SUPERVISION / 'CHILD_TERMINAL_OBSERVATION.json')
    exit_receipt = read(SUPERVISION / 'logs/qk36_numpy_assembly.EXIT.json')
    assert supervision['complete'] and supervision['failure'] is None
    assert supervision['child_exit_receipt'] == exit_receipt
    assert terminal['actual_exit_code'] == exit_receipt['exit_code'] == 0
    assert terminal['actual_direct_wait'] and exit_receipt['terminal_wait_observed']
    assert terminal['child_absent'] and terminal['child_no_CUDA_rows']
    assert exit_receipt['reason'] is None and exit_receipt['signal_refusal'] is None and exit_receipt['signals_sent'] == []
    assert not exit_receipt['retry'] and exit_receipt['attempts'] == 1
    for name in ('stdout','stderr'):
        assert binding(SUPERVISION / ('logs/qk36_numpy_assembly.'+name+'.log'))['sha256'] == exit_receipt[name+'_sha256']

    original_path = PHASE / 'pre_sigmoid_qk36_original_route_phase1_activation_root_20261010_v1/native_baselines_execution/allocation_a100/compact/ORIGINAL_COSTS.json'
    allocation = read(original_path)['training_lanes']['allocation_a100']['cells']
    allocation_by_key = {row['spec']['key']:row for row in allocation}
    allocation_cost_metadata_path = PREP_METADATA / 'pre_sigmoid_qk36_original_route_phase1_activation_preparation_root_20261010_v1/actual_metadata/allocation_a100_ORIGINAL_COST_METADATA.json'
    allocation_cost_metadata = read(allocation_cost_metadata_path)
    allocation_meta_by_key = {row['spec']['key']:row for row in allocation_cost_metadata['original_worker_costs_and_actual_exits']}
    training = []
    selected = []
    for row in collection['cells']:
        if row['original_training_route'] == 'allocation_a100':
            original = allocation_by_key[row['scientific_key']]
            meta = allocation_meta_by_key[row['scientific_key']]
            assert meta['worker_cost_binding'] == row['original_cost']
            assert meta['actual_exit_binding'] == row['original_exit']
            assert meta['original_worker_cost'] == original['worker_cost'] and meta['actual_exit'] == original['exit']
            cost, endpoint, original_exit = original['worker_cost'], original['endpoint'], original['exit']
            provenance = dict(parsed_metadata_sources=[binding(original_path),binding(allocation_cost_metadata_path)],exact_original_bindings_recorded=True,original_file_bytes_reopened=False)
        else:
            cost = checked(PREP_METADATA / row['original_cost']['path'], row['original_cost'])
            endpoint = checked(PREP_METADATA / row['original_completion']['path'], row['original_completion'])
            original_exit = checked(PREP_METADATA / row['original_exit']['path'], row['original_exit'])
            provenance = dict(exact_local_original_cost_binding=binding(PREP_METADATA / row['original_cost']['path']),original_file_bytes_reopened=True)
        assert cost['success'] and cost['error'] is None and original_exit['exit_code'] == 0
        assert endpoint['complete'] and endpoint['epochs'] == 1100 and not endpoint['TEST_scoring']
        training.append(dict(scientific_key=row['scientific_key'],route=row['original_training_route'],original_cost=row['original_cost'],original_completion=row['original_completion'],original_exit=row['original_exit'],worker_cost=cost,actual_child_exit=original_exit,tracked_F_work=endpoint['operator_work'],provenance=provenance))
        per_cell = next(value for value in cells if value['cell'] == row['cell'])
        meta = per_cell['selected_metadata']
        selected.append(dict(cell=row['cell'],condition=row['condition'],seed=row['seed'],members=row['members'],selected_epochs=meta['selected_epochs'],selected_member_global_modes=meta['selected_member_modes'],selection_rule=meta['selector'],original_route=row['original_training_route'],trainable_parameters=meta['trainable_parameters'],stored_model_tensor_bytes=meta['stored_model_tensor_bytes'],selected_checkpoint=row['selected_checkpoint'],own_selected_checkpoints=row['own_selected_checkpoints'],full_population=per_cell['full_population'],serving_cost={key:row[key] for key in ('inclusive_cell_seconds','construction_seconds','checkpoint_load_and_mode_restore_seconds','forward_and_CPU_transfer_seconds','prediction_serialization_seconds','peak_CUDA_allocated_bytes','peak_CUDA_reserved_bytes')}))

    route_costs = []
    release = read(PHASE / 'pre_sigmoid_qk36_original_route_assembly_activation_root_20261010_v1/RELEASE.json')
    freeze = checked(PHASE / release['global_native_baseline_freeze']['path'], release['global_native_baseline_freeze'])
    for route in ('allocation_a100','gpu77_a998','gpu77_8ced'):
        for phase_name, bound in (('native_baselines',freeze['routes'][route]['phase1_cost']),('candidates',release['phase2_routes'][route]['phase2_cost'])):
            cost = checked(PHASE / bound['path'], bound)
            assert cost['TRAIN_updates'] == cost['backward_calls'] == cost['Adam_steps'] == 0
            route_costs.append(dict(route=route,collection_phase=phase_name,binding=bound,cost=cost))
    assert sum(row['cost']['completed_member_forwards'] for row in route_costs) == 108
    stream_path = PHASE / 'pre_sigmoid_qk36_original_route_phase2_closure_and_assembly_preparation_root_20261010_v1/ACTUAL_RAW_STREAM_BYTES_COST_AND_FAILURES.json'
    stream = read(stream_path)
    assembly_cost = read(COMPACT / 'COST.json')
    assert assembly_cost['TRAIN_updates'] == assembly_cost['backward_calls'] == assembly_cost['member_forwards'] == 0
    tracked_names = ('Adam_steps','completed_updates','member_view_backwards','member_view_forwards','update_attempts')
    total_work = {key:sum(row['tracked_F_work'][key] for row in training) for key in tracked_names}
    cost_report = dict(schema='qk36-complete-outcome-cost-summary-v1',scope='Saved costs only. Child wall sums represent additive work, not concurrent elapsed campaign time. Per-route hardware differences and overlapping measured scopes are retained.',all72_scratch_bodies_charged=True,native_bodies=72,training_groups=36,tracked_F_work=total_work,original_training=training,original_selected_storage_and_serving=selected,phase1_and_phase2_costs=route_costs,server_only_numpy_assembly=assembly_cost,actual_assembly_exit=exit_receipt,raw_archive_stream=dict(binding=binding(stream_path),report=stream),detached_assembly_parent_OS_exit=supervision['detached_parent_OS_exit'],detached_assembly_parent_direct_wait=supervision['detached_parent_direct_wait_observed'],unknowns_retained_as_null=['Complete raw-stream wall and CPU timing','Separated transport costs','Detached assembly parent OS exit/direct wait','Assembly outer owner terminal and compact mirroring cost'])
    write(HERE / 'COST_SUMMARY.json', cost_report)

    selected_summary = []
    for row in selected:
        population = row['full_population']
        selected_summary.append({**{key:row[key] for key in ('cell','condition','seed','members','selected_epochs','selected_member_global_modes','selection_rule','original_route','trainable_parameters','stored_model_tensor_bytes')},'served_accuracy':population['served_accuracy'],'served_nll':population['served_nll'],'served_brier':population['served_brier'],'exact_pooled_correctcount':population['counts']['pool_correct'],'actual_members':population['members'],'mean_member_accuracy':population['mean_member_accuracy'],'worst_member_accuracy':population['worst_member_accuracy'],'mean_member_nll':population['mean_member_nll'],'worst_member_nll':population['worst_member_nll']})
    diagnostic_names = ('repairs','harms','net_repairs','coverage_flows.gained','coverage_flows.lost','coverage_flows.net_gained','pool_rescue_flows.appeared','pool_rescue_flows.cleared','pool_harm_flows.appeared','pool_harm_flows.cleared','common_wrong_competitor_flows.cleared','common_wrong_competitor_flows.appeared')
    diagnostic_labels = [row['comparison'] for row in checked_gates]
    diagnostics = [dict(comparison=row['comparison'],cohort=row['cohort'],reported_statistics={key:row['statistics'][key] for key in diagnostic_names}) for row in seed_summary if row['comparison'] in diagnostic_labels and row['cohort']=='full_population']
    report = dict(schema='qk36-completed-scientific-outcome-v1',UTC=datetime.now(timezone.utc).isoformat(),conclusion='Reject the fixed shared be_init pre-sigmoid Q/K operator utility hypothesis under the frozen pilot criteria. Both co-primary gates fail accuracy criteria. Same-operator sharing quality advantage is unsupported. Full-Q/K comparisons remain descriptive; no superiority claim is cleared.',analysis_action='Read and summarize saved compact statistics; compare reported values with frozen thresholds. No scientific modules or raw arrays loaded, no inference, fitting, reselection, calibration or new intervals.',frozen_protocol=protocol_binding,frozen_quality_member_rule=rule,coverage=dict(banks_required=36,banks_complete=36,selected_member_calls_attempted=108,selected_member_calls_completed=108,cohort_archives=9,ordered_contrasts=17,paired_contrast_seed_rows=51,cohort_seed_summary_rows=85,development_nodes=5274,paired_seeds=[6101,6203,6307],missing_bank_slots=[],missing_seed_slots=[],retained_analysis_failures=[],whole36_readout_available=decision['whole36_readout_available']),sign_convention='Candidate minus reference; positive accuracy differences are better, negative NLL and Brier differences are better. Accuracy contrasts are percentage points; NLL contrasts are nats.',operator_co_primary=checked_gates[:2],same_operator_sharing=checked_gates[2:],operator_utility_rejected=decision['operator_utility_rejected_if_any_co_primary_fails'],sharing_quality_advantage_supported=decision['sharing_quality_advantage_supported'],mandatory_same_family_full_qk_comparisons=decision['full_qk_same_family_comparisons'],full_qk_interpretation='Shared split mean accuracy is below full_qk. Single split has a small mixed-seed gain. Independent split has positive accuracy in every seed and a mean gain above 0.2 pp, with worse pooled NLL and member NLL. These outcomes do not clear superiority and cannot rescue failed co-primary gates.',all17_ordered_contrasts=contrasts,all36_selected_banks=selected_summary,key_paired_full_population_diagnostics=diagnostics,cost_summary=binding(HERE / 'COST_SUMMARY.json'),supervision=dict(complete=supervision['complete'],failure=supervision['failure'],actual_child_exit_code=terminal['actual_exit_code'],actual_direct_wait=terminal['actual_direct_wait'],child_absent=terminal['child_absent'],child_no_CUDA_rows=terminal['child_no_CUDA_rows'],owned_GPU_bytes=exit_receipt['max_sampled_owned_GPU_bytes'],actual_active_and_cleanup_seconds=exit_receipt['elapsed_seconds'],signals_sent=exit_receipt['signals_sent'],retry=exit_receipt['retry'],detached_parent_OS_exit=supervision['detached_parent_OS_exit'],detached_parent_direct_wait_observed=supervision['detached_parent_direct_wait_observed'],terminal_binding=binding(SUPERVISION / 'CHILD_TERMINAL_OBSERVATION.json'),result_binding=binding(SUPERVISION / 'SUPERVISION_RESULT.json')),scientific_limits=limits,additional_limits=['All 5274 development nodes were consumed for state selection and are reused for readout. These results are exploratory selected-development evidence.','Three optimizer-seed blocks on one graph. Reported df2 intervals are descriptive; nodes and members are not independent replicates.','Every family/seed native baseline freezes its own cohorts; cross-family sharing uses shared be_init/native_tied cohorts.','Independent models retain their own selected local/global modes; pooling uses actual member softmax means without arbitrary cross-bank member-index pairing.','No TEST evidence, unused confirmation, novelty clearance, mechanism causality, or graph-population generalization.','Lower costs, learned scales and raw asymmetry cannot rescue a failed co-primary.'],harvest_inventory=binding(HERE / 'HARVEST_INVENTORY.json'),full_saved_decision=binding(COMPACT / 'PILOT_DECISION.json'),publication_custody=dict(remote_original_reports_retained=True,files_larger_than_2000000_bytes=[row for row in inventory['files'] if row['bytes']>2000000],large_diagnostics_not_split=True,encoded_transport_receipt_for_custody_only=True),new_fits=0,new_member_calls=0,raw_arrays_read_locally=False,checkpoints_read_locally=False,Git_mutated=False,shared_status_edited=False)
    write(HERE / 'SCIENTIFIC_OUTCOME.json', report)
    write(HERE / 'VERIFICATION.json',dict(schema='qk36-outcome-metadata-verification-v1',exact_harvest_file_hashes_verified=len(inventory['files']),protocol_hash_verified=protocol_binding['sha256'],ordered17_contrasts_verified=True,whole36_and_all51_paired_rows_verified=True,saved_four_primary_and_sharing_gate_checks_match_frozen_thresholds=True,original36_training_costs_and_complete_endpoints_bound=True,all108_completed_original_route_inference_calls_verified=True,assembly_EXIT_and_child_terminal_agree=True,assembly_stdout_stderr_hashes_verified=True,scientific_recomputation=False,raw_arrays_loaded=False))
    manifest_files = [path for path in HERE.rglob('*') if path.is_file() and path.name not in ('MANIFEST.json','MANIFEST.sha256','SEAL.json')]
    manifest = dict(schema='qk36-completed-outcome-manifest-v1',files=[binding(path) for path in sorted(manifest_files)],raw_payloads_included=False,publication_subset_selected_by_root=True)
    write(HERE / 'MANIFEST.json',manifest)
    digest = binding(HERE / 'MANIFEST.json')['sha256']
    with (HERE / 'MANIFEST.sha256').open('x') as stream_file:
        stream_file.write(digest+'\n')
    write(HERE / 'SEAL.json',dict(schema='qk36-completed-outcome-seal-v1',manifest=binding(HERE / 'MANIFEST.json'),scientific_outcome=binding(HERE / 'SCIENTIFIC_OUTCOME.json'),cost_summary=binding(HERE / 'COST_SUMMARY.json'),complete=True,scientific_analysis_rerun=False,source_sources_unchanged=True))
    print(json.dumps(dict(complete=True,outcome=binding(HERE/'SCIENTIFIC_OUTCOME.json'),cost_summary=binding(HERE/'COST_SUMMARY.json'),manifest=binding(HERE/'MANIFEST.json'),seal=binding(HERE/'SEAL.json'),tracked_F_work=total_work,co_primary_passed=[row['passed'] for row in checked_gates[:2]],sharing_passed=[row['passed'] for row in checked_gates[2:]]),indent=2))


if __name__ == '__main__':
    main()
