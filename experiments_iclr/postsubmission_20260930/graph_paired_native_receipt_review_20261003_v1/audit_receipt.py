"""Stdlib audit of supplied TRAIN-only native receipts; no native computation."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

audit = Path(__file__).resolve().parent
phase = audit.parent
native = phase / 'graph_paired_native_qualification_cpu_gpu_root_v1'
source = phase / 'graph_paired_native_qualification_preparation_v3'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
receipt_path = native / 'QUALIFICATION.json'
r = json.loads(receipt_path.read_bytes())
checks = []


def check(name, condition):
    assert condition, name
    checks.append({'name': name, 'passed': True})


custody = json.loads((native / 'ORIGINAL_NATIVE_OUTPUT_CUSTODY.json').read_bytes())
output_hashes = {'QUALIFICATION.json': sha(receipt_path),
                 'ORIGINAL_NATIVE_OUTPUT_CUSTODY.json': sha(native / 'ORIGINAL_NATIVE_OUTPUT_CUSTODY.json')}
for row in custody['files']:
    local = native / row['local_copy']
    check('exact_native_output_custody:' + row['name'], sha(local) == row['sha256']
          and local.stat().st_size == row['bytes']
          and '/graph_paired_native_qualification_preparation_v3/runs/root01/' in row['remote_path'])
    output_hashes[row['local_copy']] = sha(local)
check('parsed_original_equals_supplied_receipt',
      json.loads((native / 'ORIGINAL_NATIVE_QUALIFICATION.json').read_bytes()) == r)

manifest_bytes = (source / 'MANIFEST.json').read_bytes()
manifest = json.loads(manifest_bytes)
bound_bytes = (source / 'BOUND_INPUTS.json').read_bytes()
bound = json.loads(bound_bytes)
check('reviewed_source_manifest_correspondence', hashlib.sha256(manifest_bytes).hexdigest() ==
      'fba0de7e94eb3974fd5df40cd996bdccb816a0748c6834925c3eca3881a743c7')
payload = {row['path']: row for row in manifest['payload']}
check('reviewed_script_descriptor', payload['prototype/qualify_squirrel17.py']['sha256'] ==
      '3d38e56dd2b87fea3507e9b96ae5987c72283582321d123f58da2ee3df3febb1')
check('reviewed_bound_inputs_metadata_hash', hashlib.sha256(bound_bytes).hexdigest() ==
      payload['BOUND_INPUTS.json']['sha256'])
expected = {row['path']: row['sha256'] for row in bound['original_records']}
before, after = r['original_preservation_before'], r['original_preservation_after']
check('all_151_bound_original_preservation_records', len(before) == len(after) == len(expected) == 151
      and before == after and all(row['preserved'] is True for row in before)
      and {row['path']: row['sha256'] for row in before} == expected)
check('numeric_binding_scope_no_extra_labels', len([row for row in bound['original_records']
      if Path(row['path']).suffix in ('.npy','.npz','.pt','.pth')]) == 7
      and set(bound['context']['source_labels']) == {'train'})
check('qualified_TRAIN_only_no_training_or_merit',
      r['schema'] == 'Squirrel17-native-paired-source-qualification-v3'
      and r['status'] == 'native_paired_source_qualified' and r['native_invocation_performed']
      and not r['validation_or_test_labels_loaded'] and not r['validation_or_test_scores_computed']
      and not r['training_or_continuation_performed'] and r['optimizer_steps'] == 0
      and not r['checkpoint_or_RNG_files_written'] and not r['scientific_merit_assessed'])
check('complete_output_and_slice_contract', r['output_contract']['shape'] == [2223,5]
      and r['output_contract']['canonical_node_order'] and r['output_contract']['exact_target_coverage']
      and r['output_contract']['TRAIN_count'] == 444
      and r['output_contract']['model_logits_gradients_dtype'] == 'FP32'
      and r['output_contract']['factor_binding']['names'] == ['stem.S','head.R']
      and r['output_contract']['factor_binding']['dimensions'] == 512
      and r['output_contract']['factor_binding']['identity'])
check('initial_native_K1_K4_equality', r['identity_audit']['passed'] and
      len(r['identity_audit']['comparisons']) == 5 and all(row['passed'] and row['finite']
      and row['max_absolute'] == row['max_relative'] == 0.0 for row in r['identity_audit']['comparisons']))

ad = r['active_precision_AD']
check('active_precision_source_and_counts', ad['passed'] and ad['source_labels'] == 'train_only'
      and ad['qualification_measurement'] == 'FP32_per_example_CE_FP64_mean_v1'
      and ad['original_training_gradient_unchanged'] and ad['epsilons'] == [0.001,0.0003]
      and (ad['vjp_forwards'],ad['vjp_calls'],ad['jvp_calls'],ad['finite_difference_forward_calls']) == (1,4,3,12))
ad_conditions = [row['dual_relative_error'] <= 2e-4 and any(
      x['logits_relative_error'] <= 0.05 and x['ce_directional_error'] <= 0.05
      for x in row['finite_differences']) for row in ad['records']]
check('independent_AD_acceptance_reconstruction', len(ad_conditions) == 3 and all(ad_conditions))
check('both_fixed_epsilons_actually_pass_each_direction', all(
      len(row['finite_differences']) == 2 and all(x['logits_relative_error'] <= 0.05
      and x['ce_directional_error'] <= 0.05 for x in row['finite_differences']) for row in ad['records']))

topology = r['topology_control']
vector = topology['permutation']
check('saved_full_universe_permutation_and_hash', len(vector) == 2223
      and sorted(vector) == list(range(2223)) and topology['seed'] == 80017
      and topology['genuine_Pi_S_PiT'] and topology['feature_label_order_unchanged']
      and hashlib.sha256(struct.pack('=' + str(len(vector)) + 'q', *vector)).hexdigest() ==
      topology['permutation_sha256'])

p = r['paired']
arms = ['common_only','train_remasked','full_node','full_node_permuted']
g_norm = math.sqrt(p['common_gradient_squared_norm'])
alpha = p['accepted_alpha']
check('four_arms_first_shared_step_no_fallback', p['arm_names'] == arms and p['status'] == 'joint_accepted'
      and not p['fallback_enabled'] and len(p['attempts']) == 1 and p['attempts'][0]['attempt'] == 0
      and alpha == p['alpha0'] == p['attempts'][0]['alpha']
      and p['attempts'][0]['joint_accepted'])
check('shared_radius_and_Armijo_formula', math.isclose(p['direction_norm_bound'], math.sqrt(1.25)*g_norm,
      rel_tol=1e-12) and math.isclose(alpha, 0.01*math.sqrt(512)/(math.sqrt(1.25)*g_norm), rel_tol=1e-12)
      and math.isclose(p['attempts'][0]['armijo_bound'], p['baseline_train_ce']-
      1e-4*alpha*p['common_gradient_squared_norm'], rel_tol=1e-12))
check('graph_partition_certificates', max(p['original_partition_relative_error'],
      p['permuted_partition_relative_error']) <= 2e-5)
geometry_summary = {}
for arm in arms:
    geometry = p['arms'][arm]
    trial = p['attempts'][0]['arms'][arm]
    check('geometry_certificates:' + arm, geometry['geometry_status'] == 'ready'
          and max(geometry['band_gradient_sum_relative_error'], geometry['descent_relative_error'],
                  geometry['tangent_mean_relative_error']) <= 2e-5
          and geometry['max_direction_norm'] <= p['direction_norm_bound']*(1+2e-5)
          and alpha*geometry['max_direction_norm']/math.sqrt(512) <= 0.01*(1+2e-5))
    check('Frobenius_cap:' + arm, math.isclose(geometry['tangent_frobenius_norm'],
          0.0 if arm == 'common_only' else 0.5*g_norm, rel_tol=2e-5, abs_tol=1e-20))
    check('TRAIN_member_pooled_acceptance:' + arm, trial['accepted'] and trial['finite']
          and trial['quality_accepted'] and trial['functional_accepted']
          and trial['failure_reasons'] == trial['closure_errors'] == []
          and len(trial['route_train_ce']) == 4
          and max(trial['route_train_ce']+[trial['pooled_train_ce']]) <= p['attempts'][0]['armijo_bound'])
    source_min = min(geometry['source_tangent_pair_rms'])
    actual_min = min(trial['actual_centered_pair_rms'])/alpha
    check('separation_guard_semantics:' + arm,
          (source_min == actual_min == 0.0) if arm == 'common_only' else
          (source_min > p['functional_rms_threshold'] and actual_min > p['functional_rms_threshold']))
    gram = geometry['full_output_tangent_gram']
    check('finite_symmetric_Gram:' + arm, geometry['full_output_tangent_finite']
          and len(gram) == 4 and all(len(row) == 4 for row in gram)
          and all(math.isfinite(x) for row in gram for x in row)
          and all(gram[i][j] == gram[j][i] for i in range(4) for j in range(4))
          and all(gram[i][i] >= 0 for i in range(4)))
    signed = trial['signed_graph_contrast']
    check('diagnostic_scope:' + arm, signed['diagnostic_only']
          and not signed['prediction_is_predictive_success']
          and signed['support_matched'] == (arm in ('full_node','full_node_permuted')))
    if arm != 'common_only':
        check('signed_arithmetic:' + arm, signed['status'] == 'available' and signed['finite']
              and signed['alpha'] == alpha and math.isclose(signed['first_order_prediction'],
              alpha*geometry['signed_graph_contrast_first_order_slope'], rel_tol=1e-12)
              and math.isclose(signed['signed_value'], sum(signed['member_signed_terms']), rel_tol=1e-12)
              and math.isclose(signed['finite_to_first_order_ratio'], signed['signed_value']/
                               signed['first_order_prediction'], rel_tol=1e-12))
    equality = r['installation'][arm]
    check('installed_warm_and_exact_closure_equality:' + arm, all(value['passed'] and value['finite']
          and value['max_absolute'] == value['max_relative'] == 0.0
          and value['atol'] == 1e-6 and value['rtol'] == 1e-5 for value in equality.values()))
    geometry_summary[arm] = {'tangent_Frobenius_norm': geometry['tangent_frobenius_norm'],
        'max_reported_relative_factor_step': alpha*geometry['max_direction_norm']/math.sqrt(512),
        'min_source_TRAIN_pair_RMS': source_min, 'min_finite_TRAIN_pair_RMS_per_alpha': actual_min}
check('declared_support_and_topology', p['arms']['train_remasked']['cotangent_support'] == 'train_remasked'
      and p['arms']['train_remasked']['topology'] == p['arms']['full_node']['topology'] == 'original'
      and p['arms']['full_node']['cotangent_support'] == p['arms']['full_node_permuted']['cotangent_support'] == 'full_node'
      and p['arms']['full_node_permuted']['topology'] == 'permuted')
gram_dist = lambda a,b: math.sqrt(sum((x-y)**2 for row_a,row_b in zip(a,b) for x,y in zip(row_a,row_b)))
support_gram_delta = gram_dist(p['arms']['full_node']['full_output_tangent_gram'],
                              p['arms']['train_remasked']['full_output_tangent_gram'])
topology_gram_delta = gram_dist(p['arms']['full_node']['full_output_tangent_gram'],
                               p['arms']['full_node_permuted']['full_output_tangent_gram'])
check('nonzero_output_Gram_support_and_topology_deltas', support_gram_delta > 0 and topology_gram_delta > 0)

check('paired_primitive_forward_counters',
      (p['vjp_forwards'],p['vjp_calls'],p['jvp_calls'],p['graph_sparse_products'],
       p['line_search_forward_calls'],p['candidate_trial_forward_calls'],p['same_alpha_common_forward_calls']) ==
      (1,13,12,6,16,16,0) and p['trial_forward_calls_by_arm'] == {arm:4 for arm in arms})
paired_forwards = p['vjp_forwards']+p['jvp_calls']+p['line_search_forward_calls']
expected_observed = 1+(ad['vjp_forwards']+ad['jvp_calls']+ad['finite_difference_forward_calls'])+paired_forwards+16
check('actual_observed_counter_reconciliation', expected_observed == 62
      and r['closure_forward_calls'] == {'calls_started':62,'calls_completed':62,'calls_failed':0})
check('outside_container_and_member_counts', r['outside_closure_forward_calls'] ==
      {'containers_started':11,'containers_completed':11,'scheduled_member_forwards':38})
check('no_resource_errors_and_preserved_donor', r['observed_resource_failures'] == []
      and r['warm_donor_state_unchanged'])
costs = [json.loads(line) for line in (native/'native_cost_trace.txt').read_bytes().splitlines() if line.strip()]
events = [json.loads(line) for line in (native/'native_interface_trace.txt').read_bytes().splitlines() if line.strip()]
check('exact_cost_trace_correspondence', costs == r['operation_costs'] and len(costs) == 36
      and all(row['status'] == 'completed' and math.isfinite(row['seconds']) and row['seconds'] >= 0 for row in costs))
check('closure_trace_exact_1_to_62', [row['receipt']['call'] for row in events
      if row['event']=='closure_forward_started'] == list(range(1,63))
      and not any(row['event']=='closure_forward_failed' for row in events))
started = [row['receipt'] for row in events if row['event']=='model_forward_started']
completed = [row['receipt'] for row in events if row['event']=='model_forward_completed']
check('outside_trace_batch_correspondence', len(started) == 9 and started == completed
      and sum(row['scheduled_members'] for row in started) == 38
      and started[0]['name']=='native_K1_K4_identity_audit' and started[0]['scheduled_members']==6)
installation_events = {row['receipt']['arm']:row['receipt']['receipt'] for row in events
                       if row['event']=='arm_installation_equality'}
check('all_four_installation_traces_equal_final_receipt', installation_events == r['installation'])
check('all_stdlib_cost_stages_completed', all(row['status']=='completed' for row in r['stdlib_operation_costs']))
check('resource_preflight_passed', r['resource_preflight']['status']=='resource_preflight_passed'
      and r['resource_preflight']['reasons']==[] and r['resource_preflight']['free_bytes'] >=
      r['resource_preflight']['required_free_bytes'])
allocated = max(row['peak_allocated_bytes'] for row in costs)
reserved = max(row['peak_reserved_bytes'] for row in costs)
check('recorded_peaks_below_preflight_free_memory', 0 <= allocated <= reserved < r['resource_preflight']['free_bytes'])
summary = {'accepted_shared_alpha':alpha, 'shared_trials':len(p['attempts']),
           'paired_closure_forwards':paired_forwards, 'observed_closure_forwards':expected_observed,
           'outside_scheduled_member_predictors':38, 'successful_member_predictor_total':expected_observed+38,
           'geometry':geometry_summary, 'output_tangent_Gram_support_delta_Frobenius':support_gram_delta,
           'output_tangent_Gram_topology_delta_Frobenius':topology_gram_delta,
           'whole_process_wall_seconds':r['whole_process_wall_seconds'],
           'paired_operation_seconds':next(row['seconds'] for row in costs if row['operation']=='sealed_actual_four_arm_shared_alpha'),
           'max_recorded_operation_peak_allocated_bytes':allocated,
           'max_recorded_operation_peak_reserved_bytes':reserved,
           'max_recorded_allocated_GiB':allocated/2**30, 'max_recorded_reserved_GiB':reserved/2**30,
           'process_maxrss_native_units':r['process_maxrss_native_units']}
result = {'schema':'independent-paired-native-receipt-audit-v1', 'checks':checks, 'summary':summary,
          'output_hashes':output_hashes, 'reviewed_source_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
          'reviewed_source_script_descriptor_sha256':payload['prototype/qualify_squirrel17.py']['sha256'],
          'original_bytes_reopened':False, 'native_GPU_remote_execution':False,
          'validation_test_data_or_scores_accessed':False, 'predictive_gain_assessed':False,
          'audit_scope':'Supplied native output/trace bytes plus reviewed source manifest and bound-input metadata only.'}
(audit/'AUDIT_CHECKS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'passed':True,'receipt_checks':len(checks),'native_construction_correspondence':True,
                  'predictive_gain_assessed':False,'summary':summary}))
