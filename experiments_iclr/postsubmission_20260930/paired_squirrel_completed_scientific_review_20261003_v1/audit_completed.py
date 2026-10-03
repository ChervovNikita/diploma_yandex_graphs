"""Stdlib audit of complete supplied development receipts; no model/data execution."""
import hashlib
import json
import math
from pathlib import Path
import statistics

review = Path(__file__).resolve().parent
phase = review.parent
fetched = phase / 'graph_paired_squirrel_completed_root_v1'
packet = fetched / 'original/graph_paired_squirrel_continuation_preparation_v2'
run = packet / 'runs/root01'
local_source = phase / 'graph_paired_squirrel_continuation_preparation_v2'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_bytes())
checks = []


def check(name, value):
    assert value, name
    checks.append(dict(name=name, passed=True))


custody = read(fetched/'SCP_FETCH_RECEIPT.json')
check('all17_fetched_originals_match_transfer_records', len(custody['files']) == 17 and all(
    row['exit_code'] == 0 and sha(packet/row['path']) == row['sha256']
    and (packet/row['path']).stat().st_size == row['bytes'] for row in custody['files']))
trace_custody = read(fetched/'TRACE_FETCH_RECEIPT.json')
check('all12_fetched_traces_match_transfer_records', len(trace_custody['files']) == 12 and all(
    row['exit_code'] == 0 and sha(packet/row['path']) == row['sha256']
    and (packet/row['path']).stat().st_size == row['bytes'] for row in trace_custody['files']))
s = read(run/'STAGE1.json')
f = read(packet/'FROZEN_STUDY.json')
m = read(packet/'MANIFEST.json')
seal = read(packet/'SEAL.json')
choice_path = phase/'graph_paired_staged_root_admission_v1/MODE_DONOR_FREEZE.json'
choice = read(choice_path)
release_path = phase/'graph_paired_staged_root_admission_v1/ROOT_EXECUTION_RELEASE_v2.json'
release = read(release_path)
manifest_hash = sha(packet/'MANIFEST.json')
check('reviewed_v2_source_and_frozen_metadata_correspondence', manifest_hash ==
      '3f2d8faa81039fd6e5ceb5fecf7ea4b46bf725b516dbb040827926a92ddde452'
      and seal['manifest_sha256'] == manifest_hash
      and sha(local_source/'prototype/continue_squirrel.py') ==
      '60fb7675318252ac43a00d847e5f43fa6f20521868cb6ca1610f75a9e39186ee'
      and (packet/'FROZEN_STUDY.json').read_bytes() == (local_source/'FROZEN_STUDY.json').read_bytes())
admission = s['immutable_pre_execution_admission']
check('root_mode_donors_release_match_before_execution_binding', admission['mode'] == choice['mode'] ==
      f['mode'] == 'staged_validation_gate' and admission['six_donors'] == choice['donors'] == f['all_six_donors']
      and admission['prepared_manifest_sha256'] == release['prepared_manifest_sha256'] == manifest_hash
      and admission['mode_donor_freeze_sha256'] == release['mode_donor_freeze_sha256'] == sha(choice_path)
      and admission['root_execution_release_sha256'] == sha(release_path)
      and choice['no_choice_changes_after_first_prospective_study_operation'] is True)
launch = read(packet/'launch_root01/START.json')
check('launch_matches_reviewed_source_route_and_release', launch['prepared_manifest_sha256'] == manifest_hash
      and launch['mode_donor_freeze_sha256'] == sha(choice_path)
      and launch['release_sha256'] == sha(release_path)
      and launch['route_uuid'] == f['expected_gpu_uuid']
      and launch['argv'][2].endswith('/graph_paired_squirrel_continuation_preparation_v2/prototype/continue_squirrel.py'))
expected_originals = {row['path']: row['sha256'] for row in f['original_records']}
check('all48_reported_originals_preserved', len(expected_originals) == len(s['originals_before']) == 48
      and s['originals_before'] == s['originals_after']
      and all(row['preserved'] for row in s['originals_before'])
      and {row['path']:row['sha256'] for row in s['originals_before']} == expected_originals)
expected_val = {c['validation_labels']['path']:c['validation_labels']['sha256'] for c in f['cells']}
check('all3_admitted_validation_metadata_preserved', len(s['validation_originals_after']) == 3
      and all(row['preserved'] for row in s['validation_originals_after'])
      and {row['path']:row['sha256'] for row in s['validation_originals_after']} == expected_val)
arms = tuple(f['arms'])
controls = ('train_remasked','common_only','full_node_permuted')
pairs = ((17,0),(29,1),(43,2))
check('complete_paired_three_blocks_twelve_selected_fits', s['status'] == 'development_comparison_complete'
      and [(b['seed'],b['source_split_index']) for b in s['blocks']] == list(pairs)
      and all(b['status'] == 'completed' and set(b['fits']) == set(arms)
              and all(v['status'] == 'selected' and v['attempted'] for v in b['fits'].values())
              for b in s['blocks']))
check('final_labels_Photo_old_cohort_closed_in_receipt', not s['Photo_launched']
      and not s['final_labels_loaded'] and not s['old_cohort_modified'] and s['final_labels_closed']
      and s['validation_blocks_loaded'] == 3
      and all(not b['final_labels_loaded'] and b['validation_labels_loaded'] for b in s['blocks']))
check('all3_native_AD_optimizer_geometry_and_resource_receipts_pass', all(
      b['optimizer_equivalence']['passed'] and b['identity']['passed'] and b['AD']['passed']
      and b['paired']['status'] == 'joint_accepted' and b['donor_unchanged']
      and b['observed_resource_failures'] == []
      and b['closure_calls'] == dict(calls_started=62,calls_completed=62,calls_failed=0)
      and b['binding']['dimensions'] == 512 for b in s['blocks']))
check('all12_fresh_warm_and_installed_outputs_pass', all(
      set(b['initializations']) == set(arms) and all(
      b['initializations'][a]['common_warm_equality']['passed']
      and b['initializations'][a]['installed_equality']['passed']
      and b['fits'][a]['initialization'] == b['initializations'][a] for a in arms)
      for b in s['blocks']))
check('twelve_original_SELECTION_copies_equal_STAGE1', all(
      read(run/f"seed{b['seed']}_split{b['source_split_index']}"/a/'SELECTION.json') == b['fits'][a]
      for b in s['blocks'] for a in arms))
check('native_configuration_selector_stopping_correspondence', all(
      v['selection']['configuration'] == 0 and v['selection']['members'] == 4
      and v['selection']['global_only'] is False and v['label_scope'] == ['train','validation']
      and v['selection']['update_cap'] == 1950 and v['selection']['patience'] == 250
      and v['selection']['selected_actual_update'] == 50+v['selection']['selected_continuation_epoch']
      and v['selection']['continuation_updates_completed'] == v['selection']['selected_continuation_epoch']+250
      and v['selection']['selection'] == 'pooled mean raw-logit validation NLL; earliest strict tie; epoch0 eligible'
      and not v['selection']['native_midpoint_saved']
      and v['selection']['continuation_updates_completed'] < 950
      for b in s['blocks'] for v in b['fits'].values()))
means = {a:sum(b['fits'][a]['selection']['primary_validation_nll'] for b in s['blocks'])/3 for a in arms}
deltas = {a:means['full_node']-means[a] for a in controls}
check('frozen_macro_means_deltas_and_gate_recomputed', means == s['development']['validation_macro_NLL']
      and deltas == s['development']['full_minus_control_NLL']
      and all(v < 0 for v in deltas.values()) and s['development']['Photo_trigger'] is True
      and s['development']['all12_arm_terminals'] and s['development']['all12_selected_fits'])
block_table = []
for b in s['blocks']:
    values = {a:b['fits'][a]['selection']['primary_validation_nll'] for a in arms}
    block_table.append(dict(seed=b['seed'], split=b['source_split_index'], validation_NLL=values,
        full_minus_control_NLL={a:values['full_node']-values[a] for a in controls},
        selected_epochs={a:b['fits'][a]['selection']['selected_continuation_epoch'] for a in arms},
        completed_updates={a:b['fits'][a]['selection']['continuation_updates_completed'] for a in arms}))
t95_df2 = math.sqrt(2*0.95**2/(1-0.95**2))
contrasts = {}
for a in controls:
    values = [b['full_minus_control_NLL'][a] for b in block_table]
    paired_mean, sd = statistics.mean(values), statistics.stdev(values)
    se = sd/math.sqrt(3)
    leave_one_out = [statistics.mean(values[:i]+values[i+1:]) for i in range(3)]
    contrasts[a] = dict(block_deltas=values, frozen_gate_mean_delta=deltas[a], paired_mean_delta=paired_mean,
        sample_SD=sd, IID_standard_error=se, nominal95_IID_paired_t_interval_df2=[paired_mean-t95_df2*se,paired_mean+t95_df2*se],
        nominal_interval_is_not_design_calibrated=True, full_node_lower_blocks=sum(x < 0 for x in values),
        leave_one_block_out_means=leave_one_out, relative_NLL_reduction_percent=-deltas[a]/means[a]*100,
        geometric_true_class_probability_ratio=math.exp(-deltas[a]))
leave_one_out_gates = {str(pairs[i][0]):all(contrasts[a]['leave_one_block_out_means'][i] < 0 for a in controls)
                       for i in range(3)}
trace_rows = []
for b in s['blocks']:
    for a in arms:
        path = run/f"seed{b['seed']}_split{b['source_split_index']}"/a/'continuation_trace.jsonl'
        if path.exists():
            rows = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
            values = [x for x in rows if 'continuation_epoch' in x and 'validation_nll' in x]
            selected = b['fits'][a]['selection']
            require_epochs = list(range(selected['continuation_updates_completed']+1))
            check(f'complete_finite_epoch_sequence:{b["seed"]}:{a}',
                  [x['continuation_epoch'] for x in values] == require_epochs
                  and all(math.isfinite(x['validation_nll']) for x in values)
                  and all(x['actual_update'] == 50+x['continuation_epoch'] for x in values[1:]))
            earliest = min(values,key=lambda x:(x['validation_nll'],x['continuation_epoch']))
            check(f'earliest_strict_trace_selection:{b["seed"]}:{a}',
                  earliest['continuation_epoch'] == selected['selected_continuation_epoch']
                  and earliest['validation_nll'] == selected['primary_validation_nll'])
            trace_rows.append(dict(seed=b['seed'],arm=a,sha256=sha(path),epochs=len(values),minimum=earliest,
                                   epoch0_validation_NLL=values[0]['validation_nll'],
                                   final_validation_NLL=values[-1]['validation_nll']))
check('all12_trace_minima_independently_recomputed', len(trace_rows) == 12)
input_hashes = [{k:row[k] for k in ('path','sha256','bytes')} for row in custody['files']+trace_custody['files']]
result = dict(schema='independent-complete-Squirrel-development-scientific-audit-v1',
    checks=checks, stage1_sha256=sha(run/'STAGE1.json'), source_manifest_sha256=manifest_hash,
    supplied_input_hashes=input_hashes,
    complete_fits=12, blocks=block_table, validation_macro_NLL=means, contrasts=contrasts,
    frozen_Photo_trigger=True, leave_one_block_out_Photo_trigger_by_omitted_seed=leave_one_out_gates,
    selection_traces_independently_recomputed=len(trace_rows), trace_selection_evidence=trace_rows,
    checkpoint_or_label_bytes_opened=False, native_numeric_or_remote_execution=False,
    nominal_interval_scope='Three IID normally distributed paired block effects approximation; uncalibrated for shared graph, overlapping roles and source-validation selection.',
    whole_study_wall_seconds=s['wall_seconds'])
(review/'AUDIT_RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:result[k] for k in ('complete_fits','validation_macro_NLL','contrasts','frozen_Photo_trigger',
      'leave_one_block_out_Photo_trigger_by_omitted_seed','selection_traces_independently_recomputed','whole_study_wall_seconds')},indent=2))
