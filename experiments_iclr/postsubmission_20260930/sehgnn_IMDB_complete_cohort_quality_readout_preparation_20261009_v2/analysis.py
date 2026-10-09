"""Thin saved-output readout; no model, fitting, source assay or new experiment."""
import csv
import json
from pathlib import Path
import statistics


def paired(values):
    assert len(values) == 3
    return dict(pairs=list(values), mean=statistics.mean(values), SD=statistics.stdev(values),
        minimum=min(values), maximum=max(values), positive_pairs=sum(value > 0 for value in values),
        leave_one_pair_out_means=[statistics.mean(values[:i]+values[i+1:]) for i in range(3)])


def flat_events(events, truth):
    """Summarize existing masks, preserving positive and negative events."""
    result = {key: value for key, value in events.items() if not key.endswith('mask')}
    for name in ('repair', 'harm'):
        mask = events[name+'_mask']
        result[name+'_positive_events'] = int((mask & truth.bool()).sum())
        result[name+'_negative_events'] = int((mask & ~truth.bool()).sum())
    return result


def confusion(predicted, truth):
    truth = truth.bool()
    return {name: mask.sum(dim=0).tolist() for name, mask in (
        ('TP', predicted & truth), ('FP', predicted & ~truth), ('FN', ~predicted & truth), ('TN', ~predicted & ~truth))}


def write_csv(path, rows):
    assert rows
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(torch, diagnostics, native, comparisons, records, costs, freeze, output):
    """Existing metrics/masks feed complete tables and prospective root rules."""
    candidate = freeze['candidate']
    primary = freeze['primary']['references']
    methods = ['shared_own_only','native_pool_credit','source_view_supervision','uncoupled_source_contrast','COMMON_cycle',candidate,*primary]
    pool_rows, member_rows, label_rows, source_rows, repair_rows, delta_rows, origins = [], [], [], [], [], [], []
    for pair in (1,2,3):
        for method in methods:
            item = diagnostics[pair,method]
            pool = item['full_pool_F1']
            pool_rows.append(dict(pair=pair,method=method,BCE=item['full_pool_BCE'],micro_F1=pool['micro_F1'],macro_F1=pool['macro_F1']))
            counts = confusion(item['full_pool_probabilities'] > .5, item['targets'])
            for label in range(5):
                label_rows.append(dict(pair=pair,method=method,member='pool',label=label,F1=pool['per_label_F1'][label],**{key:values[label] for key,values in counts.items()}))
            for member in range(4):
                score = item['full_member_F1'][member]
                member_rows.append(dict(pair=pair,method=method,member=member,BCE=item['full_member_BCE'][member],micro_F1=score['micro_F1'],macro_F1=score['macro_F1']))
                prediction = torch.where(item['member_correct'][member],item['targets'].bool(),~item['targets'].bool())
                counts = confusion(prediction,item['targets'])
                for label in range(5):
                    label_rows.append(dict(pair=pair,method=method,member=member,label=label,F1=score['per_label_F1'][label],**{key:values[label] for key,values in counts.items()}))
                for column, family in enumerate(item['family_order']):
                    source_rows.append(dict(pair=pair,method=method,member=member,family=family,U=item['U'][member][column],D=item['D'][member][column],
                        Spec_U=item['assigned_specificity']['U'],Spec_D=item['assigned_specificity']['D']))
                    details = item['family_details'][family]['members'][member]
                    for context in ('absent_peer_context','full_peer_context'):
                        repair_rows.append(dict(pair=pair,method=method,member=member,family=family,context=context,**flat_events(details[context],item['targets'])))
            origins.append(dict(pair=pair,method=method,selected_origin=records[pair,method]))
        for reference in methods:
            if reference == candidate:
                continue
            comparison = comparisons[pair,reference]
            c, r = comparison['candidate_pool_F1'],comparison['reference_pool_F1']
            row = dict(pair=pair,reference=reference,micro_F1=c['micro_F1']-r['micro_F1'],macro_F1=c['macro_F1']-r['macro_F1'],
                BCE_candidate_minus_reference=comparison['full_pool_BCE_difference_candidate_minus_reference'])
            row.update({'label_'+str(label)+'_F1':c['per_label_F1'][label]-r['per_label_F1'][label] for label in range(5)})
            delta_rows.append(row)
            repair_rows.append(dict(pair=pair,method=candidate,reference=reference,context='deployed_full_input_pool',
                **flat_events(comparison['deployed_full_input_pool_events'],diagnostics[pair,candidate]['targets'])))
        score = native[pair]['VALID']
        c = diagnostics[pair,candidate]['full_pool_F1']
        delta_rows.append(dict(pair=pair,reference='matched_native_single',micro_F1=c['micro_F1']-score['micro_F1'],
            macro_F1=c['macro_F1']-score['macro_F1'],BCE_candidate_minus_reference=diagnostics[pair,candidate]['full_pool_BCE']-score['BCE'],
            native_fresh_per_label_predictions='unavailable; no reconstruction or retrospective label gate'))
    summaries = {}
    for reference in [method for method in methods if method != candidate]+['matched_native_single']:
        rows = [row for row in delta_rows if row['reference'] == reference]
        summaries[reference] = {key:paired([row[key] for row in rows]) for key in rows[0]
            if key not in ('pair','reference','native_fresh_per_label_predictions')}
    checks = []
    def check(name,passed,observed,rule):
        checks.append(dict(name=name,passed=bool(passed),observed=observed,frozen_rule=rule))
    for reference in primary+['matched_native_single']:
        rule = freeze['primary'] if reference in primary else freeze['single_model_quality_requirement']
        values = summaries[reference]['micro_F1']
        margin = rule['mean_paired_gain_each_reference_at_least'] if reference in primary else rule['mean_paired_candidate_pool_minus_matched_single_at_least']
        floor = rule['minimum_any_pair_delta_each_reference'] if reference in primary else rule['minimum_pair_delta']
        positives = rule['positive_pairs_each_reference_at_least'] if reference in primary else rule['positive_pairs_at_least']
        check(reference+':mean_micro_gain',values['mean']>=margin,values,{'at_least':margin})
        check(reference+':minimum_pair',values['minimum']>=floor,values,{'at_least':floor})
        check(reference+':positive_pairs',values['positive_pairs']>=positives,values,{'at_least':positives})
    guards = freeze['supporting_guards']
    for reference in primary:
        values = summaries[reference]
        check(reference+':mean_BCE',values['BCE_candidate_minus_reference']['mean']<=guards['mean_deployed_BCE_candidate_minus_each_reference_at_most'],values['BCE_candidate_minus_reference'],guards)
        check(reference+':mean_macro',values['macro_F1']['mean']>=guards['mean_macro_F1_candidate_minus_each_reference_at_least'],values['macro_F1'],guards)
        for label in range(5):
            value = values['label_'+str(label)+'_F1']
            check(reference+':label'+str(label),value['mean']>=guards['mean_per_label_F1_candidate_minus_each_reference_each_label_at_least'],value,guards)
    own = freeze['own_member_competence']
    floor_details, member_delta, worst_delta = [], [], []
    for pair in (1,2,3):
        c = [score['micro_F1'] for score in diagnostics[pair,candidate]['full_member_F1']]
        r = [score['micro_F1'] for score in diagnostics[pair,'shared_own_only']['full_member_F1']]
        threshold = native[pair]['VALID']['micro_F1']-own['every_pair_every_member_micro_F1_at_least_matched_native_single_minus']
        floor_details.extend(dict(pair=pair,member=m,micro_F1=value,native_single_floor=threshold,passed=value>=threshold) for m,value in enumerate(c))
        member_delta.append(statistics.mean(c)-statistics.mean(r));worst_delta.append(min(c)-min(r))
    check('all_actual_candidate_members_vs_matched_native',all(row['passed'] for row in floor_details),floor_details,own)
    check('candidate_member_mean_vs_own_only',statistics.mean(member_delta)>=own['mean_candidate_member_micro_F1_minus_paired_own_only_member_mean_at_least'],paired(member_delta),own)
    check('candidate_worst_mean_vs_own_only',statistics.mean(worst_delta)>=own['mean_candidate_worst_member_micro_F1_minus_paired_own_only_worst_at_least'],paired(worst_delta),own)
    check('candidate_worst_minimum_pair_vs_own_only',min(worst_delta)>=own['minimum_pair_worst_member_difference_vs_own_only'],paired(worst_delta),own)
    result = dict(schema='complete-closed-IMDB-paired-quality-readout-v1',candidate=candidate,paired_contrasts=summaries,
        frozen_checks=checks,advance_to_independently_frozen_unused_confirmation=all(row['passed'] for row in checks),
        all_five_native_fresh_aggregate_results=native,native_fresh_per_label_predictions='unavailable',selected_origins=origins,costs=costs,
        uncertainty=dict(pairs=3,graphs=1,range_SD_leave_one_pair_out_reported=True,minimum_two_sided_sign_flip_p=.25,
            overlapping_roles_and_labels_or_members_not_extra_repetitions=True,role_and_optimizer_effects_confounded=True,VALID_selected=True,
            significance_equivalence_novelty_generalization_or_acceptance_claim=False),
        source_specific_interpretation='All five control contrasts and all U/D cells retained; positive U or own-only gain is insufficient; no equivalence inference.',
        sharing_advantage_identified=False,matched_untied_same_J_still_required=True)
    for name,rows in [('POOL_METRICS',pool_rows),('MEMBER_METRICS',member_rows),('LABEL_CONFUSION',label_rows),('FULL_U_D',source_rows),('REPAIR_HARM',repair_rows),('PAIRED_DELTAS',delta_rows)]:
        write_csv(output/(name+'.csv'),rows)
    (output/'QUALITY_READOUT.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    lines=['# Complete closed IMDB pilot readout','',
        'Practical advancement decision: **'+('advance to unused-outcome confirmation' if result['advance_to_independently_frozen_unused_confirmation'] else 'do not advance under the frozen joint rules')+'**.','',
        'This is three paired repetitions on one graph, with VALID reused for selection. It is not a confirmatory, novelty, equivalence or acceptance verdict.','',
        '## Paired micro-F1 contrasts','',
        '| Comparator | Pair deltas (percentage points) | Mean | SD | Range | Leave-one-pair-out means |','|---|---|---:|---:|---|---|']
    for name,values in summaries.items():
        v=values['micro_F1'];fmt=lambda x:f'{100*x:+.3f}'
        lines.append('| '+name+' | '+', '.join(map(fmt,v['pairs']))+' | '+fmt(v['mean'])+' | '+f"{100*v['SD']:.3f}"+' | '+fmt(v['minimum'])+' to '+fmt(v['maximum'])+' | '+', '.join(map(fmt,v['leave_one_pair_out_means']))+' |')
    lines+=['','## Frozen checks','']+[('- **'+('pass' if row['passed'] else 'fail')+'** '+row['name']) for row in checks]
    lines+=['','## Primary supporting changes','',
        '| Independent reference | Mean BCE change (lower is better) | Mean macro-F1 change (pp) | Five mean label-F1 changes (pp) |',
        '|---|---:|---:|---|']
    for name in primary:
        v=summaries[name]
        lines.append(f"| {name} | {v['BCE_candidate_minus_reference']['mean']:+.6f} | {100*v['macro_F1']['mean']:+.3f} | "+', '.join(f"{100*v['label_'+str(label)+'_F1']['mean']:+.3f}" for label in range(5))+' |')
    lines+=['','## Every candidate member','',
        '| Pair | Member | Micro-F1 | Matched native floor | Meets frozen floor |','|---|---:|---:|---:|---|']
    for row in floor_details:
        lines.append(f"| {row['pair']} | {row['member']} | {row['micro_F1']:.5f} | {row['native_single_floor']:.5f} | {row['passed']} |")
    lines+=['','Candidate-minus-own-only member mean deltas (pp): '+', '.join(f'{100*v:+.3f}' for v in member_delta)+'.',
        'Candidate-minus-own-only worst-member deltas (pp): '+', '.join(f'{100*v:+.3f}' for v in worst_delta)+'.',
        '', '## All five source-native results','',
        '| Source seed | Fresh VALID micro-F1 | Macro-F1 | BCE | Selected epoch |','|---|---:|---:|---:|---:|']
    for seed,item in native.items():
        v=item['VALID'];lines.append(f"| {seed} | {v['micro_F1']:.5f} | {v['macro_F1']:.5f} | {v['BCE']:.6f} | {item['selected_epoch']} |")
    lines+=['','Every arm, member, label, signed U/D cell and repair/harm stratum is in the accompanying tables. Supporting macro/per-label/BCE gates apply to both independent references. Native singles have the separate micro rule and member floor; fresh native per-label predictions are unavailable.','',
        'Costs are reported at their original inclusive scopes; nested entry/family/cell timings are not summed. All negative results are retained. Control contrasts describe unresolved mechanisms; a matched untied J counterpart and stronger comparators remain required.']
    (output/'READOUT.md').write_text('\n'.join(lines)+'\n')
    return result
