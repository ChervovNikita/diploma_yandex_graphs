"""Summarize all frozen arms from a whole-family scalar extraction."""
import argparse
import json
import math
from pathlib import Path
import statistics

def paired(values):
    assert len(values)==3 and all(math.isfinite(x) for x in values)
    mean=statistics.mean(values)
    half=4.302652729911275*statistics.stdev(values)/math.sqrt(3)
    return dict(deltas=values,mean=mean,exploratory_95pct_t_interval_df2=[mean-half,mean+half],
                interpretation='variation among three repeats on the encountered graph, after validation selection, not external confirmation')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    phase=Path(__file__).resolve().parent.parent
    assert args.evidence.resolve().is_relative_to(phase) and args.output.resolve().is_relative_to(phase)
    evidence=json.loads(args.evidence.read_text())
    assert evidence['whole_family_closed_before_quality'] and evidence['TEST_access'] is False
    groups={}
    if evidence['study']=='cmcl18':
        seeds=[9101,9203,9307]
        for r in evidence['data']['runs']:
            assert r['complete'] and r['TEST_access'] is False
            v=r['selected_scores']['VALID']
            groups.setdefault(r['condition'],{})[r['seed']]=dict(accuracy=v['accuracy'],NLL=v['NLL'],
                macro_F1=v['macro_F1'],mean_member_accuracy=statistics.mean(v['member_accuracy']),
                worst_member_accuracy=min(v['member_accuracy']),available_correct_nodes=v['available_correct_nodes'],
                no_correct_member_errors=v['no_correct_member_errors'],pool_lost_correct_alternatives=v['pool_lost_correct_alternatives'],
                strict_common_rival_errors=v['strict_common_rival_errors'],inclusive_fit_seconds=r['costs']['inclusive_seconds'],
                selected_epoch=r['selected_epoch_zero_based'],updates=r['updates'])
        own='own_floor'
        primary='private_cmcl'
        metrics=('accuracy','NLL','macro_F1','mean_member_accuracy','worst_member_accuracy','available_correct_nodes',
                 'no_correct_member_errors','pool_lost_correct_alternatives','strict_common_rival_errors','inclusive_fit_seconds')
        refs=evidence['data']['contextual_reference']['all_selected_readouts']
        for name in ('factor1_native','factor1_mean4_dropout','independent4_own','single_native','shared4_own'):
            groups['contextual_'+name]={s:dict(accuracy=refs[f'seed{s}__{name}']['VALID']['pooled']['accuracy'],
                                             NLL=refs[f'seed{s}__{name}']['VALID']['pooled']['NLL']) for s in seeds}
        mainmetrics=('accuracy','NLL')
    else:
        seeds=[1,2,3]
        for r in evidence['data']['candidate']['report']['runs']:
            v=r['fresh_selected']['recorded_scores']['VALID']
            groups.setdefault(r['condition'],{})[r['seed_spec']['role_seed']]=dict(v,fit_seconds=r['seconds'],selected_epoch=r['selected_epoch'])
        for bank in evidence['data']['reference']['report']['ensembles']:
            path=Path(bank['selected_payload']['path'])
            role=int(path.parent.name.removeprefix('role'))
            v=bank['scores']['VALID']
            groups.setdefault(bank['family'],{})[role]=v
        own='shared_own_pair4'
        primary='shared_local_mul4'
        metrics=('micro_F1','macro_F1','BCE')
        mainmetrics=metrics
    assert all(set(rows)==set(seeds) for rows in groups.values())
    means={k:{m:statistics.mean(rows[s][m] for s in seeds) for m in metrics if all(m in rows[s] for s in seeds)} for k,rows in groups.items()}
    contrasts={}
    for name,rows in groups.items():
        if name!=primary:
            contrasts[primary+'__minus__'+name]={m:paired([(groups[primary][s][m]-rows[s][m])*(100 if m!='NLL' and m!='BCE' else 1) for s in seeds]) for m in mainmetrics}
    for name,rows in groups.items():
        if name not in (own,primary) and not name.startswith('contextual_'):
            contrasts[name+'__minus__'+own]={m:paired([(rows[s][m]-groups[own][s][m])*(100 if m!='NLL' and m!='BCE' else 1) for s in seeds]) for m in mainmetrics}
    result=dict(schema='whole-family-exploratory-scalar-summary-v1',study=evidence['study'],seeds=seeds,
        primary=primary,own_reference=own,all_selected_rows=groups,means=means,paired_contrasts=contrasts,
        percent_metrics_in_contrasts='percentage points, NLL/BCE in natural-log units',
        all_fixed_arms_preserved=True,TEST_access=False,novelty_or_acceptance_established=False,
        contextual_reference_comparisons_are_not_pure_causal=True)
    args.output.mkdir(exist_ok=False)
    (args.output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    title='Complete specialist-credit results' if evidence['study']=='cmcl18' else 'Complete heterogeneous context results'
    lines=['# '+title,'','All declared fits closed before interpretation. These are development results on an encountered graph. No original paper score changed. TEST remains closed.','',
           '| Predictor | '+' | '.join(mainmetrics)+' |','|---|'+'---:|'*len(mainmetrics)]
    for name,row in means.items():
        vals=[row[m]*(100 if m not in ('NLL','BCE') else 1) for m in mainmetrics]
        lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines.extend(['','## Primary paired comparisons','',
        'Accuracy and F1 deltas below are percentage points. NLL and BCE use natural-log units. The exploratory intervals describe variation among three repeats after validation selection. They do not account for selecting checkpoints, trying several methods, or changing graphs.',''])
    for name,rows in contrasts.items():
        lines.append('### '+name)
        lines.append('')
        for m,r in rows.items():
            lo,hi=r['exploratory_95pct_t_interval_df2']
            lines.append(f"- {m}: mean {r['mean']:+.6f}, paired deltas "+', '.join(f'{x:+.6f}' for x in r['deltas'])+f', interval [{lo:+.6f}, {hi:+.6f}].')
        lines.append('')
    lines.extend(['## Interpretation limits','',
        'Every fixed arm is retained in SUMMARY.json. Positive member diversity, a positive interaction, or a small validation delta alone does not establish a useful ensemble mechanism. Strongest capable references, exact error changes and unused confirmation govern any next claim. Concurrent fits prevent an isolated speed interpretation.',''])
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(study=result['study'],primary=primary,means=means,output=str(args.output))))

if __name__=='__main__':main()
