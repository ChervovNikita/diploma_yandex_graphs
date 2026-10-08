"""Disabled whole-family opener; mechanical closure precedes every metric read."""
import argparse
import json
import math
from pathlib import Path

from stage import ARMS,EXPECTED,PHASE,SEEDS,require,source_identity,verify_closed_family,write


def collect_family(*,family_root,output,later_execution_authorized=False,comparative_opening_authorized=False):
    require(later_execution_authorized is True and comparative_opening_authorized is True,'Disabled comparative opener')
    family_root=Path(family_root).resolve(strict=True); output=Path(output)
    require(family_root.is_relative_to(PHASE) and output.resolve().is_relative_to(PHASE) and not output.exists(),
            'Fresh phase-owned summary; preserve existing endpoints')
    records=verify_closed_family(family_root); source=source_identity()
    # Only after the complete mechanical barrier may numerical readouts be loaded.
    metrics={key:json.loads((path/'PRIVATE_SELECTED_METRICS.json').read_text()) for key,(path,_) in records.items()}
    for (seed,arm),value in metrics.items():
        counts=value['member_correctcount']; members=EXPECTED[arm][0]
        require(len(value['member_accuracy'])==len(value['member_NLL'])==len(counts)==members
                and type(value['correctcount']) is int and 0<=value['correctcount']<=5274
                and all(type(x) is int and 0<=x<=5274 for x in counts)
                and all(math.isfinite(value[key]) for key in ('served_NLL','Brier','served_accuracy',
                    'mean_member_accuracy','worst_member_accuracy','mean_member_NLL','worst_member_NLL'))
                and all(math.isfinite(x) and 0<=x<=1 for x in value['member_accuracy'])
                and all(math.isfinite(x) and x>=0 for x in value['member_NLL'])
                and value['served_accuracy']==value['correctcount']/5274
                and value['member_accuracy']==[x/5274 for x in counts]
                and value['mean_member_accuracy']==sum(counts)/(members*5274)
                and value['worst_member_accuracy']==min(counts)/5274
                and value['mean_member_NLL']==sum(value['member_NLL'])/members
                and value['worst_member_NLL']==max(value['member_NLL'])
                and value['member_metric_semantics']=='Actual per-route bounded mixtures','Full-population served/member scores')
    contrasts={}; passed=True
    for ref in ('S_joint4head','U4_sharedB'):
        gains=[100*(metrics[(s,'C4')]['correctcount']-metrics[(s,ref)]['correctcount'])/5274 for s in SEEDS]
        mean=lambda field:sum(metrics[(s,'C4')][field]-metrics[(s,ref)][field] for s in SEEDS)/3
        values=dict(paired_accuracy_gain_pp=gains,mean_over3seeds_accuracy_gain_pp=sum(gains)/3,
            mean_over3seeds_served_NLL_delta=mean('served_NLL'),mean_over3seeds_delta_mean_member_accuracy_pp=100*mean('mean_member_accuracy'),
            mean_over3seeds_delta_worst_member_accuracy_pp=100*mean('worst_member_accuracy'),
            mean_over3seeds_delta_mean_member_NLL=mean('mean_member_NLL'),mean_over3seeds_delta_worst_member_NLL=mean('worst_member_NLL'))
        checks=dict(positive_each_seed=all(x>0 for x in gains),mean_gain_at_least_point2=values['mean_over3seeds_accuracy_gain_pp']>=.2,
            nonworsening_pool_NLL=values['mean_over3seeds_served_NLL_delta']<=0,
            mean_member_accuracy=values['mean_over3seeds_delta_mean_member_accuracy_pp']>=-.1,
            worst_member_accuracy=values['mean_over3seeds_delta_worst_member_accuracy_pp']>=-.2,
            mean_member_NLL=values['mean_over3seeds_delta_mean_member_NLL']<=.01,
            worst_member_NLL=values['mean_over3seeds_delta_worst_member_NLL']<=.02)
        passed=passed and all(checks.values()); contrasts[ref]=dict(values=values,checks=checks,passed=all(checks.values()))
    result=dict(schema='staged-posterior-complete-family-gate-v1',all12bank_endpoints_complete=True,
        all3seed_blocks_complete=True,source=source,gate_passed=passed,contrasts=contrasts,
        complete_metrics={str(s):{a:metrics[(s,a)] for a in ARMS} for s in SEEDS},
        one_path_secondary_cannot_rescue=True,scope='Entire staged pipeline, not isolated loss-versus-aggregation causality',
        ordinary_independent4_available=False,prior_failed_C4_or_CS06_replaced=False,
        global_superiority_or_novelty_established=False,unused_confirmation=False,scientific_launch_admitted_by_collector=False)
    write(output,result); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family-root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--later-execution-authorized',action='store_true'); parser.add_argument('--comparative-opening-authorized',action='store_true')
    a=parser.parse_args(); collect_family(family_root=a.family_root,output=a.output,later_execution_authorized=a.later_execution_authorized,
                                         comparative_opening_authorized=a.comparative_opening_authorized)


if __name__=='__main__': main()
