#!/usr/bin/env python3
"""Post-completion frozen-checkpoint analysis; no fits or original-score replay."""
import argparse
import math
from pathlib import Path
import statistics
import pilot_common as c

CONTRASTS = {
    'ensemble_live_credit': {'E_end_live':1,'E_end_detached':-1},
    'single_live_credit': {'S_end_live':1,'S_end_detached':-1},
    'ensemble_vs_generic_credit': {'E_end_live':1,'E_end_detached':-1,'S_end_live':-1,'S_end_detached':1},
    'endpoint_construction': {'E_end_live':1,'E_random_live':-1},
    'endpoint_credit_interaction': {'E_end_live':1,'E_end_detached':-1,'E_random_live':-1,'E_random_detached':1},
    'shared_vs_capable_adapted_single': {'E_end_live':1,'S_end_live':-1},
    'shared_vs_paid_single_joint': {'E_end_live':1,'S_end_joint':-1},
    'shared_vs_initially_matched_untied': {'E_end_live':1,'U_end_live':-1},
    'shared_vs_jointly_trained_native4': {'E_end_live':1,'J4_end_joint':-1},
    'learning_rule_vs_shared_joint': {'E_end_live':1,'E_end_joint':-1},
    'shared_vs_frozen_true_independent4': {'E_end_live':1,'external_true_independent4':-1},
    'shared_vs_frozen_native_single': {'E_end_live':1,'external_native_single':-1},
    'shared_vs_frozen_unchanged_unframedF4': {'E_end_live':1,'external_unchanged_unframedF4':-1},
}

QUALITY_REQUIRED = ['shared_vs_frozen_true_independent4','shared_vs_capable_adapted_single',
                    'shared_vs_paid_single_joint','shared_vs_jointly_trained_native4',
                    'learning_rule_vs_shared_joint','shared_vs_frozen_unchanged_unframedF4']
ATTRIBUTION_REQUIRED = ['ensemble_live_credit','ensemble_vs_generic_credit','endpoint_construction',
                        'endpoint_credit_interaction','shared_vs_initially_matched_untied']


def describe(values):
    if len(values)!=3 or not all(math.isfinite(v) for v in values):raise ValueError('All three paired blocks required')
    mean=statistics.mean(values);sd=statistics.stdev(values)
    radius=4.302652729911275*sd/math.sqrt(3)
    return {'values':values,'mean':mean,'sample_sd':sd,'descriptive_t95_seed_interval':[mean-radius,mean+radius],
        'mean_MRR_percentage_point_difference':100*mean,
        'higher_blocks':sum(v>0 for v in values),'tied_blocks':sum(v==0 for v in values),
        'inference_scope':'Descriptive variation over three seeds of one fixed graph/split; no significance, query independence or graph-generalization claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--execution',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();release=c.read(args.release)
    if release.get('root_analysis_approved') is not True or release.get('original_scores_recalculation') is not False or release.get('TEST_access') is not False or release.get('fits_authorized') is not False:
        raise ValueError('Separate completed-family analysis approval required; original scores remain fixed')
    c.verify_packet(release['pilot_source_manifest_sha256']);c.verify_training()
    root=args.execution.resolve(strict=True)
    if not root.is_relative_to(c.PHASE) or c.sha(root/'COLLECTION_FREEZE.json')!=release['collection_freeze_sha256']:
        raise ValueError('Exact authenticated mixed-provider collection differs')
    freeze=c.read(root/'COLLECTION_FREEZE.json');plan=c.read(root/'COHORT_PLAN.json')
    if freeze.get('schema')!='authenticated_complete_mixed_provider_paired_pilot_collection_v1' or freeze.get('complete') is not True or len(freeze.get('completed',[]))!=30 or freeze.get('comparative_scoring_performed') is not False:
        raise ValueError('No partial-family score access or missing-seed selection')
    if c.sha(root/'COLLECTION_RELEASE.json')!=freeze['collection_release_sha256'] or freeze['science_contract_sha256']!=c.sha(c.ROOT/'SCIENCE_CONTRACT.json'):
        raise ValueError('Prospective collector/science authority changed')
    if c.sha(root/'COHORT_PLAN.json')!=freeze['cohort_plan_sha256'] or c.sha(root/'EXTERNAL_ANCHORS.json')!=freeze['external_anchors_sha256']:
        raise ValueError('Prospective plan/anchors changed')
    output=c.fresh_phase_directory(args.output);output.mkdir()
    # Numerical import occurs only after completed-family/root approval.
    import inspect
    import torch
    if 'weights_only' not in inspect.signature(torch.load).parameters:raise ValueError('Safe frozen-logit load required')
    records={row['cell_id']:row for row in freeze['completed']}
    if len(records)!=30 or set(records)!={row['cell_id'] for row in plan['cells']}:
        raise ValueError('All thirty distinct prospective cells are required')
    anchors=c.read(root/'EXTERNAL_ANCHORS.json');values={b['block']:{} for b in plan['blocks']};costs=[];limited=[]
    for cell in plan['cells']:
        identity=cell['cell_id'];row=records[identity];path=c.phase_file(row['freeze_relative'])
        if c.sha(path)!=row['freeze_sha256'] or row['donor_execution_receipt']['exit_code']!=0:raise ValueError('Authoritative fit completion differs')
        value=c.read(path)
        if c.sha(c.phase_file(row['provider_admission_relative']))!=row['provider_admission_sha256']:
            raise ValueError('Actual provider admission changed after collection')
        if value.get('TEST_access') is not False or value['source_manifest_sha256']!=row['source_manifest_sha256'] or value['cohort_plan_sha256']!=freeze['cohort_plan_sha256']:
            raise ValueError('Frozen fit identity/source/heldout scope differs')
        if value['job_sha256']!=row['job_sha256'] or c.sha(c.phase_file(row['job_relative']))!=row['job_sha256']:
            raise ValueError('Frozen job identity differs')
        logits_path=c.phase_file(row['selected_VALID_logits_relative']);checkpoint=c.phase_file(row['selected_checkpoint_relative'])
        if logits_path.parent!=path.parent or checkpoint.parent!=path.parent or c.sha(logits_path)!=value['VALID_logits_sha256'] or c.sha(checkpoint)!=value['checkpoint_sha256'] or row['VALID_logits_sha256']!=value['VALID_logits_sha256'] or row['checkpoint_sha256']!=value['checkpoint_sha256']:
            raise ValueError('Selected checkpoint/logit custody differs')
        logits=torch.load(logits_path,map_location='cpu',weights_only=True)
        if logits['checkpoint_sha256']!=value['checkpoint_sha256'] or logits['selected_cycle']!=value['selected_cycle']:
            raise ValueError('Member logits do not bind the selected serving state')
        positive,negative=logits['mean_pos'],logits['mean_neg']
        if positive.shape!=(227,) or negative.shape!=(227,500) or not bool(torch.isfinite(positive).all() and torch.isfinite(negative).all()):
            raise ValueError('Full fixed VALID ranking population differs')
        rank=1+.5*((negative>=positive[:,None]).sum(1)+(negative>positive[:,None]).sum(1))
        raw=float((1/rank.float()).mean().item())
        if round(raw,4)!=value['selected_VALID_MRR']:raise ValueError('Frozen selector score disagrees with new-pilot logits')
        block=identity.split('_',1)[0];values[block][cell['cell']]=raw
        horizon_limited=value['last_cycle']==cell['schedule']['max_cycles'] and (value['last_cycle']-value['selected_cycle'])//5<11
        if horizon_limited:limited.append(identity)
        costs.append({'cell_id':identity,'raw_VALID_MRR':raw,'selected_rounded4_VALID_MRR':value['selected_VALID_MRR'],
            'selected_cycle':value['selected_cycle'],'last_cycle':value['last_cycle'],'inclusive_seconds':value['inclusive_seconds'],
            'counters':value['counters'],'peak_CUDA_allocated_bytes':value['peak_CUDA_allocated_bytes'],
            'maximum_validation_checks':cell['schedule']['max_cycles']//5,'actual_validation_checks':value['last_cycle']//5,
            'cap_before_eleven_misses':horizon_limited})
    for anchor in anchors['blocks']:
        values[anchor['block']]['external_true_independent4']=anchor['separately_trained_independent4']['raw_VALID_MRR']
        values[anchor['block']]['external_native_single']=anchor['native_single']['raw_VALID_MRR']
        values[anchor['block']]['external_unchanged_unframedF4']=anchor['unchanged_unframed_f4']['raw_VALID_MRR']
    descriptions={name:describe([sum(weight*values[block][cell] for cell,weight in terms.items()) for block in ('b0','b1','b2')])
                  for name,terms in CONTRASTS.items()}
    policy=plan['analysis_policy']
    if policy.get('strict_positive_mean') is not True or policy.get('minimum_mean_MRR_gain')!=0 or policy.get('minimum_higher_blocks')!=2:
        raise ValueError('Prospective exploratory policy differs')
    if policy.get('required_quality_contrasts')!=QUALITY_REQUIRED or policy.get('required_attribution_contrasts')!=ATTRIBUTION_REQUIRED:
        raise ValueError('All prospective strong-quality and explanation contrasts must be retained')
    flags={name:descriptions[name]['mean']>0 and descriptions[name]['higher_blocks']>=2 for name in QUALITY_REQUIRED}
    mechanism={name:descriptions[name]['mean']>0 and descriptions[name]['higher_blocks']>=2 for name in ATTRIBUTION_REQUIRED}
    result={'scope':'complete_paired_pilot_development_only','plan_sha256':freeze['cohort_plan_sha256'],
        'collection_freeze_sha256':c.sha(root/'COLLECTION_FREEZE.json'),'original_anchor_scores_copied_verbatim':True,
        'provider_provenance':freeze['donor_blocks'],
        'provider_source_and_runtime_by_fit':[{key:row[key] for key in ('cell_id','provider','hostname','GPU_UUID','source_manifest_sha256','runtime_versions','job_relative','job_sha256','provider_admission_relative','provider_admission_sha256')} for row in freeze['completed']],
        'original_score_or_checkpoint_recalculation':False,'TEST_access':False,'blocks':values,'contrasts':descriptions,'costs':costs,
        'horizon_limited_cells':limited,'prospective_lead_policy':policy,'lead_conditions':flags,
        'replicated_exploratory_quality_lead':all(flags.values()),'attribution_conditions':mechanism,
        'complete_proposed_explanation_supported_in_this_pilot':all(flags.values()) and all(mechanism.values()),
        'horizon_limited':bool(limited),'confirmation_required':True,
        'horizon_status_definition':'Cap-before-eleven-misses denotes incomplete native stopping-rule exposure; it does not prove recent improvement.',
        'novelty_clearance':False,'confirmed_superiority':False,'acceptance_verdict':None,
        'limits':['Joint native4 is not a separately trained ensemble; true-independent anchors are separate immutable references.',
                  'External schedules/support/selection and data/dropout draws differ despite aligned initialization seeds.',
                  'Prespecified blocks may use different admitted providers/runtimes; paired cell contrasts retain their own block provider and every actual provider binding.',
                  'Three seeds of one already-used fixed VALID split are development evidence, not an independent confirmation.',
                  'All contrasts and descriptive intervals are retained; no favorable subgroup, seed, checkpoint or method selection.']}
    c.write(output/'RESULTS.json',result)
    print('Completed fixed-family development analysis: '+str(output))


if __name__=='__main__':main()
