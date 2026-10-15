"""Frozen21-fit descriptors/gates and scalar paired-ranking diagnostics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
E=HERE.parents[1]
PHASE=E.parent
SEEDS=[101,202,303]
PRODUCER=E/'research/molbace_pairwise_competence_v1'


def require(ok,message):
    if not ok:raise ValueError(message)


def load(path,name):
    definition=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(definition)
    sys.modules[name]=module;definition.loader.exec_module(module);return module


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def helpers():return load(E/'pilot_20261015_structural_private_risk_v1/run.py','_pair_existing_process_helpers')


def prior():return load(E/'research/pubmed_cheb_transfer_driver_v1/study.py','_pair_existing_fit_gate')


def serial():return load(E/'research/pubmed_cheb_transfer_driver_v1/driver.py','_pair_existing_serial_owner').serial


def diagnostics():return load(E/'pilot_20261015_bn_geometry_correction_v1/complete_route_analysis_v2/analyze.py','_pair_existing_ranking_diagnostics')


def jobs(spec):
    producer=load(PRODUCER/'extension.py','_pair_frozen_descriptor_builder');rows=[]
    for seed in SEEDS:
        descriptors=[('native_common',body) for body in range(4)]+[(arm,None) for arm in ('common','structured','permuted')]
        for arm,body in descriptors:
            job=producer.descriptor(spec['project_root'],seed,arm,body)
            index=len(rows);value=digest(job)
            rows.append({'index':index,'id':job['id'],'cohort_seed':seed,'kind':arm,'job':job,'job_hash':value,
                'token':str(index).zfill(3)+'_'+value[:16]})
    require(len(rows)==len({r['id'] for r in rows})==21,'Exactly21 unique cold trajectories required')
    return rows


def terminal_gate(end,outcomes,rows):
    require(end.get('status')=='complete' and end.get('TEST_accessed')is False and end.get('outcomes')==outcomes and
        len(rows)==len(outcomes)==21 and [r.get('id') for r in outcomes]==[r['id'] for r in rows] and
        all(r.get('status')=='complete' and r.get('returncode')==0 and not r.get('timed_out') and
            len(r.get('RESULT',{}).get('sha256',''))==64 for r in outcomes),
        'All21 ordered terminal successful fits required before any new collection/quality summary')


def collection_gate(end,rows):
    require(end.get('status')=='complete' and end.get('TEST_accessed')is False and len(rows)==9 and
        [r.get('id') for r in end.get('outcomes',[])]==[r['id'] for r in rows] and
        all(r.get('status')in ('complete','complete_with_sanity_warning') and r.get('returncode')==0 and
            not r.get('timed_out') and r.get('RESULT',{}).get('sha256') for r in end['outcomes']),
        'All9 shared selected collections required; no partial-family analysis')


def collection(value,fit,receipt):
    require(value.get('status')in ('complete','complete_with_sanity_warning') and value.get('role')=='VALID' and
        value.get('TEST_accessed')is False and value.get('job_id')==fit['job_id'] and value.get('job_hash')==fit['job_hash'] and
        value.get('selected_epoch')==fit['selected_epoch'] and value.get('fit_receipt')==receipt and
        value.get('selected_checkpoint')==fit['selected_checkpoint'] and value.get('saved_VALID_export')==fit['selected_VALID_export'] and
        value.get('members')==4 and value.get('evaluated_VALID_graphs')==151 and
        value['actual_work']['TRAIN_callbacks']==value['actual_work']['optimizer_steps']==0 and
        value['actual_work']['native_route_forward_attempts']==value['actual_work']['native_route_forwards_completed']==20 and
        not value.get('failure') and not value.get('close_error'),'Exact complete zero-training own-selected collection required')


def accept_export(torch,value,fit):
    require(value.get('schema')=='own-selected-VALID-logits-v1' and value.get('role')=='VALID' and value.get('TEST_accessed')is False and
        value.get('job_id')==fit['job_id'] and value.get('job_hash')==fit['job_hash'] and value.get('selected_epoch')==fit['selected_epoch'] and
        value.get('selected_checkpoint')==fit['selected_checkpoint'] and value.get('validation')==fit['selected_VALID'],
        'Authentic own-selected VALID export required')
    z,y,ids=value['raw_logits'],value['labels'],value['ids']
    require(tuple(z.shape)==tuple(y.shape)==(151,1) and ids.shape==(151,) and z.dtype==torch.float32 and ids.dtype==torch.long and
        torch.unique(ids).numel()==151 and torch.isfinite(z).all().item() and torch.isfinite(y).all().item() and
        int((y==1).sum())==130 and int((y==0).sum())==21,'Whole finite original MolBACE VALID151 required')


def alternatives(reference,candidate,old_pool,new_pool):
    require(len(reference)==len(candidate)>0,'Aligned pair columns required')
    before=[any(v>0 for v in col) for col in reference];after=[any(v>0 for v in col) for col in candidate]
    common=[all(v<0 for v in col) for col in reference];new_common=[all(v<0 for v in col) for col in candidate]
    credit=lambda v:1. if v>0 else .5 if v==0 else 0.
    delta=[credit(b)-credit(a) for a,b in zip(old_pool,new_pool)]
    return {'acquired_correct_alternatives':sum(not a and b for a,b in zip(before,after)),
        'lost_correct_alternatives':sum(a and not b for a,b in zip(before,after)),
        'new_alternative_served':sum(not a and b and p>0 for a,b,p in zip(before,after,new_pool)),
        'new_alternative_still_lost':sum(not a and b and p<=0 for a,b,p in zip(before,after,new_pool)),
        'reference_common_reversal_acquires_correct_member':sum(w and b for w,b in zip(common,after)),
        'reference_common_reversal_served_repairs':sum(w and p>0 for w,p in zip(common,new_pool)),
        'new_common_reversals':sum(not a and b for a,b in zip(common,new_common)),
        'cleared_common_reversals':sum(a and not b for a,b in zip(common,new_common)),
        'tie_aware_repair_credit':sum(max(v,0.) for v in delta),'tie_aware_harm_credit':sum(max(-v,0.) for v in delta)}


def decision(rows):
    means={bank:statistics.mean(r['pool_AUC'] for r in rows if r['bank']==bank) for bank in dict.fromkeys(r['bank'] for r in rows)}
    require(all(sorted(r['seed'] for r in rows if r['bank']==bank)==SEEDS for bank in means),'Every decision uses all3 seeds')
    recipe=all(means['structured']>means[bank] for bank in ('residual_F','common','permuted'))
    advance=recipe and all(means['structured']>means[bank] for bank in ('BCE_I4','pair_I4'))
    return {'mean_AUC':means,'fixed_structural_recipe_passes':recipe,'main_goal_advancement_screen_passes':advance,
        'rule':'Strict mean served AUC improvements over residual_F/common/permuted; advancement also BCE_I4/pair_I4. No retuning or superiority claim.',
        'proper_loss_and_each_seed':'Signed BCE and each paired seed remain visible; mean AUC screen does not erase harms.',
        'competence_and_complementarity':'Inspect mean/worst member AUC/BCE, absolute common reversals, alternatives, pooling gain and served repairs/harms. No new threshold.'}
