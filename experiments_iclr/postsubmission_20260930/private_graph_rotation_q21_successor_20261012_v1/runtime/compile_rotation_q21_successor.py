"""Append five source-declared serving procedures to four frozen Q21 studies.

Metadata only. Every prior run dictionary, weighted table and method is exact.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path

from adapters.private_rotation import ARMS,I4,CONTRACT,receipt
from prove_q21_rotation_overlap import MAPPING
from spec import canonical,fingerprint,validate

DOMAIN=[{'learning_rate':lr,'dropout':p} for lr in (.0005,.001,.002) for p in (.1,.2,.4)]
SEARCH_SEEDS=(77101,77103)
COMPARATIVE_SEEDS=(77201,77203,77207)
METHODS={'shared4_paired_local':'private_graph_rotation_shared4_paired_local',
 'paired_local_M1':'private_graph_rotation_control_paired_local_M1',
 'rademacher_allmap_M1':'private_graph_rotation_control_rademacher_allmap_M1',
 'paired_local_I4':'private_graph_rotation_control_paired_local_genuine_I4',
 'rademacher_allmap_I4':'private_graph_rotation_control_rademacher_allmap_genuine_I4'}
ADDED_EXPECTED={'atomic_fit_groups':1836,'full_budget_optimizer_updates':9180000,'search_selection_jobs':12,
 'bank_exports':900,'scalar_CAL_fits':900,'evaluation_jobs':900,'genuine_I4_banks':360,'all_run_records':4548,
 'planned_TRAIN_route_forwards':12960000,'planned_TRAIN_backward_calls':9180000,'planned_TRAIN_VJP_calls':0}


def append_study(original,bindings,runtime_files,packet_prefix,proof):
    study=copy.deepcopy(original);runs=study['runs'];old=list(original['runs']);banks=[];selections={}
    cohorts={(r['split'],r['seed']) for r in old if r['kind']=='ensemble_assembly'}
    splits=range(20 if study['dataset']=='wikics' else 10)
    if study['dataset'] not in ('wikics','amazon-photo') or cohorts!={(i,s) for i in splits for s in COMPARATIVE_SEEDS}:
        raise ValueError('Four existing full20/full10 Q21 studies required')
    backbone=old[0]['config']['backbone'];old_runtime=study['source']['frozen_runtime_package']
    def template(kind,split,seed=None):
        return next(r for r in old if r['kind']==kind and r['split']==split and (seed is None or r['seed']==seed)
                    and r['method']=='private_bootstrap_control_factorized_allmap_M1')
    def common(arm,split,stage):
        c=copy.deepcopy(template('model_fit',split)['config'])
        for key in ('bootstrap_arm','bootstrap_sources','bootstrap_weights','source_contract','selected_config_from','initialization_seed'):c.pop(key,None)
        c.update(rotation_arm=arm,rotation_sources=bindings,rotation_contract=CONTRACT,members=ARMS[arm][1],
          architecture='internal_be' if ARMS[arm][1]==4 else 'factorized_single',stage=stage,
          recipe_scope='literal source constructor/private RNG/TRAIN unit;new roles/full5000/firstSELECT/commonCAL;no legacy trajectory claim')
        return c
    def add(identifier,kind,arm,split,seed,body,config,parents=()):
        if kind=='model_fit':config['initialization_seed']=seed+1000003*body
        adapter=(packet_prefix+'/runtime/adapters/private_rotation.py:make_adapter' if kind=='model_fit' else old_runtime+'/adapters/'+
          ('search.py:select_configuration' if kind=='selection' else 'node_endpoints.py:'+{'ensemble_assembly':'export_bank','calibration':'scalar_calibration','evaluation':'evaluate_bank'}[kind]))
        row={'run_id':identifier,'kind':kind,'method':METHODS[arm],'split':split,'seed':seed,'body':body,'adapter':adapter,
          'config':config,'parents':list(parents),'budget':copy.deepcopy(template('model_fit',0)['budget']) if kind=='model_fit' else {},
          'selection':copy.deepcopy(template('model_fit',0)['selection']) if kind=='model_fit' else {}}
        runs.append(row);return row
    for arm in ARMS:
        candidates=[]
        for point in DOMAIN:
            key=fingerprint(point)[:20];ids=[]
            for seed in SEARCH_SEEDS:
                ids.append(add('rotation-search-'+arm+'-'+key+'-s'+str(seed),'model_fit',arm,0,seed,0,{**common(arm,0,'search'),**point})['run_id'])
            candidates.append({'config_id':key,'parameters':point,'run_ids':ids})
        selections[arm]=add('rotation-select-'+arm,'selection',arm,0,0,0,
          {'stage':'search_selection','configurations':candidates,'ranking_metric':'accuracy','ranking_direction':'max','reservation_mib':1,'estimated_seconds':1},
          [r for c in candidates for r in c['run_ids']])['run_id']
    for split in splits:
        for seed in COMPARATIVE_SEEDS:
            fits={}
            for arm in ARMS:
                row=add('rotation-compare-sp'+str(split)+'-'+arm+'-s'+str(seed),'model_fit',arm,split,seed,0,
                  {**common(arm,split,'comparative'),**DOMAIN[0],'selected_config_from':selections[arm]},[selections[arm]])
                fits[arm]=[row['run_id']]
            for arm,member in I4.items():
                first=next(r for r in runs if r['run_id']==fits[member][0]);fits[arm]=[first['run_id']]
                for body in (1,2,3):
                    fits[arm].append(add(first['run_id']+'-i4-b'+str(body),'model_fit',member,split,seed,body,copy.deepcopy(first['config']),first['parents'])['run_id'])
            for arm,parents in fits.items():
                prefix='rotation-bank-sp'+str(split)+'-'+arm+'-s'+str(seed)
                export_c=copy.deepcopy(template('ensemble_assembly',split,seed)['config']);export_c['fit_parents']=parents
                export=add(prefix,'ensemble_assembly',arm,split,seed,0,export_c,parents)['run_id']
                cal_c=copy.deepcopy(template('calibration',split,seed)['config']);cal_c['export_parent']=export
                cal=add(prefix+'-cal','calibration',arm,split,seed,0,cal_c,[export])['run_id']
                eval_c=copy.deepcopy(template('evaluation',split,seed)['config']);eval_c.update(export_parent=export,calibration_parent=cal)
                evaluation=add(prefix+'-evaluate','evaluation',arm,split,seed,0,eval_c,[export,cal])['run_id']
                banks.append({'dataset':study['dataset'],'backbone':backbone,'arm':arm,'split':split,'seed':seed,'fit_parents':parents,
                  'export':export,'calibration':cal,'evaluation':evaluation,'genuine_I4':arm in I4,'body0_reused':arm in I4,
                  'served_views':4 if arm in I4 else ARMS[arm][1]})
    study['source']['files']+=runtime_files
    study['source']['private_rotation_within_Q21_extension']={'schema':'private-rotation-Q21-source-lineage-v1',
      'prior_spec_hash':fingerprint(original),'prior_records':len(old),'prior_run_dicts_preserved_exactly':True,
      'new_runtime_package':packet_prefix+'/runtime','finite_bank_mapping_proof':proof,'mapped_three_bank_methods':MAPPING,
      'mapped_three_duplicate_fits':0,'new_procedures':METHODS,'new_canonical_method_count':1,
      'historical_trajectory_or_failure_equivalence':False,'existing_M1_global_RNG_provenance_preserved':True}
    study['comparisons'].append({'candidates':list(MAPPING.values())+[METHODS['shared4_paired_local']],
      'references':['private_bootstrap_control_ordinary_M1','private_bootstrap_control_ordinary_genuine_I4',
        'private_bootstrap_control_factorized_allmap_M1','private_bootstrap_control_factorized_allmap_genuine_I4',
        'private_bootstrap_control_paired_graph_M1','private_bootstrap_control_paired_graph_genuine_I4',
        'private_bootstrap_control_rank1_lora_graph_M1','private_bootstrap_control_rank1_lora_graph_genuine_I4',
        METHODS['paired_local_M1'],METHODS['paired_local_I4'],METHODS['rademacher_allmap_M1'],METHODS['rademacher_allmap_I4'],
        'private_bootstrap_control_original_gnnm'],'original_GNNM_reference':'private_bootstrap_control_original_gnnm',
      'scope':'Existing whole20/10 masks and three paired comparative seeds. Three finite bank mechanisms map to existing Q21 groups. Added local-pair and first-input-Rademacher source-scoped M1/I4 are honest controls;global/private RNG difference is provenance,not isolated causality or legacy trajectory equality.'})
    check=copy.deepcopy(study);check['runs']=check['runs'][:len(old)];check['source']=copy.deepcopy(original['source']);check['comparisons']=check['comparisons'][:-1]
    if check!=original or runs[:len(old)]!=old:raise ValueError('Existing Q21 rows or study contracts changed')
    validate(study);added=runs[len(old):];new_fits=[r for r in added if r['kind']=='model_fit'];starts=Counter(r['config']['stage'] for r in new_fits)
    counts={'atomic_fit_groups':len(new_fits),'paid_optimizer_objects':len(new_fits),'physical_body_instances':len(new_fits),
      'maximum_optimizer_updates':len(new_fits)*5000,'full_budget_optimizer_updates':len(new_fits)*5000,'search_selection_jobs':3,
      'comparative_banks':len(banks),'bank_exports':len(banks),'scalar_CAL_fits':len(banks),'evaluation_jobs':len(banks),
      'genuine_I4_banks':sum(b['genuine_I4'] for b in banks),'reused_I4_body0':sum(b['body0_reused'] for b in banks),
      'all_run_records':len(added),'actual_optimizer_updates':0,'actual_predictive_scores':0,
      'planned_TRAIN_route_forwards':sum(r['config']['members']*5000 for r in new_fits),
      'planned_TRAIN_backward_calls':len(new_fits)*5000,'planned_TRAIN_VJP_calls':0,'optimizer_starts_by_stage':dict(starts)}
    proof_row={'study_id':study['study_id'],'prior_spec_hash':fingerprint(original),'previous_records':len(old),'appended_records':len(added),
      'prior_records_equal':True,'data_roles_equal':study['data']==original['data'],'prior_comparisons_equal':True,
      'original_GNNM_fits':sum(r['kind']=='model_fit' and r['config'].get('is_original_gnnm',False) for r in old),
      'added_original_GNNM_fits':0,'added_counts':counts}
    return study,counts,banks,proof_row


def prepare(root,packet):
    root=Path(root).resolve();packet=Path(packet).resolve();prefix=str(packet.relative_to(root));prior=packet/'lineage/prior_q21/matrix'
    output=packet/'matrix'
    if output.exists():raise ValueError('Fresh immutable output required')
    old_plan=json.loads((prior/'PLAN.json').read_text());old_summary=json.loads((prior/'SUMMARY.json').read_text())
    if fingerprint(old_plan)!=old_summary['plan_hash'] or old_plan['scientific_launch_enabled'] is not False:raise ValueError('Complete disabled prior Q21 required')
    proof=receipt(root,prefix+'/Q21_FINITE_BANK_EQUIVALENCE.json');bindings=json.loads((packet/'SOURCE_BINDINGS.json').read_text())
    source_files=[receipt(root,str(p.relative_to(root))) for p in sorted((packet/'runtime').rglob('*.py'))]
    source_files+=list(bindings.values())+[proof,receipt(root,prefix+'/SOURCE_PROPOSAL.json'),receipt(root,prefix+'/lineage/AUTHORIZED_DEVELOPED_NEXT_PORTS.json')]
    prepared=[];added_counts=Counter();stages=Counter();bank_rows=[];preservation=[]
    for row in old_plan['studies']:
        path=prior/'studies'/Path(row['path']).name;original=json.loads(gzip.decompress(path.read_bytes()))
        if fingerprint(original)!=row['spec_hash']:raise ValueError('Mirrored Q21 source contract changed')
        study,counts,banks,kept=append_study(original,bindings,source_files,prefix,proof)
        prepared.append(study);preservation.append(kept);bank_rows.extend(banks);stages.update(counts['optimizer_starts_by_stage'])
        added_counts.update({k:v for k,v in counts.items() if type(v)==int})
    if len(prepared)!=4 or {s['study_id'] for s in prepared}!={r['study_id'] for r in preservation}:raise ValueError('Four exact study IDs required')
    for key,n in ADDED_EXPECTED.items():
        if added_counts[key]!=n:raise ValueError('Incremental count differs: '+key)
    output.mkdir();(output/'studies').mkdir();plan=copy.deepcopy(old_plan);relative=str(output.relative_to(root));plan['studies']=[]
    plan['output_root']=relative+'/execution'
    for gate in plan['global_gates'].values():gate['path']=relative+'/gate_receipts/'+Path(gate['path']).name
    for study in prepared:
        path=output/'studies'/(study['study_id']+'.json.gz');path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
        plan['studies'].append({'path':str(path.relative_to(root)),'spec_hash':fingerprint(study)})
        group=next(g for g in plan['groups'] if g['id']==study['study_id']);group['run_keys']=[study['study_id']+'/'+r['run_id'] for r in study['runs']]
        group['estimated_seconds']=sum(r['config']['estimated_seconds'] for r in study['runs'])
    plan['private_rotation_Q21_successor']={'previous_plan_hash':fingerprint(old_plan),'within_study_extension':True,
      'added_runtime':prefix+'/runtime','existing_runtime_retained':True,'CUDA_qualification_pending':True}
    counts=copy.deepcopy(old_summary['counts'])
    for k,n in added_counts.items():counts[k]=counts.get(k,0)+n
    counts['optimizer_starts_by_stage']={k:counts['optimizer_starts_by_stage'].get(k,0)+stages[k] for k in set(counts['optimizer_starts_by_stage'])|set(stages)}
    fits=[r for s in prepared for r in s['runs'] if r['kind']=='model_fit']
    counts['planned_TRAIN_route_forwards']=sum(r['budget']['maximum']*r['config']['members'] for r in fits)
    counts['planned_TRAIN_backward_calls']=sum(r['budget']['maximum']*(0 if 'bootstrap_weights' in r['config'] else 4 if r['config'].get('is_original_gnnm') else 1) for r in fits)
    counts['planned_TRAIN_VJP_calls']=sum(r['budget']['maximum']*2 for r in fits if 'bootstrap_weights' in r['config'])
    summary=copy.deepcopy(old_summary);summary.update(schema='private-bootstrap-with-private-rotation-disabled-successor-v1',plan_hash=fingerprint(plan),counts=counts,
      banks=old_summary['banks']+bank_rows,prior_plan_hash=fingerprint(old_plan),added_counts={**dict(added_counts),'optimizer_starts_by_stage':dict(stages)},
      appended_procedures=METHODS,canonical_bank_mapping=MAPPING,new_canonical_method_count=1,complete_prior_preservation=preservation,
      numerical_CPU_source_checks='PENDING_ROOT_CPU_ONLY',source_and_data_arrays_decoded=False,CUDA_executions=0,scientific_fits=0,
      old_Q21_weighted_procedures_modified=False,old_original_GNNM_fits_retained=sum(r['original_GNNM_fits'] for r in preservation),added_original_GNNM_fits=0,
      status='DISABLED_INCREMENTAL_SOURCE_PORT_CPU_RESOURCE_MASTER_PROTOCOL_AND_GO_PENDING')
    if counts['atomic_fit_groups']!=7020 or counts['all_run_records']!=18420 or counts['full_budget_optimizer_updates']!=35100000:raise ValueError('Extended exact totals required')
    for name,value in [('PLAN.json',plan),('SUMMARY.json',summary)]:
        (output/name).write_text(canonical(value)+'\n')
    entry={'id':'private_bootstrap_native_v1','plan':receipt(root,relative+'/PLAN.json'),'summary':receipt(root,relative+'/SUMMARY.json'),
      'plan_hash':fingerprint(plan),'coverage':[proof,receipt(root,prefix+'/SOURCE_PROPOSAL.json')]}
    (packet/'MASTER_INPUT_ENTRY.json').write_text(json.dumps(entry,indent=2)+'\n')
    return summary,entry


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True)
    a=p.parse_args();s,e=prepare(a.project_root,a.packet);print(canonical({'added':s['added_counts'],'extended':s['counts'],'entry':e,'scientific_fits':0,'CUDA':0,'GO':False}))
