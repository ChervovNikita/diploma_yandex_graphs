"""Prepare a finite expanded queue on allocation, without launching children."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
import statistics
import sys
from pathlib import Path
from launch_contracts import (HERE,PHASE,EXPANSION,WORKERS,guard,receipt,verify,read,write,digest,
                              require,check_host_policy,qualification_blocks,legacy)

INDEX_READY_SHA='6f9f204b5fa74c18ce67af06954a73a3427a219dc53d60c3cdd11465f94e7edd'


def cost_catalogue(cost):
    raw=read(cost['raw_cost_ledger']['path'],cost['raw_cost_ledger'])
    require(raw['complete_fits']==raw['planned_fits']==24 and len(raw['jobs'])==24,'Complete24-fit cost ledger required')
    jobs={r['job_id']:r for r in raw['jobs']}
    require(sum(r['trained_epochs'] for r in cost['fits'])==3680,'Preserve all3680 observed epochs')
    for row in cost['fits']:
        ledger=jobs[row['job_id']]
        require(row['invocation_wall_seconds']==ledger['known_invocation_wall_seconds'],'Cost copy differs from raw ledger')
        require(sum(row['optimizer_calls']['completed'])==ledger['observed_optimizer_completed'],'Optimizer cost mismatch')
        require(row['trained_epochs']==ledger['completed_epochs'],'Epoch cost mismatch')
    mappings={'native_single':'native_single','gnnm_boundary_native':'gnnm_boundary_native',
              'gnnm_graph_alignment_variance':'task_geometry_alignment_variance'}
    classes=[]
    for procedure,scientific in mappings.items():
        own=[r for r in cost['fits'] if r['procedure']==procedure]
        wall=[r['invocation_wall_seconds'] for r in own]
        per_epoch=[r['invocation_wall_seconds']/r['trained_epochs'] for r in own]
        classes.append({'preset_id':cost['preset_id'],'scientific_method_id':scientific,'observed_source_procedure':procedure,
            'sample_count':len(own),'measured_completed_fit_invocation_wall_seconds':{'min':min(wall),'median':statistics.median(wall),'max':max(wall)},
            'measured_invocation_wall_per_completed_epoch_seconds':{'min':min(per_epoch),'median':statistics.median(per_epoch),'max':max(per_epoch)},
            'observed_epochs':[r['trained_epochs'] for r in own],'observed_fit_ids':[r['job_id'] for r in own],
            'weight_kind':'observed_same_case_prior;not_target_fit_measurement','target_track':'matched_extended_only',
            'contention':cost['contention'],'runtime_reporting_generation_equivalent':False,
            'runtime_generation_and_seed_uncertainty':'Pilot and target runtime/reporting generations and cohort seeds differ; observed cost is a scheduling prior, never an ETA or target measurement',
            'mature_resource_or_pair_qualification':False})
    mismatch=[r for r in cost['fits'] if r['procedure'] not in mappings]
    return {'schema':'expanded-measured-cost-catalogue-v1','authority':cost['raw_cost_ledger'],'recorded_worker_wall_sum_seconds':raw['known_invocation_wall_seconds_sum'],
        'observed_optimizer_attempts':raw['observed_optimizer_attempts'],'observed_optimizer_completed':raw['observed_optimizer_completed'],
        'pilot_elapsed_seconds':cost['recorded_pilot_elapsed_seconds'],'worker_wall_sum_is_not_elapsed':True,
        'observed_fit_count':24,'observed_epochs':3680,'case_classes':classes,
        'source_mismatched_proxy_observations':mismatch,'proxy_observations_assigned_numeric_target_weights':0,
        'source_reference_numeric_weights':None,'unknown_cases_numeric_weight':None,'campaign_ETA':None,
        'resource_limits_inferred_from_torch_allocator_peaks':False}


def prepare(name,host_file,admission_file,placement_file=None):
    guard()
    require(name.replace('_','').isalnum() and '/'not in name,'Fresh simple output name required')
    output=HERE/name;require(not output.exists(),'Preserve prior preparation; choose fresh generation')
    hosts=read(HERE/host_file);enabled=check_host_policy(hosts)
    admissions=read(HERE/admission_file);require(admissions['schema']=='expanded-launch-admissions-v1','Explicit admission schema required')
    parent=EXPANSION/'expanded_job_binding_v1/generated_v2'
    ready=read(parent/'READINESS.json',{'path':str(parent/'READINESS.json'),'sha256':INDEX_READY_SHA})
    index=read(ready['artifacts']['index']['path'],ready['artifacts']['index'])['fits']
    banks=read(ready['artifacts']['banks']['path'],ready['artifacts']['banks'])['logical_banks']
    require(len(index)==len({r['id']for r in index})==6460,'Exact6460-fit index required')
    require(sum(r['status']=='blocked'for r in index)==90,'Retain current honest source-context blocks')
    pins=read(HERE/'REUSED_SUPERVISOR_PINS.json');old=legacy(pins)
    cost_input=read(HERE/'MOLBACE24_COST_INPUT.json')
    verify({'path':str(PHASE/cost_input['report']['path_relative_to_phase']),**{k:cost_input['report'][k]for k in ('sha256','bytes')}})
    catalogue=cost_catalogue(cost_input)
    case_by_method={r['scientific_method_id']:r for r in catalogue['case_classes']}
    # Verify fixed protocol/config and exact callback/extension semantics before cost transfer.
    target_representatives={}
    for row in index:
        if row['cell_id']==cost_input['preset_id'] and row['track']=='long_convergence' and row['method_id']in case_by_method:
            if row['method_id']not in target_representatives:target_representatives[row['method_id']]=read(row['descriptor']['path'],row['descriptor'])
    require(set(target_representatives)==set(case_by_method),'All measured comparator cases must actually exist')
    for method,job in target_representatives.items():
        require(job['recipe']['native_fixed_config']==cost_input['native_fixed_config'],'Measured comparator native settings differ')
        require(job['callback_sha256']==cost_input['callback_sha256'],'Measured comparator callback differs')
        expected=copy.deepcopy(cost_input['policy']);expected['track']='matched_extended'
        require(job['policy']==expected,'Measured pilot controller differs beyond explicitly disclosed track label')
        if method=='task_geometry_alignment_variance':
            source=next(r for r in cost_input['method_definitions']if r['id']=='gnnm_graph_alignment_variance')
            require(job['session_transform']==source['session_transform'] and job['method_settings']==source['method_settings'],'Geometry cost comparator source/settings mismatch')
    tasks=[];groups=defaultdict(list)
    for row in index:
        group=row['id'].rsplit('/',1)[0]
        task={'id':row['id'],'kind':'fit','cell_id':row['cell_id'],'family':row['family'],'protocol_track':row['track'],
              'scientific_method_id':row['method_id'],'cohort_seed':row['cohort_seed'],'initialization_seed':row['initialization_seed'],
              'split':row['split'],'body':row['body'],'science_contract_hash':row['expanded_contract_hash'],
              'source_descriptor':row['descriptor'],'source_blocked_record':row.get('blocked_record'),'runtime_job_hash':row['runtime_job_hash'],
              'source_status':row['status'],'source_blocked_reason':row.get('blocked_reason'),'group_id':group,
              'dependencies':[],'output_relative':'fits/'+digest(row['id']), 'estimated_seconds':None,'resources':None,
              'same_selected_export_bookkeeping':'actual serving and applicable same-call mean exported by this own fit;zero separate scheduler scoring jobs',
              'cost_case_prior':None}
        if row['track']=='long_convergence'and row['cell_id']==cost_input['preset_id']and row['method_id']in case_by_method:
            case=case_by_method[row['method_id']]
            task['cost_case_prior']={'measured_case_id':row['cell_id']+'/'+row['method_id'],
                'observed_median_completed_fit_wall_seconds':case['measured_completed_fit_invocation_wall_seconds']['median'],
                'sample_count':case['sample_count'],'kind':case['weight_kind'],'is_target_fit_measurement':False,'is_campaign_ETA':False}
        tasks.append(task);groups[group].append(task)
    for bank in banks:
        if not bank['id'].endswith('/native-I4'):continue
        deps=bank['fit_ids'];members=[next(r for r in index if r['id']==x)for x in deps]
        require(len(set(deps))==4 and[m['body']for m in members]==[0,1,2,3],'I4 needs four independent bodies')
        require([m['initialization_seed']-m['cohort_seed']for m in members]==[0,100003,200003,300007],'Independent native seeds changed')
        group=bank['id'].rsplit('/',1)[0]
        task={'id':bank['id']+'/aggregate-own-selected-VALID','kind':'aggregate_I4_exports','cell_id':members[0]['cell_id'],
            'protocol_track':'long_convergence','science_contract_hash':digest({'bank_id':bank['id'],'fit_contracts':[m['expanded_contract_hash']for m in members],
                'aggregation':'mean_raw_logits_after_four_own_selectors','VALID_only':True}),
            'dependencies':deps,'group_id':group,'bank_id':bank['id'],'output_relative':'banks/'+digest(bank['id']),
            'runtime_job_hash':None,'source_status':'export_handler_bridge_qualification_pending','cost_case_prior':None,'resources':None,
            'extra_training_fits':0,'model_or_dataset_reconstruction':False,'member_prediction_forward_calls':0,
            'M1_exact_reuse':deps[0],'selected_pool_reselection':False,'estimated_seconds':None}
        tasks.append(task);groups[group].append(task)
    require(len(tasks)==6730 and len(groups)==540,'Finitefit/export/group accounting changed')
    if placement_file:
        placement=read(HERE/placement_file)
        require(set(placement['group_to_worker'])==set(groups),'Placement must partition every complete cohort group')
        owner=placement['group_to_worker'];require(set(owner.values()).issubset(enabled),'No disabled/unverified host assignments')
    else:
        owner={g:enabled[int(digest(g),16)%len(enabled)]for g in groups}
    for task in tasks:
        task['worker']=owner[task['group_id']]
        admission=admissions['tasks'].get(task['id'],{})
        task['qualification_blocks']=qualification_blocks(task,admission,hosts['workers'][task['worker']])
        if admission.get('resources'):
            require(admission.get('measured_cost'),'Resource entry requires a measured scope/cost receipt')
            sys_contract=sys.modules.get('contracts')
            sys_contract.resource_contract(PHASE,admission['resources'],admission['measured_cost'])
            task['resources']=admission['resources']
        task['execution_binding']=admission.get('execution_binding')
    indexed={r['id']:r for r in tasks}
    require(len(indexed)==6730,'Unique physicalfit/export task IDs required')
    for task in tasks:
        for dep in task['dependencies']:require(indexed[dep]['worker']==task['worker'],'Keep native I4 exports local to one verified worker')
    output.mkdir()
    artifacts={}
    artifacts['cost_catalogue']=write(output/'COST_CATALOGUE.json',catalogue)
    artifacts['placement']=write(output/'PLACEMENT.json',{'schema':'expanded-group-placement-v1','group_to_worker':owner,
        'assignment_method':'explicit reviewed placement'if placement_file else'stable group-ID hash across currently assignment-enabled profiles',
        'load_balancing_or_ETA_claim':False,'science_recompiled':False,'disabled_workers_assigned':0})
    manifest={'schema':'expanded-finite-launch-skeleton-v1','status':'source_preparation_all_execution_blocked',
        'descriptor_readiness':receipt(parent/'READINESS.json'),'descriptor_index':ready['artifacts']['index'],
        'host_policy':receipt(HERE/host_file),'admissions':receipt(HERE/admission_file),'legacy_supervisor_sources':pins,
        'worker_slots_maximum':2,'workers':list(WORKERS),'assignment_enabled_workers':enabled,'tasks':tasks,
        'science_contract_hashes_preserved':True,'protocols_retained':['native_replica','long_convergence'],
        'protocol_order':'separate source-reference and matched queues;source-reference-first default is queue ordering only',
        'source_reference_independent_fits':270,'matched_native_body_fits':1080,'shared_method_fits':5110,'I4_export_only_tasks':270,
        'main_M1_extra_fit_or_scoring_tasks':0,'shared_mean_extra_scoring_tasks':0,'output_root':str(EXPANSION/'campaign_outputs/expanded_REQUIRES_REVIEW'),
        'queues_by_worker_and_protocol':{w:{track:[t['id']for t in tasks if t['worker']==w and t['protocol_track']==track]for track in('native_replica','long_convergence')}for w in WORKERS},
        'artifacts':artifacts,'human_GO':None,'source_execution_bridge_enabled':False,'automatic_retries':0,'TEST_access':False,'campaign_ETA':None}
    artifacts['manifest']=write(output/'PLAN.json',manifest)
    blocks=Counter(reason for t in tasks for reason in t['qualification_blocks'])
    summary={'schema':'expanded-launch-preparation-summary-v1','status':'DRY_RUN_SOURCE_PREPARATION_ONLY','fit_intentions':6460,'I4_export_only_tasks':270,'total_queue_tasks':6730,
        'fit_source_descriptors':6370,'source_context_blocked_fits':90,'cohort_groups':540,
        'assigned_fits_by_worker':{w:sum(t['kind']=='fit'and t['worker']==w for t in tasks)for w in WORKERS},
        'assigned_export_tasks_by_worker':{w:sum(t['kind']=='aggregate_I4_exports'and t['worker']==w for t in tasks)for w in WORKERS},
        'all_execution_blocked':all(t['qualification_blocks']for t in tasks),'qualification_block_counts':dict(blocks),
        'target_fits_with_measured_case_priors':sum(t['cost_case_prior']is not None for t in tasks),
        'target_fits_with_actual_measured_costs':0,'numeric_source_mismatched_proxy_weights':0,'observed_pilot_cost_fits':24,
        'observed_pilot_worker_wall_seconds':catalogue['recorded_worker_wall_sum_seconds'],'observed_pilot_optimizer_transitions':catalogue['observed_optimizer_completed'],
        'campaign_ETA':None,'load_balance_claim':False,'resource_profiles_qualified':0,'children_started':0,'models_constructed':0,'TEST_accessed':False,
        'source_or_data_tensors_opened':0,'server77_contacted':False,'source_runtime_protocol_changes':False,'artifacts':artifacts}
    summary_pin=write(output/'SUMMARY.json',summary)
    return {'summary':summary_pin,'manifest':artifacts['manifest'],'fits':6460,'I4_export_only_tasks':270,'measured_case_prior_target_fits':summary['target_fits_with_measured_case_priors'],'children_started':0}


if __name__=='__main__':
    import sys
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-name',default='generated_v1');p.add_argument('--hosts',default='HOST_POLICY.json');p.add_argument('--admissions',default='ADMISSIONS_TEMPLATE.json');p.add_argument('--placement')
    a=p.parse_args();import json;print(json.dumps(prepare(a.output_name,a.hosts,a.admissions,a.placement),sort_keys=True))
