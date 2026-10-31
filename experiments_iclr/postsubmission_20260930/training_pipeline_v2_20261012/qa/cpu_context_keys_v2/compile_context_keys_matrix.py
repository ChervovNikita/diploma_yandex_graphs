"""Finite disabled fixed-context-key WikiCS20 packet; metadata only."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import itertools
import json
from pathlib import Path

from adapters.context_keys import ARMS,REFERENCES,SOURCE_PINS,CONTRACT,file_sha
from compile_native_matrix import bind_custody,file_binding
from partition import assign,DEFAULT_WORKERS
from prepare_node_data import verify_allocation
from spec import fingerprint,validate

SEARCH_SEEDS=(71101,71103)
COMPARATIVE_SEEDS=(71201,71203,71207)
DOMAIN=[{'learning_rate':lr,'dropout':dropout} for lr,dropout in itertools.product((.0005,.001,.002),(.1,.2,.4))]
SOURCE_METHODS={arm:'fixed_context_keys_'+arm.lower() for arm in ARMS}
REFERENCE_METHODS={'reference_ordinary_M1':'ordinary_native_single','reference_factorized_M1':'allmap_factor_single',
    'reference_original_GNNM':'original_boundary_GNNM'}
I4_REFERENCE_METHODS={'reference_ordinary_I4':'ordinary_genuine_independent4','reference_factorized_I4':'allmap_factor_genuine_independent4'}


def make_study(custody,data_prefix,source,source_files,runtime,profile):
    if custody['dataset_id']!='wikics' or {s['split_id'] for s in custody['splits']}!=set(range(20)):raise ValueError('Exact prepared WikiCS20 split roster required')
    graph=custody['model_graph']
    if graph['features']<128 or graph['classes']<2:raise ValueError('Native multiclass and tall query QR dimensions required')
    if profile.get('reservation_mib',0)<=0 or profile.get('estimated_seconds_per_update',0)<=0 or profile.get('wall_seconds',0)<=0:raise ValueError('Explicit positive provisional resource declarations required')
    rows={s['split_id']:s for s in custody['splits']};files={r['path']:r for r in custody['files']}
    def bound_input(relative):return {**files[relative],'path':data_prefix+'/'+relative}
    def inputs(split,field):return {name:bound_input(path) for name,path in rows[split][field].items()}
    for row in rows.values():
        if any(int(count)<2 for count in row['role_class_counts']['train'].values()) or len(row['role_class_counts']['train'])!=graph['classes']:
            raise ValueError('Every actual TRAIN class needs nonself classmates; no repair')
    runs=[];selections={};banks=[]
    def add(identifier,kind,method,split,seed,body,adapter,config,parents=()):
        row={'run_id':identifier,'kind':kind,'method':method,'split':split,'seed':seed,'body':body,'adapter':runtime+'/'+adapter,
            'config':config,'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':5000} if kind=='model_fit' else {},
            'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'save_every':1} if kind=='model_fit' else {},'parents':list(parents)}
        runs.append(row);return identifier
    def config(arm,split,seed,body,stage,point):
        members=4 if arm.startswith('shared4_') or arm=='reference_original_GNNM' else 1
        architecture=('original_gnnm' if arm=='reference_original_GNNM' else 'single' if arm in ('ordinaryM1_key_mixture','reference_ordinary_M1')
            else 'internal_be' if members==4 else 'factorized_single')
        return {'context_key_arm':arm,'source_arm':arm,'architecture':architecture,'backbone':'SAGE','members':members,
            'classes':graph['classes'],'target_law':'multiclass_ce','width':128,'depth':2,'weight_decay':0.,
            'context_coefficient':.05,'node_inputs':inputs(split,'node_inputs'),'context_key_sources':source_files,
            'paired_seed':seed,'initialization_seed':seed+1000003*body,'stage':stage,'fusion':'mean_logits' if arm=='reference_original_GNNM' else 'mean_probabilities',
            'query_map_initialization_law':'paired_seed+5000011; same initial bytes, separate trained objects',
            'assigned_context_law':'own run.body exactly0..3' if arm=='genuinefactorizedI4_key_assigned' else None,
            'recipe_scope':'pinned_fixed_key_mechanism_standardized_node_roles_full5000',
            'reservation_mib':profile['reservation_mib'],'estimated_seconds':profile['estimated_seconds_per_update']*5000*members,**point}
    search_arms=[a for a in ARMS if a!='genuinefactorizedI4_key_mixture']+list(REFERENCES)
    for arm in search_arms:
        candidates=[]
        for point in DOMAIN:
            config_id=fingerprint(point)[:20];identifiers=[]
            for seed in SEARCH_SEEDS:
                for body in range(4 if arm=='genuinefactorizedI4_key_assigned' else 1):
                    identifier='search-'+arm+'-'+config_id+'-s'+str(seed)+'-b'+str(body)
                    identifiers.append(add(identifier,'model_fit',SOURCE_METHODS.get(arm,REFERENCE_METHODS.get(arm)),0,seed,body,
                        'adapters/context_keys.py:make_adapter',config(arm,0,seed,body,'search',point)))
            candidates.append({'config_id':config_id,'parameters':point,'run_ids':identifiers})
        selections[arm]=add('select-'+arm,'selection',arm,0,0,0,'adapters/search.py:select_configuration',
            {'stage':'search_selection','configurations':candidates,'ranking_metric':'accuracy','ranking_direction':'max',
             'ranking_scope':'mean of own-selected body/seed accuracy; assigned law ranks all4 separately paid contexts; not pooled I4 HPO accuracy',
             'reservation_mib':256,'estimated_seconds':60},[i for c in candidates for i in c['run_ids']])
    selections['genuinefactorizedI4_key_mixture']=selections['factorizedM1_key_mixture']
    comparative=[]
    for split in range(20):
        for seed in COMPARATIVE_SEEDS:
            fitted={}
            for arm in ARMS+REFERENCES:
                fitted[arm]=[]
                for body in range(4 if arm.startswith('genuinefactorizedI4_') else 1):
                    cfg=config(arm,split,seed,body,'comparative',DOMAIN[0]);cfg['selected_config_from']=selections[arm]
                    identifier='compare-sp'+str(split)+'-'+arm+'-s'+str(seed)+'-b'+str(body)
                    fitted[arm].append(add(identifier,'model_fit',SOURCE_METHODS.get(arm,REFERENCE_METHODS.get(arm)),split,seed,body,
                        'adapters/context_keys.py:make_adapter',cfg,[selections[arm]]));comparative.append(identifier)
            for bank,arm in (('reference_ordinary_I4','reference_ordinary_M1'),('reference_factorized_I4','reference_factorized_M1')):
                fitted[bank]=fitted[arm][:]
                for body in (1,2,3):
                    cfg=config(arm,split,seed,body,'comparative',DOMAIN[0]);cfg['selected_config_from']=selections[arm]
                    identifier='compare-sp'+str(split)+'-'+bank+'-s'+str(seed)+'-b'+str(body)
                    fitted[bank].append(add(identifier,'model_fit',REFERENCE_METHODS[arm],split,seed,body,'adapters/context_keys.py:make_adapter',cfg,[selections[arm]]));comparative.append(identifier)
            endpoint_paths={name:value['path'] for name,value in inputs(split,'endpoint_inputs').items()}
            for label,fits in fitted.items():
                name=SOURCE_METHODS.get(label,REFERENCE_METHODS.get(label,I4_REFERENCE_METHODS.get(label)))
                identifier='bank-sp'+str(split)+'-'+label+'-s'+str(seed);fusion='mean_logits' if label=='reference_original_GNNM' else 'mean_probabilities'
                export=add(identifier,'ensemble_assembly',name,split,seed,0,'adapters/node_endpoints.py:export_bank',
                    {'stage':'bank_export','fit_parents':fits,'endpoint_inputs':endpoint_paths,'fusion':fusion,
                     'reservation_mib':profile['reservation_mib'],'estimated_seconds':profile.get('estimated_export_seconds',60)},fits)
                cal=add(identifier+'-cal','calibration',name,split,seed,0,'adapters/node_endpoints.py:scalar_calibration',
                    {'stage':'calibration','export_parent':export,'endpoint_inputs':endpoint_paths,'fusion':fusion,'placement':'after_serving_pool',
                     'steps':500,'learning_rate':.01,'reservation_mib':512,'estimated_seconds':10},[export])
                evaluate=add(identifier+'-evaluate','evaluation',name,split,seed,0,'adapters/node_endpoints.py:evaluate_bank',
                    {'stage':'evaluation','export_parent':export,'calibration_parent':cal,'endpoint_inputs':endpoint_paths,'fusion':fusion,
                     'reservation_mib':512,'estimated_seconds':10},[export,cal])
                banks.append({'arm':label,'split':split,'seed':seed,'fit_parents':fits,'routes':4 if len(fits)==4 or label.startswith('shared4_') or label=='reference_original_GNNM' else 1,
                    'body0_fit_reused':label in I4_REFERENCE_METHODS,'export':export,'calibration':cal,'evaluation':evaluate})
    roles={role:{str(split):inputs(split,'node_inputs' if role in ('train','select') else 'endpoint_inputs')[role if role in ('train','select') else role+'_targets'] for split in range(20)} for role in ('train','select','cal','evaluate')}
    spec={'schema_version':1,'study_id':'fixed-context-keys-wikics-v1','stage':'development','dataset':'wikics',
        'hypothesis':'Source fixed-key, live-peer, moment and shuffled-Q mechanisms versus capable ordinary/factorized M1, original GNNM and separately selected I4 on prepared WikiCS20.',
        'data':{'task':'node_classification','preprocessing':custody['request']['graph_policy'],'roles':roles,
            'prior_exposure':custody['request']['evaluation_exposure'],'evidence':{'custody_request_hash':custody['request_hash'],
                'model_graph':graph,'split_role_counts':{str(k):v['role_counts'] for k,v in rows.items()},'source_contract':CONTRACT,
                'HPO_shared_only_for_member_equivalent_mixture':True,'source_ten_arms_comparative_fits_per_seed_split':16,
                'assigned_I4_context_law':'paid body m uses only factual dictionary/Q_m; different assignments are explicit under one body-indexed law',
                'source_historical_recipe_reproduction':False,'protected_confirmation':False}},
        'source':source,'runs':runs,'comparisons':[{'references':list(REFERENCE_METHODS.values())+list(I4_REFERENCE_METHODS.values()),
            'candidates':list(SOURCE_METHODS.values()),'seed_ids':list(COMPARATIVE_SEEDS),'split_ids':list(range(20)),
            'source_aliases_preserved':True,'no_novelty_or_acceptance_guarantee':True}],
        'release':{'policy':'whole_roster','primary_endpoint':'whole_closed_raw_and_scalar_CAL_evaluations',
            'global_plan_gates_required':True,'scientific_launch_enabled':False,'human_GO_required':True},
        'resources':{'wall_seconds':profile['wall_seconds'],'profile':profile,'status':'PROVISIONAL_UNMEASURED_CONTEXT_TARGET_QUERY_CUDA_QUALIFICATION_REQUIRED'}}
    validate(spec)
    fits=[r for r in runs if r['kind']=='model_fit'];by_stage=Counter(r['config']['stage'] for r in fits)
    counts={'model_fits':len(fits),'search_model_fits':by_stage['search'],'comparative_model_fits':by_stage['comparative'],
        'optimizer_objects_per_fit':1,'physical_slow_bodies_per_fit':1,'registered_optimizer_updates':len(fits)*5000,
        'native_factual_forwards':sum(r['config']['members']*5000 for r in fits),
        'query_maps':sum(r['config']['context_key_arm'] in ARMS for r in fits),'search_selection_jobs':len(search_arms),
        'bank_exports':len(banks),'scalar_CAL_fit_jobs':len(banks),'registered_scalar_CAL_updates':500*len(banks),
        'evaluation_jobs':len(banks),'served_source_arm_banks':600,'served_reference_banks':300,'run_records':len(runs),
        'actual_model_fits':0,'actual_optimizer_updates':0,'actual_predictive_scores':0}
    return spec,banks,counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','custody','source-dir','frozen-runtime','resources','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();verify_allocation();root=args.project_root.resolve();output=args.output.resolve()
    if not output.is_relative_to(root) or output.exists():raise ValueError('Fresh confined output required')
    custody,data,bindings=bind_custody(root,args.custody)
    runtime=args.frozen_runtime.resolve();prefix=str(runtime.relative_to(root));source_files={}
    for name,pin in SOURCE_PINS.items():
        row=file_binding(root,args.source_dir/(name+'.py'))
        if row['sha256']!=pin:raise ValueError('Pinned context-key source changed: '+name)
        source_files[name]=row;bindings.append(row)
    for path in sorted(runtime.rglob('*.py')):
        if '__pycache__' not in path.parts:bindings.append(file_binding(root,path))
    bindings.append(file_binding(root,args.resources));bindings.append(file_binding(root,Path(__file__)))
    profile=json.loads(args.resources.read_text());source={'git_commit':'pinned_source_files_and_frozen_runtime_authoritative','files':bindings}
    study,banks,counts=make_study(custody,str(data.relative_to(root)),source,source_files,prefix,profile)
    keys=[study['study_id']+'/'+r['run_id'] for r in study['runs']];seconds=sum(r['config']['estimated_seconds'] for r in study['runs'])
    capacities=profile['worker_capacities_cuda_bytes']
    if set(capacities)!=set(DEFAULT_WORKERS):raise ValueError('All three planning bounds required')
    placement=assign([{'id':study['study_id'],'run_keys':keys,'study_ids':[study['study_id']],
        'maximum_cuda_bytes':profile['reservation_mib']*1024**2,'estimated_seconds':seconds,'allowed_workers':['allocation/gpu0']}],capacities)
    output.mkdir(parents=True);path=output/'STUDY_DISABLED.json';path.write_text(json.dumps(study,indent=2)+'\n')
    plan={'schema_version':1,'slots_per_gpu':2,'runtime_package':prefix,'runtime_launcher':prefix+'/launcher.py',
        'studies':[{'path':str(path.relative_to(root)),'spec_hash':fingerprint(study)}],'groups':placement['groups'],
        'barriers':{},'worker_capacities_cuda_bytes':capacities,'estimated_gpu_seconds':placement['estimated_gpu_seconds'],
        'estimates_are_measurements':False,'scientific_launch_enabled':False,'human_GO_required':True}
    summary={'schema':'disabled-fixed-context-key-packet-v1','plan_hash':fingerprint(plan),'counts':counts,
        'source_contract':CONTRACT,'status':'PREPARED_DISABLED_PENDING_CUDA_AND_ROOT_MASTER_QUALIFICATION',
        'scientific_launch_enabled':False,'human_GO_required':True,'GO_receipt_created':False,
        'training_started':False,'benchmark_optimizer_fits_executed':0,'predictive_scores_computed':0,'role_tensor_deserializations':0,
        'resources_provisional':True,'source_archived_recipe_reproduction':False,'source_semantic_scope':'ten exact equations/ownership/order controls; standardized roles/HPO/horizon',
        'blockers':['CUDA_target_dictionary_query_map_memory_rate_qualification','root_combined_master_global_closures','human_GO']}
    for name,value in (('PLAN.json',plan),('SUMMARY.json',summary),('BANKS_DISABLED.json',banks)):(output/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'counts':counts,'study_hash':fingerprint(study),'plan_hash':fingerprint(plan),'scientific_launch_enabled':False}))


if __name__=='__main__':main()
