"""Disabled native GNCL/retrieval/decoder matrix; source/custody metadata only."""
import argparse
from collections import Counter
import copy,gzip,json
from pathlib import Path
from adapters.common_native_wrapper import DATASETS, SOURCE_ANCHOR, receipt
from adapters.native_combinations import CONTRACT,FAMILIES,CONTROLS,CONTROL_I4,arm_shape,render_sources
from spec import canonical,fingerprint,validate,confined
import hashlib
DOMAIN=[{'learning_rate':lr,'dropout':dropout} for lr in (.0005,.001,.002) for dropout in (.1,.2,.4)]
SEARCH_SEEDS=(88101,88103)
COMPARATIVE_SEEDS=(88201,88203,88207)
CONTROL_METHODS={'ordinary_M1':'ordinary_native_single','ordinary_I4':'ordinary_genuine_independent4',
 'factorized_allmap_M1':'allmap_factor_single','equipped_I4':'allmap_factor_genuine_independent4',
 'shared4_unchanged':'allmap_factor_shared4','original_gnnm':'original_boundary_GNNM'}
CANONICAL={'gncl_supcon':{'shared4_GNCL':'gncl_supcon_shared4_gncl','shared4_own_SupCon':'gncl_supcon_shared4_own_supcon',
 'shared4_GNCL_SupCon':'gncl_supcon_shared4_gncl_supcon','ordinary_M1_SupCon':'gncl_supcon_ordinary_m1_supcon',
 'factorized_M1_SupCon':'gncl_supcon_factorized_m1_supcon','genuine_factorized_I4_SupCon':'gncl_supcon_genuine_factorized_i4_supcon'},
 'nonlocal_label_retrieval':{k:'nonlocal_label_retrieval_'+k.lower() for k in (*FAMILIES['nonlocal_label_retrieval']['arms'],*FAMILIES['nonlocal_label_retrieval']['i4'])},
 'shared_recurrent_class_decoder':{k:'shared_recurrent_class_decoder_'+k.lower() for k in (*FAMILIES['shared_recurrent_class_decoder']['arms'],*FAMILIES['shared_recurrent_class_decoder']['i4'])}}
def method(family,arm):return CONTROL_METHODS[arm] if arm in CONTROL_METHODS else CANONICAL[family][arm]

def bind_metadata(root, custody_path, receipt_path):
    """Verify raw copied metadata and preserve original payload receipts unread."""
    (custody_path, receipt_path) = (Path(custody_path), Path(receipt_path))
    original = json.loads(receipt_path.read_text())
    raw = custody_path.read_bytes()
    if len(raw) != original['bytes'] or hashlib.sha256(raw).hexdigest() != original['sha256']:
        raise ValueError('Copied custody bytes differ from allocation receipt')
    custody = json.loads(raw)
    dataset = custody['dataset_id']
    expected = {'wikics': (11701, 300, 10, 442907), 'amazon-photo': (7650, 745, 8, 245812)}
    if dataset not in DATASETS or custody.get('schema') != 'node-provider-role-custody-v1' or custody.get('status') != 'prepared' or (custody.get('benchmark_fits') != 0) or (custody.get('predictive_scores') != 0) or (tuple((custody['model_graph'][k] for k in ('nodes', 'features', 'classes', 'edges'))) != expected[dataset]) or (custody['model_graph']['self_edges'] != expected[dataset][0]) or (custody['provider_pyg_version'] != '2.7.0') or (sorted((row['split_id'] for row in custody['splits'])) != list(range(DATASETS[dataset]))):
        raise ValueError('Exact prepared homogeneous complete WikiCS20/Photo10 provider roster required')
    original_path = Path(original['path'])
    if original_path != Path('training_pipeline_v2_20261012/prepared_data') / dataset / 'CUSTODY.json':
        raise ValueError('Original allocation custody path differs')
    data_dir = original_path.parent
    files = custody['files']
    if len({row['path'] for row in files}) != len(files):
        raise ValueError('Ambiguous payload receipt paths')
    for row in files:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or type(row['bytes']) is not int or (row['bytes'] < 1) or (len(row['sha256']) != 64):
            raise ValueError('Invalid metadata payload receipt')
    bindings = [receipt(root, custody_path), receipt(root, receipt_path)]
    deferred = [{'path': str(data_dir / row['path']), 'bytes': row['bytes'], 'sha256': row['sha256']} for row in files]
    return (custody, data_dir, bindings, deferred)

def make_study(custody,data_dir,family,backbone,source,sources,runtime,reuse_body0=False):
    dataset=custody['dataset_id'];split_rows={row['split_id']:row for row in custody['splits']};files={row['path']:row for row in custody['files']}
    for row in split_rows.values():
        if min(row['role_class_counts']['train'].values())<2:raise ValueError('Every complete TRAIN class needs at least two examples')
    def inputs(split,field):return {key:{'path':str(data_dir/rel),'bytes':files[rel]['bytes'],'sha256':files[rel]['sha256']} for key,rel in split_rows[split][field].items()}
    def common(arm,split,stage):
        members,bodies=arm_shape(family,arm)
        architecture='original_gnnm' if arm=='original_gnnm' else 'joint_untied' if bodies==4 else 'single' if arm.startswith('ordinary_M1') else 'factorized_single' if members==1 else 'internal_be'
        return {'combination_family':family,'combination_arm':arm,'backbone':backbone,'architecture':architecture,'members':members,
            'physical_slow_bodies':bodies,'source_optimizer_count':1,'depth':2,'width':128,'rank':16,'classes':custody['model_graph']['classes'],
            'train_count':split_rows[split]['role_counts']['train'],'weight_decay':0.,'provider_graph':custody['model_graph'],
            'wrapper_sources':sources,'node_inputs':inputs(split,'node_inputs'),'is_original_gnnm':arm=='original_gnnm',
            'fusion':'mean_logits' if arm=='original_gnnm' else 'mean_probabilities','stage':stage,'source_contract':CONTRACT,
            'reservation_mib':32768,'estimated_seconds':1,'scheduling_weight_status':'unmeasured legacy seconds placeholder; no ETA/admission',
            'resource_evidence':'root qualification/three-GPU assignment pending','recipe_scope':'literal source constructor/init/TRAIN; standardized provider/HPO/controller adaptation'}
    runs,selections,banks=[],{},[];arms=FAMILIES[family]['arms']+CONTROLS;i4={**FAMILIES[family]['i4'],**CONTROL_I4}
    def add(identifier,kind,arm,split,seed,body,config,parents=()):
        entry='native_combinations.py:make_adapter' if kind=='model_fit' else 'search.py:select_configuration' if kind=='selection' else 'node_endpoints.py:'+{'ensemble_assembly':'export_bank','calibration':'scalar_calibration','evaluation':'evaluate_bank'}[kind]
        runs.append({'run_id':identifier,'kind':kind,'method':method(family,arm),'split':split,'seed':seed,'body':body,
            'adapter':runtime+'/adapters/'+entry,'config':config,'parents':list(parents),
            'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':5000} if kind=='model_fit' else {},
            'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':5000} if kind=='model_fit' else {}})
        return identifier
    for arm in arms:
        candidates=[]
        for point in DOMAIN:
            config_id=fingerprint(point)[:20]
            ids=[add('search-'+arm+'-'+config_id+'-s'+str(seed),'model_fit',arm,0,seed,0,{**common(arm,0,'search'),**point}) for seed in SEARCH_SEEDS]
            candidates.append({'config_id':config_id,'parameters':point,'run_ids':ids})
        selections[arm]=add('select-'+arm,'selection',arm,0,0,0,{'stage':'search_selection','configurations':candidates,'ranking_metric':'accuracy','ranking_direction':'max','reservation_mib':1,'estimated_seconds':1},[rid for row in candidates for rid in row['run_ids']])
    for split in sorted(split_rows):
      for seed in COMPARATIVE_SEEDS:
        fits={arm:[add('compare-sp'+str(split)+'-'+arm+'-s'+str(seed),'model_fit',arm,split,seed,0,
            {**common(arm,split,'comparative'),**SOURCE_ANCHOR,'selected_config_from':selections[arm]},[selections[arm]])] for arm in arms}
        for bank_arm,arm in i4.items():
            # Full constructor/init/TRAIN/HPO/SELECT/budget equality proves
            # body0 is the paired M1 checkpoint; bodies1-3 stay fresh/disjoint.
            fits[bank_arm]=([fits[arm][0]] if reuse_body0 else []) + [add('compare-sp'+str(split)+'-'+bank_arm+'-s'+str(seed)+'-b'+str(body),'model_fit',arm,split,seed,body,
                {**common(arm,split,'comparative'),**SOURCE_ANCHOR,'selected_config_from':selections[arm]},[selections[arm]]) for body in range(1 if reuse_body0 else 0,4)]
        for arm,parents in fits.items():
            single=i4.get(arm,arm);fusion=common(single,split,'comparative')['fusion'];endpoint_inputs={key:row['path'] for key,row in inputs(split,'endpoint_inputs').items()}
            prefix='bank-sp'+str(split)+'-'+arm+'-s'+str(seed)
            export=add(prefix,'ensemble_assembly',arm,split,seed,0,{'stage':'bank_export','fit_parents':parents,'endpoint_inputs':endpoint_inputs,'fusion':fusion,'reservation_mib':32768,'estimated_seconds':1},parents)
            cal=add(prefix+'-cal','calibration',arm,split,seed,0,{'stage':'calibration','export_parent':export,'endpoint_inputs':endpoint_inputs,'fusion':fusion,'placement':'after_serving_pool','steps':500,'learning_rate':.01,'reservation_mib':512,'estimated_seconds':1},[export])
            eva=add(prefix+'-evaluate','evaluation',arm,split,seed,0,{'stage':'evaluation','export_parent':export,'calibration_parent':cal,'endpoint_inputs':endpoint_inputs,'fusion':fusion,'reservation_mib':512,'estimated_seconds':1},[export,cal])
            banks.append({'arm':arm,'canonical_id':method(family,arm),'split':split,'seed':seed,'fit_parents':parents,'export':export,'calibration':cal,'evaluation':eva,'genuine_I4':arm in i4,'body0_reused':arm in i4 and reuse_body0,'served_views':4 if arm in i4 else arm_shape(family,single)[0]})
    roles={role:{str(split):inputs(split,'node_inputs' if role in ('train','select') else 'endpoint_inputs')[role if role in ('train','select') else role+'_targets'] for split in sorted(split_rows)} for role in ('train','select','cal','evaluate')}
    study={'schema_version':1,'study_id':'native-combinations-v1-'+family.replace('_','-')+'-'+dataset+'-'+backbone.lower(),'stage':'development','dataset':dataset,
        'hypothesis':'Source '+family+' procedures can change served new-graph generalization relative to native, factorized, independent, unchanged shared and literal original GNNM controls.',
        'data':{'task':'node_classification','preprocessing':custody['request']['graph_policy'],'roles':roles,'prior_exposure':custody['request']['evaluation_exposure'],
            'evidence':{'custody_request_hash':custody['request_hash'],'model_graph':custody['model_graph'],'split_roster':sorted(split_rows),'search_split':0,'source_exact_training_recipe':False}},
        'source':source,'runs':runs,'comparisons':[{'references':list(CONTROL_METHODS.values()),'candidates':list(CANONICAL[family].values()),
            'scope':'every complete registered mask and fixed comparative seed; retain outcomes/costs/failures','interpretation':'development; masks/seeds are not independent graphs; no acceptance or accuracy guarantee'}],
        'release':{'policy':'whole_roster','primary_endpoint':'all raw and after-pool CAL500 banks','global_plan_gates_required':True},
        'resources':{'wall_seconds':1,'status':'UNMEASURED_DISABLED_CANDIDATE','wall_seconds_is_placeholder':True}}
    validate(study);fits=[r for r in runs if r['kind']=='model_fit'];stages=Counter(r['config']['stage'] for r in fits)
    counts={'optimizer_starts_by_stage':dict(stages),'atomic_fit_groups':len(fits),'paid_optimizer_objects':len(fits),
        'physical_body_instances':sum(r['config']['physical_slow_bodies'] for r in fits),'full_budget_optimizer_updates':len(fits)*5000,
        'logical_backward_calls':sum(4 if r['config']['is_original_gnnm'] else 1 for r in fits)*5000,
        'full_view_native_TRAIN_forwards':sum(r['config']['members'] for r in fits)*5000,'search_selection_jobs':len(selections),
        'comparative_banks':len(banks),'bank_exports':len(banks),'scalar_CAL_fits':len(banks),'scalar_CAL_optimizer_updates':len(banks)*500,
        'evaluation_jobs':len(banks),'genuine_I4_banks':sum(r['genuine_I4'] for r in banks),'reused_I4_body0':sum(r['body0_reused'] for r in banks),'all_run_records':len(runs),
        'actual_optimizer_updates':0,'actual_predictive_scores':0}
    source_costs={'decoder_support_preparations':sum(r['config']['combination_arm'] not in CONTROLS for r in fits) if family=='shared_recurrent_class_decoder' else 0,
        'query_draws':sum(r['config']['combination_arm'] not in CONTROLS for r in fits)*5000 if family=='nonlocal_label_retrieval' else 0,
        'decoder_train_route_steps':sum(5*r['config']['members'] for r in fits if r['config']['combination_arm'] not in CONTROLS)*5000 if family=='shared_recurrent_class_decoder' else 0,
        'cost_policy':'count support/query/native forwards/Gram/similarity/recurrent/selection/export and scalar CAL; tensor sizes in runtime counts; time/resource qualification pending'}
    return study,counts,banks,source_costs


def compile_packet(root,packet,output,reuse_body0=False):
    root,packet,output=map(lambda p:Path(p).resolve(),(root,packet,output))
    if not packet.is_relative_to(root) or not output.is_relative_to(packet) or output.exists():raise ValueError('Fresh owned output required')
    bindings=json.loads((packet/'SOURCE_BINDINGS.json').read_text());runtime=str((packet/'runtime').relative_to(root))
    source_files=[receipt(root,p) for p in sorted((packet/'runtime').rglob('*.py'))]+list(bindings.values())+[receipt(root,packet/'SOURCE_BINDINGS.json')]
    output.mkdir();(output/'studies').mkdir();studies,prepared,banks,adaptations=[],[],[],[];totals,stages=Counter(),Counter()
    for dataset in DATASETS:
        custody,data_dir,metadata,deferred=bind_metadata(root,packet/'metadata'/dataset/'CUSTODY.json',packet/'metadata'/dataset/'RECEIPT.json')
        source={'git_commit':'PENDING_ROOT_SOURCE_COMMIT_BINDING','source_commit_status':'pending_root_owned','files':source_files+metadata+deferred,
            'working_tree_fingerprints_authoritative':True,'frozen_runtime_package':runtime,'official_author_protocol_replication':False,
            'payload_bindings_status':'original custody declarations; payloads unread/unverified locally; root allocation admission required'}
        for family,row in FAMILIES.items():
            # Metadata-only complete TRAIN class checks; actual payload validation remains admission-time.
            for split in custody['splits']:
                adaptations.append({'dataset':dataset,'split':split['split_id'],**render_sources(root,bindings,family,custody['model_graph']['features'],custody['model_graph']['classes'],split['role_counts']['train'])})
            for backbone in row['backbones']:
                study,counts,study_banks,costs=make_study(custody,data_dir,family,backbone,copy.deepcopy(source),bindings,runtime,reuse_body0)
                rel=str((output/'studies'/(study['study_id']+'.json.gz')).relative_to(root));(root/rel).write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
                studies.append({'path':rel,'spec_hash':fingerprint(study)});prepared.append({'study_id':study['study_id'],'dataset':dataset,'family':family,'backbone':backbone,'counts':counts,'source_costs':costs})
                banks.extend({'study_id':study['study_id'],**r} for r in study_banks);totals.update({k:v for k,v in counts.items() if type(v) is int});stages.update(counts['optimizer_starts_by_stage'])
    totals['optimizer_starts_by_stage']=dict(stages)
    assert len(studies)==14 and totals['atomic_fit_groups']==(13230 if reuse_body0 else 15120) and totals['physical_body_instances']==(14364 if reuse_body0 else 16254) and totals['comparative_banks']==7290 and totals['genuine_I4_banks']==1890 and totals['reused_I4_body0']==(1890 if reuse_body0 else 0)
    prefix=str(output.relative_to(root));groups=[{'id':r['study_id'],'gpu_assignment':'allocation/gpu0','allowed_workers':['allocation/gpu0'],'maximum_cuda_bytes':32768*1024**2,
        'estimated_seconds':r['counts']['all_run_records'],'assignment_status':'preview placeholder; root final staging pending',
        'run_keys':[r['study_id']+'/'+run['run_id'] for run in json.loads(gzip.decompress((root/path['path']).read_bytes()))['runs']]} for r,path in zip(prepared,studies)]
    plan={'schema_version':1,'slots_per_gpu':2,'torch_threads_per_fit':2,'output_root':prefix+'/execution','runtime_launcher':runtime+'/launcher.py','runtime_package':runtime,
        'worker_capacities_cuda_bytes':{'allocation/gpu0':78*1024**3},'studies':studies,'groups':groups,'global_gates':{},'barriers':{},
        'scientific_launch_enabled':False,'human_GO_required':True,'scientific_roster_approved':False,'resource_admission_status':'pending_root_qualification',
        'source_commit_status':'pending_root_owned','preparation_scope':'17 canonical native source conditions; prospective compatible extensions; roster pending',
        'resource_estimates_are_measurements':False,'scheduling_weights_are_ETA':False,
        'host_bindings':{'allocation':{'status':'root_owned_live_verification_pending','hostname':'anogena-2-0','gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac']}}}
    for name,src,target in [('whole_search',['search','search_selection'],['comparative']),('whole_comparative',['comparative'],['bank_export','calibration','evaluation']),('whole_calibration',['calibration'],['evaluation'])]:
        plan['global_gates'][name]={'path':prefix+'/gate_receipts/'+name+'.json','source_stages':src,'required_for_stages':target,'required_hosts':['allocation'],
            'receipt_contract':'status=complete;plan_hash=current_plan_hash;closed_phase=gate_name;host_receipts_keys=required_hosts'}
    summary={'schema':'native-combinations-disabled-preparation-v1','status':'DISABLED_SOURCE_RESOURCE_ROSTER_MASTER_AND_GO_PENDING','plan_hash':fingerprint(plan),'counts':dict(totals),
        'studies':prepared,'banks':banks,'domain':DOMAIN,'source_anchor':SOURCE_ANCHOR,'source_anchor_domain_membership_count':DOMAIN.count(SOURCE_ANCHOR),
        'search_seed_ids':list(SEARCH_SEEDS),'comparative_seed_ids':list(COMPARATIVE_SEEDS),'source_contract':CONTRACT,'canonical_conditions':CANONICAL,
        'newly_scoped_canonical_methods':17,'exact_member0_reuse_enabled':reuse_body0,'I4_HPO_policy':'component M1 HPO; four distinct own-selected comparative bodies; source body0 trajectory equals paired M1 and is reused iff exact-body0-reuse flag is set; no pooled I4 HPO claim',
        'scientific_training_started':False,'GO_created':False,'real_tensors_opened':False,'git_operations':False,'source_exact_recipe':False,
        'controls_reuse_status':'paid self-contained controls now; candidates for exact full-protocol coalescing with common v2 before root final roster; no external artifact reuse'}
    for name,value in [('PLAN.json',plan),('SUMMARY.json',summary),('PROVIDER_AST_BINDINGS.json',adaptations)]: (output/name).write_text(canonical(value)+'\n')
    return {'counts':dict(totals),'plan_hash':fingerprint(plan),'studies':len(studies),'scientific_fits':0,'GO':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('project-root','packet','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--exact-body0-reuse',action='store_true')
    args=parser.parse_args();print(canonical(compile_packet(args.project_root,args.packet,args.output,args.exact_body0_reuse)))
