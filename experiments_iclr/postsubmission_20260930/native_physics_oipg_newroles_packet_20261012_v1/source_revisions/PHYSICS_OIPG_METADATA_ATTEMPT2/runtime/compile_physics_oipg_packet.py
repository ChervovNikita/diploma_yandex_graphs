"""Finite disabled Physics10 + exact OIPG4 + packed coupled source roster.

Standard library metadata only; no graph, role arrays or numerical packages load.
"""
import argparse,copy,csv,gzip,json,subprocess
from pathlib import Path
from compile_master_plan import receipt,graph_check,typed_counts,reference_matches,bank_roster,load_packet
from spec import canonical,fingerprint,validate
from tabm_stopping import budget,selection

PHYSICS={'O4_stem_IID':'physics_joint_dropout_o4_stem_iid','O4_stem_SIGN':'physics_joint_dropout_o4_stem_sign',
 'O4_graph_IID':'physics_joint_dropout_o4_graph_iid','O4_graph_SIGN':'physics_joint_dropout_o4_graph_sign',
 'O4_graph_BALANCED':'physics_joint_dropout_o4_graph_balanced','factor_M1_stem':'boundary_factor_single',
 'factor_M1_graph':'boundary_factor_single','MS4_IID':'physics_joint_dropout_factor_m1_ms4_graph_iid',
 'MS4_SIGN':'physics_joint_dropout_factor_m1_ms4_graph_sign','coupled_I4':'physics_joint_dropout_unshared_coupled_i4',
 'original_gnnm':'control_Physics_native_original_GNNM','source_meanSAGE':'control_Physics_source_meanSAGE'}
OIPG={p:'internal_be_allocation_'+p.lower() for p in 'OIPG'}
COUPLED='ordinary_fourbody_coupled_alignment_residual'
PHYSICS_CANONICAL=list(dict.fromkeys(list(PHYSICS.values())[:10]+['boundary_factor_genuine_independent4']))
CANONICAL=PHYSICS_CANONICAL+list(OIPG.values())+[COUPLED]
WIKI_SEEDS=(111601,111603,111607)
SEARCH_SEED_OFFSET=210000011

def source_bindings(root,packet):
 p=packet/'sources';physics={k:receipt(root,p/'physics'/n) for k,n in {'native':'native_models.py','author':'author_model.py',
  'joint':'joint_forward.py','samplers':'samplers.py','independent':'independent_controller.py'}.items()}
 oipg={'portable':receipt(root,p/'portable_internal_be_public_interface_20261007_v2/portable.py'),
  'method':receipt(root,p/'public_internal_be_allocation_controls_20261007_v1/method.py'),
  'native':receipt(root,p/'native_polynormer/model.py'),
  'auditor_directory':str((p/'public_internal_be_private_steering_adapter_20261007_v1').relative_to(root)),
  'files':[receipt(root,q) for q in sorted(p.rglob('*')) if q.is_file()]}
 return physics,oipg

def new_study(root,packet,dataset,custody,source_files,commit):
 runtime=str(packet.relative_to(root))+'/runtime';runs=[];banks=[]
 def add(identifier,kind,method,split,seed,body,config,parents=(),native_arm=None):
  if kind=='model_fit':factory='physics_joint.py:make_adapter' if native_arm else 'oipg.py:make_adapter'
  elif kind=='selection':factory='oipg.py:choose_lambda'
  else:factory='source_endpoints.py:'+{'ensemble_assembly':'export_bank','calibration':'scalar_calibration','evaluation':'evaluate_bank'}[kind]
  b=budget(native_arm) if native_arm else {'unit':'epochs' if config.get('oipg_policy')=='fourbody_coupled' else 'optimizer_updates','policy':'full_budget','maximum':1100}
  sel=selection(native_arm) if native_arm else {'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':1101}
  runs.append({'run_id':identifier,'kind':kind,'method':method,'split':split,'seed':seed,'body':body,'adapter':runtime+'/adapters/'+factory,
   'config':config,'budget':b if kind=='model_fit' else {},'selection':sel if kind=='model_fit' else {},'parents':list(parents)})
  return identifier
 def endpoints(name,method,split,seed,fit_ids,fusion,arm):
  inp=copy.deepcopy(custody['splits'][split]['endpoint_inputs']);paths={k:r['path'] for k,r in inp.items()}
  base={'endpoint_inputs':paths,'endpoint_input_receipts':inp,'fusion':fusion,'resource_profile_status':'UNQUALIFIED_PROVISIONAL_ROOT_ADMISSION_PENDING'}
  export=add(name,'ensemble_assembly',method,split,seed,0,{**base,'stage':'bank_export','fit_parents':fit_ids,'reservation_mib':32768,'estimated_seconds':30.},fit_ids)
  cal=add(name+'-cal','calibration',method,split,seed,0,{**base,'stage':'calibration','export_parent':export,'placement':'after_serving_pool',
   'steps':500,'learning_rate':.01,'reservation_mib':512,'estimated_seconds':10.,'CAL_recipe_authority':'literal_source_Physics' if dataset=='physics' else 'prospective_common_scalar_CAL_after_native_mean_probability_pool'},[export])
  evaluation=add(name+'-evaluate','evaluation',method,split,seed,0,{**base,'stage':'evaluation','export_parent':export,'calibration_parent':cal,
   'reservation_mib':512,'estimated_seconds':10.},[export,cal])
  banks.append({'method':method,'arm':arm,'split':split,'seed':seed,'fit_parents':fit_ids,'export':export,'calibration':cal,'evaluation':evaluation,
   'genuine_I4':len(fit_ids)==4,'coupled_fourbody_not_genuine_I4':arm in ('coupled_I4','fourbody_coupled')})
  return export,cal,evaluation
 roles={r:{str(s['split_id']):s['roles'][r] for s in custody['splits']} for r in ('train','select','cal','evaluate')}
 study={'schema_version':1,'study_id':'native-Physics-joint-dropout-newroles-v1' if dataset=='physics' else 'native-WikiCS-OIPG-coupled-newroles-v1',
  'stage':'development','dataset':'Coauthor-Physics' if dataset=='physics' else 'WikiCS',
  'hypothesis':'Literal source procedures compared with fresh own M1, genuinely separately selected I4 and compatible original GNNM on encountered development graph roles; all source budgets and failures retained.',
  'data':{'task':'node_classification','preprocessing':'Complete immutable raw-verified graph; pinned PyG positive-entry feature binarization for Physics, literal raw WikiCS features/support; no feature normalization added.',
   'roles':roles,'prior_exposure':custody['evaluation_scope'],'evidence':{'actual_custody':receipt(root,packet/('DATASET_'+dataset+'_CUSTODY.json')),
    'graph':custody['graph'],'provider_files':[custody['raw']],'geometry':custody['geometry'],'fresh_roles':True,'official_raw_masks_used':False,'confirmation_eligibility':False}},
  'source':{'git_commit':commit,'files':source_files,'working_tree_fingerprints_authoritative':True,'frozen_runtime_package':runtime,
   'data_bindings_rechecked_by_adapter_and_endpoint_before_deserialization':True,'official_author_protocol_replication':False},
  'runs':runs,'comparisons':[],'release':{'policy':'whole_roster','primary_endpoint':'All raw native serving and fixed CAL500 proper-score banks; complete case/seed cohorts with source bounds and failures.',
   'global_plan_gates_required':True,'unsupported_source_exporter_keeps_full_selected_states':True},
  'resources':{'wall_seconds':172800,'status':'DISABLED_UNQUALIFIED_ROOT_RESOURCE_MASTER_AND_GO_PENDING','finite_window_is_not_ETA':True}}
 return study,runs,banks,add,endpoints

def physics_config(arm,split,seed,body,custody,sources):
 source=arm=='source_meanSAGE';coupled=arm=='coupled_I4';members=4 if arm.startswith('O4') or arm in ('coupled_I4','original_gnnm') else 1
 return {'physics_arm':arm,'physics_sources':sources,'physics_placement':'stem_pre_gelu' if 'stem' in arm or arm=='original_gnnm' else 'first_normalized_sage_input',
  'architecture':'source_TABM_joint_dropout','backbone':'SAGE','depth':2,'width':32 if source else 128,'dropout':.4 if source else .2,
  'classes':custody['geometry'][2],'learning_rate':.001,'weight_decay':0,'members':members,'physical_slow_bodies':4 if coupled else 1,
  'source_optimizer_count':4 if coupled else 1,'source_TRAIN_forwards_per_round':4 if arm.startswith('O4') or arm.startswith('MS4') or arm in ('coupled_I4','original_gnnm') else 1,
  'initialization_seed':seed+1000003*body,'historical_state_inherited':False,'fusion':'mean_logits','node_inputs':copy.deepcopy(custody['splits'][split]['node_inputs']),
  'stage':'comparative','is_original_gnnm':arm=='original_gnnm','source_recipe':'literal_TABM_per_ten_saved_best_p300_cap5000' if not source else 'literal_meanSAGE_width32_TF1Adam_L2_latest_total_SELECT_p50_cap100000',
  'reservation_mib':32768,'estimated_seconds':300.,'resource_profile_status':'UNQUALIFIED_PROVISIONAL_ROOT_ADMISSION_PENDING'}

def oipg_config(policy,strength,split,seed,body,custody,sources,stage='comparative'):
 native=policy=='native_M1';coupled=policy=='fourbody_coupled';members=1 if native else 4;owners=4 if coupled else 1
 return {'oipg_policy':policy,'base_arm':'single' if native else 'independent4_contrastive' if coupled else 'be_unit','allocation_lambda':strength,
  'oipg_sources':sources,'architecture':'source_internal_BE_OIPG','backbone':'native_WikiPolynormer','width':512,'depth':7,'global_depth':2,'classes':10,
  'learning_rate':.001,'weight_decay':0,'members':members,'physical_slow_bodies':owners,'source_optimizer_count':owners,'source_local_epochs':100,
  'source_TRAIN_forwards_per_update':2*members,'initialization_seed':seed+1009*body,'historical_state_inherited':False,'fusion':'mean_probabilities',
  'node_inputs':copy.deepcopy(custody['splits'][split]['node_inputs']),'stage':stage,'source_recipe':'literal_native1100epoch_100local_localbest_optimizer_restore_live_RNG_1000global',
  'objective_log_authority':'common_own_plus_auxiliary_monitor; source_component_scalars_retained; mixed_permission_vector_field_not_asserted_to_be_one_global_scalar_objective',
  'coupled_recipe':{'alignment_weight':.05,'residual_weight':.05,'temperature':.2,'max_objects':512,'own_gradients_unscaled':True,'source_auxiliary_multiplier':4} if coupled else None,
  'reservation_mib':32768,'estimated_seconds':300.,'resource_profile_status':'UNQUALIFIED_PROVISIONAL_ROOT_ADMISSION_PENDING'}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--project-root',required=True);parser.add_argument('--packet',required=True);args=parser.parse_args()
 root=Path(args.project_root).resolve();packet=root/args.packet;matrix=packet/'matrix'
 if matrix.exists():raise FileExistsError('Preserve prior metadata attempt; fresh packet or archived provisional revision required')
 custody=json.loads((packet/'DATA_CUSTODY.json').read_text())['datasets'];ps,os=source_bindings(root,packet)
 files=[receipt(root,p) for folder in ('runtime','sources') for p in sorted((packet/folder).rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.txt','.sh')]
 files += [receipt(root,packet/n) for n in ('DESIGN_DECISION.json','DATA_INPUT_BINDINGS.json','DATA_CUSTODY.json','DATASET_physics_CUSTODY.json','DATASET_wikics_CUSTODY.json','SOURCE_SNAPSHOT_RECEIPTS.json','SYNTHETIC_CPU_PROOF.json','CONTROLLER_TYPE_AMENDMENT.json','CANONICAL_SOURCE_COMPLETION.json')]
 files=list({r['path']:r for r in files}.values());commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
 studies=[];all_banks=[]
 study,runs,banks,add,endpoints=new_study(root,packet,'physics',custody['physics'],files,commit)
 for split,s in enumerate(custody['physics']['splits']):
  seed=s['source_block'];fits={}
  for arm,method in PHYSICS.items():
   fits[arm]=[add(f'compare-sp{split}-{arm}-s{seed}-b{body}','model_fit',method,split,seed,body,physics_config(arm,split,seed,body,custody['physics'],ps),native_arm=arm)
    for body in range(4 if arm.startswith('factor_M1') else 1)]
  for arm,method in PHYSICS.items():endpoints(f'bank-sp{split}-{arm}-s{seed}',method,split,seed,[fits[arm][0]],'mean_logits',arm)
  for arm in ('factor_M1_stem','factor_M1_graph'):endpoints(f'bank-sp{split}-genuine-I4-{arm}-s{seed}','boundary_factor_genuine_independent4',split,seed,fits[arm],'mean_logits',arm+'_genuine_I4')
 study['comparisons']=[{'candidates':PHYSICS_CANONICAL,'references':['boundary_factor_single','boundary_factor_genuine_independent4','control_Physics_native_original_GNNM','control_Physics_source_meanSAGE'],
  'scope':'Literal14banks x3fresh role/source blocks; exact new M1 body0 is reused by corresponding equipped I4; native meanSAGE and TABM backbones/horizons differ and remain disclosed.'}]
 studies.append(study);all_banks+=banks
 study,runs,banks,add,endpoints=new_study(root,packet,'wikics',custody['wikics'],files,commit)
 for split in range(10):
  for seed in WIKI_SEEDS:
   fit_banks=[]
   for policy in 'OIPG':
    parents=[];config=oipg_config(policy,0. if policy=='O' else .05,split,seed,0,custody['wikics'],os)
    if policy!='O':
     cells=[]
     for strength in (.05,.5):
      label='lambda005' if strength==.05 else 'lambda05';search_seed=seed+SEARCH_SEED_OFFSET
      rid=add(f'search-sp{split}-{policy}-{label}-s{seed}','model_fit',OIPG[policy],split,search_seed,0,oipg_config(policy,strength,split,search_seed,0,custody['wikics'],os,'search'))
      cells.append({'config_id':label,'parameters':{'allocation_lambda':strength},'run_ids':[rid]})
     choice=add(f'choice-sp{split}-{policy}-s{seed}','selection',OIPG[policy],split,seed,0,{'stage':'search_selection','configurations':cells,'lambda_grid':[.05,.5],
      'prospective_numeric_choice':True,'both_candidates_full_source_horizon_required':True,'ranking':'SELECT_accuracy; exact ties choose smaller_lambda','reservation_mib':128,'estimated_seconds':1.},[r for cell in cells for r in cell['run_ids']])
     parents=[choice];config['selected_config_from']=choice
    rid=add(f'compare-sp{split}-{policy}-s{seed}','model_fit',OIPG[policy],split,seed,0,config,parents)
    fit_banks.append((policy,OIPG[policy],[rid],'mean_probabilities'))
    if policy!='O':
     zero=add(f'compare-sp{split}-{policy}-lambda0-s{seed}','model_fit','control_OIPG_'+policy+'_lambda0',split,seed,0,oipg_config(policy,0.,split,seed,0,custody['wikics'],os))
     fit_banks.append((policy+'_lambda0','control_OIPG_'+policy+'_lambda0',[zero],'mean_probabilities'))
   native=[add(f'compare-sp{split}-native-M1-s{seed}-b{body}','model_fit','control_WikiPolynormer_native_M1',split,seed,body,oipg_config('native_M1',0.,split,seed,body,custody['wikics'],os)) for body in range(4)]
   fit_banks += [('native_M1','control_WikiPolynormer_native_M1',[native[0]],'mean_probabilities'),('native_genuine_I4','control_WikiPolynormer_genuine_I4',native,'mean_probabilities')]
   coupled=add(f'compare-sp{split}-fourbody-coupled-s{seed}','model_fit',COUPLED,split,seed,0,oipg_config('fourbody_coupled',0.,split,seed,0,custody['wikics'],os))
   fit_banks.append(('fourbody_coupled',COUPLED,[coupled],'mean_probabilities'))
   original=add(f'compare-sp{split}-original-GNNM-s{seed}','model_fit','control_WikiCS_native_original_GNNM',split,seed,0,physics_config('original_gnnm',split,seed,0,custody['wikics'],ps),native_arm='original_gnnm')
   fit_banks.append(('original_gnnm','control_WikiCS_native_original_GNNM',[original],'mean_logits'))
   for arm,method,parents,fusion in fit_banks:endpoints(f'bank-sp{split}-{arm}-s{seed}',method,split,seed,parents,fusion,arm)
 study['comparisons']=[{'candidates':list(OIPG.values())+[COUPLED],'references':['control_WikiPolynormer_native_M1','control_WikiPolynormer_genuine_I4','control_WikiCS_native_original_GNNM'],
  'scope':'10fresh WikiCS roles x3paired model seeds; native512Polynormer1100 vs literal source SAGE128 originalGNNM5000/p300 control; no published-score claim.'},
  {'candidates':[OIPG[p] for p in 'IPG'],'references':['control_OIPG_'+p+'_lambda0' for p in 'IPG'],
   'scope':'Symmetric prospective .05/.5 full1100 TRAIN/SELECT search then distinct new comparative seed; zero-strength computational programs kept separately; O exact numeric aliases one fit.'}]
 studies.append(study);all_banks+=banks
 for study in studies:
  validate(study)
  if not all(r['matched'] for r in reference_matches(study)):raise ValueError('Incomplete reference cohorts')
 graph_check(studies,{});counts=typed_counts(studies);typed_banks=[dict(b,study_id=s['study_id']) for s in studies for b in bank_roster(s)]
 assert len(CANONICAL)==15 and len(set(CANONICAL))==15 and counts['totals']['atomic_fit_run_groups']==624 and counts['totals']['bank_exports']==372
 assert counts['totals']['genuine_separately_selected_I4_banks']==36 and counts['totals']['run_records']==1830
 assert counts['model_optimizer_step_call_total'] is None and [counts['model_optimizer_step_call_minimum'],counts['model_optimizer_step_call_maximum']]==[721053,1443000]
 matrix.mkdir();(matrix/'studies').mkdir();prefix=args.packet+'/matrix';study_rows=[]
 for study in studies:
  path=matrix/'studies'/(study['study_id']+'.json.gz');path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0));study_rows.append({'path':str(path.relative_to(root)),'spec_hash':fingerprint(study)})
 gates={'whole_search':{'path':prefix+'/gate_receipts/whole_search.json','source_stages':['search','search_selection'],'required_for_stages':['comparative'],'required_hosts':['allocation']},
  'whole_comparative':{'path':prefix+'/gate_receipts/whole_comparative.json','source_stages':['comparative'],'required_for_stages':['bank_export'],'required_hosts':['allocation']},
  'whole_bank_exports':{'path':prefix+'/gate_receipts/whole_bank_exports.json','source_stages':['bank_export'],'required_for_stages':['calibration'],'required_hosts':['allocation']},
  'whole_calibration':{'path':prefix+'/gate_receipts/whole_calibration.json','source_stages':['calibration'],'required_for_stages':['evaluation'],'required_hosts':['allocation']}}
 plan={'schema_version':1,'slots_per_gpu':2,'torch_threads_per_fit':2,'output_root':prefix+'/execution','runtime_launcher':args.packet+'/runtime/launcher.py','runtime_package':args.packet+'/runtime',
  'worker_capacities_cuda_bytes':{'allocation/gpu0':85899345920},'studies':study_rows,'groups':[{'id':s['study_id'],'gpu_assignment':'allocation/gpu0','allowed_workers':['allocation/gpu0'],
   'maximum_cuda_bytes':34359738368,'estimated_seconds':sum(r['config']['estimated_seconds'] for r in s['runs']),'run_keys':[s['study_id']+'/'+r['run_id'] for r in s['runs']]} for s in studies],
  'global_gates':gates,'barriers':{},'scientific_launch_enabled':False,'human_GO_required':True,
  'host_bindings':{'allocation':{'status':'verified_at_compilation','hostname':'anogena-2-0','gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac']}}}
 summary={'schema':'Physics-OIPG-source-disabled-summary-v1','status':'SOURCE_DATA_PREPARED_DISABLED_ROOT_RESOURCE_MASTER_GO_PENDING','plan_hash':fingerprint(plan),
  'canonical_ids':CANONICAL,'counts':counts,'banks':all_banks,'prospective_design':receipt(root,packet/'DESIGN_DECISION.json'),'training_started':False,'actual_model_fits':0,'actual_predictive_scores':0,'GO_receipt_created':False,
  'source_choices':'Physics literal native source panel; OIPG prospective symmetric .05/.5 selected strength, zero controls retained; packed fourbody fixed source independent4_contrastive.',
  'resource_closure':'UNQUALIFIED; no actual-input callbacks/model constructors/updates/scores; all estimates are scheduling placeholders, not ETA.'}
 for name,value in [('PLAN.json',plan),('SUMMARY.json',summary),('BANK_ROSTER.json',{'schema':'Physics-OIPG-declared-banks-v1','banks':all_banks,'compiler_typed_banks':typed_banks})]:(matrix/name).write_text(canonical(value)+'\n')
 entry={'id':'native_Physics_OIPG_coupled_newroles_v1','plan':receipt(root,matrix/'PLAN.json'),'plan_hash':summary['plan_hash'],'summary':receipt(root,matrix/'SUMMARY.json'),
  'coverage':[receipt(root,packet/'CANONICAL_SOURCE_COMPLETION.json'),receipt(root,matrix/'BANK_ROSTER.json')]}
 (packet/'MASTER_INPUT_ENTRIES.json').write_text(json.dumps([entry],indent=2,sort_keys=True)+'\n')
 loaded=load_packet(root,entry);assert loaded['counts']==counts
 metadata={'schema':'Physics-OIPG-exact-disabled-typed-compilation-v1','entry':entry,'counts':counts,'references':loaded['references'],'canonical_ids':CANONICAL,'source_files':len(files),
  'per_study_counts':{s['study_id']:typed_counts([s]) for s in studies},'all_selected_exports_before_CAL':True,'all_CAL_before_evaluation':True,
  'full_selected_source_states_must_remain_until_root_exporter_audit':True,'models_constructed':0,'optimizer_updates':0,'target_arrays_decoded':False,'GO':False}
 (packet/'COMPILATION_METADATA.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'entry':entry,'counts':counts,'references':loaded['references'],'actual_input_models':0,'GO':False}))

if __name__=='__main__':main()
