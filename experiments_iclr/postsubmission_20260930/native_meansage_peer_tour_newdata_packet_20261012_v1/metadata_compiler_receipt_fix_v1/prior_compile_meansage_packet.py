"""Stdlib-only fixed native source recipe and complete Photo10x3 roster."""
import argparse,copy,gzip,hashlib,json,subprocess
from collections import Counter
from pathlib import Path
from adapters.meansage_native import CONTRACT,PINS,BUDGET,SELECTION
from spec import canonical,fingerprint,validate
from compile_master_plan import graph_check,typed_counts,reference_matches,receipt

SEEDS=(98401,98403,98407)
PEER_CANDIDATES={a:'meansage_peer_'+a.lower() for a in ('PA1','IA4','SA4','SM4','SX4','UX4')}
TOUR_CANDIDATES={a:'meansage_tour_scan_'+a.lower() for a in ('PV1','WI4','V1','SW4','C4','R4')}

def make_study(root,packet,custody,prep,sources,source_metadata,family,git_commit):
 prefix=str(packet.relative_to(root));runtime=prefix+'/runtime';data='training_pipeline_v2_20261012/prepared_data/amazon-photo';known={r['path']:r for r in custody['files']}
 def inputs(split,field):return {k:{**known[v],'path':data+'/'+v} for k,v in custody['splits'][split][field].items()}
 def endpoints(split):return {k:v['path'] for k,v in inputs(split,'endpoint_inputs').items()}
 candidates=PEER_CANDIDATES if family=='peer' else TOUR_CANDIDATES
 arm_methods={'P1':'control_'+family+'_ordinary_M1','I4':'control_'+family+'_factorized_M1','S4':'control_'+family+'_shared_S4','original_gnnm':'control_'+family+'_original_GNNM',**candidates}
 bank_methods={'P1':[(0,'control_'+family+'_ordinary_M1'),(None,'control_'+family+'_ordinary_I4')],
  'I4':[(0,'control_'+family+'_factorized_M1'),(None,'control_'+family+'_factorized_I4')],
  'S4':[(None,'control_'+family+'_shared_S4')],'original_gnnm':[(None,'control_'+family+'_original_GNNM')]}
 if family=='peer':
  arm_bodies={'P1':4,'I4':4,'S4':1,'original_gnnm':1,'PA1':4,'IA4':4,'SA4':1,'SM4':1,'SX4':1,'UX4':1}
  bank_methods.update({'PA1':[(0,candidates['PA1']),(None,'control_peer_ordinary_self_I4')],
   'IA4':[(0,'control_peer_factor_self_M1'),(None,candidates['IA4'])],
   **{a:[(None,candidates[a])] for a in ('SA4','SM4','SX4','UX4')}})
 else:
  arm_bodies={'P1':4,'I4':4,'S4':1,'original_gnnm':1,'PV1':1,'WI4':4,'V1':1,'SW4':1,'C4':1,'R4':1}
  bank_methods.update({'WI4':[(0,'control_tour_factor_scanner_M1'),(None,candidates['WI4'])],
   **{a:[(None,candidates[a])] for a in ('PV1','V1','SW4','C4','R4')}})
 runs=[];banks=[]
 def add(identifier,kind,method,split,seed,body,c,parents=()):
  fn={'model_fit':'meansage_native.py:make_adapter','ensemble_assembly':'meansage_endpoints.py:export_bank','calibration':'meansage_endpoints.py:scalar_calibration','evaluation':'meansage_endpoints.py:evaluate_bank'}[kind]
  row={'run_id':identifier,'kind':kind,'method':method,'split':split,'seed':seed,'body':body,'adapter':runtime+'/adapters/'+fn,'config':c,
    'budget':copy.deepcopy(BUDGET) if kind=='model_fit' else {},'selection':copy.deepcopy(SELECTION) if kind=='model_fit' else {},'parents':list(parents)}
  runs.append(row);return identifier
 for split in range(10):
  for seed in SEEDS:
   fits={}
   for arm,body_count in arm_bodies.items():
    original=arm=='original_gnnm';m=4 if original or arm in ('S4','SA4','SM4','SX4','UX4','PV1','V1','SW4','C4','R4') else 1
    fits[arm]=[]
    for body in range(body_count):
     c={'meansage_family':family,'meansage_arm':arm,'meansage_sources':sources,'architecture':'original_gnnm' if original else 'native_meansage_'+family,
      'backbone':'SAGE','width':128 if original else 32,'depth':2,'classes':8,'learning_rate':.001,'dropout':.4,'weight_decay':0,
      'source_optimizer_count':1,'physical_slow_bodies':4 if arm=='UX4' else 1,'members':m,'is_original_gnnm':original,
      'initialization_seed':seed+1000003*body,'source_TRAIN_forwards_per_update':4 if original else 1,'source_TRAIN_backwards_per_update':4 if original else 1,
      'source_complete_body_constructors':4 if arm in ('I4','IA4','WI4','UX4') else 1,
      'source_constructor_selection':'full_four_factory_choose_run_body' if arm in ('I4','IA4','WI4') else 'whole_single_source_factory',
      'historical_state_inherited':False,'fusion':'mean_logits' if original else 'mean_probabilities','node_inputs':inputs(split,'node_inputs'),
      'meansage_incidence':prep['incidence'] if family=='peer' else None,
      'meansage_tours':prep['tours'][str(seed)] if family=='tour' and arm not in ('P1','I4','S4','original_gnnm') else None,
      'stage':'comparative','recipe_scope':'native_fixed_source_recipe_new_Photo_roles_geometry_seeds',
      'reservation_mib':32768,'estimated_seconds':100000.,'resource_profile_status':'UNQUALIFIED_CONSERVATIVE_PLACEHOLDER; root actual-input resource qualification required',
      'parameter_count_whole_source_factory':None if original else prep['parameter_counts'][family][arm]}
     identifier=f'compare-sp{split}-{arm}-s{seed}-b{body}';fits[arm].append(add(identifier,'model_fit',arm_methods[arm],split,seed,body,c))
   for arm,labels in bank_methods.items():
    for pick,method in labels:
     parents=fits[arm] if pick is None else [fits[arm][pick]];inp=endpoints(split);fusion='mean_logits' if arm=='original_gnnm' else 'mean_probabilities'
     name=f'bank-sp{split}-{method}-s{seed}'
     export=add(name,'ensemble_assembly',method,split,seed,0,{'stage':'bank_export','fit_parents':parents,'endpoint_inputs':inp,'fusion':fusion,'reservation_mib':32768,'estimated_seconds':30.},parents)
     cal=add(name+'-cal','calibration',method,split,seed,0,{'stage':'calibration','export_parent':export,'endpoint_inputs':inp,'fusion':fusion,
      'placement':'before_member_probability_pool' if fusion=='mean_probabilities' else 'after_mean_logits','steps':500,'learning_rate':.05,'reservation_mib':512,'estimated_seconds':10.},[export])
     evaluation=add(name+'-evaluate','evaluation',method,split,seed,0,{'stage':'evaluation','export_parent':export,'calibration_parent':cal,'endpoint_inputs':inp,'fusion':fusion,'reservation_mib':512,'estimated_seconds':10.},[export,cal])
     banks.append({'arm':arm,'method':method,'split':split,'seed':seed,'fit_parents':parents,'export':export,'calibration':cal,'evaluation':evaluation,'genuine_I4':len(parents)==4,
      'logical_views':4 if len(parents)==4 else 4 if arm in ('S4','SA4','SM4','SX4','UX4','PV1','V1','SW4','C4','R4','original_gnnm') else 1})
 roles={r:{str(split):inputs(split,'node_inputs' if r in ('train','select') else 'endpoint_inputs')[r if r in ('train','select') else r+'_targets'] for split in range(10)} for r in ('train','select','cal','evaluate')}
 references=['control_'+family+'_'+s for s in ('ordinary_M1','ordinary_I4','factorized_M1','factorized_I4','shared_S4','original_GNNM')]
 references+=['control_peer_factor_self_M1','control_peer_ordinary_self_I4'] if family=='peer' else ['control_tour_factor_scanner_M1']
 study={'schema_version':1,'study_id':'native-meansage-'+family+'-photo-v1','stage':'development','dataset':'amazon-photo',
  'hypothesis':'Complete native '+family+' source conditions compared with ordinary/factorized/shared and genuine own-selected independent baselines, plus literal full original GNNM, on new Photo roles.',
  'data':{'task':'node_classification','preprocessing':custody['request']['graph_policy'],'roles':roles,'prior_exposure':custody['request']['evaluation_exposure'],
   'evidence':{'model_graph':custody['model_graph'],'custody_request_hash':custody['request_hash'],'source_contract':CONTRACT,'actual_graph_only_preparation':prep,
    'fixed_recipe_no_hyperparameter_search':True,'source_role_geometry_seed_adaptations_explicit':True}},
  'source':{'git_commit':git_commit,'files':source_metadata,'working_tree_fingerprints_authoritative':True,'frozen_runtime_package':runtime,'official_author_protocol_replication':False},
  'runs':runs,'comparisons':[{'references':references,'candidates':list(candidates.values()),'scope':'complete Photo10 masks x3 seeds; fixed native recipe; no search; independent own fits; cap costs and actual stopping costs separately reported'}],
  'release':{'policy':'whole_roster','primary_endpoint':'all raw and native member-temperature CAL500 banks','global_plan_gates_required':True},
  'resources':{'wall_seconds':172800,'status':'DISABLED_UNQUALIFIED_ROOT_RESOURCE_MASTER_AND_GO_PENDING'}}
 validate(study);return study,banks

def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True);args=p.parse_args();root=Path(args.project_root).resolve();packet=root/args.packet
 custody_read=json.loads((packet/'PHOTO_CUSTODY_READ.json').read_text());custody=custody_read['custody'];prep=json.loads((packet/'prepared_graph/CUSTODY.json').read_text())
 if custody['dataset_id']!='amazon-photo' or len(custody['splits'])!=10 or custody['benchmark_fits']!=0 or prep['target_tensors_opened'] or not prep['raw_binary']:raise ValueError('Complete untouched Photo custody and GRAPH-only binary preparation required')
 from adapters.meansage_native import counts
 peer,tour=counts(745,8);prep['parameter_counts']={'peer':peer,'tour':tour}
 sources=json.loads((packet/'EXACT_SOURCE_BINDINGS.json').read_text());sources={k:{x:v for x,v in row.items() if x in ('path','bytes','sha256')} for k,row in sources.items()}
 if any(sources[k]['sha256']!=v for k,v in PINS.items()):raise ValueError('Exact source pins')
 metadata=[]
 for directory in ('runtime','sources','source_orchestration'):
  metadata.extend(receipt(root,path) for path in sorted((packet/directory).rglob('*')) if path.is_file() and path.suffix in ('.py','.json'))
 metadata.extend(receipt(root,packet/name) for name in ('EXACT_SOURCE_BINDINGS.json','ORCHESTRATION_SOURCE_BINDINGS.json','SYNTHETIC_CPU_PROOF.json','ROOT_V14_COMPOSITION.json','INITIAL_CPU_PROOF_FAILURE.json','prepared_graph/CUSTODY.json'))
 metadata += [{**r,'path':'training_pipeline_v2_20261012/prepared_data/amazon-photo/'+r['path']} for r in custody['files']]
 metadata += [custody_read['custody_receipt'],prep['incidence']]+list(prep['tours'].values())
 metadata=list({r['path']:r for r in metadata}.values())
 git_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
 matrix=packet/'matrix';matrix.mkdir();entries=[];allstudies=[];allbanks=[];reports=[]
 for family in ('peer','tour'):
  target=matrix/family;target.mkdir();study,banks=make_study(root,packet,custody,prep,sources,metadata,family,git_commit);graph_check([study],{})
  counts_result=typed_counts([study]);refs=reference_matches(study)
  if any(not r['matched'] for r in refs):raise ValueError('Required complete reference cohorts')
  (target/'studies').mkdir();path=target/'studies'/(study['study_id']+'.json.gz');path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0));prefix=str(target.relative_to(root))
  plan={'schema_version':1,'slots_per_gpu':2,'torch_threads_per_fit':2,'output_root':prefix+'/execution','runtime_launcher':args.packet+'/runtime/launcher.py','runtime_package':args.packet+'/runtime',
   'worker_capacities_cuda_bytes':{'allocation/gpu0':85899345920},'studies':[{'path':str(path.relative_to(root)),'spec_hash':fingerprint(study)}],
   'groups':[{'id':study['study_id'],'gpu_assignment':'allocation/gpu0','allowed_workers':['allocation/gpu0'],'maximum_cuda_bytes':34359738368,
     'estimated_seconds':sum(r['config']['estimated_seconds'] for r in study['runs']),'run_keys':[study['study_id']+'/'+r['run_id'] for r in study['runs']]}],
   'global_gates':{'whole_comparative':{'path':prefix+'/gate_receipts/whole_comparative.json','source_stages':['comparative'],'required_for_stages':['bank_export','calibration','evaluation'],'required_hosts':['allocation']},
     'whole_calibration':{'path':prefix+'/gate_receipts/whole_calibration.json','source_stages':['calibration'],'required_for_stages':['evaluation'],'required_hosts':['allocation']}},
   'barriers':{},'scientific_launch_enabled':False,'human_GO_required':True,
   'host_bindings':{'allocation':{'status':'verified_at_compilation','hostname':'anogena-2-0','gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac']}}}
  summary={'schema':'native-meansage-'+family+'-disabled-summary-v1','plan_hash':fingerprint(plan),'status':'DISABLED_RESOURCE_MASTER_AND_GO_PENDING','fixed_seeds':list(SEEDS),
   'fixed_native_point':{'learning_rate':.001,'dropout':.4},'tuning_fits':0,'source_contract':CONTRACT,'counts':counts_result,'banks':banks,'canonical_ids':list((PEER_CANDIDATES if family=='peer' else TOUR_CANDIDATES).values()),
   'optimizer_updates_exact_before_execution':False,'optimizer_update_bounds':[counts_result['model_optimizer_step_call_minimum'],counts_result['model_optimizer_step_call_maximum']],
   'training_started':False,'actual_model_fits':0,'actual_predictive_scores':0,'GO_receipt_created':False}
  (target/'PLAN.json').write_text(canonical(plan)+'\n');(target/'SUMMARY.json').write_text(canonical(summary)+'\n')
  entries.append({'id':'native_meansage_'+family+'_photo_v1','plan':receipt(root,target/'PLAN.json'),'plan_hash':summary['plan_hash'],'summary':receipt(root,target/'SUMMARY.json'),
    'coverage':[receipt(root,packet/'EXACT_SOURCE_BINDINGS.json'),receipt(root,packet/'CANONICAL_SOURCE_COMPLETION.json')]})
  allstudies.append(study);allbanks+=banks;reports.append({'family':family,'counts':counts_result,'reference_matches':refs})
 combined=typed_counts(allstudies);(packet/'MASTER_INPUT_ENTRIES.json').write_text(json.dumps(entries,indent=2,sort_keys=True)+'\n')
 (packet/'COMPILATION_METADATA.json').write_text(json.dumps({'schema':'native-meansage-disabled-compilation-v1','entries':entries,'combined_counts':combined,'family_reports':reports,
  'source_files':len(metadata),'canonical_ids':list(PEER_CANDIDATES.values())+list(TOUR_CANDIDATES.values()),'scientific_fits':0,'predictive_scores':0,'GO':False},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'entries':entries,'counts':combined,'source_files':len(metadata),'science':False,'GO':False}))

if __name__=='__main__':main()
