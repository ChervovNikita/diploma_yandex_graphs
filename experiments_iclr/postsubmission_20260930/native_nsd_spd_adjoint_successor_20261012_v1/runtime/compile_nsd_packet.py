"""Disabled finite native NSD4 plus geometry1 binary-source roster."""
import argparse,collections,gzip,hashlib,json
from pathlib import Path
from spec import canonical,fingerprint,validate
SEARCH=(98101,98103);COMPARE=(98201,98203,98207);LRS=(.005,.01,.02)
LEGACY=('assigned_fixed','assigned_resampled','layer_shared_incidence','node_capacity_right','joint_multi_transport_single','ordinary_M1','factorized_M1','original_gnnm')
CHOICES=('identity','random_admissible','live_action_rank','classifier_response_rank')
CANONICAL={'assigned_fixed':'nsd_shared_incidence_assigned','assigned_resampled':'nsd_shared_incidence_assigned','layer_shared_incidence':'nsd_shared_layer_incidence','node_capacity_right':'nsd_node_capacity_right','joint_multi_transport_single':'nsd_joint_multi_transport_single','live_action_rank':'nsd_geometry_live_action_initialized_geometry'}

def receipt(root,p):
 b=p.read_bytes();return {'path':str(p.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def compile_packet(root,packet):
 root=Path(root).resolve();packet=Path(packet).resolve();out=packet/'matrix'
 if out.exists():raise ValueError('Fresh finite compiler output required')
 custody=json.loads((packet/'CUSTODY.json').read_text());B=json.loads((packet/'SOURCE_BINDINGS.json').read_text());runtime=str((packet/'runtime').relative_to(root));out.mkdir();(out/'studies').mkdir();entries=[]
 source=[receipt(root,p) for p in sorted(packet.rglob('*')) if p.is_file() and (p.suffix in ('.py','.json','.md') or p.name=='LICENSE') and not any(x in p.parts for x in ('matrix','transport','receipts','controller_base','public_provider_cache','data'))]
 source.extend(custody['files'])
 for family in ('legacy','geometry'):
  runs=[];banks=[]
  def method(arm,choice=None):
   if arm=='geometry_branch':return CANONICAL.get(choice,'control_nsd_geometry_'+choice)
   if arm=='assigned_fixed' or arm=='assigned_resampled':return CANONICAL[arm]+'_'+arm.split('_')[1]
   return CANONICAL.get(arm,'control_nsd_'+arm)
  def config(stage,arm,sp,seed,body,lr,choice=None):
   views=1 if arm in ('ordinary_M1','factorized_M1','joint_multi_transport_single') else 4
   paths=1 if arm in ('ordinary_M1','factorized_M1','original_gnnm') else 4
   forwards=4 if arm=='original_gnnm' else paths;backwards=4 if arm in ('original_gnnm','geometry_warm','geometry_branch') else 1
   c={'stage':stage,'nsd_arm':arm,'nsd_sources':B,'node_inputs':custody['splits'][sp]['node_inputs'],'classes':2,'initialization_seed':seed+100003*body,'learning_rate':lr,'dropout':.2,'members':views,'architecture':'original_gnnm' if arm=='original_gnnm' else 'single' if views==1 else 'internal_be','backbone':'native_DiscreteGeneralSheafDiffusion','source_optimizer_count':1,'physical_slow_bodies':1,'source_complete_native_paths':paths,'source_TRAIN_forwards_per_update':forwards,'source_TRAIN_backwards_per_update':backwards,'source_constructor_count':paths,'source_optimizer_constructor_count':1,'is_original_gnnm':arm=='original_gnnm','fusion':'mean_logits' if arm=='original_gnnm' else 'mean_probabilities','source_recipe':'d4_f16_L4 full nonlinear General augmented block-degree SVD;native maps,left/right,ELU,residual;no LP/HP/sparse learner','optimizer_law':'AdamW zero_decay original BEBlock adaptation' if arm=='original_gnnm' else 'source Adam;sheaf and other decay .0005','historical_results_outcomes_or_authorization_inherited':False,'reservation_mib':49152,'estimated_seconds':2000000,'resource_evidence':'PROVISIONAL_UNMEASURED;root complete native sparse forward/backward/setup/export/serialization qualification required'}
   if choice:c['geometry_choice']=choice
   return c
  def fit(id,stage,arm,sp,seed,body,lr,parent=None,screen=None,choice=None):
   c=config(stage,arm,sp,seed,body,lr,choice);steps=100 if arm=='geometry_warm' else 400 if arm=='geometry_branch' else 500;parents=[]
   if parent:
    c['lifecycle']={'schema':'paid-NSD-geometry-branch-v1','parent_fit_ids':[parent],'new_optimizer':True,'parameter_initialization':'terminal100_slow_Adam_retained'};c['screen_parent']=screen;parents=[parent,screen]
   row={'run_id':id,'kind':'model_fit','method':'paid_nsd_geometry_identity_warm' if arm=='geometry_warm' else method(arm,choice),'split':sp,'seed':seed,'body':body,'adapter':runtime+'/adapters/nsd.py:make_adapter','config':c,'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':steps},'selection':{'metric':'auroc','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':arm=='geometry_branch','patience_steps':501},'parents':parents};runs.append(row);return id
  def screen(id,stage,warm,sp,seed,lr):
   c=config('geometry_screen_'+stage,'geometry_warm',sp,seed,0,lr);c.update(warm_parent=warm,common_native_forwards=17,extra_offnode_spmm=17,source_optimizer_count=0,source_constructor_count=4,source_optimizer_constructor_count=0)
   runs.append({'run_id':id,'kind':'selection','method':'paid_nsd_geometry_common_TRAIN_screen','split':sp,'seed':seed,'body':0,'adapter':runtime+'/adapters/nsd_screen.py:run_screen','config':c,'budget':{},'selection':{},'parents':[warm]});return id
  selectors={};search_records={a:[] for a in (LEGACY if family=='legacy' else (*CHOICES,'ordinary_M1','factorized_M1','original_gnnm'))}
  for i,lr in enumerate(LRS):
   cell={a:[] for a in search_records}
   for seed in SEARCH:
    if family=='geometry':
     warm=fit('search-warm-lr'+str(i)+'-s'+str(seed),'search','geometry_warm',0,seed,0,lr);scr=screen('search-screen-lr'+str(i)+'-s'+str(seed),'search',warm,0,seed,lr)
     for choice in CHOICES:cell[choice].append(fit('search-'+choice+'-lr'+str(i)+'-s'+str(seed),'search','geometry_branch',0,seed,0,lr,warm,scr,choice))
    for arm in (LEGACY if family=='legacy' else ('ordinary_M1','factorized_M1','original_gnnm')):cell[arm].append(fit('search-'+arm+'-lr'+str(i)+'-s'+str(seed),'search',arm,0,seed,0,lr))
   for arm in cell:search_records[arm].append({'config_id':str(i),'parameters':{'learning_rate':lr},'run_ids':cell[arm]})
  for arm,cells in search_records.items():
   id='select-'+arm;selectors[arm]=id;runs.append({'run_id':id,'kind':'selection','method':method('geometry_branch',arm) if arm in CHOICES else method(arm),'split':0,'seed':0,'body':0,'adapter':runtime+'/adapters/search.py:select_configuration','config':{'stage':'search_selection','configurations':cells,'ranking_metric':'auroc','ranking_direction':'max','reservation_mib':1,'estimated_seconds':1},'budget':{},'selection':{},'parents':[id for cell in cells for id in cell['run_ids']]})
  for sp in range(10):
   for i,lr in enumerate(LRS):
    for seed in COMPARE:
     prefix='compare-sp'+str(sp)+'-lr'+str(i)+'-s'+str(seed);fits={}
     if family=='geometry':
      warm=fit(prefix+'-warm','comparative','geometry_warm',sp,seed,0,lr);scr=screen(prefix+'-screen','comparative',warm,sp,seed,lr)
      for choice in CHOICES:fits[choice]=[fit(prefix+'-'+choice,'comparative','geometry_branch',sp,seed,0,lr,warm,scr,choice)]
     for arm in (LEGACY if family=='legacy' else ('ordinary_M1','factorized_M1','original_gnnm')):fits[arm]=[fit(prefix+'-'+arm,'comparative',arm,sp,seed,0,lr)]
     for arm in (('ordinary_M1','factorized_M1','joint_multi_transport_single') if family=='legacy' else ('ordinary_M1','factorized_M1')):fits[arm+'_I4']=fits[arm]+[fit(prefix+'-'+arm+'-I4-b'+str(b),'comparative',arm,sp,seed,b,lr) for b in (1,2,3)]
     ep={k:v['path'] for k,v in custody['splits'][sp]['endpoint_inputs'].items()}
     for arm,parents in fits.items():
      name=method('geometry_branch',arm) if arm in CHOICES else method(arm);id=prefix+'-bank-'+arm;fusion='mean_logits' if arm=='original_gnnm' else 'mean_probabilities'
      for kind,suffix,adapter,c,pa in [('ensemble_assembly','','export_bank',{'stage':'bank_export','fit_parents':parents,'endpoint_inputs':ep,'fusion':fusion},parents),('calibration','-cal','scalar_calibration',{'stage':'calibration','export_parent':id,'endpoint_inputs':ep,'fusion':fusion,'steps':500,'learning_rate':.01},[id]),('evaluation','-evaluate','evaluate_bank',{'stage':'evaluation','export_parent':id,'calibration_parent':id+'-cal','endpoint_inputs':ep,'fusion':fusion},[id,id+'-cal'])]:
       runs.append({'run_id':id+suffix,'kind':kind,'method':name,'split':sp,'seed':seed,'body':0,'adapter':runtime+'/adapters/nsd_endpoints.py:'+adapter,'config':{**c,'reservation_mib':49152 if kind=='ensemble_assembly' else 512,'estimated_seconds':30000},'budget':{},'selection':{},'parents':pa})
      banks.append({'method':name,'arm':arm,'split':sp,'learning_rate':lr,'seed':seed,'fit_parents':parents,'export':id,'genuine_I4':arm.endswith('_I4')})
  role0=custody['splits'][0];roles={k:role0['node_inputs'][k] if k in ('train','select') else role0['endpoint_inputs'][k+'_targets'] for k in ('train','select','cal','evaluate')}
  study_id='native-nsd-'+family+'-twitch-de-musae-raw-v2';candidates=[method(a) for a in LEGACY[:5]] if family=='legacy' else [method('geometry_branch',a) for a in CHOICES];refs=[method(a) for a in ('ordinary_M1','factorized_M1','original_gnnm')]+[method(a+'_I4') for a in ('ordinary_M1','factorized_M1')]
  study={'schema_version':1,'study_id':study_id,'stage':'development','dataset':'twitch-de-musae-raw','hypothesis':'Exact native full-General NSD source operators on a new compatible public binary task,including cold native/factorized M1,genuine I4,and actual source BEBlock boundary baseline.',
   'data':{'task':'node_classification','preprocessing':custody['graph_policy'],'roles':roles,'prior_exposure':custody['prior_exposure'],'evidence':{'model_graph':custody['model_graph'],'source_exact_update_callback':True,'one_graph_ten_role_masks':True,'source_binary_AUROC_NLL_and_margin_preserved':True,'all_three_LR_comparative_cells_explicit':True,'HPO_selectors_descriptive_only':True,'source_total500_complete_updates':True}},
   'source':{'git_commit':'owned_exact_byte_pins_authoritative','files':source,'frozen_runtime_package':runtime,'historical_authorization_or_outcomes_inherited':False},'runs':runs,
   'comparisons':[{'candidates':candidates,'references':refs,'original_GNNM_reference':method('original_gnnm'),'scope':'3LRcells x10fixedrole masks x3pairedseeds,one binarypublicgraph;sourcecenter.01 primary;allmask/LR/siblingoutcomes retained'}],
   'release':{'policy':'whole_roster','primary_endpoint':'own source AUROC/NLL/earlier-epoch selection;source probability mean except literal original boundary raw-logit mean;raw and common scalarCAL outcomes both retained','global_plan_gates_required':True},'resources':{'wall_seconds':10**12,'status':'PROVISIONAL_UNMEASURED_DISABLED'}}
  validate(study);path=out/'studies'/(study_id+'.json.gz');path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0));prefix=str(out.relative_to(root));plan={'schema_version':1,'slots_per_gpu':2,'torch_threads_per_fit':2,'output_root':prefix+'/execution','runtime_package':runtime,'runtime_launcher':runtime+'/launcher.py','worker_capacities_cuda_bytes':{'allocation/gpu0':78*1024**3},'studies':[{'path':str(path.relative_to(root)),'spec_hash':fingerprint(study)}],'groups':[{'id':study_id,'gpu_assignment':'allocation/gpu0','allowed_workers':['allocation/gpu0'],'maximum_cuda_bytes':49152*1024**2,'estimated_seconds':sum(r['config']['estimated_seconds'] for r in runs),'run_keys':[study_id+'/'+r['run_id'] for r in runs]}],'global_gates':{},'barriers':{},'scientific_launch_enabled':False,'human_GO_required':True,'host_bindings':{'allocation':{'hostname':'anogena-2-0','gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac'],'status':'root_runtime_gate_required'}}}
  for name,src,dst in [('whole_search',['search','search_selection','geometry_screen_search'],['comparative','geometry_screen_comparative']),('whole_comparative',['comparative','geometry_screen_comparative'],['bank_export','calibration','evaluation']),('whole_calibration',['calibration'],['evaluation'])]:plan['global_gates'][name]={'path':prefix+'/'+family+'_gate_receipts/'+name+'.json','source_stages':src,'required_for_stages':dst,'required_hosts':['allocation']}
  f=[r for r in runs if r['kind']=='model_fit'];screenrows=[r for r in runs if r['config']['stage'].startswith('geometry_screen')];counts={'atomic_fit_groups':len(f),'declared_model_optimizer_updates':sum(r['budget']['maximum'] for r in f),'paid_optimizer_objects':len(f),'physical_slow_body_instances':len(f),'training_complete_body_constructors':sum(r['config']['source_constructor_count'] for r in f),'training_optimizer_constructors':len(f),'search_selection_jobs':len(selectors),'bank_exports':len(banks),'scalar_CAL_fits':len(banks),'scalar_CAL_updates':500*len(banks),'evaluation_jobs':len(banks),'common_TRAIN_screens':len(screenrows),'screen_complete_native_forwards':17*len(screenrows),'screen_extra_offnode_spmm':17*len(screenrows),'screen_complete_body_constructors':4*len(screenrows),'screen_optimizer_constructors':0,'new_paid100_warm':sum(r['config']['nsd_arm']=='geometry_warm' for r in f),'paid400_continuation_branches':sum(r['config']['nsd_arm']=='geometry_branch' for r in f),'genuine_I4_banks':sum(b['genuine_I4'] for b in banks),'original_GNNM_fit_groups':sum(r['config']['is_original_gnnm'] for r in f),'all_run_records':len(runs),'actual_scientific_fits':0,'actual_optimizer_updates':0,'CUDA':0,'GO':False}
  expected=(1578,789000,990) if family=='legacy' else (1308,577200,810);assert (len(f),counts['declared_model_optimizer_updates'],len(banks))==expected
  summary={'schema':'native-NSD-disabled-finite-source-v1','plan_hash':fingerprint(plan),'counts':counts,'banks':banks,'canonical_method_ids':sorted(set(CANONICAL[a] for a in (LEGACY[:5] if family=='legacy' else ('live_action_rank',)))),'canonical_aliases':{method(a):CANONICAL[a] for a in LEGACY[:5]} if family=='legacy' else {method('geometry_branch','live_action_rank'):CANONICAL['live_action_rank']},'historical_join':'UNRESOLVED_SEPARATE','root_equivalent_typed_controller_adoption_required':True,'root_protocol_resource_master_GO_gates_pending':True,'scientific_launch_enabled':False,'human_GO_required':True,'GO':False}
  pd=out/family;pd.mkdir();(pd/'PLAN.json').write_text(canonical(plan)+'\n');(pd/'SUMMARY.json').write_text(canonical(summary)+'\n');entries.append({'id':'native_nsd_'+family+'_twitch_de_musae_raw_v2','plan':receipt(root,pd/'PLAN.json'),'summary':receipt(root,pd/'SUMMARY.json'),'plan_hash':fingerprint(plan),'coverage':[receipt(root,packet/'SOURCE_BINDINGS.json')]})
 (packet/'MASTER_INPUT_ENTRIES.json').write_text(json.dumps(entries,indent=2)+'\n');return entries

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True);a=p.parse_args();print(json.dumps({'entries':compile_packet(a.project_root,a.packet),'models':0,'optimizer_updates':0,'CUDA':0,'GO':False}))
