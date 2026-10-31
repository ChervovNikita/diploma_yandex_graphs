"""Finite disabled20warm/60post NCN source outer-cycle packet and exact costs."""
import argparse,collections,gzip,hashlib,json
from pathlib import Path
from adapters.preaggregation_growth import ARMS
from spec import canonical,fingerprint,validate

SEARCH=(99101,99103);COMPARE=(99201,99203,99207);LRS=(.0005,.001,.002)


def receipt(root,p):
 b=p.read_bytes();return {'path':str(p.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def compile_packet(root,packet):
 root=Path(root).resolve();packet=Path(packet).resolve();out=packet/'matrix'
 if out.exists():raise ValueError('Fresh disabled finite output')
 custody=json.loads((packet/'CUSTODY.json').read_text());B=json.loads((packet/'SOURCE_BINDINGS.json').read_text());runtime=str((packet/'runtime').relative_to(root));runs=[];banks=[]
 if not custody['endpoint_preparation']:raise ValueError('Complete public link roles must be prepared')
 def method(a):return 'preaggregation_growth_'+a if a in ARMS else 'control_preaggregation_growth_'+a
 def config(stage,arm,seed,body,lr,phase):
  count=4 if arm=='independent_graph_growth4' else 1;members=4 if arm in ('no_growth','graph_growth','unfiltered_growth','independent_graph_growth4','original_gnnm') else 1
  return {'stage':stage,'growth_stage':phase,'growth_arm':arm,'growth_sources':B,'link_inputs':custody['link_inputs'],'calibration_draw':custody['calibration_draws'][str(seed)],'initialization_seed':seed+5*body,
   'factor_seed':seed+4000037,'learning_rate':lr,'members':members,'source_optimizer_count':count,'physical_slow_bodies':count,'source_complete_episodes_per_cycle':61,'source_Adam_passes_per_episode':3,'train_minibatches_per_epoch':183,
   'source_cycle_unit':'full TRAIN3870 order,61 outer64/tail episodes,four inner256 lists,three native Adam transitions per episode/owner',
   'source_constructor_count':5 if arm=='independent_graph_growth4' else 2 if phase=='post' else 1,'source_optimizer_constructor_count':count,
   'source_TRAIN_calibration_calls':int(phase=='post' and arm!='no_growth'),'source_SVD_calls':4 if arm in ('graph_growth','independent_graph_growth4') else 1 if arm in ('unfiltered_growth','capable_single_rank8') else 0,
   'historical_donor_outcomes_authorization_or_GO_inherited':False,'is_original_gnnm':arm=='original_gnnm','fusion':'mean_logits','selector':'first rounded4 complete SELECT MRR every5;terminal20 TRAIN-only warm custody',
   'architecture':'joint_untied' if count==4 else 'original_gnnm' if arm=='original_gnnm' else 'internal_be' if members==4 else 'single',
   'reservation_mib':8192*count,'estimated_seconds':5000000,'resource_evidence':'provisional unmeasured;root resource/native sparse/checkpoint/query endpoints/concurrent admission pending'}
 def fit(id,stage,arm,split,seed,body,lr,phase,parent=None):
  c=config(stage,arm,seed,body,lr,phase);budget=20 if phase=='warm' else 60 if phase=='post' else 80;parents=[] if parent is None else [parent]
  if parent:c['lifecycle']={'schema':'paid-parent-continuation-v1','parent_fit_ids':[parent],'new_optimizer':True,'parameter_initialization':'selected_parent_no_optimizer_reuse'}
  row={'run_id':id,'kind':'model_fit','method':'paid_growth_native_warm' if phase=='warm' else method(arm),'split':split,'seed':seed,'body':body,'adapter':runtime+'/adapters/preaggregation_growth.py:make_adapter','config':c,
   'budget':{'unit':'epochs','policy':'full_budget','maximum':budget},'selection':{'metric':'training_completion' if phase=='warm' else 'MRR','direction':'max','ties':'first','evaluation_interval':20 if phase=='warm' else 5,'checkpoint0':False,'patience_steps':budget+1},'parents':parents}
  runs.append(row);return id
 select={}
 for arm in (*ARMS,'ordinary_M1','original_gnnm'):
  cs=[]
  for i,lr in enumerate(LRS):
   ids=[]
   for seed in SEARCH:
    if arm in ARMS:
     warm='search-warm-lr'+str(i)+'-s'+str(seed)
     if not any(r['run_id']==warm for r in runs):fit(warm,'search','ordinary_M1',i,seed,0,lr,'warm')
     id=fit('search-'+arm+'-lr'+str(i)+'-s'+str(seed),'search',arm,i,seed,0,lr,'post',warm)
    else:id=fit('search-'+arm+'-lr'+str(i)+'-s'+str(seed),'search',arm,i,seed,0,lr,'cold_control')
    ids.append(id)
   cs.append({'config_id':str(i),'parameters':{'learning_rate':lr},'run_ids':ids})
  id='select-'+arm;select[arm]=id;runs.append({'run_id':id,'kind':'selection','method':method(arm),'split':0,'seed':0,'body':0,'adapter':runtime+'/adapters/search.py:select_configuration',
   'config':{'stage':'search_selection','configurations':cs,'ranking_metric':'MRR','ranking_direction':'max','reservation_mib':1,'estimated_seconds':1},'parents':[x for p in cs for x in p['run_ids']],'budget':{},'selection':{}})
 for i,lr in enumerate(LRS):
  for seed in COMPARE:
   warm=fit('compare-warm-lr'+str(i)+'-s'+str(seed),'comparative','ordinary_M1',i,seed,0,lr,'warm');fits={}
   for arm in ARMS:fits[arm]=[fit('compare-'+arm+'-lr'+str(i)+'-s'+str(seed),'comparative',arm,i,seed,0,lr,'post',warm)]
   for arm in ('ordinary_M1','original_gnnm'):fits[arm]=[fit('compare-'+arm+'-lr'+str(i)+'-s'+str(seed),'comparative',arm,i,seed,0,lr,'cold_control')]
   fits['ordinary_I4']=[fits['ordinary_M1'][0]]+[fit('compare-ordinary_M1-lr'+str(i)+'-s'+str(seed)+'-i4-b'+str(b),'comparative','ordinary_M1',i,seed,b,lr,'cold_control') for b in (1,2,3)]
   for arm,parents in fits.items():
    name=method(arm);id='bank-'+arm+'-lr'+str(i)+'-s'+str(seed);ep=custody['endpoint_inputs']
    for kind,suffix,adapter,endpoint_config,pa in [('ensemble_assembly','', 'export_bank',{'stage':'bank_export','fit_parents':parents,'endpoint_inputs':ep,'fusion':'mean_logits'},parents),
      ('calibration','-cal','scalar_calibration',{'stage':'calibration','export_parent':id,'endpoint_inputs':ep,'fusion':'mean_logits','steps':500,'learning_rate':.01},[id]),
      ('evaluation','-evaluate','evaluate_bank',{'stage':'evaluation','export_parent':id,'calibration_parent':id+'-cal','endpoint_inputs':ep,'fusion':'mean_logits'},[id,id+'-cal'])]:
     runs.append({'run_id':id+suffix,'kind':kind,'method':name,'split':i,'seed':seed,'body':0,'adapter':runtime+'/adapters/growth_endpoints.py:'+adapter,'config':{**endpoint_config,'reservation_mib':32768 if kind=='ensemble_assembly' else 512,'estimated_seconds':10000},'parents':pa,'budget':{},'selection':{}})
    banks.append({'method':name,'arm':arm,'lr':lr,'split':i,'seed':seed,'fit_parents':parents,'export':id,'genuine_cold_I4':arm=='ordinary_I4','source_copied_warm4_pooled_selector':arm=='independent_graph_growth4'})
 source=[receipt(root,p) for p in sorted(packet.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.md') and not any(x in p.parts for x in ('matrix','transport','receipts'))]
 source.extend(custody['link_inputs'].values());source.extend(custody['calibration_draws'].values())
 roles={k:custody['link_inputs'][k] if k in ('train','select') else custody['endpoint_inputs'][k+'_targets'] for k in ('train','select','cal','evaluate')}
 study={'schema_version':1,'study_id':'independent-preaggregation-growth-citeseer-heart-v2','stage':'development','dataset':'citeseer-heart','hypothesis':'Source-specific nonlinear preaggregation growth after newly paid native warm versus competent native single,cold genuine I4,and actual original source BEBlock link adaptation;all complete source cycle costs retained.',
  'data':{'task':'link_prediction','preprocessing':'exact official HeaRT features;TRAIN-only symmetric unweighted target-union-masked support,no loops;native PureConv adds self once;full perquery500negative pools','roles':roles,'prior_exposure':custody['prior_exposure'],
   'evidence':{'model_graph':{'nodes':3327,'features':3703},'source_exact_update_callback':True,'source20warm60post_unit_preserved':True,'source_recipe_replication':False,'source_CAL_evaluate_adaptation':'public TEST disjoint query-level half split;wholeVALID227SELECT;no heldout guided basis','LR_paired_cells':'all3prospectively declared LRs are complete comparison cohorts;source.001 center isprimary recipe;separate7HPO selectors are descriptive development decisions,no unscheduled fits'}},
  'source':{'git_commit':'exact_owned_source_byte_pins_authoritative_root_commit_pending','files':source,'frozen_runtime_package':runtime,'historical_authorization_or_donor_inherited':False},'runs':runs,
  'comparisons':[{'candidates':[method(a) for a in ARMS],'references':[method(a) for a in ('ordinary_M1','ordinary_I4','original_gnnm')],'original_GNNM_reference':method('original_gnnm'),'scope':'all3LR cells x3paired comparative seeds,one public graph;source copied-warm pooled4 remains a distinct condition;genuine coldI4 has4own-selected complete bodies'}],
  'release':{'policy':'whole_roster','primary_endpoint':'source.001center complete SELECT-selected source mean-logit ranks;paired LR sensitivity and raw/commonCAL binary-NLL/ranking both retained','global_plan_gates_required':True},'resources':{'wall_seconds':1000000000,'status':'PROVISIONAL_UNMEASURED_DISABLED'}}
 validate(study);out.mkdir();(out/'studies').mkdir();path=out/'studies'/('independent-preaggregation-growth-citeseer-heart-v2.json.gz');path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
 prefix=str(out.relative_to(root));plan={'schema_version':1,'slots_per_gpu':2,'torch_threads_per_fit':2,'output_root':prefix+'/execution','runtime_package':runtime,'runtime_launcher':runtime+'/launcher.py','worker_capacities_cuda_bytes':{'allocation/gpu0':78*1024**3},
  'studies':[{'path':str(path.relative_to(root)),'spec_hash':fingerprint(study)}],'groups':[{'id':study['study_id'],'gpu_assignment':'allocation/gpu0','allowed_workers':['allocation/gpu0'],'maximum_cuda_bytes':32768*1024**2,'estimated_seconds':sum(r['config']['estimated_seconds'] for r in runs),'run_keys':[study['study_id']+'/'+r['run_id'] for r in runs]}],
  'global_gates':{},'barriers':{},'scientific_launch_enabled':False,'human_GO_required':True,'host_bindings':{'allocation':{'hostname':'anogena-2-0','gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac'],'status':'root_runtime_gate_required'}}}
 for name,src,dst in [('whole_search',['search','search_selection'],['comparative']),('whole_comparative',['comparative'],['bank_export','calibration','evaluation']),('whole_calibration',['calibration'],['evaluation'])]:plan['global_gates'][name]={'path':prefix+'/gate_receipts/'+name+'.json','source_stages':src,'required_for_stages':dst,'required_hosts':['allocation']}
 f=[r for r in runs if r['kind']=='model_fit'];counts={'atomic_fit_groups':len(f),'paid_optimizer_objects':sum(r['config']['source_optimizer_count'] for r in f),'physical_body_instances':sum(r['config']['physical_slow_bodies'] for r in f),
  'registered_source_outer_cycles':sum(r['budget']['maximum'] for r in f),'aggregate_owner_source_cycles':sum(r['budget']['maximum']*r['config']['source_optimizer_count'] for r in f),'known_optimizer_step_calls':sum(r['budget']['maximum']*183*r['config']['source_optimizer_count'] for r in f),
  'search_selection_jobs':7,'bank_exports':len(banks),'scalar_CAL_fits':len(banks),'evaluation_jobs':len(banks),'scalar_CAL_updates':len(banks)*500,'all_run_records':len(runs),'genuine_cold_I4_banks':9,
  'source_copied_warm4_jointly_selected_banks':9,'new_paid_warm_donors':sum(r['config']['growth_stage']=='warm' for r in f),'original_GNNM_fit_groups':sum(r['config']['is_original_gnnm'] for r in f),
  'source_constructor_calls':sum(r['config']['source_constructor_count'] for r in f),'source_optimizer_constructor_calls':sum(r['config']['source_optimizer_constructor_count'] for r in f),'TRAIN_gradient_calibrations':sum(r['config']['source_TRAIN_calibration_calls'] for r in f),'CPU_float64_SVD_calls':sum(r['config']['source_SVD_calls'] for r in f),'actual_model_fits':0,'actual_optimizer_updates':0}
 assert counts['atomic_fit_groups']==147 and counts['paid_optimizer_objects']==192 and counts['known_optimizer_step_calls']==2206980 and counts['all_run_records']==370
 summary={'schema':'independent-preaggregation-growth-disabled-source-v1','plan_hash':fingerprint(plan),'counts':counts,'banks':banks,'canonical_method_ids':[method(a) for a in ARMS],'source_training_started':False,'historical_join':'UNRESOLVED_SEPARATE','controller_type_amendment_required':'owned runtime adds only explicit NCN multiAdam/source-cycle/terminal20 paid-parent semantics;root must adopt equivalent typed branch before combined ingestion',
  'parameter_growth':'4096newParameters once per source growth condition;sourcebase/newgrowthalltrainable;nofallback/redraw;not ongoing iterative constructors','resource_scope':'provisional_unmeasured;RESOURCE_PROPOSAL.json required','scientific_launch_enabled':False,'human_GO_required':True,'GO':False}
 (out/'PLAN.json').write_text(canonical(plan)+'\n');(out/'SUMMARY.json').write_text(canonical(summary)+'\n');entry={'id':'independent_preaggregation_growth_citeseer_heart_v2','plan':receipt(root,out/'PLAN.json'),'summary':receipt(root,out/'SUMMARY.json'),'plan_hash':fingerprint(plan),'coverage':[receipt(root,packet/'SOURCE_BINDINGS.json')]}
 (packet/'MASTER_INPUT_ENTRY.json').write_text(json.dumps(entry,indent=2)+'\n');return entry

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True);a=p.parse_args();print(canonical({'entry':compile_packet(a.project_root,a.packet),'fits':0,'CUDA':0,'GO':False}))
