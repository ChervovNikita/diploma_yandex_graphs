"""Disabled independent role-bound source studies; no numeric/data import."""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path

from adapters.common_native_wrapper import CONTRACT as COMMON_CONTRACT,SOURCES
from adapters.native_label_families import NEIGHBOR_ARMS,SPARSE_ARMS
from spec import canonical,fingerprint,validate

DOMAIN=[{'learning_rate':lr,'dropout':d} for lr in (.0005,.001,.002) for d in (.1,.2,.4)]
SEARCH=(99101,99103);COMPARE=(99201,99203,99207)
FAMILIES={
 'neighborhood_distribution':{'arms':list(NEIGHBOR_ARMS[:4])+['ordinary_M1','factorized_plain_M1','original_gnnm'],
   'I4':{NEIGHBOR_ARMS[4]:NEIGHBOR_ARMS[3],'ordinary_I4':'ordinary_M1','factorized_plain_I4':'factorized_plain_M1'},'native_readouts':True},
 'sparse_feature_kernel':{'arms':list(SPARSE_ARMS[:5])+['ordinary_M1','factorized_plain_M1','original_gnnm'],
   'I4':{SPARSE_ARMS[5]:SPARSE_ARMS[4],'ordinary_feature_I4':SPARSE_ARMS[3],'ordinary_I4':'ordinary_M1','factorized_plain_I4':'factorized_plain_M1'},'native_readouts':False},
 'shared_message':{'arms':['exact_member','shared_mean','detached_shared_mean','single_D64','single_matched','original_gnnm'],
   'I4':{'ordinary_I4':'single_D64','matched_I4':'single_matched'},'native_readouts':False},
 'private_hop':{'arms':['private_missinghop','native_shared4_own','ordinary_M1','factor1_native','original_gnnm'],
   'I4':{'ordinary_I4':'ordinary_M1','factor1_I4':'factor1_native'},'native_readouts':False}}


def receipt(root,path):
 raw=path.read_bytes();return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def method(family,arm):
 if family=='neighborhood_distribution' and arm in NEIGHBOR_ARMS:return (family+'_'+arm).lower()
 if family=='sparse_feature_kernel' and arm in SPARSE_ARMS:return (family+'_'+arm).lower()
 if family=='shared_message' and arm in ('exact_member','shared_mean','detached_shared_mean'):
  return {'exact_member':'shared_message_exact_member','shared_mean':'shared_message_shared_mean_message','detached_shared_mean':'shared_message_detached_mean_message'}[arm]
 if family=='private_hop' and arm=='private_missinghop':return 'private_hop_native_private_hop'
 return 'control_'+family+'_'+arm.lower()


def make_study(root,packet,family,dataset,bindings,derived):
 spec=FAMILIES[family];common=family not in ('shared_message','private_hop');hop=family=='private_hop';custody=json.loads((packet/'metadata'/dataset/'CUSTODY.json').read_text());shape=custody['model_graph']
 splits={s['split_id']:s for s in custody['splits']};files={r['path']:r for r in custody['files']};prefix=str(packet.relative_to(root));runtime=prefix+'/runtime'
 data_prefix='massive_polyformer_packet_20261012_v1/data/pubmed/' if hop else 'training_pipeline_v2_20261012/prepared_data/'+dataset+'/'
 def inputs(split,key):return {k:{'path':data_prefix+v,**{f:files[v][f] for f in ('bytes','sha256')}} for k,v in splits[split][key].items()}
 def members(arm):return 4 if arm in list(NEIGHBOR_ARMS[:3])+list(SPARSE_ARMS[:3])+['exact_member','shared_mean','detached_shared_mean','original_gnnm','private_missinghop','native_shared4_own'] else 1
 sources=bindings[family];runs=[];banks=[];selections={};seed_roster=sorted({s+1000003*b for s in (*SEARCH,*COMPARE) for b in range(4)})
 def config(arm,split,stage):
  c={'ported_family':family,'family_arm':arm,'backbone':'PolyFormer_mono_K2' if hop else 'SAGE' if common else 'SOURCE_MEAN_SAGE','architecture':'original_gnnm' if arm=='original_gnnm' else 'internal_be' if members(arm)==4 else 'single' if arm in ('ordinary_M1',SPARSE_ARMS[3],'single_D64','single_matched') else 'factorized_single',
   'members':members(arm),'physical_slow_bodies':1,'source_optimizer_count':1,'width':256 if hop else 128 if common else 64,'depth':2,'rank':16,'classes':shape['classes'],
   'weight_decay':0. if hop and arm=='original_gnnm' else .001 if hop else 0. if common else .0001,'wrapper_sources':bindings['common'],'family_sources':sources,'node_inputs':inputs(split,'node_inputs'),
   'provider_graph':shape,'wrapper_arm':'original_gnnm' if arm=='original_gnnm' else 'ordinary_M1' if arm=='ordinary_M1' else 'factorized_allmap_M1',
   'source_contract':COMMON_CONTRACT,'is_original_gnnm':arm=='original_gnnm','stage':stage,'fusion':'mean_logits' if family=='shared_message' or arm=='original_gnnm' else 'mean_probabilities',
   'reservation_mib':32768,'estimated_seconds':5000000,'resource_evidence':'unmeasured explicit scheduling placeholder;root source/resource/protocol gates required',
   'source_only_independent_new_data':True,'historical_authorization_or_outcomes_inherited':False,
   'source_constructor_count':1 if hop or common or arm=='original_gnnm' else 2,'source_optimizer_constructor_count':2 if hop and arm=='factor1_native' else 1,
   'full_budget_extension':'source update callback fixed;common engine full5000;old horizon/earlystop not claimed exact'}
  if family=='sparse_feature_kernel':c['sparse_support']=derived['sparse_supports'][dataset]
  if hop:c.update(hop_sources=bindings['private_hop'],polyformer_files=bindings['polyformer_files'],boundary_source=bindings['common']['native'],
    source_native_recipe='exact PubMed monoK2/H256/L2/heads8/FFN32/q1.6/multi2/dprate.8;dropout neighborhood .25/.5/.75;Adam attention/baseLR ratio.1;source decays1e-8/.001',
    source_feature_normalization_gate='provider raw-feature normalization/cache custody remains root-review gate;no old feature tensor inherited')
  if family=='shared_message':c.update(shared_source_root=prefix+'/sources',shared_source_imports=bindings['shared_imports'],shared_original_train=bindings['common']['gnnm_train'],
    mean_operator=derived['mean_operators'][dataset],shared_task_index=2 if dataset=='wikics' else 3,shared_seed_roster=seed_roster,
    source_selector='NLL every5;first strict minimum;final_primary,selected_secondary',source_model_dropout_interface='rebind existing p/dropout plain fields after exact source constructor/copy;weights unchanged')
  return c
 def add(identifier,kind,arm,split,seed,body,c,parents=(),served_method=None):
  if kind=='model_fit':c['initialization_seed']=seed+(1009 if hop else 1000003)*body
  adapter=runtime+'/adapters/'+(('private_hop_port.py:make_adapter' if hop else 'native_label_families.py:make_adapter' if common else 'shared_message_port.py:make_adapter') if kind=='model_fit' else 'search.py:select_configuration' if kind=='selection' else
    'label_family_endpoints.py:'+{'ensemble_assembly':'export_bank','calibration':'scalar_calibration','evaluation':'evaluate_bank'}[kind])
  row={'run_id':identifier,'kind':kind,'method':served_method or method(family,arm),'split':split,'seed':seed,'body':body,'adapter':adapter,'config':c,'parents':list(parents),
   'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':5000} if kind=='model_fit' else {},
   'selection':{'metric':'accuracy' if common or hop else 'pooled_nll','direction':'max' if common or hop else 'min','ties':'first','evaluation_interval':1 if common or hop else 5,'checkpoint0':False,'patience_steps':5000} if kind=='model_fit' else {}}
  runs.append(row);return row
 for arm in spec['arms']:
  candidates=[]
  domain=[{'learning_rate':lr,'dropout':d} for lr in (.0025,.005,.01) for d in (.25,.5,.75)] if hop else DOMAIN
  for point in domain:
   key=fingerprint(point)[:20];ids=[add('search-'+arm+'-'+key+'-s'+str(s),'model_fit',arm,0,s,0,{**config(arm,0,'search'),**point})['run_id'] for s in SEARCH]
   candidates.append({'config_id':key,'parameters':point,'run_ids':ids})
  selections[arm]=add('select-'+arm,'selection',arm,0,0,0,{'stage':'search_selection','configurations':candidates,'ranking_metric':'accuracy' if common or hop else 'pooled_nll','ranking_direction':'max' if common or hop else 'min','reservation_mib':1,'estimated_seconds':1},[r for c in candidates for r in c['run_ids']])['run_id']
 for split in sorted(splits):
  for seed in COMPARE:
   fits={}
   for arm in spec['arms']:
    row=add('compare-sp'+str(split)+'-'+arm+'-s'+str(seed),'model_fit',arm,split,seed,0,{**config(arm,split,'comparative'),**(domain[0] if hop else DOMAIN[0]),'selected_config_from':selections[arm]},[selections[arm]]);fits[arm]=[row['run_id']]
   for arm,single in spec['I4'].items():
    first=next(r for r in runs if r['run_id']==fits[single][0]);fits[arm]=[first['run_id']]
    for body in (1,2,3):fits[arm].append(add(first['run_id']+'-i4-b'+str(body),'model_fit',single,split,seed,body,copy.deepcopy(first['config']),first['parents'])['run_id'])
   served=[]
   for arm,parents in fits.items():
    served.append((arm,parents,'final' if family=='shared_message' else 'selected',None,method(family,arm)))
    if family=='shared_message':served.append((arm,parents,'selected',None,method(family,arm)+'_selected_secondary'))
    elif family=='neighborhood_distribution' and arm in NEIGHBOR_ARMS:served.append((arm,parents,'selected','native_at_corrected_selected',method(family,arm)+'_native_at_corrected_selected'))
   for arm,parents,checkpoint,readout,name in served:
    single=spec['I4'].get(arm,arm);fusion=config(single,split,'comparative')['fusion'];prefix_id='bank-sp'+str(split)+'-'+arm+'-s'+str(seed)+('-secondary' if checkpoint=='selected' and family=='shared_message' else '-native' if readout else '')
    endpoint={k:v['path'] for k,v in inputs(split,'endpoint_inputs').items()}
    export=add(prefix_id,'ensemble_assembly',arm,split,seed,0,{'stage':'bank_export','fit_parents':parents,'endpoint_inputs':endpoint,'fusion':fusion,'checkpoint_role':checkpoint,'readout':readout,'reservation_mib':32768,'estimated_seconds':10000},parents,name)['run_id']
    cal=add(prefix_id+'-cal','calibration',arm,split,seed,0,{'stage':'calibration','export_parent':export,'endpoint_inputs':endpoint,'fusion':fusion,'placement':'after_serving_pool','steps':500,'learning_rate':.01,'reservation_mib':512,'estimated_seconds':1000},[export],name)['run_id']
    ev=add(prefix_id+'-evaluate','evaluation',arm,split,seed,0,{'stage':'evaluation','export_parent':export,'calibration_parent':cal,'endpoint_inputs':endpoint,'fusion':fusion,'reservation_mib':512,'estimated_seconds':1000},[export,cal],name)['run_id']
    banks.append({'arm':arm,'method':name,'split':split,'seed':seed,'fit_parents':parents,'export':export,'calibration':cal,'evaluation':ev,'genuine_I4':arm in spec['I4'],'body0_reused':arm in spec['I4'],'checkpoint_role':checkpoint,'readout':readout,'served_views':4 if arm in spec['I4'] else members(single)})
 roles={role:{str(s):inputs(s,'node_inputs' if role in ('train','select') else 'endpoint_inputs')[role if role in ('train','select') else role+'_targets'] for s in splits} for role in ('train','select','cal','evaluate')}
 original=method(family,'original_gnnm')
 canonical_arms=list(NEIGHBOR_ARMS) if family=='neighborhood_distribution' else list(SPARSE_ARMS) if family=='sparse_feature_kernel' else ['exact_member','shared_mean','detached_shared_mean'] if family=='shared_message' else ['private_missinghop']
 candidates=[method(family,a) for a in canonical_arms]
 references=[method(family,a) for a in list(spec['arms'])+list(spec['I4']) if a not in canonical_arms]
 source_files=[receipt(root,p) for p in sorted((packet/'runtime').rglob('*.py'))]+[receipt(root,p) for p in sorted((packet/'sources').rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.md')]
 source_files+=[receipt(root,packet/'metadata'/dataset/'CUSTODY.json'),receipt(root,packet/'lineage/HISTORICAL_JOIN_UNRESOLVED.json')]
 source_files+=list(inputs(0,'node_inputs').values())
 if family in ('sparse_feature_kernel','shared_message'):source_files.append(derived['sparse_supports' if family=='sparse_feature_kernel' else 'mean_operators'][dataset])
 study={'schema_version':1,'study_id':'independent-'+family+'-v3-'+dataset,'stage':'development','dataset':dataset,
  'hypothesis':'Independent source-specific '+family+' on complete new-data roles versus competent own single/genuine I4 and actual originalGNNM; no historical authorization/result inheritance.',
  'data':{'task':'node_classification','preprocessing':custody['request']['graph_policy'],'roles':roles,'prior_exposure':custody['request']['evaluation_exposure'],
    'evidence':{'model_graph':shape,'source_exact_training_recipe':False,'source_backbone':'source native PolyFormer PubMed monoK2/H256' if hop else 'native SAGE H128' if common else 'source meanSAGE D64;deterministic parameter-matched single width',
    'source_preparation':'full incoming-nonself degree buckets;duplicates retained' if family=='neighborhood_distribution' else 'label-free raw-X cosine top20 support union;source exact learned metric' if family=='sparse_feature_kernel' else 'exact ordered rawX/PX/P2X K2 token provider;complete23 factor sites;root normalization gate' if hop else 'full factual incoming record-mean forward/adjoint CSR'}},
  'source':{'git_commit':'independent_source_pins_authoritative;root_commit_binding_pending','files':source_files,'frozen_runtime_package':runtime,'historical_authorization_or_checkpoint_inherited':False},
  'runs':runs,'comparisons':[{'candidates':candidates,'references':references,'original_GNNM_reference':original,'scope':('PubMed public split0 and three paired comparative seeds; all outcomes/failures/costs retained;development' if hop else 'all20WikiCS/10Photo masks and three paired comparative seeds; all outcomes/failures/costs retained;development,not independent graphs')}],
  'release':{'policy':'whole_roster','primary_endpoint':'final-primary plus earliest-NLL selected secondary for shared-message;corrected primary/native-at-selected secondary for neighborhood;all raw and common CAL banks','global_plan_gates_required':True},
  'resources':{'wall_seconds':1000000000,'status':'PROVISIONAL_UNMEASURED_DISABLED'}}
 validate(study);f=[r for r in runs if r['kind']=='model_fit'];counts={'atomic_fit_groups':len(f),'paid_optimizer_objects':len(f),'physical_body_instances':len(f),'full_budget_optimizer_updates':len(f)*5000,
  'search_selection_jobs':len(selections),'bank_exports':len(banks),'scalar_CAL_fits':len(banks),'evaluation_jobs':len(banks),'genuine_I4_banks':sum(b['genuine_I4'] for b in banks),
  'reused_I4_body0':sum(b['body0_reused'] for b in banks),'unique_genuine_I4_acquisitions':len({tuple(b['fit_parents']) for b in banks if b['genuine_I4']}),'unique_reused_I4_body0_fits':len({b['fit_parents'][0] for b in banks if b['body0_reused']}),'all_run_records':len(runs),'source_model_constructor_calls':sum(r['config']['source_constructor_count'] for r in f),
  'source_optimizer_constructor_calls':sum(r['config']['source_optimizer_constructor_count'] for r in f),'original_GNNM_fit_groups':sum(r['config']['is_original_gnnm'] for r in f),'actual_model_fits':0,'actual_optimizer_updates':0}
 return study,counts,banks


def compile_batch(root,packet):
 root=Path(root).resolve();packet=Path(packet).resolve();bindings=json.loads((packet/'SOURCE_BINDINGS.json').read_text());derived=json.loads((packet/'PREPARED_DERIVED_GRAPH_INPUTS.json').read_text());entries=[]
 for family in FAMILIES:
  output=packet/'packets'/family/'matrix'
  if output.exists():raise ValueError('Fresh disabled output required')
  output.mkdir(parents=True);(output/'studies').mkdir();totals=Counter();banks=[];studies=[];groups=[]
  for dataset in (('pubmed',) if family=='private_hop' else ('wikics','amazon-photo')):
   study,counts,served=make_study(root,packet,family,dataset,bindings,derived);path=output/'studies'/(study['study_id']+'.json.gz');path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
   studies.append({'path':str(path.relative_to(root)),'spec_hash':fingerprint(study)});totals.update(counts);banks.extend({'study_id':study['study_id'],**b} for b in served)
   groups.append({'id':study['study_id'],'gpu_assignment':'allocation/gpu0','allowed_workers':['allocation/gpu0'],'maximum_cuda_bytes':32768*1024**2,'estimated_seconds':sum(r['config']['estimated_seconds'] for r in study['runs']),'run_keys':[study['study_id']+'/'+r['run_id'] for r in study['runs']]})
  prefix=str(output.relative_to(root));runtime=str((packet/'runtime').relative_to(root));plan={'schema_version':1,'slots_per_gpu':2,'torch_threads_per_fit':2,'output_root':prefix+'/execution','runtime_package':runtime,'runtime_launcher':runtime+'/launcher.py',
   'worker_capacities_cuda_bytes':{'allocation/gpu0':78*1024**3},'studies':studies,'groups':groups,'global_gates':{},'barriers':{},'scientific_launch_enabled':False,'human_GO_required':True,
   'host_bindings':{'allocation':{'hostname':'anogena-2-0','gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac'],'status':'root_runtime_gate_required'}}}
  for name,source,target in [('whole_search',['search','search_selection'],['comparative']),('whole_comparative',['comparative'],['bank_export','calibration','evaluation']),('whole_calibration',['calibration'],['evaluation'])]:
   plan['global_gates'][name]={'path':prefix+'/gate_receipts/'+name+'.json','source_stages':source,'required_for_stages':target,'required_hosts':['allocation']}
  family_domain=[{'learning_rate':lr,'dropout':d} for lr in (.0025,.005,.01) for d in (.25,.5,.75)] if family=='private_hop' else DOMAIN
  summary={'schema':'independent-fitted-label-source-family-disabled-v1','family':family,'plan_hash':fingerprint(plan),'counts':dict(totals),'banks':banks,'domain':family_domain,'source_anchor':{'learning_rate':.005,'dropout':.5} if family=='private_hop' else {'learning_rate':.001,'dropout':.2},
   'canonical_condition_count':5 if family=='neighborhood_distribution' else 6 if family=='sparse_feature_kernel' else 1 if family=='private_hop' else 3,'historical_authorization_join':'UNRESOLVED_SEPARATE_NOT_INHERITED',
   'scientific_training_started':False,'GO_created':False,'numerical_CPU_proof':receipt(root,packet/'receipts/FINAL_CPU_QA_RESULT.json'),'resource_scope':'provisional_unmeasured'}
  for name,value in [('PLAN.json',plan),('SUMMARY.json',summary)]: (output/name).write_text(canonical(value)+'\n')
  entry={'id':'independent_'+family+'_v3','plan':receipt(root,output/'PLAN.json'),'summary':receipt(root,output/'SUMMARY.json'),'plan_hash':fingerprint(plan),'coverage':[receipt(root,packet/'lineage/HISTORICAL_JOIN_UNRESOLVED.json')]}
  (output.parent/'MASTER_INPUT_ENTRY.json').write_text(json.dumps(entry,indent=2)+'\n');entries.append(entry)
 (packet/'MASTER_INPUT_ENTRIES.json').write_text(json.dumps(entries,indent=2)+'\n');return entries


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True);a=p.parse_args();print(canonical({'entries':compile_batch(a.project_root,a.packet),'CUDA':0,'scientific_fits':0,'GO':False}))
