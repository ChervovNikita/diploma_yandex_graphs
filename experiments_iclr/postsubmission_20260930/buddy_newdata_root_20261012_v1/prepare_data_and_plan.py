"""Owned CPU-only source/cache preparation and disabled BUDDY run roster."""
import copy
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def save(path,value):
 path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
 assert socket.gethostname()=='anogena-2-0'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
 packet=Path(__file__).resolve().parent;phase=packet.parent;repo=phase.parents[1]
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
 source=packet/'source';provider=packet/'providers/datasketch';provider.mkdir(parents=True)
 cmd=[str(repo/'.venv/bin/python'),'-m','pip','install','--no-deps','--disable-pip-version-check','--target',str(provider),'datasketch==1.6.5']
 r=subprocess.run(cmd,cwd=packet,capture_output=True,text=True,timeout=180)
 save(packet/'PROVIDER_INSTALL.json',{'command':cmd,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'active_environment_modified':False});assert r.returncode==0
 sys.path.insert(0,str(provider));sys.path.insert(0,str(source))
 import torch
 import numpy as np
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 from models import single,FactorizedBUDDY
 import cache_builder as builder
 source_contract=json.loads((phase/'buddy_complete_data_cache_preparation_v3/CONTRACT.json').read_text())
 spec=importlib.util.spec_from_file_location('buddy_preparation_native',phase/'buddy_complete_data_cache_preparation_v3/prepare_cache.py');prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
 archive=phase/'buddy_official_archive_staging_v1/root_transfer_v2/collab.zip';datasetroot=packet/'provider_data';cache=packet/'cache'
 save(packet/'PROGRESS.json',{'stage':'native_archive_staging','scientific_fits':0})
 staged=prep.stage_archive(archive,datasetroot,source_contract)
 dataset=builder.offline_dataset(datasetroot);graph=dataset[0]
 raw=prep.qualify_public_graph(dataset,graph,datasetroot,source_contract,builder)
 train,_,_=builder.official_split_file(dataset,'train');valid,_,_=builder.official_split_file(dataset,'valid')
 topology=prep.qualify_train_split(train,graph,raw,source_contract,builder)
 validation=prep.qualify_validation(valid,source_contract,builder)
 save(packet/'PROGRESS.json',{'stage':'native_full_cache_build','scientific_fits':0,'topology':topology})
 builder.build(__import__('argparse').Namespace(dataset_root=str(datasetroot),output=str(cache)))
 contents=prep.qualify_cache_contents(cache,train,valid,graph,topology,source_contract,builder)
 # Native cache contains only TRAIN/SELECT labels. The separate endpoint is
 # built from frozen opaque query pairs, with no targets or positive counts.
 oldplan=json.loads((phase/'link_boundary_packet_20261012_v3/matrix/PLAN.json').read_text())
 prior=json.loads(gzip.decompress((phase/oldplan['studies'][0]['path']).read_bytes()))
 query_binding=prior['data']['evidence']['endpoint_inputs']['evaluate_queries']
 sys.path.insert(0,str(packet/'runtime'))
 from adapters.common_native_wrapper import bound,receipt
 query=torch.load(bound(phase,query_binding),map_location='cpu',weights_only=True);assert set(query)=={'ids','pairs'}
 common= torch.load(cache/'common.pt',map_location='cpu',weights_only=True)
 sf=builder.native_hashes(json.loads((source/'CONFIG.json').read_text())).get_subgraph_features(query['pairs'],common['hashes'],common['cards'],65536);sf[:,[4,5]]=0
 torch.save({'ids':query['ids'],'links':query['pairs'],'sf':sf},cache/'query.pt')
 del common,sf
 provider_files=[receipt(phase,q) for q in sorted(provider.rglob('*')) if q.is_file() and '__pycache__' not in q.parts]
 save(packet/'PROVIDER_MANIFEST.json',{'distribution':'datasketch1.6.5','files':provider_files,'runtime_loader':'BUDDY native cached fitting does not import datasketch; provider retained for exact cache regeneration','active_environment_modified':False})
 save(packet/'DATA_PREPARATION.json',{'archive':receipt(phase,archive),'staged':staged,'topology':topology,'validation':validation,'contents':contents,'native_cache_manifest':receipt(phase,cache/'manifest.json'),'label_free_endpoint':receipt(phase,cache/'query.pt'),'endpoint_queries':query_binding,'heldout_targets_opened':False,'no_future_positive_filtering':True,'cache_construction_not_training':True,'scientific_fits':0})
 bindings={name:receipt(phase,source/(name+'.py')) for name in ('guards','models','cache_builder')}
 original=receipt(phase,phase/'common_wrapper_native_source_20261010_v1/models.py')
 c={'buddy_sources':bindings,'original_BE_source':original,'native_config':json.loads((source/'CONFIG.json').read_text()),'learning_rate':.02,'dropout_multiplier':1.}
 # CPU finite-gradient/ownership and source model parity checks. No step,
 # benchmark fitting or predictive benchmark scores are performed.
 from adapters import buddy as port
 from engine import inspect_ownership
 rows=[];torch.manual_seed(620);fixture=(torch.randn(8,8),torch.randn(8,2,128),torch.arange(1,9).float(),torch.arange(2,10).float())
 for arm in port.ARMS:
  local={**c,'buddy_arm':arm};model,opt,mods,details=port.construct(torch,phase,local,74201,0,'cpu');inspect_ownership(__import__('types').SimpleNamespace(model=model,optimizer=opt))
  z=model(*fixture).reshape(8,-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.ones_like(z));assert torch.isfinite(loss);loss.backward()
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()) and not opt.state
  rows.append({'arm':arm,'finite_gradient':True,'route_count':z.shape[1],'parameters':sum(p.numel() for p in model.parameters()),'optimizer_commits':0})
 save(packet/'CPU_QA.json',{'status':'PASS','cases':rows,'fixture_nodes_or_edges':'synthetic','scientific_fits':0,'optimizer_commits':0,'CUDA_initialized':False})
 save(packet/'PROGRESS.json',{'stage':'disabled_plan_compile','scientific_fits':0})
 compile_plan(packet,phase,prior,c,port.CONTRACT,rows,receipt)
 save(packet/'COMPLETE.json',{'status':'DATA_SOURCE_DISABLED_ROSTER_READY_RESOURCE_AND_GO_PENDING','scientific_fits':0,'benchmark_scores':0,'optimizer_commits':0,'GO_created':False})


def compile_plan(packet,phase,prior,c,contract,qa,receipt):
 from spec import canonical,fingerprint,validate
 sys.path.insert(0,str(phase/'master_controller_source_20261012_v16/runtime'))
 import compile_master_plan as master
 out=packet/'matrix';out.mkdir();(out/'studies').mkdir()
 study=copy.deepcopy(prior);study['study_id']='native-buddy-collab-newdata-v1';study['runs']=[];runs=study['runs'];study['hypothesis']='The retained shared-map BUDDY ensemble may improve source predictor quality on the same temporal link roles, compared with native M1, independently trained I4, original-GNNM boundary adaptation and a parameter-matched single predictor.'
 study['data']['preprocessing']='Exact pinned HLL/MinHash structural cache, native sign_k0 endpoints, weighted TRAIN-only collab support; fixed one TRAIN negative per positive; closed-label opaque endpoint cache.'
 study['data']['evidence'].update(buddy_source_contract=contract,retained_source_canonical_ids=['buddy_shared_cache_factorized4'],data_preparation=receipt(phase,packet/'DATA_PREPARATION.json'),CPU_QA=receipt(phase,packet/'CPU_QA.json'),original_GNNM_link={'status':'explicit_BUDDY_multiinput_task_adaptation','contract':contract['original_gnnm'],'original_task_replication':False})
 study['source']['files']=[receipt(phase,q) for q in sorted((packet/'runtime').rglob('*.py'))]+[receipt(phase,q) for q in sorted((packet/'source').rglob('*')) if q.is_file() and '__pycache__' not in q.parts]+[original:=c['original_BE_source']]+[receipt(phase,packet/'CPU_QA.json'),receipt(phase,packet/'DATA_PREPARATION.json'),receipt(phase,packet/'PROVIDER_MANIFEST.json')]
 runtime=str((packet/'runtime').relative_to(phase));inputs={name:receipt(phase,packet/'cache'/filename) for name,filename in (('common','common.pt'),('train','train.pt'),('select','valid.pt'),('query','query.pt'))}
 # Explicit data receipts are checked independently by the factory. Add their
 # current custody as source/data closure for lossless remote transfer.
 study['source']['files'] += list(inputs.values())
 arms=('native1024','single256','factorized4','matched_single','original_gnnm');methods={'native1024':'source_BUDDY_native1024','single256':'source_BUDDY_native256_M1','factorized4':'buddy_shared_cache_factorized4','matched_single':'source_BUDDY_parameter_matched_M1','original_gnnm':'original_GNNM_BUDDY_boundary','independent4':'source_BUDDY_native256_genuine_I4'}
 domain=[{'learning_rate':lr,'dropout_multiplier':d} for lr in (.005,.02,.08) for d in (.5,1.,2.)];search=(74101,74103);compare=(74201,74203,74207);selects={};banks=[]
 def config(arm,seed,body,stage):
  return {**copy.deepcopy(c),'stage':stage,'buddy_arm':arm,'backbone':'native_BUDDY','architecture':'original_gnnm' if arm=='original_gnnm' else 'internal_be' if arm=='factorized4' else 'single','members':4 if arm in ('factorized4','original_gnnm') else 1,'physical_slow_bodies':1,'source_optimizer_count':1,'is_original_gnnm':arm=='original_gnnm','initialization_seed':seed+10000*body,'buddy_inputs':inputs,'provider_nodes':235868,'provider_features':128,'training':{'batch_size':1024,'eval_batch_size':65536,'tail_policy':'native_keep_tail_and_merge_singleton'},'train_minibatches_per_epoch':2303,'metric':{'name':'hits@50','direction':'max'},'fusion':'mean_raw_scores','source_contract':contract,'reservation_mib':16384,'estimated_seconds':1,'resource_evidence':'unmeasured source/control/endpoint/concurrency','scheduling_weight_status':'placeholder_not_ETA'}
 def add(rid,kind,arm,seed,body,cc,parents=()):
  entry='buddy.py:make_adapter' if kind=='model_fit' else 'link_search.py:select_configuration' if kind=='selection' else 'buddy_endpoints.py:'+('export_bank' if kind=='ensemble_assembly' else 'evaluate_bank')
  runs.append({'run_id':rid,'kind':kind,'method':methods[arm],'split':0,'seed':seed,'body':body,'adapter':runtime+'/adapters/'+entry,'config':cc,'parents':list(parents),'budget':{'unit':'epochs','policy':'full_budget','maximum':100} if kind=='model_fit' else {},'selection':{'metric':'hits@50','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False} if kind=='model_fit' else {}});return rid
 for arm in arms:
  candidates=[]
  for point in domain:
   key=fingerprint(point)[:20];ids=[add(f'search-{arm}-{key}-s{seed}','model_fit',arm,seed,0,{**config(arm,seed,0,'search'),**point}) for seed in search];candidates.append({'config_id':key,'parameters':point,'run_ids':ids})
  selects[arm]=add('select-'+arm,'selection',arm,0,0,{'stage':'search_selection','configurations':candidates,'ranking_metric':'hits@50','ranking_direction':'max','reservation_mib':1,'estimated_seconds':1},[rid for row in candidates for rid in row['run_ids']])
 ep=study['data']['evidence']['endpoint_inputs'];metric={'name':'hits@50','direction':'max'}
 for seed in compare:
  fits={arm:[add(f'compare-{arm}-s{seed}','model_fit',arm,seed,0,{**config(arm,seed,0,'comparative'),'selected_config_from':selects[arm]},[selects[arm]])] for arm in arms}
  fits['independent4']=[fits['single256'][0]]+[add(f'compare-independent4-s{seed}-b{body}','model_fit','single256',seed,body,{**config('single256',seed,body,'comparative'),'selected_config_from':selects['single256']},[selects['single256']]) for body in (1,2,3)]
  for arm,parents in fits.items():
   rid=f'bank-{arm}-s{seed}';ex=add(rid,'ensemble_assembly',arm,seed,0,{'stage':'bank_export','fit_parents':parents,'query_input':ep['evaluate_queries'],'support_policy':'full_TRAIN_only','fusion':'mean_raw_scores','reservation_mib':16384,'estimated_seconds':1},parents)
   ev=add(rid+'-evaluate','evaluation',arm,seed,0,{'stage':'evaluation','export_parent':ex,'query_input':ep['evaluate_queries'],'target_input':ep['evaluate_targets'],'support_policy':'full_TRAIN_only','fusion':'mean_raw_scores','metric':metric,'reservation_mib':512,'estimated_seconds':1},[ex]);banks.append({'method':methods[arm],'seed':seed,'fits':parents,'export':ex,'evaluate':ev,'genuine_I4':arm=='independent4','body0_reused':arm=='independent4','CAL':False})
 study['comparisons']=[{'references':[methods[k] for k in ('native1024','single256','independent4','matched_single','original_gnnm')],'candidates':[methods['factorized4']],'scope':'native100epochs x2303batches, nine-point symmetric HPO x2seeds,three paired comparison seeds','interpretation':'development; source model/recipe baseline, not full published score replication; training-only evaluation support disclosed; preserve all outcomes/costs'}]
 validate(study);path=out/'studies'/(study['study_id']+'.json.gz');path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
 plan=copy.deepcopy(json.loads((phase/'link_boundary_packet_20261012_v3/matrix/PLAN.json').read_text()));plan.update(studies=[{'path':str(path.relative_to(phase)),'spec_hash':fingerprint(study)}],output_root=str(out.relative_to(phase))+'/execution')
 plan['groups'][0].update(study_ids=[study['study_id']],run_keys=[study['study_id']+'/'+r['run_id'] for r in runs],estimated_seconds=len(runs),maximum_cuda_bytes=16384*1024**2,assignment_status='disabled_resource_pending')
 for gate in plan['global_gates'].values():gate['path']=str(out.relative_to(phase))+'/gate_receipts/'+Path(gate['path']).name
 fits=[r for r in runs if r['kind']=='model_fit'];assert len(fits)==114
 summary={'schema':'native-BUDDY-newdata-disabled-source-roster-v1','status':'DISABLED_RESOURCE_AND_GO_PENDING','plan_hash':fingerprint(plan),'counts':master.typed_counts([study]),'fit_groups':114,'known_optimizer_calls':26254200,'banks':18,'canonical_conditions':['buddy_shared_cache_factorized4'],'source_contract':contract,'domain':domain,'source_anchor':{'learning_rate':.02,'dropout_multiplier':1.},'search_seeds':search,'comparative_seeds':compare,'bank_details':banks,'exact_M1_I4_body0_fit_references':3,'scientific_fits':0,'actual_optimizer_commits':0,'GO_created':False}
 assert summary['counts']['totals']['atomic_fit_run_groups']==114
 for name,value in [('PLAN.json',plan),('SUMMARY.json',summary)]:save(out/name,value)
 entry={'id':'native_BUDDY_shared_cache_newdata_source_port','plan':receipt(phase,out/'PLAN.json'),'summary':receipt(phase,out/'SUMMARY.json'),'plan_hash':fingerprint(plan),'coverage':[receipt(phase,packet/'CPU_QA.json'),receipt(phase,packet/'DATA_PREPARATION.json'),receipt(phase,packet/'PROVIDER_MANIFEST.json')]}
 save(packet/'MASTER_INPUT_ENTRY.json',entry);print(json.dumps({'entry':entry,'fit_groups':114,'banks':18,'scientific_fits':0}),flush=True)


if __name__=='__main__':
 packet=Path(__file__).resolve().parent
 try:main()
 except BaseException as error:
  import traceback
  save(packet/'FAILURE.json',{'error':str(error),'type':type(error).__name__,'traceback':traceback.format_exc(),'scientific_fits':0,'optimizer_commits':0});raise
