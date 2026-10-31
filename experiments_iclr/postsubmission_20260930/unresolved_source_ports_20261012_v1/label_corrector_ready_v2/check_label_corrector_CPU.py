import copy,gc,json,os,sys,tempfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(P/'runtime'))
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from adapters.label_corrector import construct,Adapter,WORK
from engine import capture_checkpoint,restore_checkpoint,inspect_ownership,_equal_tree
B=json.loads((P/'LABEL_CORRECTOR_SOURCE_BINDINGS.json').read_text());g=torch.Generator().manual_seed(83001);nodes=600;graph={'x':torch.randn(nodes,300,generator=g),'edge_index':torch.stack([torch.arange(nodes).repeat(3),torch.cat([(torch.arange(nodes)+1)%nodes,(torch.arange(nodes)+13)%nodes,torch.arange(nodes)])])};train={'ids':torch.arange(580),'y':torch.arange(580)%10};select={'ids':torch.arange(580,600),'y':torch.arange(20)%10};cases=[];maps=0
for condition in WORK:
 seed=6101;provenance={'synthetic':True,'condition':condition,'actual_model_seed':seed};state=construct(torch,R,B,condition,seed,'cpu',graph,train);a=Adapter(torch,state,graph,train,select,condition,provenance);inspect_ownership(a);core=a.core;detach=[]
 if core is not None:
  original=core.train_corrector_step
  def observed(H,base,*args,**kwargs):
   before={n:None if p.grad is None else p.grad.clone() for n,p in a.native.model.named_parameters()};assert not H.requires_grad and H.grad_fn is None and not base.requires_grad and base.grad_fn is None;v=original(H,base,*args,**kwargs)
   assert all((p.grad is None and before[n] is None) or (p.grad is not None and before[n] is not None and torch.equal(p.grad,before[n])) for n,p in a.native.model.named_parameters());detach.append(True);return v
  core.train_corrector_step=observed
 out1=a.train_step(1);v1=a.validate(1);a.observe_validation(1,v1);chosen=a.local['step'];saved=copy.deepcopy(a.local['snapshot']);out2=a.train_step(2);v2=a.validate(2);a.observe_validation(2,v2)
 # Bounded synthetic lifecycle uses source own-local epoch1 and live endepoch2, not100 numerical updates.
 a.local['step']=chosen;a.local['snapshot']=saved;old=a.s.common.native_rng_state(a.native);mask=None if core is None else core.mask_generator.get_state().clone();native_steps=a.native.steps;core_steps=None if core is None else core.steps;counts=copy.deepcopy(a.counters);a.local_transition();a.s.common.require_same_native_rng(a.native,old,'synthetic coherent local restore');assert a.native.steps==native_steps and a.counters==counts
 if core is not None:assert core.steps==core_steps and torch.equal(core.mask_generator.get_state(),mask) and len(detach)==2
 assert a.native.model.models[0].body._global is True and a.parameter_epoch==chosen
 before_rng=a.s.common.native_rng_state(a.native);pred=a.predict_rows(select['ids'],2);a.s.common.require_same_native_rng(a.native,before_rng,'global selected serving');assert pred.shape==(WORK[condition][0],20,10)
 snapshot=capture_checkpoint(torch,a,fit_id='synthetic-'+condition,step=2,validation=v2,decision={'save_selected':True,'stop_requested':False});buffers={} if core is None else {n:v.clone() for n,v in core.named_buffers()};assert not any(n.startswith('corrector.') and n in a.model.role_buffers for n in snapshot['model_state'])
 with tempfile.TemporaryDirectory() as tmp:
  q=Path(tmp)/'selected.pt';torch.save(snapshot,q);size=q.stat().st_size;fresh=Adapter(torch,construct(torch,R,B,condition,seed,'cpu',graph,train),graph,train,select,condition,provenance);inspect_ownership(fresh);restore_checkpoint(torch,q,fresh,expected_fit_id='synthetic-'+condition);assert _equal_tree(torch,snapshot['adapter_state'],fresh.capture_state());p=fresh.predict_rows(select['ids'],2);assert torch.equal(p,pred)
  if fresh.core is not None:assert all(torch.equal(dict(fresh.core.named_buffers())[n],v) for n,v in buffers.items())
  parameter_count=sum(p.numel() for p in a.model.parameters());core_parameters=0 if core is None else sum(p.numel() for p in core.parameters())
  # Foreign immutable role identity is rejected before state adoption; no numeric kernel rerun.
  bad=copy.deepcopy(snapshot['adapter_state']);bad['role_identity']['TRAIN_ids_sha256']='0'*64
  try:fresh.restore_state(bad)
  except ValueError:pass
  else:raise AssertionError('Wrong immutable role authority admitted')
  del fresh
 commits=out1['operation_counts']['optimizer_updates']+out2['operation_counts']['optimizer_updates'];maps+=commits;cases.append({'condition':condition,'status':'passed','synthetic_complete_epochs':2,'committed_Adam_maps':commits,'learned_parameters':parameter_count,'corrector_parameters':core_parameters,'checkpoint_preview_bytes':size,'source_native_train_step_literal':condition!='original_GNNM','source_original_boundary_and2view8backwards':condition=='original_GNNM','oldviewB_detached_andnative_gradients_unchanged':True if core is not None else None,'source_nativeownlocal_coherent_restore':True,'synthetic_restore_boundary':'two completeupdates to ownlocalepoch1;source100 guard retained in production','live_endlocal_native_RNG_mask_andlogical_counters_preserved':True,'fresh_whole_state_reconstruction':True,'fixed_GRAPH_TRAIN_buffers_checked_neverloaded':True,'operation_counts_per_epoch':out1['operation_counts'],'capture_counts':None if core is None else dict(a.s.capture.counts),'core_counters':None if core is None else dict(core.counters),'scientific_fits':0,'GO':False});del a,state,core;gc.collect()
report={'schema':'label-corrector4-new-composition-synthetic-CPU-v1','status':'passed','cases':cases,'synthetic_complete_epochs':12,'committed_Adam_maps':maps,'actual_input_arrays_opened':0,'old_closed_zerooutput_or_gradient_kernel_tests_rerun':False,'scientific_fits':0,'CUDA_compute':0,'GO':False};(P/'LABEL_CORRECTOR_PIPELINE_CPU_QA.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','cases':len(cases),'synthetic_epochs':12,'committed_Adam_maps':maps,'GO':False}))
