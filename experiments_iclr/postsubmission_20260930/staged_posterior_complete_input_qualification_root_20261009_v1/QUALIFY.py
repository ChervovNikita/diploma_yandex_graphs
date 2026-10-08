"""Root-authorized full-input unscored staged-posterior qualification only."""
from pathlib import Path
import json,hashlib,importlib.util,sys,time,signal,copy,resource,socket,subprocess,os
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930');R=P.parents[1];OUT=P/'staged_posterior_complete_input_qualification_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
SOURCE=P/'label_only_staged_posterior_full_WikiCS_source_20261008_v1';assert hashlib.sha256((SOURCE/'MANIFEST.json').read_bytes()).hexdigest()=='ab2be08e841293e95d23d2098c128e0c989087e6a6215101474ce00a2ea0a3e8'
OUT.mkdir(exist_ok=False);began=time.monotonic();result=dict(complete=False,native_training_updates=0,scientific_fit=False,development_scores_computed=False,TEST_access=False,discarded_updates=0,new_native_captures=0,source_manifest_sha256='ab2be08e841293e95d23d2098c128e0c989087e6a6215101474ce00a2ea0a3e8');stage=cached=None
signal.signal(signal.SIGALRM,lambda n,f:(_ for _ in ()).throw(TimeoutError('Qualification180s budget')));signal.setitimer(signal.ITIMER_REAL,180)
try:
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2,0);torch.cuda.reset_peak_memory_stats(0)
 spec=importlib.util.spec_from_file_location('_root_staged_qualification',SOURCE/'stage.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 train,valid,origin=m.load_roles(later_execution_authorized=True);stage=m.make_stage(train_data=train,origin=origin,seed=6101,later_execution_authorized=True,purpose='engineering_qualification')
 result['new_native_captures']+=stage.work['native_capture_calls'];result['role_shapes']={k:list(v.shape) for k,v in train.items() if torch.is_tensor(v)}
 result['constructor_parameter_counts']={a:sum(x.numel() for x in b.parameters()) for a,b in stage.banks.items()}
 for epoch in (1,2):
  stage.train_step(label_epoch=epoch);result['discarded_updates']+=1;pred=stage.serve_ids(valid['ids'])
  if epoch==1:
   first={a:v['served_probabilities'].detach().clone() for a,v in pred.items()};snap=stage.snapshot()
 live={a:(b.mask_generator.get_state().clone(),b.steps,copy.deepcopy(b.counters)) for a,b in stage.banks.items()}
 stage.restore_learned(snap);assert stage.label_epoch==2 and stage.parameter_label_epoch==1
 for a,b in stage.banks.items():assert torch.equal(b.mask_generator.get_state(),live[a][0]) and b.steps==live[a][1] and b.counters==live[a][2]
 restored=stage.serve_ids(valid['ids']);diff={a:float((restored[a]['served_probabilities']-first[a]).abs().max()) for a in m.ARMS};assert max(diff.values())<2e-6
 stage.train_step(label_epoch=3);result['discarded_updates']+=1;after=stage.serve_ids(valid['ids']);snap3=stage.snapshot();cache=stage.capture_state()
 torch.save(cache,OUT/'QUALIFICATION_CAPTURE.pt');torch.save(snap3,OUT/'QUALIFICATION_STATE.pt')
 cache=torch.load(OUT/'QUALIFICATION_CAPTURE.pt',map_location='cpu',weights_only=False);snap3=torch.load(OUT/'QUALIFICATION_STATE.pt',map_location='cpu',weights_only=False)
 cached=m.make_stage(train_data=train,origin=origin,seed=6101,later_execution_authorized=True,purpose='engineering_qualification',capture_cache=cache)
 assert cached.work['native_capture_calls']==0;cached.restore_learned(snap3);prediction=cached.serve_ids(valid['ids']);cache_diff={a:float((prediction[a]['served_probabilities']-after[a]['served_probabilities']).abs().max()) for a in m.ARMS};assert max(cache_diff.values())<2e-6
 stage.train_step(label_epoch=4);result['discarded_updates']+=1;stage.serve_ids(valid['ids']);stage.fixed();cached.fixed()
 result.update(complete=True,constructor_setup=stage.run['new_setup_timings_seconds'],restore_probability_max_abs=diff,cache_probability_max_abs=cache_diff,native_capture_identity=stage.capture_identity,common_query_and_frozen_native_guards_passed=True,live_mask_and_work_retained_on_restore=True,nonlinear_Adam_owner_checked=True,cache_reconstructed_without_native_forward=True,selected_serving_and_gate_validation_deferred_until_all12_science_closed=True,work=stage.work,bank_counts={a:dict(b.counters) for a,b in stage.banks.items()})
except BaseException as e:
 result['error']=dict(type=type(e).__name__,message=str(e));raise
finally:
 signal.setitimer(signal.ITIMER_REAL,0)
 result.update(seconds=time.monotonic()-began,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(0) if 'torch' in globals() and torch.cuda.is_initialized() else None,automatic_retry=False,qualification_states_and_costs_preserved=True,quality_or_promotion_claim=False)
 (OUT/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(complete=result['complete'],seconds=result['seconds'],discarded_updates=result['discarded_updates'],source_manifest_sha256=result['source_manifest_sha256'])))
