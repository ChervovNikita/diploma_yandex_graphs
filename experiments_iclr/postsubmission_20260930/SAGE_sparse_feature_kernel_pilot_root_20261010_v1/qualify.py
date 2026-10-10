"""Nine disposable full-graph TRAIN cases; no VALID scoring or pilot fits."""
from pathlib import Path
import hashlib,importlib.util,json,socket,subprocess,sys,time
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';H=Path(__file__).resolve().parent

def main():
 assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
 frozen=json.loads((H/'QUALIFICATION_FREEZE.json').read_text())
 for row in frozen['bound_files']:assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
 out=H/'actual_qualification_v1';out.mkdir(exist_ok=False)
 source=P/'native_SAGE_sparse_feature_kernel_source_20261010_v1/run_family.py'
 spec=importlib.util.spec_from_file_location('sparse_feature_kernel_actual_train',source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 cfg=H/'CONFIG.json';start=time.perf_counter();family=module.Family(json.loads(cfg.read_text()),out,hashlib.sha256(cfg.read_bytes()).hexdigest());torch=family.torch;torch.set_num_threads(2)
 def forbidden(*args,**kwargs):raise RuntimeError('No VALID quality during disposable qualification')
 family.metrics=forbidden
 support=family.feature_support_metadata;assert support['new_directed_pairs']>0 and not support['labels_or_predictions_used']
 cases=[];trajectories=0;updates=0;provider=None
 for arm in module.ARMS:
  bank=arm.startswith('shared4_');members=4 if bank else 1
  factorized=arm!='ordinary_M1_feature_kernel';mode='common' if arm=='shared4_common_kernel' else 'fixed' if arm=='shared4_fixed_feature_union' else 'private'
  kept=[];identities=[]
  for member in range(4 if arm=='genuine_factorized_I4_feature_kernel' else 1):
   model,opt,streams=family.make(7301,member,'baseline',factorized,members,mode)
   if arm=='genuine_factorized_I4_feature_kernel':
    ids={id(v) for v in model.parameters()};assert all(not ids.intersection(old) for old in identities);identities.append(ids);kept.append((model,opt,streams))
   model.train();opt.zero_grad(set_to_none=True)
   with family.scope(streams,0) if not bank else __import__('contextlib').nullcontext():logits=family.logits(model,streams,bank,family.train_ids)
   trajectories+=members;assert tuple(logits.shape)==(members,580,10) and torch.isfinite(logits).all()
   loss=family.F.cross_entropy(logits.flatten(0,1),family.data.y[family.train_ids].repeat(members));assert torch.isfinite(loss);loss.backward()
   assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
   kernel=model._feature_kernel;grad=None
   if mode!='fixed':
    assert kernel.projection.grad is not None and kernel.projection.grad.abs().sum().item()>0
    assert kernel.metric_raw.grad is not None and kernel.metric_raw.grad.abs().sum().item()>0
    assert any(v is kernel.projection for g in opt.param_groups for v in g['params']) and any(v is kernel.metric_raw for g in opt.param_groups for v in g['params'])
    grad={'projection_absolute_sum':float(kernel.projection.grad.abs().sum()),'metric_absolute_sum':float(kernel.metric_raw.grad.abs().sum())}
   assert kernel._current_weights is None and kernel._current_edges is None
   desc=kernel.describe();provider=desc['native_PyG_source']
   opt.step();updates+=1
   saved={k:v.detach().clone() for k,v in model.state_dict().items() if k.startswith('_feature_kernel.')}
   if saved:
    with torch.no_grad():kernel.projection.add_(1)
    state=model.state_dict();state.update(saved);model.load_state_dict(state,strict=True)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in saved.items())
   cases.append(dict(arm=arm,member=member,routes=members,mode=mode,TRAIN_loss=float(loss.detach()),kernel_gradients=grad,registered_kernel_parameters=desc['registered_parameters'],provider=provider,weighted_mean_calls=desc['counters']['weighted_mean_calls'],cache_cleared_after_factual_forward=True,all_gradients_finite=True,new_parameters_restored=True))
   del logits,loss,kernel,desc,saved
   if arm!='genuine_factorized_I4_feature_kernel':del model,opt,streams
  kept.clear();torch.cuda.empty_cache()
 torch.cuda.synchronize()
 v=dict(qualified=True,cases=cases,seconds=time.perf_counter()-start,native_forwards=9,native_member_trajectories=trajectories,full_parameter_backwards=9,discarded_optimizer_updates=updates,scientific_fits=0,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),support=support,native_provider=provider,VALID_labels_loaded_only_by_existing_role_loader=True,VALID_quality_scored=False,TEST_access=False,scientific_quality_evidence=False,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
 assert len(cases)==9 and trajectories==18 and updates==9
 (out/'QUALIFICATION.json').write_text(json.dumps(v,indent=2)+'\n')
 print(json.dumps(dict(qualified=True,cases=9,seconds=v['seconds'],peak_GiB=v['cuda_peak_allocated_bytes']/1024**3,new_edges=support['new_directed_pairs'],quality_evidence=False)))
if __name__=='__main__':main()
