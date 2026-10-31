"""Two root-authorized exact failed NSD source callbacks;zero-update localization."""
from pathlib import Path
import argparse,copy,gzip,hashlib,importlib.util,json,os,socket,subprocess,sys,time
HOST='anogena-2-0';UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def receipt(root,p):
 b=Path(p).read_bytes();return {'path':str(Path(p).relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def module(root,row,name):
 p=root/row['path'];assert receipt(root,p)==row;s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 assert socket.gethostname()==HOST and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[UUID]
 root=Path(a.root).resolve();out=root/a.output;assert not out.exists();out.mkdir(parents=True)
 prior=json.loads((root/'nsd_resource_admission_root_20261012_v1/ENGINEERING_ADMISSION.json').read_text());assert prior['root_source_review_confirmed'] and prior['max_actual_optimizer_updates']==0 and a.case in prior['approved_case_ids'] and a.case in ('nsd-layer_shared_incidence','nsd-node_capacity_right')
 base=module(root,{'path':'native_nsd_resource_source_review_20261012_v1/qualify_nsd.py','bytes':27516,'sha256':'81d10d693954d7a28039375e8a8fa6b64118f59aef823825526d7a4aed277eb7'},'nsd_exact_reviewed_resource_base');h=module(root,prior['role_optimizer_cap_helper'],'nsd_localization_denial_cap');z=module(root,prior['tensor_commit_tree_helper'],'nsd_localization_tensor_denial')
 proposal=json.loads(base.bound(root,prior['proposal_receipt']).read_text());case=next(c for c in proposal['requested_cases'] if c['case_id']==a.case);study=json.loads(gzip.decompress(base.bound(root,prior['case_study_receipts'][a.case]).read_bytes()));run=next(r for r in study['runs'] if r['run_id']==case['source_run_id']);assert base.sha(run)==case['source_run_hash']
 runtime=root/prior['runtime_package'];sys.path.insert(0,str(runtime));results={'case':a.case,'source_run_hash':case['source_run_hash'],'seed':run['config']['initialization_seed'],'scope':'exact failed actual GRAPH/TRAIN source callback;original RNG/recipe,not alternative seed;optimizer denied;no selectors/scores','normalizers':[],'parameter_gradients':[],'optimizer_updates':0,'GO':False};tick=time.monotonic()
 try:
  import torch
  torch.set_num_threads(2);cap,before_usage,after_init=h.configure_cuda_allocator(torch,out);torch.cuda.reset_peak_memory_stats()
  inputs=run['config']['node_inputs'];allowed=[base.bound(root,inputs[k]) for k in ('graph','train')];custody=json.loads(base.bound(root,prior['custody_receipt']).read_text());forbidden=[root/row['node_inputs']['select']['path'] for row in custody['splits']]+[root/row['endpoint_inputs'][r+'_targets']['path'] for row in custody['splits'] for r in ('cal','evaluate')]
  with h.OptimizerDenial(torch) as denial,h.RoleIO(torch,root,allowed,forbidden,out) as io:
   clone,engineering_run,stub=base.engineering_clone(torch,h,root,study,run,out);io.allowed.add(stub.resolve())
   from run_study import load_factory
   adapter=load_factory(root,engineering_run['adapter'])(torch=torch,project_root=root,study=clone,run=engineering_run,device='cuda:0')
   for name in ('validate','observe_validation','evaluate','export'):
    if hasattr(adapter,name):setattr(adapter,name,lambda *a,**k:(_ for _ in ()).throw(h.ForbiddenRoleAccess('Selectors/target metrics prohibited')))
   params=list(adapter.model.parameters());params_before=h.tensor_digest(torch,adapter.model.named_parameters());rng_before=z.tree_digest(torch,adapter.capture_state());hooks=[];current={'member':None,'layer':None};lap=None
   def grad_stats(t):
    v=t.detach();finite=torch.isfinite(v);return {'coordinates':v.numel(),'nonfinite':int((~finite).sum()),'NaN':int(torch.isnan(v).sum()),'infinite':int(torch.isinf(v).sum())}
   for m,member in enumerate(adapter.model.members):
    for l,learner in enumerate(member.sheaf_learners):
     def mark(module,args,m=m,l=l):current.update(member=m,layer=l)
     hooks.append(learner.register_forward_pre_hook(mark))
    lap=member.laplacian_builder.normalise.__globals__['lap']
   original=lap.batched_sym_matrix_pow
   def observed_power(A,p):
    output=original(A,p);diag=A.detach().diagonal(dim1=-2,dim2=-1);off=A.detach()-torch.diag_embed(diag);values=diag.sort(dim=-1,descending=True).values;gaps=values[:,:-1]-values[:,1:];repeated=(gaps==0).any(-1);good=values>values.max(-1,keepdim=True).values*4*torch.finfo(values.dtype).eps
    row={'member':current['member'],'layer':current['layer'],'power':p,'shape':list(A.shape),'finite_input':bool(torch.isfinite(A).all()),'maximum_off_diagonal_absolute':float(off.abs().max()),'positive_diagonal_minimum':float(diag.min()),'largest_diagonal':float(diag.max()),'minimum_diagonal_gap':float(gaps.min()),'exact_repeated_nodes':int(repeated.sum()),'first_repeated_node_ids':torch.where(repeated)[0][:20].cpu().tolist(),'masked_singular_values':int((~good).sum()),'finite_forward':bool(torch.isfinite(output).all())};results['normalizers'].append(row)
    def input_grad(g):
     row['input_adjoint']=grad_stats(g);bad=(~torch.isfinite(g)).any(-1).any(-1);row['bad_adjoint_nodes']=int(bad.sum());row['bad_and_repeated_nodes']=int((bad&repeated).sum());row['bad_nonrepeated_nodes']=int((bad&~repeated).sum())
     row['first_bad_node_ids']=torch.where(bad)[0][:20].cpu().tolist()
    A.register_hook(input_grad);output.register_hook(lambda g:row.update(output_adjoint=grad_stats(g)))
    return output
   lap.batched_sym_matrix_pow=observed_power
   try:
    with z.TensorCommitDenial(torch,params,[]) as commits:
     ignored=adapter.train_step(1);del ignored
    torch.cuda.synchronize()
   finally:
    lap.batched_sym_matrix_pow=original
    for hook in hooks:hook.remove()
   for name,param in adapter.model.named_parameters():
    if param.grad is not None:
     stats=grad_stats(param.grad)
     if stats['nonfinite']:results['parameter_gradients'].append({'name':name,'shape':list(param.shape),**stats})
   results.update(parameters_unchanged=params_before==h.tensor_digest(torch,adapter.model.named_parameters()),optimizer_attempts=denial.denied_step_calls,optimizer_kernels=denial.actual_optimizer_kernels,protected_commits_denied=commits.denied,heldout_tensor_loads=io.denied,source_rng_before_hash=rng_before,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),full_source_callback_completed=True)
   assert results['parameters_unchanged'] and results['optimizer_attempts']==1 and not results['optimizer_kernels'] and not commits.denied and not io.denied
   stub.unlink()
  results['status']='LOCALIZATION_COMPLETE_NONFINITE_GRADIENTS_PRESERVED'
 except BaseException as error:
  results.update(status='LOCALIZATION_FAILED_RETAINED',error={'type':type(error).__name__,'message':str(error)});raise
 finally:
  results['wall_seconds']=time.monotonic()-tick;(out/'RESULT.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'case':a.case,'status':results['status'],'nonfinite_parameter_names':[v['name'] for v in results['parameter_gradients']],'normalizer_repeated_counts':[v['exact_repeated_nodes'] for v in results['normalizers']],'optimizer_updates':0,'GO':False}))
if __name__=='__main__':main()
