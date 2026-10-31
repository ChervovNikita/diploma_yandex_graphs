"""Reviewed-execution boundary for molecular/link engineering; no scientific fits.

PT/author direct builders remain blueprint-only and are explicitly rejected by
this program. Real execution requires a fresh source freeze and root-reviewed
hash; module import contains no numerical dependency or data read.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import signal
import sys
import time

CAP_HELPER_SHA='effa70e31e64881322ccfb5a6e7e3903ccdb6f887c694a97f6b63d0bd2e267e0'
BLUEPRINT_SHA='b188a3f3ba9a01e064df9545b1f5213e7b43efc3278b0237c301b9f49244891a'
MASTER_HASH='081b1358d83468c8062844288f28c9f741984176d441466df8c7daded82a57ef'
BLUEPRINT='practical_resource_extension_blueprint_20261012_v2/BLUEPRINT.json'
CAP_HELPER='practical_resource_qualifier_source_20261012_v2/runtime/qa/practical_zero_update_resources.py'


def load_cap_helpers(root):
    path=root/CAP_HELPER
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CAP_HELPER_SHA:
        raise ValueError('Exact previously reviewed denial/cap helper required')
    spec=importlib.util.spec_from_file_location('reviewed_resource_denial_cap_v2',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def synthetic_molecular_select(torch):
    return {'x':torch.zeros((1,9),dtype=torch.long),'edge_index':torch.empty((2,0),dtype=torch.long),
            'edge_attr':torch.empty((0,3),dtype=torch.long),'node_ptr':torch.tensor([0,1]),
            'edge_ptr':torch.tensor([0,0]),'ids':torch.tensor([2**62]),'y':torch.zeros((1,1))}


def synthetic_link_select(torch,graph,train,k):
    """Only public endpoints and actual TRAIN exclusions; no target files."""
    n=len(graph['x']);pairs=train['positive']
    keys=(torch.minimum(pairs[:,0],pairs[:,1])*n+torch.maximum(pairs[:,0],pairs[:,1])).sort().values
    chosen=[];budget=1_000_000;tested=0
    for first in range(n-1):
        if tested>=budget:break
        last=min(n,first+1+budget-tested)
        second=torch.arange(first+1,last,dtype=torch.long);tested+=len(second)
        candidates=first*n+second;positions=torch.searchsorted(keys,candidates)
        safe=positions<len(keys)
        overlaps=safe & (keys[positions.clamp(max=max(0,len(keys)-1))]==candidates)
        for endpoint in second[~overlaps][:k+1-len(chosen)].tolist():chosen.append([first,endpoint])
        if len(chosen)==k+1:break
    if len(chosen)!=k+1:
        raise ValueError('Bounded public GRAPH/TRAIN synthetic interface construction lacks enough nonTRAIN pairs')
    return {'positive':torch.tensor(chosen[:1],dtype=torch.long),'negative':torch.tensor(chosen[1:],dtype=torch.long)}


def engineering_clone(torch,h,root,study,case,output):
    run=copy.deepcopy(case['declared_source_run']);family=case['family']
    key='molecular_inputs' if family=='molecular' else 'link_inputs';inputs=run['config'][key]
    if family=='molecular':pack=synthetic_molecular_select(torch)
    else:
        graph=torch.load(h.bound(root,inputs['graph']),map_location='cpu',weights_only=True)
        train=torch.load(h.bound(root,inputs['train']),map_location='cpu',weights_only=True)
        pack=synthetic_link_select(torch,graph,train,run['config']['metric']['hits_k'])
    stub=output/'SYNTHETIC_UNUSED_SELECT_INTERFACE.pt';torch.save(pack,stub)
    row=h.receipt(root,stub);old=inputs['select'];clone=copy.deepcopy(study);clone['stage']='engineering'
    clone['data']['roles']['select']=copy.deepcopy(row)
    for declared in clone['runs']:
        if declared['kind']=='model_fit' and declared['config'].get(key,{}).get('select')==old:
            declared['config'][key]['select']=copy.deepcopy(row)
    run['config'][key]['select']=copy.deepcopy(row)
    return clone,run,stub


def measure_epochs(torch,h,adapter,denial,output,warm_iterations=3,capture=None):
    for name in ('validate','observe_validation','evaluate','export'):
        if hasattr(adapter,name):
            setattr(adapter,name,lambda *a,**k:(_ for _ in ()).throw(h.ForbiddenRoleAccess('Original selector/endpoint prohibited')))
    h.allocate_adam_moments(torch,adapter)
    before=h.tensor_digest(torch,adapter.model.named_parameters());moment_before=h.tensor_digest(torch,h.optimizer_tensors(adapter))
    device=next(adapter.model.parameters()).device
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    times=[];operations=[];gradient_bytes=0
    for epoch in range(1,warm_iterations+2):
        sync();start=time.perf_counter();ignored=adapter.train_step(epoch);sync();times.append(time.perf_counter()-start)
        counts=ignored.get('operation_counts',{})
        operations.append({k:v for k,v in counts.items() if 'optimizer' not in k and type(v) is int})
        del ignored  # Returned source losses/nominal updates/metrics are never saved.
        gradient_bytes=max(gradient_bytes,sum(p.grad.numel()*p.grad.element_size() for p in adapter.model.parameters() if p.grad is not None))
    after=h.tensor_digest(torch,adapter.model.named_parameters());moment_after=h.tensor_digest(torch,h.optimizer_tensors(adapter))
    if before!=after or moment_before!=moment_after or denial.denied_functional_calls:
        raise ValueError('Parameter/moment equality failed or optimizer kernel attempted')
    if not denial.denied_step_calls:raise ValueError('Actual source TRAIN path did not reach intercepted optimizer actions')
    serialization={'scope':'unmeasured'}
    if capture is not None:
        start=time.perf_counter();payload=capture(torch,adapter,fit_id='ENGINEERING_RANDOM_NEVER_PAID_OR_SELECTED',step=0,validation={},decision={});clone_seconds=time.perf_counter()-start
        path=output/'RANDOM_ENGINEERING_SNAPSHOT.pt';start=time.perf_counter();torch.save({'engineering_only':True,'never_scientific_ancestor':True,'never_selected':True,'payload':payload},path);save_seconds=time.perf_counter()-start
        serialization={'scope':'random_full_root_state_serializer_proxy_only','bytes':path.stat().st_size,'clone_seconds':clone_seconds,
                       'save_seconds':save_seconds,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected_checkpoint':False}
        path.unlink();del payload
    import statistics
    return {'scope':'RANDOM_TRAIN_FULL_EPOCH_FWD_BACKWARD_ZERO_UPDATE_PROXY_NOT_ETA',
            'parameters_sha256_before':before,'parameters_sha256_after':after,'moments_sha256_before':moment_before,'moments_sha256_after':moment_after,
            'parameter_bytes':sum(p.numel()*p.element_size() for p in adapter.model.parameters()),'gradient_bytes':gradient_bytes,
            'optimizer_moment_bytes':sum(v.numel()*v.element_size() for _,v in h.optimizer_tensors(adapter)),
            'optimizer_objects':len(h.optimizers(adapter)),'cold_TRAIN_epoch_seconds':times[0],'warm_TRAIN_epoch_seconds':times[1:],
            'warm_TRAIN_epoch_median_seconds':statistics.median(times[1:]),'actual_source_TRAIN_operation_counts_without_nominal_updates':operations,
            'denied_optimizer_step_calls':denial.denied_step_calls,'actual_optimizer_updates':0,'actual_optimizer_kernel_calls':0,
            'zero_update_equality_proof_complete':True,'original_selector_or_endpoint_qualification':False,
            'all_fit_ETA':None,'serialization':serialization,'scientific_fits':0,'accuracy_calls':0,'scores_emitted':0}


def terminal_failure(h,error,owner,started):
    value=h.terminal_failure(error,owner,started);types=[];seen=set();current=error
    while current is not None and id(current) not in seen:
        seen.add(id(current));types.append(type(current).__name__);current=current.__cause__ or current.__context__
    value['failure_chain_types']=types
    if any('OutOfMemory' in name for name in types):value['status']='RESOURCE_DEFERRED'
    return value


def execute(root,h,case_id,output):
    path=h.local(root,BLUEPRINT)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=BLUEPRINT_SHA:raise ValueError('Exact reviewed blueprint changed')
    blueprint=json.loads(path.read_text());master=json.loads(h.bound(root,blueprint['master_binding']).read_text())
    if h.digest(master)!=MASTER_HASH or master['scientific_launch_enabled'] is not False:raise ValueError('Exact disabled v3 master changed')
    matches=[c for c in blueprint['cases'] if c['case_id']==case_id]
    if len(matches)!=1:raise ValueError('One exact reviewed extension case required')
    case=matches[0]
    if case['family'] not in {'molecular','link'}:raise ValueError('PT/author remain direct-builder blueprint only; never call paid-parent or native CLI')
    raw=h.bound(root,case['study_binding']).read_bytes();study=json.loads(gzip.decompress(raw) if case['study_binding']['path'].endswith('.gz') else raw)
    if h.digest(study)!=case['source_spec_hash'] or not any(e['spec_hash']==case['source_spec_hash'] and e['path']==case['study_binding']['path'] for e in master['studies']):raise ValueError('Exact source study outside v3 master')
    run=case['declared_source_run']
    if run!=next(r for r in study['runs'] if r['run_id']==run['run_id']):raise ValueError('Actual source configuration changed')
    if Path(run['adapter'].partition(':')[0]).name not in {'molecular.py','molecular_boundary.py','link_boundary.py'} or run['adapter'].partition(':')[2]!='make_adapter':raise ValueError('Factory outside reviewed modality scope')
    for row in case['source_files']:h.bound(root,row)
    inputs=run['config']['molecular_inputs' if case['family']=='molecular' else 'link_inputs']
    allowed=set().union(*(h.role_paths(v) for k,v in inputs.items() if k in {'graph','train'}))
    forbidden=set(case['prohibited_original_role_paths'])
    import torch
    sys.dont_write_bytecode=True;torch.set_num_threads(2)
    cap,before_usage,after_usage=h.configure_cuda_allocator(torch,output);torch.cuda.reset_peak_memory_stats(0)
    guard=h.RoleIO(torch,root,[root/p for p in allowed],[root/p for p in forbidden],output)
    with h.OptimizerDenial(torch) as denial,guard:
        clone,resource_run,stub=engineering_clone(torch,h,root,study,case,output);guard.allowed.add(stub.resolve())
        from run_study import load_factory
        from engine import capture_checkpoint
        start=time.perf_counter();adapter=load_factory(root,run['adapter'])(torch=torch,project_root=root,study=clone,run=resource_run,device='cuda:0');torch.cuda.synchronize();constructor=time.perf_counter()-start
        if denial.denied_step_calls or denial.denied_functional_calls:raise ValueError('Factory attempted update before TRAIN proxy')
        value=measure_epochs(torch,h,adapter,denial,output,3,capture_checkpoint)
        value.update(constructor_seconds=constructor,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                     process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,tensor_load_paths=guard.tensor_load_paths,
                     heldout_original_file_accesses=guard.denied,synthetic_unused_SELECT_interface=h.receipt(root,stub),original_spec_unchanged=True,
                     input_authority='explicit_engineering_clone_actual_GRAPH_TRAIN_plus_wholly_synthetic_unused_SELECT')
    value.update(case_id=case_id,family=case['family'],master_hash=MASTER_HASH,blueprint_sha256=BLUEPRINT_SHA,
                 helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),allocator_cap=cap,GPU_usage_before_init=before_usage,
                 GPU_usage_after_init=after_usage,GPU_usage_at_completion=h.GPU_process_usage(),normal_child_allocator_cap_only=True)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--case-id',required=True);parser.add_argument('--reviewed-helper-sha256',required=True)
    parser.add_argument('--case-timeout-seconds',type=int,default=600)
    args=parser.parse_args()
    import subprocess
    assert subprocess.check_output(['hostname'],text=True).strip()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
    root=Path(args.project_root).resolve();h=load_cap_helpers(root);output=h.local(root,args.output);output.mkdir(exist_ok=False)
    owner=h.own_identity();started=time.monotonic();current=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'OWNER_START.json').write_text(json.dumps({'owner':owner,'time_unix':time.time(),'helper_sha256':current,'case_id':args.case_id},indent=2)+'\n')
    try:
        if args.reviewed_helper_sha256!=current:raise ValueError('Exact root-reviewed fresh helper required')
        if not 1<=args.case_timeout_seconds<=900:raise ValueError('Finite timeout1..900 required')
        def expired(signum,frame):raise h.ResourceTimeout('Owned modality resource deadline/termination')
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.alarm(args.case_timeout_seconds)
        value=execute(root,h,args.case_id,output);value.update(status='TRAIN_PROXY_COMPLETE',owner=owner,elapsed_wall_seconds=time.monotonic()-started,automatic_retry=False)
        (output/'RESULT.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        print(h.canonical({'resource_proxy_completed':True,'actual_optimizer_updates':0,'scientific_fits':0,'scores_emitted':0}))
    except BaseException as error:
        value=terminal_failure(h,error,owner,started);(output/'TERMINAL_FAILURE.json').write_text(json.dumps(value,indent=2)+'\n')
        print(h.canonical({'resource_proxy_completed':False,'status':value['status'],'failure_type':value['failure_type'],'scientific_fits':0,'scores_emitted':0,'automatic_retry':False}));raise SystemExit(2)
    finally:signal.alarm(0)


if __name__=='__main__':main()
