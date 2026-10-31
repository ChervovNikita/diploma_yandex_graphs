"""Root-admitted exact native Q/K GRAPH/TRAIN resource check; no dispatch owner.

Preparation is standard-library only. Numerical imports occur only in a
separately admitted single-case invocation on the verified allocation. Reuse
reviewed RoleIO/OptimizerDenial/TensorCommitDenial unchanged; no new guard.
"""
from __future__ import annotations
from contextlib import ExitStack
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import signal
import socket
import statistics
import subprocess
import sys
import time
from unittest import mock

HOST='anogena-2-0';UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
H_SHA='effa70e31e64881322ccfb5a6e7e3903ccdb6f887c694a97f6b63d0bd2e267e0'
Z_SHA='259ea644b251012da75c7010746e231cc70243cfd462eaef9b04464350adab41'
ARMS=('full_qk_be_init','original_gnnm')
DATASETS=('wikics','amazon-photo')
EXPECTED={'model_forwards':8,'backward_calls':8,'autograd_grad_VJP_calls':0,'optimizer_attempts':1}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def fingerprint(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def local(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Project-relative path required')
    p=root/p
    if p.is_symlink() or not p.resolve().is_relative_to(root):raise ValueError('Confined exact source required')
    return p
def receipt(root,path):
    raw=path.read_bytes();return {'path':str(path.resolve().relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bound(root,row):
    p=local(root,row['path'])
    if receipt(root,p)!=row:raise ValueError('Exact binding changed')
    return p
def module(root,row,name):
    p=bound(root,row);s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def route():
    if socket.gethostname()!=HOST or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()!=[UUID]:
        raise ValueError('Exact authorized allocation hostname/sole UUID required')


def warm_steps(iterations=4):
    if iterations!=4:raise ValueError('Exactly one cold plus three restored warm callbacks')
    return (1,)*iterations


def validate_proposal(proposal):
    if (proposal.get('schema')!='native-polynormer-qk-resource-proposal-v1' or proposal.get('scientific_GO') is not False
            or proposal.get('execution_authorized') is not False or proposal.get('max_actual_optimizer_updates')!=0):
        raise ValueError('Disabled engineering-only proposal required')
    cases=proposal['requested_cases']
    if len(cases)!=8 or {(c['dataset'],c['arm'],c['branch']) for c in cases}!={(d,a,b) for d in DATASETS for a in ARMS for b in ('local','global')}:
        raise ValueError('Exact finite eight-case roster required')
    for c in cases:
        if (c['expected_actions_per_cycle']!=EXPECTED or c['shape_config']!={'width':512,'depth':7,'global_layers':2,'heads':1,'members':4,'classes':c['model_graph_shape']['classes']}
                or c['representative_mask']!=0 or c['train_rows']!=max(c['all_mask_TRAIN_counts'].values())
                or c['warm_callback_steps']!=list(warm_steps())):
            raise ValueError('Exact source geometry/two-view callback law required')
    return cases


def state_hashes(torch,h,z,adapter):
    return {'parameters':h.tensor_digest(torch,adapter.model.named_parameters()),
            'moments':h.tensor_digest(torch,h.optimizer_tensors(adapter)),
            'buffers':h.tensor_digest(torch,adapter.model.named_buffers()),
            'external':z.tree_digest(torch,adapter.capture_state())}


def tensor_bytes(torch,value):
    if torch.is_tensor(value):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(torch,v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(tensor_bytes(torch,v) for v in value)
    return 0


def node_probe(torch,h,z,root,case,study,run,output,admission):
    inputs=run['config']['node_inputs']
    allowed=[bound(root,inputs[key]) for key in ('graph','train')]
    forbidden=set().union(*(h.role_paths(study['data']['roles'][r]) for r in ('select','cal','evaluate')))
    for row in study['runs']:forbidden.update(h.role_paths(row['config'].get('endpoint_inputs',{})))
    if not set(allowed).isdisjoint(local(root,p) for p in forbidden):raise ValueError('TRAIN authority overlaps prohibited roles')
    runtime=local(root,admission['runtime_package']);sys.path.insert(0,str(runtime))
    with h.OptimizerDenial(torch) as denial,h.RoleIO(torch,root,allowed,[local(root,p) for p in forbidden],output) as io_guard:
        clone,engineering_run,stub=h.engineering_clone(torch,root,study,{'run':run},output)
        io_guard.allowed.add(stub.resolve());clone['runs']=[engineering_run]
        from run_study import load_factory
        adapter=load_factory(root,engineering_run['adapter'])(torch=torch,project_root=root,study=clone,run=engineering_run,device='cuda:0')
        for name in ('validate','observe_validation','evaluate','export'):
            if hasattr(adapter,name):setattr(adapter,name,lambda *a,**k:(_ for _ in ()).throw(h.ForbiddenRoleAccess('Selectors/metrics/endpoints prohibited')))
        # Explicit random untrained geometry, never a learned local selection.
        adapter.model.set_global(case['branch']=='global')
        session=adapter.source_session
        if session.steps!=0 or len(session.optimizers)!=1 or adapter.optimizer is not session.optimizers[0]:
            raise ValueError('One fresh exact native source Adam owner required')
        if adapter.capture_state()['external']['local']['snapshot'] is not None:raise ValueError('No fabricated selected local snapshot')
        h.allocate_adam_moments(torch,adapter)
        moments=[v for _,v in h.optimizer_tensors(adapter)]
        extra_workspace=[torch.empty_like(p) for p in adapter.model.parameters() for _ in range(2)]
        initial=copy.deepcopy(adapter.capture_state());modes={n:m.training for n,m in adapter.model.named_modules()}
        before=state_hashes(torch,h,z,adapter)
        if len(adapter.train_ids)!=case['train_rows']:raise ValueError('Actual maximum TRAIN shape changed')
        counts={};observations=[];times=[];gradient_bytes=0
        old_grad,old_backward=torch.autograd.grad,torch.Tensor.backward
        def counted_grad(*a,**kw):counts['autograd_grad_VJP_calls']+=1;return old_grad(*a,**kw)
        def counted_backward(value,*a,**kw):counts['backward_calls']+=1;return old_backward(value,*a,**kw)
        def counted_forward(*ignored):counts['model_forwards']+=1
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(torch.autograd,'grad',counted_grad))
            stack.enter_context(mock.patch.object(torch.Tensor,'backward',counted_backward))
            for wrapper in adapter.model.models:stack.callback(wrapper.body.register_forward_hook(counted_forward).remove)
            for step in warm_steps():
                adapter.restore_state(copy.deepcopy(initial))
                for name,m in adapter.model.named_modules():m.training=modes[name]
                if session.steps!=0 or any(w.body._global!=(case['branch']=='global') for w in adapter.model.models):
                    raise ValueError('Random state/explicit branch not restored before step1')
                counts.update({k:0 for k in EXPECTED});attempts=denial.denied_step_calls
                torch.cuda.synchronize();tick=time.perf_counter()
                with z.TensorCommitDenial(torch,list(adapter.model.parameters()),moments) as commits:
                    ignored=adapter.train_step(step);del ignored
                torch.cuda.synchronize();times.append(time.perf_counter()-tick)
                counts['optimizer_attempts']=denial.denied_step_calls-attempts
                if counts!=EXPECTED or commits.denied or denial.denied_functional_calls:raise ValueError('Exact native source eight-forward/eight-backward/one denied Adam law changed')
                if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in adapter.model.parameters()):raise FloatingPointError('Nonfinite random source gradient')
                gradient_bytes=max(gradient_bytes,sum(p.grad.numel()*p.element_size() for p in adapter.model.parameters() if p.grad is not None))
                current=state_hashes(torch,h,z,adapter)
                if current['parameters']!=before['parameters'] or current['moments']!=before['moments']:raise ValueError('Parameter/manual-moment bytes changed')
                observations.append(dict(counts))
        adapter.restore_state(copy.deepcopy(initial))
        for name,m in adapter.model.named_modules():m.training=modes[name]
        if before!=state_hashes(torch,h,z,adapter):raise ValueError('All native source RNG/counters/branch/buffers not restored')
        adapter.model.eval();torch.cuda.synchronize();tick=time.perf_counter()
        with torch.no_grad():
            logits=adapter.logits_fn(adapter.model,False,0)
            if logits.shape!=(4,case['model_graph_shape']['nodes'],case['model_graph_shape']['classes']) or not torch.isfinite(logits).all():
                raise ValueError('Finite native full-graph serving geometry required')
            serving_shape=list(logits.shape);del logits
        torch.cuda.synchronize();serving_seconds=time.perf_counter()-tick
        adapter.restore_state(copy.deepcopy(initial))
        for name,m in adapter.model.named_modules():m.training=modes[name]
        from engine import capture_checkpoint,restore_checkpoint,_atomic_checkpoint
        if Path(sys.modules['engine'].__file__).resolve()!=runtime/'engine.py':raise ValueError('Frozen v6 serializer differs')
        tick=time.perf_counter();payload=capture_checkpoint(torch,adapter,fit_id='ENGINEERING_RANDOM_UNTRAINED_UNSELECTED_NATIVE_QK',step=0,validation={},decision={});clone_seconds=time.perf_counter()-tick
        if payload['adapter_state']['external']['local']['snapshot'] is not None:raise ValueError('Only actually empty nested local slot allowed')
        missing_local_snapshot_tensor_bytes=tensor_bytes(torch,payload['model_state'])+tensor_bytes(torch,payload['optimizer_state'])
        snapshot=output/'DISPOSABLE_RANDOM_NATIVE_QK_STATE.pt';io_guard.allowed.add(snapshot.resolve())
        tick=time.perf_counter();size=_atomic_checkpoint(torch,snapshot,payload);save_seconds=time.perf_counter()-tick;saved=receipt(root,snapshot);del payload
        torch.rand(3,device='cuda:0');adapter.model.eval();tick=time.perf_counter()
        restore_checkpoint(torch,snapshot,adapter,expected_fit_id='ENGINEERING_RANDOM_UNTRAINED_UNSELECTED_NATIVE_QK',verify=True)
        restore_seconds=time.perf_counter()-tick
        if before!=state_hashes(torch,h,z,adapter):raise ValueError('Exact native source/branch/global/private RNG restore differs')
        snapshot.unlink();stub_receipt=receipt(root,stub);stub.unlink()
        return {'scope':'random actual-GRAPH/TRAIN source geometry, no optimizer kernels or selection',
                'branch':case['branch'],'branch_is_untrained_engineering_geometry':True,'warm_callback_steps':list(warm_steps()),
                'cycle_action_counts':observations,'cold_callback_seconds':times[0],'warm_callback_seconds':times[1:],'warm_median_seconds':statistics.median(times[1:]),
                'parameter_bytes':sum(p.numel()*p.element_size() for p in adapter.model.parameters()),'gradient_bytes':gradient_bytes,
                'manual_moment_bytes':sum(v.numel()*v.element_size() for v in moments),'extra_workspace_bytes':sum(v.numel()*v.element_size() for v in extra_workspace),
                'full_graph_serving_shape':serving_shape,'full_graph_serving_seconds':serving_seconds,
                'state_hashes_before':before,'state_hashes_after':state_hashes(torch,h,z,adapter),
                'input_load_paths':io_guard.tensor_load_paths,'heldout_accesses':io_guard.denied,
                'unused_SELECT_stub':{**stub_receipt,'removed':True,'synthetic_never_validated':True},
                'random_engine_restore':{'snapshot':saved,'bytes':size,'clone_seconds':clone_seconds,'save_seconds':save_seconds,'restore_seconds':restore_seconds,'removed':True,'selected_checkpoint':False},
                'missing_learned_local_snapshot_geometry':{'qualified':False,'actual_nested_slot':None,
                    'estimated_additional_model_plus_Adam_tensor_bytes':missing_local_snapshot_tensor_bytes,
                    'estimate_scope':'same-shape tensor payload only; no fabricated selected snapshot or trained-state serialization qualification'},
                'source_nominal_update_counters_are_not_optimizer_evidence':True,'optimizer_attempts':denial.denied_step_calls,
                'optimizer_functional_attempts':denial.denied_functional_calls,'actual_optimizer_kernels':denial.actual_optimizer_kernels,
                'actual_optimizer_updates':0,'selected_metrics':0,'actual_endpoint_targets_opened':False,'extra_workspace_lifetime':'retained through capture/restore'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','admission','case-id','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args();route();root=Path(args.project_root).resolve();output=local(root,args.output)
    admission_path=local(root,args.admission);admission=json.loads(admission_path.read_text());self_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (admission.get('schema')!='root-approved-zero-update-resource-admission-v1' or admission.get('root_source_review_confirmed') is not True
            or admission.get('helper_sha256')!=self_hash or admission.get('scientific_GO') is not False
            or admission.get('max_actual_optimizer_updates')!=0 or args.case_id not in admission.get('approved_case_ids',[])):
        raise ValueError('Fresh exact root source engineering admission required')
    proposal=json.loads(bound(root,admission['proposal_receipt']).read_text());cases=validate_proposal(proposal)
    for row in admission['runtime_dependency_receipts']:bound(root,row)
    case=next(c for c in cases if c['case_id']==args.case_id)
    raw=bound(root,case['study_receipt']).read_bytes();study=json.loads(gzip.decompress(raw))
    if fingerprint(study)!=case['study_hash'] or study['dataset']!=case['dataset']:raise ValueError('Frozen source study changed')
    run=next(r for r in study['runs'] if r['run_id']==case['source_run_id'])
    if fingerprint(run)!=case['source_run_hash'] or run['config']['polynormer_qk_arm']!=case['arm']:raise ValueError('Exact source run differs')
    if output.exists():raise ValueError('Fresh output required; no automatic retry')
    if admission['role_optimizer_cap_helper']['sha256']!=H_SHA or admission['tensor_commit_tree_helper']['sha256']!=Z_SHA:
        raise ValueError('Unchanged reviewed h/z helper bytes required')
    h=module(root,admission['role_optimizer_cap_helper'],'native_qk_inherited_h')
    z=module(root,admission['tensor_commit_tree_helper'],'native_qk_inherited_z')
    old=signal.getsignal(signal.SIGALRM)
    def timeout(*ignored):raise h.ResourceTimeout('Bounded native Q/K engineering case')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(case['timeout_seconds']);output.mkdir();owner=h.own_identity();started=time.monotonic()
    (output/'START.json').write_text(json.dumps({'case_id':args.case_id,'owner':owner,'admission':receipt(root,admission_path),'helper_sha256':self_hash,'scientific_fits':0,'GO':False},indent=2)+'\n')
    try:
        import torch
        torch.set_num_threads(2);cap,before_usage,after_init=h.configure_cuda_allocator(torch,output);torch.cuda.reset_peak_memory_stats()
        result=node_probe(torch,h,z,root,case,study,run,output,admission)
        result.update(status='RANDOM_NATIVE_QK_ZERO_UPDATE_ENGINEERING_COMPLETE',case_id=args.case_id,owner=owner,elapsed_wall_seconds=time.monotonic()-started,
                      CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      current_child_allocator_cap=cap,GPU_usage_before_init=before_usage,GPU_usage_after_init=after_init,GPU_usage_at_completion=h.GPU_process_usage(),
                      process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scientific_fits=0,GO=False,predictive_scores=0,loss_values_emitted=0,whole_master_admitted=False)
        (output/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
        print(json.dumps({'case_id':args.case_id,'status':result['status'],'actual_optimizer_updates':0,'fits':0,'GO':False}))
    except BaseException as e:
        failure=h.terminal_failure(e,owner,started);failure.update(case_id=args.case_id,helper_sha256=self_hash,GO=False,automatic_retry=False)
        (output/'FAILURE.json').write_text(json.dumps(failure,indent=2)+'\n');raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':main()
