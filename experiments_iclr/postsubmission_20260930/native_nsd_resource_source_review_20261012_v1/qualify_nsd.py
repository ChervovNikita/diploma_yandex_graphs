"""Ten full native NSD callbacks plus one first-action capacity source proposal; root executes only.

Imports numerical APIs only after exact allocation identity and root receipt
checks. Reuses previously reviewed optimizer/role/commit denial helpers. A
fresh reviewed engineering receipt is required; this program cannot make GO.
No remote connection, fit registry, selection, evaluator or optimizer kernel.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest import mock
HOST = 'anogena-2-0'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
ORIGINAL_FIVE = {'romanempire', 'amazonratings', 'minesweeper', 'questions', 'tolokers'}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def verify_route():
    if socket.gethostname() != HOST:
        raise ValueError('Exact authorized allocation hostname required')
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    if result.returncode or result.stdout.split() != [UUID]:
        raise ValueError('Exact sole authorized allocation UUID required')

def local(root, relative):
    value = Path(relative)
    result = root / value
    if value.is_absolute() or '..' in value.parts or result.is_symlink() or (not result.resolve().is_relative_to(root)):
        raise ValueError('Confined non-symlink project path required')
    return result

def receipt(root, path):
    raw = Path(path).read_bytes()
    return {'path': str(Path(path).resolve().relative_to(root)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def bound(root, row):
    path = local(root, row['path'])
    if receipt(root, path) != row:
        raise ValueError('Exact source/request/input binding changed')
    return path

def module(root, row, name):
    path = bound(root, row)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

def state_hashes(torch, h, z, adapter):
    return {'parameters': h.tensor_digest(torch, adapter.model.named_parameters()), 'moments': h.tensor_digest(torch, h.optimizer_tensors(adapter)), 'buffers': h.tensor_digest(torch, adapter.model.named_buffers()), 'external': z.tree_digest(torch, adapter.capture_state())}

def sync(torch, model):
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def require_finite_gradients(torch, model):
    if any((p.grad is not None and (not torch.isfinite(p.grad).all().item()) for p in model.parameters())):
        raise FloatingPointError('Nonfinite exact-source random-state gradient')

def source_static_tensors(torch, adapter):
    """Native plain topology/indices and actual input storage;L diagnostics stay mutable."""
    values = [('TRAIN_ids', adapter.train_ids), ('TRAIN_y', adapter.train_y),
              ('synthetic_SELECT_ids', adapter.select_ids), ('synthetic_SELECT_y', adapter.select_y)]
    values += list(adapter.model.named_buffers())
    for name, body in adapter.model.named_modules():
        values += [(name+'/'+key, value) for key,value in vars(body).items()
                   if torch.is_tensor(value) and key != 'L']
    for index,cell in enumerate(adapter.logits_fn.__closure__ or ()):
        if torch.is_tensor(cell.cell_contents):values.append(('input_closure'+str(index),cell.cell_contents))
    return values


def source_laws(adapter):
    return {'optimizer':type(adapter.optimizer).__name__,
            'groups':[{key:value for key,value in group.items() if key!='params'}
                      for group in adapter.optimizer.param_groups],
            'requires_grad':{name:bool(p.requires_grad) for name,p in adapter.model.named_parameters()},
            'modes':{name:bool(body.training) for name,body in adapter.model.named_modules()}}


def engineering_clone(torch,h,root,study,run,output):
    """Exact GRAPH/TRAIN plus a disjoint synthetic SELECT interface;no actual heldout hash."""
    clone=copy.deepcopy(study);run=copy.deepcopy(run);inputs=run['config']['node_inputs']
    graph=torch.load(bound(root,inputs['graph']),map_location='cpu',weights_only=True)
    train=torch.load(bound(root,inputs['train']),map_location='cpu',weights_only=True)
    mask=torch.ones(len(graph['x']),dtype=torch.bool);mask[train['ids']]=False
    unused=torch.where(mask)[0]
    if not len(unused):raise ValueError('Disjoint nonTRAIN node needed only for unused synthetic interface')
    stub=output/'SYNTHETIC_UNUSED_SELECT_INTERFACE.pt'
    torch.save({'ids':unused[:1].long(),'y':torch.zeros(1,dtype=torch.long)},stub)
    replacement=receipt(root,stub);inputs['select']=replacement;clone['stage']='engineering'
    clone['data']['roles']['select']=copy.deepcopy(replacement)
    clone['data']['roles']['train']=copy.deepcopy(inputs['train']);clone['runs']=[run]
    return clone,run,stub


def load_owned_engine(root,admission):
    row=admission.get('engine_override_receipt')
    if row is not None:return module(root,row,'engine')
    import engine
    if receipt(root,engine.__file__)!=admission['source_engine_receipt']:
        raise ValueError('Exact owned NSD engine required')
    return engine


def node_probe(torch,h,z,root,case,study,run,output,admission):
    actual_inputs=run['config']['node_inputs'];allowed=[bound(root,actual_inputs[k]) for k in ('graph','train')]
    custody=json.loads(bound(root,admission['custody_receipt']).read_text())
    if custody['dataset']!='twitch-de-musae-raw' or custody['model_graph']!={'nodes':9498,'features':3170,'classes':2,'edges':306276}:
        raise ValueError('Fresh authoritative raw3170 source shape required')
    forbidden=[local(root,row['node_inputs']['select']['path']) for row in custody['splits']]
    forbidden += [local(root,row['endpoint_inputs'][role+'_targets']['path']) for row in custody['splits'] for role in ('cal','evaluate')]
    if not set(allowed).isdisjoint(forbidden):raise ValueError('Actual GRAPH/TRAIN file authority overlaps forbidden target files')
    runtime=local(root,admission['runtime_package']);sys.path.insert(0,str(runtime))
    engine=load_owned_engine(root,admission)
    supplied=module(root,admission['nsd_helpers_receipt'],'nsd_reviewed_root_helpers')
    with h.OptimizerDenial(torch) as denial,h.RoleIO(torch,root,allowed,forbidden,output) as io_guard:
        clone,engineering_run,stub=engineering_clone(torch,h,root,study,run,output)
        io_guard.allowed.add(stub.resolve())
        from run_study import load_factory
        engineering_run['adapter']=admission['resource_factory']
        if case.get('resource_active_geometry_prototype'):
            engineering_run['config']['resource_active_geometry_prototype']=True
        adapter=load_factory(root,engineering_run['adapter'])(torch=torch,project_root=root,study=clone,run=engineering_run,device='cuda:0')
        for name in ('validate','observe_validation','evaluate','export'):
            if hasattr(adapter,name):setattr(adapter,name,lambda *a,**k:(_ for _ in ()).throw(h.ForbiddenRoleAccess('Selectors/targets/scores prohibited')))
        if len(adapter.train_ids)!=case['train_rows'] or adapter.provenance['native_args']['input_dim']!=3170:
            raise ValueError('Exact complete raw3170 GRAPH/TRAIN shape differs')
        engine.inspect_ownership(adapter)
        moments_scope=supplied.allocate_source_moments(torch,adapter)
        engine.inspect_ownership(adapter)
        moments=[v for _,v in h.optimizer_tensors(adapter)]
        extra_workspace=[torch.empty_like(p) for p in adapter.model.parameters() if p.requires_grad for _ in range(2)]
        static=source_static_tensors(torch,adapter);fixed_hash=h.tensor_digest(torch,static)
        fixed_versions=[v._version for _,v in static]
        initial=copy.deepcopy(adapter.capture_state());initial_optimizer_hash=z.tree_digest(torch,adapter.optimizer.state_dict())
        modes={name:bool(body.training) for name,body in adapter.model.named_modules()}
        flags={name:bool(p.requires_grad) for name,p in adapter.model.named_parameters()}
        before=state_hashes(torch,h,z,adapter);initial_laws=source_laws(adapter)
        native=[body for body in adapter.model.modules() if type(body).__name__=='DiscreteGeneralSheafDiffusion']
        if len(native)!=case['source_complete_native_modules']:raise ValueError('Whole original native module/body ownership differs')
        counts={key:0 for key in ('native_body_forward_calls','backward_calls','autograd_grad_VJP_calls','optimizer_attempts')}
        cycle_counts=[];times=[];gradient_bytes=0
        original_grad,original_backward=torch.autograd.grad,torch.Tensor.backward
        def counted_grad(*a,**k):counts['autograd_grad_VJP_calls']+=1;return original_grad(*a,**k)
        def counted_backward(value,*a,**k):counts['backward_calls']+=1;return original_backward(value,*a,**k)
        def restore_initial():
            if z.tree_digest(torch,adapter.optimizer.state_dict())!=initial_optimizer_hash:raise ValueError('Source optimizer state changed before restoration')
            adapter.restore_state(copy.deepcopy(initial))
            for name,body in adapter.model.named_modules():body.training=modes[name]
            for name,p in adapter.model.named_parameters():p.requires_grad_(flags[name]);p.grad=None
            engine.inspect_ownership(adapter)
        with ExitStack() as stack:
            for body in native:stack.callback(body.register_forward_hook(lambda *a:counts.__setitem__('native_body_forward_calls',counts['native_body_forward_calls']+1)).remove)
            stack.enter_context(mock.patch.object(torch.autograd,'grad',counted_grad));stack.enter_context(mock.patch.object(torch.Tensor,'backward',counted_backward))
            for index in range(2):
                restore_initial();counts.update({k:0 for k in counts});attempts=denial.denied_step_calls
                sync(torch,adapter.model);tick=time.perf_counter()
                with z.TensorCommitDenial(torch,list(adapter.model.parameters())+[v for _,v in static],moments) as commits:
                    ignored=adapter.train_step(index+1)
                    reported=dict(ignored.get('operation_counts',{}));del ignored
                sync(torch,adapter.model);times.append(time.perf_counter()-tick)
                counts['optimizer_attempts']=denial.denied_step_calls-attempts
                expected={'native_body_forward_calls':case['expected_native_TRAIN_forwards'],'backward_calls':case['expected_backward_calls'],'autograd_grad_VJP_calls':0,'optimizer_attempts':1}
                if counts!=expected or reported.get('backward_calls')!=expected['backward_calls']:
                    raise ValueError('Whole old-state native source forward/backward/Adam-attempt callback differs')
                if commits.denied or denial.denied_functional_calls:raise ValueError('Protected source tensor/functional optimizer action attempted')
                require_finite_gradients(torch,adapter.model);engine.inspect_ownership(adapter)
                gradient_bytes=max(gradient_bytes,sum(p.grad.numel()*p.grad.element_size() for p in adapter.model.parameters() if p.grad is not None))
                current=state_hashes(torch,h,z,adapter)
                if current['parameters']!=before['parameters'] or current['moments']!=before['moments'] or h.tensor_digest(torch,static)!=fixed_hash or any(v._version!=version for (_,v),version in zip(static,fixed_versions)):
                    raise ValueError('Actual Parameters/moments/immutable native inputs or topology changed')
                cycle_counts.append(dict(counts))
        restore_initial()
        if before!=state_hashes(torch,h,z,adapter) or initial_laws!=source_laws(adapter):raise ValueError('Source RNG/modes/flags/ownership/optimizer law not exactly restored')
        adapter.model.eval();sync(torch,adapter.model);tick=time.perf_counter()
        with z.TensorCommitDenial(torch,list(adapter.model.parameters())+[v for _,v in static],moments) as commits,torch.no_grad():
            logits=adapter.logits_fn(adapter.model,False,0)
            if tuple(logits.shape)!=(case['shape_config']['members'],9498,2) or not torch.isfinite(logits).all():raise ValueError('Whole source finite raw3170 serving shape required')
            serving_shape=list(logits.shape);serving_cpu_bytes=logits.numel()*logits.element_size();del logits
        sync(torch,adapter.model);serving_seconds=time.perf_counter()-tick
        if commits.denied or h.tensor_digest(torch,static)!=fixed_hash:raise ValueError('Serving mutated protected source state')
        # Preserve the populated native L diagnostics in the complete real engine snapshot.
        serving_state=state_hashes(torch,h,z,adapter);serving_laws=source_laws(adapter)
        tick=time.perf_counter();payload=engine.capture_checkpoint(torch,adapter,fit_id='ENGINEERING_RANDOM_UNTRAINED_UNSELECTED',step=0,validation={},decision={});clone_seconds=time.perf_counter()-tick
        snapshot=output/'DISPOSABLE_RANDOM_COMPLETE_NSD_STATE.pt';io_guard.allowed.add(snapshot.resolve());tick=time.perf_counter();size=engine._atomic_checkpoint(torch,snapshot,payload);save_seconds=time.perf_counter()-tick;saved=receipt(root,snapshot);del payload
        torch.rand(3,device='cuda:0');adapter.model.train();tick=time.perf_counter();engine.restore_checkpoint(torch,snapshot,adapter,expected_fit_id='ENGINEERING_RANDOM_UNTRAINED_UNSELECTED',verify=True);restore_seconds=time.perf_counter()-tick
        if serving_state!=state_hashes(torch,h,z,adapter) or serving_laws!=source_laws(adapter) or z.tree_digest(torch,adapter.optimizer.state_dict())!=initial_optimizer_hash:raise ValueError('Complete populated-L native/source/Adam/modes/flags engine restore mismatch')
        snapshot.unlink();restore_initial()
        if before!=state_hashes(torch,h,z,adapter) or initial_laws!=source_laws(adapter):raise ValueError('Original untrained source lifecycle not exactly restored')
        role_counts={role:max(row['counts'][role] for row in custody['splits']) for role in ('cal','evaluate')}
        CPU_bank={'qualified':False}
        views={role:[torch.zeros((1,role_counts[role],2),dtype=torch.float32) for _ in range(4)] for role in role_counts}
        banks={role:torch.cat(value,dim=0) for role,value in views.items()};bank_path=output/'DISPOSABLE_CPU_BANK_SHAPE.pt';io_guard.allowed.add(bank_path.resolve());tick=time.perf_counter();torch.save({'engineering_dummy_shape_only':True,'role_tensors':banks},bank_path);bank_save_seconds=time.perf_counter()-tick;tick=time.perf_counter();loaded=torch.load(bank_path,map_location='cpu',weights_only=True);bank_load_seconds=time.perf_counter()-tick
        if not loaded['engineering_dummy_shape_only'] or any(not torch.equal(banks[k],loaded['role_tensors'][k]) for k in banks):raise ValueError('Whole dummy node bank shape serializer differs')
        bank_receipt=receipt(root,bank_path);bank_path.unlink();coordinates=sum(v.numel() for v in banks.values());CPU_bank={'qualified':True,'dummy_values_no_predictions':True,'roles':role_counts,'served_routes':4,'retained_route_logits_bytes':4*coordinates,'cat_transient_bytes':4*coordinates,'snapshot':bank_receipt,'save_seconds':bank_save_seconds,'load_seconds':bank_load_seconds,'removed':True,'actual_endpoint_ID_or_target_access':False};del views,banks,loaded
        stub_receipt=receipt(root,stub);stub.unlink()
        return {'cold_callback_seconds':times[0],'warm_callback_seconds':times[1:],'warm_median_seconds':statistics.median(times[1:]),'cycle_action_counts':cycle_counts,'source_state_hashes_before':before,'source_state_hashes_after':state_hashes(torch,h,z,adapter),'source_optimizer_mode_flag_laws':initial_laws,'source_frozen_moment_scope':moments_scope,'source_ownership':engine.inspect_ownership(adapter),'source_complete_native_modules':len(native),'immutable_static_input_hash':fixed_hash,'full_graph_serving_shape':serving_shape,'full_graph_serving_seconds':serving_seconds,'full_graph_logits_CPU_bytes':serving_cpu_bytes,'parameter_bytes':sum(p.numel()*p.element_size() for p in adapter.model.parameters()),'gradient_bytes':gradient_bytes,'manual_moment_bytes':sum(v.numel()*v.element_size() for v in moments),'extra_workspace_bytes':sum(v.numel()*v.element_size() for v in extra_workspace),'input_load_paths':io_guard.tensor_load_paths,'denied_role_reads':io_guard.denied,'unused_SELECT_stub':{**stub_receipt,'removed':True,'synthetic_one_row_never_validated':True},'random_complete_engine_restore':{'snapshot':saved,'bytes':size,'clone_seconds':clone_seconds,'save_seconds':save_seconds,'load_and_verify_restore_seconds':restore_seconds,'removed':True,'selected_checkpoint':False,'populated_native_L_diagnostics_retained':True,'engine':receipt(root,engine.__file__)},'optimizer_attempts':denial.denied_step_calls,'optimizer_functional_attempts':denial.denied_functional_calls,'original_optimizer_kernels':denial.actual_optimizer_kernels,'actual_optimizer_updates':0,'CPU_bank_export_proxy':CPU_bank,'real_CAL_evaluate_export':False,'trained_warm_or_screen_eligibility_claim':False}


def first_action_probe(torch,h,z,root,case,study,run,output,admission):
    """Untrained actual raw3170 first-action/common-V capacity;no optimizer or NLL."""
    import numpy as np
    runtime=local(root,admission['runtime_package']);sys.path.insert(0,str(runtime));load_owned_engine(root,admission)
    from adapters.nsd import factory,construct,helpers,plain_capture,plain_restore,factors
    inputs=run['config']['node_inputs'];allowed=[bound(root,inputs[k]) for k in ('graph','train')]
    custody=json.loads(bound(root,admission['custody_receipt']).read_text());forbidden=[local(root,row['node_inputs']['select']['path']) for row in custody['splits']]+[local(root,row['endpoint_inputs'][role+'_targets']['path']) for row in custody['splits'] for role in ('cal','evaluate')]
    with h.OptimizerDenial(torch) as denial,h.RoleIO(torch,root,allowed,forbidden,output) as io_guard:
        graph=torch.load(allowed[0],map_location='cpu',weights_only=True);train=torch.load(allowed[1],map_location='cpu',weights_only=True);ids=train['ids'].to('cuda:0');del train
        if graph['x'].shape!=(9498,3170) or len(ids)!=case['train_rows']:raise ValueError('Whole fresh raw3170 first-action input shape required')
        c=copy.deepcopy(run['config']);c.update(nsd_arm='geometry_warm',device='cuda:0');B=c['nsd_sources'];e,f,args=factory(torch,root,B,c,graph,'cuda:0');rng=helpers(torch,root,B);model=construct(torch,root,B,e,f,c,c['initialization_seed']);model.eval()
        for p in factors(model):p.requires_grad_(False)
        e.require(model.assert_ownership()['complete_native_paths']==4,'Four complete original geometry members/builders')
        fixed=[(name+'/'+key,value) for name,body in model.named_modules() for key,value in vars(body).items() if torch.is_tensor(value) and key!='L']+list(model.named_buffers())
        x=graph['x'].to('cuda:0');fixed+=[('actual_raw_X',x),('TRAIN_ids_only',ids)];p0=h.tensor_digest(torch,model.named_parameters());fixed0=h.tensor_digest(torch,fixed);flags={n:bool(p.requires_grad) for n,p in model.named_parameters()};modes={n:bool(b.training) for n,b in model.named_modules()};initial_rng=rng.capture_rng(np,torch);initial_plain=plain_capture(torch,model);counts={k:0 for k in ('screen_native_builder_returns','extra_offnode_spmm_attempts','extra_offnode_spmm_completed','offnode_sparse_copy_bytes_total','screen_action_CPU_bytes_total')};member=model.members[0];times=[];common=None;shape=None;common_bytes=0
        try:
            with z.TensorCommitDenial(torch,list(model.parameters())+[v for _,v in fixed],[]) as commits:
                for index in range(2):
                    rng.restore_rng(np,torch,initial_rng);plain_restore(model,initial_plain,'cuda:0');sync(torch,model);tick=time.perf_counter()
                    with torch.no_grad(),e.init.FirstActionCapture(torch,member,ids,counts,common) as capture:member(x)
                    sync(torch,model);times.append(time.perf_counter()-tick)
                    if common is None:common=capture.baseline_value;common_bytes=common.numel()*common.element_size()
                    shape=list(capture.action.shape)
                    if not torch.isfinite(capture.action).all():raise FloatingPointError('Untrained actual first-action capacity must be finite')
                if commits.denied or denial.denied_step_calls or denial.denied_functional_calls:raise ValueError('First-action source capacity attempted a protected action')
        finally:
            rng.restore_rng(np,torch,initial_rng);plain_restore(model,initial_plain,'cuda:0')
            for n,b in model.named_modules():b.training=modes[n]
            for n,p in model.named_parameters():p.requires_grad_(flags[n]);p.grad=None
        if p0!=h.tensor_digest(torch,model.named_parameters()) or fixed0!=h.tensor_digest(torch,fixed) or z.tree_digest(torch,initial_rng)!=z.tree_digest(torch,rng.capture_rng(np,torch)) or z.tree_digest(torch,initial_plain)!=z.tree_digest(torch,plain_capture(torch,model)):
            raise ValueError('First-action full source Parameters/topology/flags/RNG/plain state not restored')
        if counts['extra_offnode_spmm_completed']!=2:raise ValueError('Both actual offnode actions required')
        return {'first_callback_seconds':times[0],'retained_common_V_callback_seconds':times[1],'source_complete_native_modules':4,'native_body_forwards':2,'extra_actual_offnode_spmm':2,'source_action_counts':counts,'common_value_bytes':common_bytes,'CPU_action_shape':shape,'source_parameter_hash_before':p0,'source_parameter_hash_after':h.tensor_digest(torch,model.named_parameters()),'immutable_static_input_hash':fixed0,'restored_rng_and_source_plain_diagnostics':True,'input_load_paths':io_guard.tensor_load_paths,'denied_role_reads':io_guard.denied,'optimizer_objects':0,'optimizer_attempts':denial.denied_step_calls,'original_optimizer_kernels':denial.actual_optimizer_kernels,'actual_optimizer_updates':0,'TRAIN_target_values_used':False,'trained100_warm_or_17screen_or_rank_eligibility_claim':False,'untrained_shape_and_capacity_only':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify_route()
    root = Path(args.project_root).resolve()
    output = local(root, args.output)
    admission_path = local(root, args.admission)
    admission = json.loads(admission_path.read_text())
    self_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if admission.get('schema') != 'root-approved-zero-update-resource-admission-v1' or admission.get('root_source_review_confirmed') is not True or admission.get('helper_sha256') != self_hash or (admission.get('scientific_GO') is not False) or (args.case_id not in admission.get('approved_case_ids', [])) or (admission.get('max_actual_optimizer_updates') != 0):
        raise ValueError('Fresh root exact-source engineering admission required; no scientific GO')
    proposal = json.loads(bound(root, admission['proposal_receipt']).read_text())
    for dependency in admission['runtime_dependency_receipts'] + admission['source_dependency_receipts']:
        bound(root, dependency)
    case = next((c for c in proposal['requested_cases'] if c['case_id'] == args.case_id))
    study_row = admission['case_study_receipts'][args.case_id]
    raw = bound(root, study_row).read_bytes()
    study = json.loads(gzip.decompress(raw))
    if study['dataset'].lower().replace('_', '').replace('-', '') in ORIGINAL_FIVE:
        raise ValueError('Original-paper datasets excluded for every method')
    run = next((r for r in study['runs'] if r['run_id'] == case['source_run_id']))
    if sha(run) != case['source_run_hash']:
        raise ValueError('Exact prospective source run changed')
    if output.exists():
        raise ValueError('Fresh output required; failures never overwritten or silently retried')
    h = module(root, admission['role_optimizer_cap_helper'], 'new_source_inherited_role_optimizer_cap')
    z = module(root, admission['tensor_commit_tree_helper'], 'new_source_inherited_tensor_commit_tree')
    timeout = case['timeout_seconds']
    old = signal.getsignal(signal.SIGALRM)

    def timed_out(*ignored):
        raise h.ResourceTimeout('Declared source engineering child deadline')
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(timeout)
    output.mkdir(parents=True)
    owner = h.own_identity()
    started = time.monotonic()
    (output / 'START.json').write_text(json.dumps({'case_id': args.case_id, 'owner': owner, 'admission': receipt(root, admission_path), 'helper_sha256': self_hash, 'timeout_seconds': timeout, 'scientific_fits': 0, 'GO': False}, indent=2) + '\n')
    try:
        import torch
        torch.set_num_threads(2)
        (cap, before_usage, after_init) = h.configure_cuda_allocator(torch, output)
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        if case['family'] not in ('native_NSD_source','native_NSD_first_action_capacity'):
            raise ValueError('Only the reviewed whole raw3170 native NSD source/capacity cases are admitted')
        result = (first_action_probe if case['family']=='native_NSD_first_action_capacity' else node_probe)(torch,h,z,root,case,study,run,output,admission)
        result.update(schema='new-source-zero-update-resource-result-v1', status='RANDOM_SOURCE_ZERO_UPDATE_ENGINEERING_COMPLETE', case_id=args.case_id, owner=owner, constructor_and_probe_seconds=time.monotonic() - tick, elapsed_wall_seconds=time.monotonic() - started, CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(), CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(), current_child_allocator_cap=cap, GPU_usage_before_init=before_usage, GPU_usage_after_init=after_init, GPU_usage_at_completion=h.GPU_process_usage(), process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, scientific_fits=0, GO_created=False, scores_emitted=0, loss_values_emitted=0, heldout_target_files_opened=0, actual_optimizer_updates=0, whole_master_admitted=False)
        (output / 'RESULT.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps({'case_id': args.case_id, 'status': result['status'], 'actual_optimizer_updates': 0, 'scientific_fits': 0}))
    except BaseException as error:
        failure = h.terminal_failure(error, owner, started)
        failure.update(case_id=args.case_id, helper_sha256=self_hash, GO_created=False, optimizer_updates_authorized=0, automatic_retry=False)
        (output / 'FAILURE.json').write_text(json.dumps(failure, indent=2, sort_keys=True) + '\n')
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
if __name__ == '__main__':
    main()
