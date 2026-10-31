"""One root-admitted full-input Bayesian resource case, zero optimizer maps.

Four explicit case IDs exist; this program does not loop a catalogue or fit.
"""
import argparse
import builtins
import copy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def pin(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',required=True)
    parser.add_argument('--case',required=True);a=parser.parse_args();release=json.loads(Path(a.release).read_text())
    if (release.get('execution_enabled') is not True or release.get('schema')!='Bayesian-BSNN-resource-root-release-v1'
            or release.get('case_id')!=a.case or not release.get('root_authorization_reference')
            or release.get('scientific_fits_authorized') is not False
            or release.get('optimizer_updates_authorized') is not False
            or release.get('TRAIN_callbacks')!=2 or release.get('program_pin')!=pin(__file__)):
        raise ValueError('One exact independent root source-resource admission required')
    if release.get('cases_pin')!=pin(HERE/'CASES.json'):
        raise ValueError('Exact four-case geometry/source roster required')
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Wrong allocation route')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or 'torch' in sys.modules:
        raise ValueError('Fresh native CUDA device0 process required')
    runtime=release['runtime']
    if (Path(sys.executable).resolve()!=Path(runtime['interpreter']).resolve()
            or pin(sys.executable)!=runtime['interpreter_pin']
            or {name:importlib.metadata.version(name) for name in runtime['versions']}!=runtime['versions']
            or not runtime['binary_file_pins']):
        raise ValueError('Root-bound source interpreter/numerical versions required')
    for row in runtime['binary_file_pins']:
        if pin(row['path'])!={k:row[k] for k in ('bytes','sha256')}:
            raise ValueError('Root-bound source numerical binary bytes changed')
    phase=Path(release['research_root']).resolve();case=json.loads((HERE/'CASES.json').read_text())['cases'][a.case]
    output=(phase/release['output']).resolve()
    if not output.is_relative_to(phase) or output.exists():raise ValueError('Fresh root-bound project output required')
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<release['reservation_mib']+release['headroom_mib']:raise ValueError('Present VRAM cannot admit this source case')
    output.mkdir();tick=time.monotonic();attempt={'case_id':a.case,'status':'RUNNING','scientific_fits':0,'optimizer_updates':0}
    def write(name,value):
        p=output/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(p)
    write('ATTEMPT.json',attempt)
    try:
        for row in release['source_files']:
            path=(phase/row['path']).resolve()
            if not path.is_relative_to(phase) or pin(path)!={k:row[k] for k in ('bytes','sha256')}:
                raise ValueError('Root-bound current source changed')
        forbidden={(phase/row).resolve() for row in release['forbidden_target_paths']}
        forbidden.add((phase/case['run']['config']['node_inputs']['select']['path']).resolve())
        original_open=builtins.open;original_io=io.open
        def guarded_open(old):
            def call(path,*args,**kwargs):
                if not isinstance(path,int) and Path(path).resolve() in forbidden:
                    raise ValueError('SELECT/CAL/EVALUATE target read denied')
                return old(path,*args,**kwargs)
            return call
        builtins.open,io.open=guarded_open(original_open),guarded_open(original_io)
        import torch
        import numpy as np
        torch.cuda.set_device(0)
        profile={'deterministic':torch.are_deterministic_algorithms_enabled(),
                 'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
                 'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,
                 'TF32_cudnn':torch.backends.cudnn.allow_tf32}
        if profile!=runtime['profile']:raise ValueError('Native source numerical flags changed')
        sys.path.insert(0,str(phase/case['runtime']))
        from adapters import bsnn_bayesian as port
        from adapters.common_native_wrapper import bound
        c=case['run']['config'];B=c['bayesian_bsnn_sources']
        graph=torch.load(bound(phase,c['node_inputs']['graph']),map_location='cpu',weights_only=True)
        train=torch.load(bound(phase,c['node_inputs']['train']),map_location='cpu',weights_only=True)
        model,opt,h,s,args=port.construct(torch,phase,B,graph,c['bayesian_bsnn_variant'],c['bayesian_bsnn_arm'],c['initialization_seed'],'cuda:0')
        if (tuple(graph['x'].shape)!=(9498,3170) or graph['edge_index'].shape[1]!=306276
                or len(train['ids'])!=case['TRAIN_rows'] or args!=case['native_args']):
            raise ValueError('Exact complete source graph/TRAIN/native geometry changed')
        if (type(opt) is not torch.optim.Adam or opt.defaults['capturable'] or opt.defaults['amsgrad']
                or opt.defaults['differentiable'] or opt.defaults['betas']!=(.9,.999)
                or opt.defaults['eps']!=1e-8 or any(g['lr']!=.01 or g['weight_decay']!=.0005 for g in opt.param_groups)):
            raise ValueError('Synthetic mature moments require the exact native noncapturable Adam schema')
        # No actual SELECT authority is loaded or scored; the adapter constructor
        # receives only a declared synthetic unused stub.
        occupied=torch.zeros(len(graph['x']),dtype=torch.bool);occupied[train['ids']]=True
        disjoint=torch.where(~occupied)[0]
        if not len(disjoint):raise ValueError('One disjoint synthetic unused SELECT row required')
        synthetic_select={'ids':disjoint[:1].long(),'y':torch.tensor([0],dtype=torch.long)}
        adapter=port.attach(torch,model,opt,h,s,args,graph,train,synthetic_select,c['bayesian_bsnn_arm'],c['initialization_seed'],{'resource_case':a.case,'real_SELECT_unread':True})
        original_step=opt.step
        class OptimizerDenied(RuntimeError):pass
        def denied_step(*_,**__):raise OptimizerDenied('ROOT_RESOURCE_OPTIMIZER_DISPATCH_DENIED')
        opt.step=denied_step
        initial_model=copy.deepcopy(model.state_dict());initial_external=copy.deepcopy(adapter.capture_state())
        initial_flags={n:p.requires_grad for n,p in model.named_parameters()}
        initial_modes={n:b.training for n,b in model.named_modules()}
        callbacks=[];durations=[]
        for index in range(2):
            model.load_state_dict(initial_model);adapter.restore_state(initial_external);opt.zero_grad(set_to_none=True)
            started=time.monotonic()
            try:adapter.train_step(1)
            except Exception as error:
                if 'ROOT_RESOURCE_OPTIMIZER_DISPATCH_DENIED' not in str(error):raise
                counts=error.operation_counts
                if counts['optimizer_updates']!=0 or counts['backward_calls']!=adapter.members:
                    raise ValueError('Exact complete Bayesian TRAIN callback was not denied only at dispatch')
                callbacks.append(counts)
            else:raise ValueError('An optimizer map slipped into a resource case')
            torch.cuda.synchronize(0);durations.append(time.monotonic()-started)
            if not s.exact(torch,initial_model,model.state_dict()):raise ValueError('Source resource callback changed model tensors')
            if index==0:
                # Reserve mature source Adam moments without optimizer kernels.
                # Native never-used leaves retain None/no moment state.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        opt.state[parameter]={'step':torch.tensor(1.,dtype=torch.float32),
                                              'exp_avg':torch.zeros_like(parameter),
                                              'exp_avg_sq':torch.ones_like(parameter)}
            elif any(p.grad is not None and p not in opt.state for p in model.parameters()):
                raise ValueError('Two source draws expose a later-used leaf outside the mature moment envelope')
        # Conservative additional Adam work buffers; no optimizer kernel runs.
        workspace=[torch.empty_like(p) for p in model.parameters() if p.grad is not None for _ in range(2)]
        populated_leaves=len(opt.state)
        adapter.restore_state(initial_external);model.load_state_dict(initial_model);model.eval()
        with torch.no_grad():
            four=adapter.serving_draws(1,4)
            if adapter.members==1:
                sixteen=adapter.serving_draws(1,16)
                if not torch.equal(four,sixteen[:4]):raise ValueError('MC16 source-prefix draw equality failed')
        state=copy.deepcopy(adapter.capture_state());optimizer_state=copy.deepcopy(opt.state_dict());buffer=io.BytesIO()
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'adapter':state},buffer)
        buffer.seek(0);restored=torch.load(buffer,map_location='cpu',weights_only=True)
        model.load_state_dict(restored['model']);opt.load_state_dict(restored['optimizer']);adapter.restore_state(restored['adapter'])
        if not s.exact(torch,state,adapter.capture_state()):raise ValueError('Complete SciPy/plain/source stream serializer restore failed')
        if not s.exact(torch,optimizer_state,opt.state_dict()):raise ValueError('Complete mature source Adam serializer restore failed')
        if not s.exact(torch,initial_model,model.state_dict()):raise ValueError('Resource lifecycle changed native model tensors')
        adapter.restore_state(initial_external)
        for n,p in model.named_parameters():p.requires_grad_(initial_flags[n])
        for n,b in model.named_modules():b.training=initial_modes[n]
        if not s.exact(torch,initial_external,adapter.capture_state()):raise ValueError('Outer/private native RNG and plain state were not restored')
        result={'schema':'Bayesian-BSNN-source-resource-result-v1','status':'PASS','case_id':a.case,
                'actual_GRAPH_TRAIN_only':True,'real_SELECT_CAL_EVALUATE_read':False,'source_TRAIN_callbacks':callbacks,
                'callback_seconds':durations,'source_used_leaves_with_synthetic_mature_Adam_moments':populated_leaves,
                'never_used_None_leaves_without_moments':sum(p.grad is None for p in model.parameters()),
                'mature_moment_states_are_synthetic_not_learned':True,'complete_checkpoint_bytes':len(buffer.getvalue()),
                'native_args':args,'plain_native_state_and_SciPy_restored':True,
                'additional_unstepped_Adam_workspace_bytes':sum(v.numel()*v.element_size() for v in workspace),
                'runtime':runtime,'source_cases_pin':release['cases_pin'],'program_pin':release['program_pin'],
                'source_FILES':release['source_files'],'target_denial_paths':release['forbidden_target_paths'],
                'posterior_source_serving_draws':4*adapter.members,
                'native_M1_MC16_extra_count_control_draws':16 if adapter.members==1 else 0,
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(0),
                'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(0),
                'CPU_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'wall_seconds':time.monotonic()-tick,'science_state_donor':False,'scientific_fits':0,'optimizer_updates':0,
                'actual_optimizer_kernel_timing_qualified':False,'convergence_or_ETA_qualified':False}
        opt.step=original_step;builtins.open,io.open=original_open,original_io
        write('RESULT.json',result);attempt.update(status='PASS',result_pin=pin(output/'RESULT.json'))
    except BaseException as error:
        attempt.update(status='FAILED',error_type=type(error).__name__,error=str(error),failures_preserved=True,wall_seconds=time.monotonic()-tick)
        write('ATTEMPT.json',attempt);raise
    write('ATTEMPT.json',attempt);print(json.dumps({'status':'PASS','case_id':a.case,'scientific_fits':0,'optimizer_updates':0}))


if __name__=='__main__':main()
