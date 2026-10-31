"""Bounded synthetic CPU source/gradient/state tests, no scientific training."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact authorized allocation route required')
    root=Path(args.project_root).resolve();here=Path(__file__).resolve().parent;output=root/args.output
    if output.exists() or not output.resolve().is_relative_to(root):raise ValueError('Fresh confined synthetic QA output required')
    output.mkdir();os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(here/'runtime'))
    import torch
    from adapters.native_polynormer_qk import make_adapter as native_adapter,receipt,SOURCES,CONTRACT
    from adapters.label_stage import make_adapter,ARMS,SOURCE_PINS,source_module
    from engine import capture_checkpoint,restore_checkpoint,_equal_tree
    import importlib.util
    hp=root/'private_source_zero_update_helpers_20261012_v2/runtime/qa/practical_zero_update_resources.py'
    if hashlib.sha256(hp.read_bytes()).hexdigest()!='effa70e31e64881322ccfb5a6e7e3903ccdb6f887c694a97f6b63d0bd2e267e0':raise ValueError('Exact reviewed optimizer denial helper required')
    spec=importlib.util.spec_from_file_location('paid_label_cpu_inherited_h',hp);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    torch.set_num_threads(2);sources={k:receipt(root,root/path) for k,(path,sha) in SOURCE_PINS.items()};ns={k:receipt(root,root/path) for k,(path,sha) in SOURCES.items()}
    results=[];started=time.monotonic()
    try:
        for C,features in ((10,300),(8,745)):
            fixture=output/('C'+str(C));fixture.mkdir();N=16;gen=torch.Generator().manual_seed(23)
            graph={'x':torch.randn(N,features,generator=gen),
                   'edge_index':torch.stack((torch.arange(N).repeat(3),torch.cat((torch.roll(torch.arange(N),1),torch.roll(torch.arange(N),-1),torch.arange(N)))))}
            train={'ids':torch.arange(6),'y':torch.arange(6)%C};select={'ids':torch.arange(6,10),'y':torch.arange(4)%C}
            for key,value in (('graph',graph),('train',train),('select',select)):torch.save(value,fixture/(key+'.pt'))
            inputs={k:receipt(root,fixture/(k+'.pt')) for k in ('graph','train','select')}
            study={'stage':'engineering','dataset':'synthetic_source_state_C'+str(C),'data':{'synthetic':True},'source':{'files':list(ns.values())+list(sources.values())}}
            config={'polynormer_qk_arm':'bare_native_single','members':1,'physical_slow_bodies':1,'source_optimizer_count':1,
                    'width':512,'depth':7,'global_layers':2,'heads':1,'local_transition_updates':100,'learning_rate':.001,
                    'in_dropout':.5,'dropout':.5,'global_dropout':.5,'classes':C,'source_contract':CONTRACT,'node_inputs':inputs,'polynormer_qk_sources':ns}
            native=native_adapter(torch=torch,project_root=root,study=study,run={'run_id':'CPU_RANDOM_NATIVE','seed':81101,'body':0,'config':config},device='cpu')
            cap=source_module(root,sources['capture'],'capture',C)[0].NativeSingleCapture(native.source_session,N,512,C)
            native.model.eval();state=native.capture_state()
            try:
                cap.begin('SERVE',1)
                with torch.no_grad():unused=native.logits_fn(native.model,False,0);H,base=cap.finish();del unused
            finally:cap.close()
            if not _equal_tree(torch,state,native.capture_state()):raise ValueError('Native eval capture changed source streams/RNG')
            cache={'schema':'ENGINEERING_RANDOM_NOT_SELECTED_NO_SCIENTIFIC_ANCESTRY','native_fit_id':'CPU_RANDOM_NOT_PAID',
                   'native_selected_step':0,'native_logical_steps':0,'native_global':False,'H':H,'native_logits':base}
            for arm in ARMS:
                cfg={'paid_label_arm':arm,'label_cycles':5000,'label_learning_rate':.001,'classes':C,'node_inputs':inputs,'label_sources':sources}
                run={'run_id':'CPU_'+arm+'_C'+str(C),'seed':81101,'body':0,'config':cfg}
                model=make_adapter(torch=torch,project_root=root,study=study,run=run,device='cpu',execution_context={'engineering_synthetic':True,'synthetic_cache':cache})
                no=None
                try:make_adapter(torch=torch,project_root=root,study=study,run=run,device='cpu')
                except ValueError as e:no=str(e)
                if not no or 'Paid native' not in no:raise ValueError('Missing paid native context did not fail before input/model construction')
                before={name:p.detach().clone() for name,p in model.model.named_parameters()}
                h.allocate_adam_moments(torch,model)
                with h.OptimizerDenial(torch) as denial:
                    first=model.train_step(1)
                    if first['operation_counts']['optimizer_updates']!=ARMS[arm][6] or denial.denied_step_calls!=ARMS[arm][6]:raise ValueError('Exact source owner counts changed')
                    if any(not torch.equal(before[n],p) for n,p in model.model.named_parameters()):raise ValueError('Synthetic parameter changed under inherited denial')
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.model.parameters()):raise ValueError('Nonfinite source gradient')
                native_before=(H.clone(),base.clone())
                with torch.no_grad():logits=model.logits_fn(model.model,False,1)
                if logits.shape!=(ARMS[arm][3],N,C) or not torch.isfinite(logits).all():raise ValueError('Source class-provider serving differs')
                if not torch.equal(H,native_before[0]) or not torch.equal(base,native_before[1]) or H.requires_grad or base.requires_grad:raise ValueError('Frozen native capture changed')
                pending_state=model.capture_state()
                payload=capture_checkpoint(torch,model,fit_id=run['run_id'],step=1,validation={},decision={});path=fixture/(arm+'_state.pt');torch.save(payload,path)
                model.bank.mask_generator.manual_seed(3);model.bank.counters['mask_draws']+=10
                restore_checkpoint(torch,path,model,expected_fit_id=run['run_id'],verify=True)
                if not _equal_tree(torch,pending_state,model.capture_state()):raise ValueError('Full source mask/counter/erasure state restore differs')
                path.unlink()
                results.append({'arm':arm,'classes':C,'source_factory_gradients_serving_restore':'pass','members':ARMS[arm][3],
                                'source_optimizer_owners':ARMS[arm][6],'denied_optimizer_attempts':denial.denied_step_calls,
                                'parameters':sum(p.numel() for p in model.model.parameters()),'actual_optimizer_updates':0})
                del model,payload,logits
            # Live source gate derivative, unit gate/zero-message invariant; no label/native fit.
            gates,_=source_module(root,sources['gate_hook'],'gate_hook',C);gate=gates.make_query_value_gate(torch=torch,later_execution_authorized=True)
            message=torch.randn(4,64,generator=gen);q=torch.randn(4,512,generator=gen)
            gated=gate(message,q)
            if not torch.equal(message,gated) or not torch.equal(gate(torch.zeros_like(message),q),torch.zeros_like(message)):raise ValueError('Source zero-start multiplicative gate changed')
            gated.sum().backward()
            if gate.A.grad is None or not torch.isfinite(gate.A.grad).all() or not gate.A.grad.any():raise ValueError('Source gate has no active receiver gradient')
            results.append({'classes':C,'source_gate_zero_start_zero_message_active_derivative':'pass','source_native_capture':'one_random_eval_forward_not_paid_selected'});del native
        if torch.cuda.is_initialized():raise ValueError('CPU QA initialized CUDA')
        result={'status':'passed','scope':'synthetic CPU source engineering only; no data array, teacher fit, selected native model or predictive score',
                'cases':results,'torch':torch.__version__,'elapsed_seconds':time.monotonic()-started,'actual_optimizer_updates':0,'scientific_fits':0,'predictive_scores':0,'CUDA':False,'GO':False}
        (output/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'status':'passed','source_arms_classes_tested':18,'gates_tested':2,'optimizer_updates':0,'fits':0,'CUDA':False,'GO':False}))
    except BaseException as e:
        (output/'FAILURE.json').write_text(json.dumps({'status':'failed','error':str(e),'traceback':traceback.format_exc(),'completed':results,'optimizer_updates':0,'fits':0,'GO':False},indent=2)+'\n');raise


if __name__=='__main__':main()
