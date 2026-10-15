"""Prepared finite real BUDDY private-BN smoke; execute only after plan review."""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import socket
import subprocess
import sys
import time
import traceback


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def pin(path):
    path=Path(path).resolve();raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('CPU-only smoke requires CUDA hidden')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--context-manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.root).resolve();output=Path(args.output).resolve();package=Path(__file__).resolve().parent;expansion=package.parent;phase=expansion.parent
    assert all(p.is_relative_to(root) for p in (package,output,Path(args.context_manifest).resolve())) and not output.exists()
    output.mkdir(parents=True);resource.setrlimit(resource.RLIMIT_AS,(64*1024**3,64*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1800,1810))
    started=time.perf_counter();report={'schema':'native-BUDDY-private-BN-finite-CPU-smoke-v1','status':'running','stage':'before_import',
        'native_construction_attempts':0,'native_constructions':0,'actual_optimizer_transitions':0,'optimizer_attempts':0,'VALID_calls':0,
        'native_model_forwards':{'baseline_TRAIN':0,'candidate_TRAIN':0,'candidate_VALID':0},'reverse_collections':0,
        'device':'cpu','threads':2,'address_space_limit_bytes':64*1024**3,'scientific_fits':0,'TEST_accessed':False,'GPU_updates':0,
        'script':pin(__file__),'memory_events_before':Path('/sys/fs/cgroup/memory.events').read_text()}
    def write():
        usage=resource.getrusage(resource.RUSAGE_SELF);report.update(elapsed_seconds=time.perf_counter()-started,process_peak_RSS_KiB=usage.ru_maxrss,process_user_seconds=usage.ru_utime,process_system_seconds=usage.ru_stime)
        temporary=output/'RESULT.json.part';temporary.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');temporary.replace(output/'RESULT.json')
    write();adapter=None;baseline=None;handles=[]
    try:
        import numpy as np
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        def seed():random.seed(0);np.random.seed(0);torch.random.default_generator.manual_seed(0)
        smoke=phase/'reference_wrapper_redesign_20261013_v1/smoke_20261014/MANIFEST.json'
        case=next(row for row in json.loads(smoke.read_text())['cases'] if row['id']=='collab-buddy/gnnm_boundary_native')
        recipe=copy.deepcopy(case['recipe']);recipe.update(_repository_root=str(root),_seed=0,seed=0)
        settings={'manifest_sha256':pin(package/'MANIFEST.json')['sha256'],'banks_sha256':pin(expansion/'private_bn_state_v3/banks.py')['sha256'],
            'context_manifest':str(Path(args.context_manifest).resolve()),'context_manifest_sha256':pin(args.context_manifest)['sha256']}
        job={'schema':'native-full-epoch-fit-v1','id':'BUDDY-private-BN-CPU-engineering/split0/seed0','cohort_id':'BUDDY-private-BN-CPU-engineering/split0/seed0',
            'family':'link','procedure':'gnnm_private_bn_state_v3','base_procedure':'gnnm_boundary_native','seed':0,'split':0,'body':0,'recipe':recipe,
            'data_root':case['data_root'],'device':'cpu','TEST_access':False,'callback_module':str(package/'callback.py'),'callback_sha256':pin(package/'callback.py')['sha256'],
            'method_settings':settings,'policy':{'family':'link','track':'matched_extended','maximum_epochs':1000,'minimum_epochs':100,'patience':50,
                'metric':'hits@50','direction':'max','ties':'first','evaluate_TRAIN_each_epoch':False,'evaluation_mode':'VALID_only_disclosed_adaptation'}}
        source=load(package/'callback.py','_BUDDY_private_bn_callback');library=source.source();pins=library.bindings(root,settings['manifest_sha256'])
        policy=load(expansion/'link_policy_v1/policy.py','_BUDDY_private_bn_policy').create_policy(job,root)
        native_path=expansion/'link_native_integration_v1/session.py';native_source=load(native_path,'_BUDDY_private_bn_oracle_source')
        context_library=load(expansion/'link_expansion_ports_BUDDY_v2/session.py','_BUDDY_private_bn_context')
        base_job=copy.deepcopy(job);base_job.update(procedure='gnnm_boundary_native',callback_module=str(native_path),callback_sha256=pin(native_path)['sha256'],method_settings={})
        report.update(job=job,source_receipts=pins,smoke_manifest=pin(smoke),context_manifest=pin(args.context_manifest),torch=str(torch.__version__),stage='construct_zero_step_baseline')
        write();seed();report['native_construction_attempts']+=1;baseline=native_source.create_adapter(torch,base_job,policy,root);report['native_constructions']+=1
        n=baseline.session.native;context,context_pin=context_library.native_context(n,baseline.session,job)
        oracle_options=[{k:copy.deepcopy(v) for k,v in group.items() if k!='params'} for group in n.optimizer.param_groups]
        def baseline_count(module,arguments,result):report['native_model_forwards']['baseline_TRAIN']+=1;write()
        baseline_handle=n.model.register_forward_hook(baseline_count);handles.append(baseline_handle)
        n.model.train();n.optimizer.zero_grad();ids=n._next_batch();expected=context['TRAIN_batches'][0];links=n.links[ids];labels=n.labels[ids]
        inputs=n._hydrate(links)
        if not library.same(torch,ids,expected['sampler_ids']) or not library.same(torch,links,expected['links']) or not library.same(torch,labels,expected['labels']) or any(not library.same(torch,a,b) for a,b in zip(inputs,expected['inputs'])):
            raise AssertionError('Zero-step baseline differs from the authenticated first native TRAIN batch')
        report['stage']='zero_step_baseline_original_joint_BCE_backward';write();outputs=[];losses=[]
        for member in range(4):
            with n._route(member,True):
                value=n.model(*inputs,None,None);outputs.append(value.reshape(-1));losses.append(n.native_loss(value,labels))
        F=torch.stack(losses).mean();F.backward();report['reverse_collections']+=1
        oracle_outputs=torch.stack([value.detach().clone() for value in outputs]);oracle_loss=float(F.detach())
        oracle_gradients={name:None if value.grad is None else value.grad.detach().clone() for name,value in n.model.named_parameters()}
        if not torch.isfinite(oracle_outputs).all() or any(value is not None and not torch.isfinite(value).all() for value in oracle_gradients.values()):raise FloatingPointError('Nonfinite baseline oracle')
        baseline.assert_identity();report['baseline']={'optimizer_transitions':0,'TRAIN_queries':1024,'positive':int((labels==1).sum()),'negative':int((labels==0).sum()),
            'native_joint_BCE':oracle_loss,'route_forwards':4,'reverse_collections':1,'sampler_consumption':'one zero-step oracle batch; no trained epoch/clock claim'}
        baseline_handle.remove();handles.remove(baseline_handle);baseline.close();baseline=None
        del n,ids,links,labels,inputs,outputs,losses,F,value;gc.collect()
        first_route_outputs=[]
        def construct():
            seed();report['native_construction_attempts']+=1;write();current=source.create_adapter(torch,copy.deepcopy(job),policy,root);report['native_constructions']+=1
            if any(p.device.type!='cpu' for p in current.model.parameters()):raise AssertionError('Non-CPU model')
            if [{k:v for k,v in group.items() if k!='params'} for group in current.optimizers[0].param_groups]!=oracle_options:
                raise AssertionError('Original native optimizer options changed')
            def count(module,arguments,result):
                role='candidate_TRAIN' if module.training else 'candidate_VALID';report['native_model_forwards'][role]+=1
                if module.training and report['actual_optimizer_transitions']==0:first_route_outputs.append(result.reshape(-1).detach().clone())
                write()
            handle=current.model.register_forward_hook(count);handles.append(handle)
            optimizer=current.optimizers[0];original=optimizer.step
            def step(*a,**kw):
                report['optimizer_attempts']+=1
                if report['actual_optimizer_transitions']==0:
                    observed=dict(current.model.named_parameters())
                    if set(observed)!=set(oracle_gradients) or len(first_route_outputs)!=4:raise AssertionError('Added learned parameters or missing original route outputs')
                    maximum=0.
                    for name,p in observed.items():
                        target=oracle_gradients[name]
                        if (p.grad is None)!=(target is None):raise AssertionError('Changed gradient support: '+name)
                        if target is not None:maximum=max(maximum,float((p.grad-target).abs().max()))
                    output_difference=float((torch.stack(first_route_outputs)-oracle_outputs).abs().max())
                    if maximum>1e-6 or output_difference>1e-6:raise AssertionError('Private BUDDY BN changed TRAIN output/gradient law')
                    report['TRAIN_parity']={'max_output_abs_difference':output_difference,'max_gradient_abs_difference':maximum,'tolerance':1e-6,'additional_learned_parameters':0,'first_old_state_native_joint_BCE_unchanged':True}
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in current.model.parameters()):raise FloatingPointError('Nonfinite candidate gradient')
                value=original(*a,**kw);report['actual_optimizer_transitions']+=1;write();return value
            optimizer.step=step;return current,handle
        report['stage']='construct_first_private_BN_source';write();adapter,forward_handle=construct();adapter.assert_identity();n=adapter.session.native
        report['source_and_method_metadata']=adapter.metadata()
        def expect_guard(name,operation):
            try:operation()
            except RuntimeError:report.setdefault('ownership_guards',{})[name]=True
            else:raise AssertionError('Missing ownership guard: '+name)
        expect_guard('direct_model_forward',lambda:n.model())
        for name,ref in zip(adapter.session.manager.names,adapter.session.manager._refs):expect_guard('direct_BN_forward/'+name,lambda ref=ref:ref()())
        expect_guard('direct_model_only_state_export',lambda:n.model.state_dict())
        expect_guard('direct_model_only_state_load',lambda:n.model.load_state_dict({}))
        expect_guard('direct_native_sampler',lambda:n._next_batch())
        if n.batch_cursor!=0 or report['native_model_forwards']['candidate_TRAIN']!=0:raise AssertionError('Ownership guard mutated native task state')
        report['stage']='first_source_update';write();report['first_update']=adapter.session.train_update();report['reverse_collections']+=1
        if abs(report['first_update']['loss']-oracle_loss)>1e-6:raise AssertionError('Native joint TRAIN criterion changed')
        def valid():
            value=adapter.validate();report['VALID_calls']+=1
            if (value['complete_VALID_positive_records'],value['complete_VALID_negative_records'])!=(60084,100000) or value['complete_source_VALID'] is not True or value['TEST_accessed'] is not False or value.get('materialized_VALID_context_verified') is not True or value.get('private_BN_eval_state_unchanged') is not True:
                raise AssertionError('Incomplete VALID, context or read-only private-BN state')
            if not torch.isfinite(value['outputs']).all() or not torch.equal(value['query_endpoints'],torch.cat((adapter.session.native.valid['edge'],adapter.session.native.valid['edge_neg'])).cpu()):raise AssertionError('Source VALID identity or finite output failed')
            report.setdefault('VALID_diagnostics',[]).append({'call':report['VALID_calls'],'metrics':value['metrics'],'loss':value['loss'],'private_BN_branch_calls_by_route':value['private_BN_branch_calls_by_route'],'private_BN_eval_state_unchanged':True,'materialized_context_verified':True})
            write();return value
        report['stage']='first_full_VALID';write();first=valid();saved=adapter.state();checkpoint=output/'after_one_update.pt';torch.save({'session':saved,'job':job},checkpoint)
        report['checkpoint']=pin(checkpoint);first_outputs=first['outputs'].clone();first_mean=first['same_bank_mean_outputs'].clone();first_labels=first['labels'].clone();first_queries=first['query_endpoints'].clone()
        bank_state={name:value.clone() for name,value in saved['private_BN_banks'].items()}
        forward_handle.remove();handles.remove(forward_handle);adapter.close();adapter=None;del n,first,saved;gc.collect()
        report['stage']='construct_fresh_source_and_restore';write();adapter,forward_handle=construct();references={name:id(p) for name,p in adapter.model.named_parameters()}
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);adapter.load_state(saved['session'])
        if references!={name:id(p) for name,p in adapter.model.named_parameters()}:raise AssertionError('Fresh restore replaced original Parameters')
        restored=adapter.state()['private_BN_banks']
        if any(not torch.equal(value,restored[name]) for name,value in bank_state.items()):raise AssertionError('Route BN bank restoration differs')
        report['private_BN_banks_restored_bit_exact']=True
        report['stage']='fresh_restore_full_VALID';write();replay=valid()
        report['restored_served_bit_exact']=torch.equal(replay['outputs'],first_outputs);report['restored_mean_bit_exact']=torch.equal(replay['same_bank_mean_outputs'],first_mean)
        if not report['restored_served_bit_exact'] or not report['restored_mean_bit_exact'] or not torch.equal(replay['labels'],first_labels) or not torch.equal(replay['query_endpoints'],first_queries):raise AssertionError('Fresh complete VALID replay changed')
        adapter.load_state(saved['session']);report['stage']='resumed_second_source_update';write();report['resumed_second_update']=adapter.session.train_update();report['reverse_collections']+=1
        report['stage']='final_full_VALID';write();final=valid();torch.save({'session':adapter.state(),'job':job},output/'after_two_updates.pt');report['final_checkpoint']=pin(output/'after_two_updates.pt')
        if report['native_constructions']!=3 or report['actual_optimizer_transitions']!=2 or report['VALID_calls']!=3 or adapter.session.native.updates!=2 or report['native_model_forwards']!={'baseline_TRAIN':4,'candidate_TRAIN':8,'candidate_VALID':24} or report['reverse_collections']!=3:raise AssertionError('Finite budget or native restored clock differs')
        report.update(status='passed',stage='complete',native_update_clock=2,source_native_Parameter_IDs_preserved=True,original_BN_affine_Parameter_IDs_preserved=True,
            original_joint_TRAIN_loss_preserved=True,active_route_BN_counters=adapter.session._counts(),canonical_native_BN_buffers_unchanged=True,
            complete_source_minibatches_and_full_VALID=True,full_epoch_or_CUDA_or_quality_qualification=False)
    except BaseException as exc:report.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(limit=18)})
    finally:
        for handle in handles:
            try:handle.remove()
            except Exception:pass
        for current in (adapter,baseline):
            if current is not None:
                try:current.close()
                except Exception as exc:report.setdefault('close_errors',[]).append(str(exc))
        report['memory_events_after']=Path('/sys/fs/cgroup/memory.events').read_text();write()
    print(json.dumps({k:report[k] for k in ('status','stage','native_constructions','actual_optimizer_transitions','VALID_calls','native_model_forwards','reverse_collections','elapsed_seconds')}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
