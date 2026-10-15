"""One fixed BUDDY method: two source-sized CPU updates, full VALID, fresh restore."""
from __future__ import annotations
import argparse
import copy
import os
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import resource
import socket
import subprocess
import sys
import time
import traceback


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--method', required=True); parser.add_argument('--output', required=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only qualification requires CUDA hidden')
    args = parser.parse_args(); root = Path(args.root).resolve(); output = Path(args.output).resolve()
    package = Path(__file__).resolve().parent; expansion = package.parents[1]; phase = expansion.parent
    if not all(path.is_relative_to(root) for path in (package,output,phase)) or output.exists():
        raise ValueError('Fresh own repository output required')
    output.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_AS, (64*1024**3,64*1024**3))
    started = time.perf_counter()
    report = {'schema':'native-BUDDY-interaction-v1-actual-assignment-CPU-v1', 'status':'running','method':args.method,
        'device':'cpu','threads':2,'address_space_limit_bytes':64*1024**3,'source_joint_batch':1024,'native_training_reduction':'joint mixed-source BCE mean',
        'actual_optimizer_transitions':0,'optimizer_attempts':0,'VALID_calls':0,'native_constructions':0,
        'TEST_accessed':False,'scientific_fit':False,'GPU_models_or_updates':0,'script':pin(__file__),
        'native_model_forwards_attempted':0,'native_model_forwards_completed':0,'native_constructions_attempted':0,'VALID_attempts':0,'actual_reverse_collections_attempted':0,'actual_reverse_collections_completed':0,'stage':'before_numerical_import','memory_events_before':Path('/sys/fs/cgroup/memory.events').read_text()}
    def write():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(elapsed_seconds=time.perf_counter()-started, process_user_seconds=usage.ru_utime,
            process_system_seconds=usage.ru_stime, process_peak_RSS_KiB=usage.ru_maxrss)
        temporary = output/'RESULT.json.part'; temporary.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n'); temporary.replace(output/'RESULT.json')
    write()
    adapter = None
    try:
        import numpy as np
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        random.seed(0); np.random.seed(0); torch.random.default_generator.manual_seed(0)
        manifest = phase/'reference_wrapper_redesign_20261013_v1/smoke_20261014/MANIFEST.json'
        case = next(row for row in json.loads(manifest.read_text())['cases'] if row['id']=='collab-buddy/gnnm_boundary_native')
        jobs=json.loads((package/'JOBS.json').read_text())
        if args.method not in jobs:raise ValueError('Only the four frozen BUDDY interaction jobs are admitted')
        job=copy.deepcopy(jobs[args.method]);method_package=expansion/'interaction_BUDDY_v1';callback=method_package/'callback.py'
        if job['recipe']['_repository_root']!=str(root) or pin(method_package/'MANIFEST.json')['sha256']!=job['method_settings']['manifest_sha256']:
            raise ValueError('Frozen native repository/method source differs')
        if pin(job['method_settings']['context_manifest'])['sha256']!=job['method_settings']['context_manifest_sha256']:
            raise ValueError('Authenticated native context changed')
        policy_source = load(expansion/'link_policy_v1/policy.py','_qualify_BUDDY_policy')
        policy = policy_source.create_policy(job,root)
        source = load(callback,'_qualify_BUDDY_callback')
        report.update(source_smoke_case=case['id'],smoke_manifest=pin(manifest),callback=pin(callback),method_manifest=pin(method_package/'MANIFEST.json'),
            torch=str(torch.__version__),job=job,stage='construct_first_native_source')
        write()
        def construct():
            report['native_constructions_attempted'] += 1; write()
            current = source.create_adapter(torch,copy.deepcopy(job),policy,root)
            report['native_constructions'] += 1
            if any(p.device.type != 'cpu' for p in current.model.parameters()): raise AssertionError('Non-CPU model parameter')
            def pre(module,inputs):
                report['native_model_forwards_attempted']+=1;write()
            def post(module,inputs,result):
                report['native_model_forwards_completed']+=1;write()
            current.model.register_forward_pre_hook(pre);current.model.register_forward_hook(post)
            optimizer = current.optimizers[0]; original = optimizer.step
            def step(*a,**kw):
                report['optimizer_attempts'] += 1
                value = original(*a,**kw)
                report['actual_optimizer_transitions'] += 1; write(); return value
            optimizer.step = step
            return current
        def valid(current):
            report['VALID_attempts'] += 1; write()
            value = current.validate(); report['VALID_calls'] += 1
            if (value['complete_VALID_positive_records'],value['complete_VALID_negative_records']) != (60084,100000) \
                    or value.get('complete_source_VALID') is not True or value.get('TEST_accessed') is not False:
                raise AssertionError('Incomplete source VALID or TEST disclosure')
            if not torch.isfinite(value['outputs']).all(): raise AssertionError('Nonfinite served full VALID')
            if not torch.equal(value['query_endpoints'],torch.cat((current.session.native.valid['edge'],current.session.native.valid['edge_neg'])).cpu()):
                raise AssertionError('Source VALID endpoint identities/order changed')
            if value.get('materialized_VALID_context_verified') is not True or current.session._depth:
                raise AssertionError('Both full native BUDDY VALID roles must authenticate')
            report.setdefault('VALID_diagnostics',[]).append({'metrics':value['metrics'],'loss':value['loss'],
                'same_bank_mean_metrics':value['same_bank_mean_metrics'],'same_bank_mean_loss':value['same_bank_mean_loss'],
                'source_criterion_preserved':True,'native_BUDDY_forwards':value['work']['native_BUDDY_forwards']})
            sidecar=source.source().selected_mean_sidecar(current.session)
            if sidecar['source_updates']!=current.session.native.updates or not torch.equal(sidecar['actual_served_raw_logits'],value['outputs']) \
                    or not torch.equal(sidecar['same_bank_mean_raw_logits'],value['same_bank_mean_outputs']) or sidecar['extra_forward_calls']!=0:
                raise AssertionError('Same-call served/mean sidecar identities or source update clock differ')
            return value
        report['additional_old_state_gradient_collections'] = 0
        report['actual_call_assignment_checks'] = []
        def exact(a,b,path='state'):
            if isinstance(a,torch.Tensor):
                if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a.cpu(),b.cpu()):
                    raise AssertionError('Exact restored tensor mismatch at '+path)
            elif isinstance(a,dict):
                if not isinstance(b,dict) or set(a)!=set(b):raise AssertionError('Restored keys differ at '+path)
                for key in a:exact(a[key],b[key],path+'.'+str(key))
            elif isinstance(a,(tuple,list)):
                if type(a)is not type(b) or len(a)!=len(b):raise AssertionError('Restored sequence differs at '+path)
                for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'['+str(i)+']')
            elif type(a)is not type(b) or a!=b:raise AssertionError('Restored scalar differs at '+path)
        def no_stale_sidecar(current):
            try:source.source().selected_mean_sidecar(current.session)
            except ValueError:return
            raise AssertionError('Sidecar survived TRAIN/restore without fresh current VALID')
        def train(current):
            original_build=current.session.objectives.build
            original_grad=torch.autograd.grad
            original_backward=torch.Tensor.backward
            optimizer=current.optimizers[0];original_step=optimizer.step
            n=current.session.native
            shared=tuple(p for p in n.source_parameter_references.values() if p.requires_grad)
            private=tuple(p for p in n.core.private_parameters() if p.requires_grad)
            all_parameters=shared+private;ids=[id(p) for p in all_parameters]
            owned=[id(p) for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
            if len(ids)!=len(set(ids)) or len(owned)!=len(set(owned)) or set(ids)!=set(owned):
                raise AssertionError('Declared native shared/private optimizer partition is incomplete/aliased')
            live={};calls=[];returned={};checked=[False];before_transition=report['actual_optimizer_transitions']
            def build(*pargs,**kwargs):
                terms=original_build(*pargs,**kwargs);live['terms']=terms
                live['versions']={id(p):p._version for p in all_parameters}
                if terms['scalar_update']:
                    if terms['shared_objective'] is not terms['private_objective'] or terms['logged_objective'] is not terms['shared_objective']:
                        raise AssertionError('Scalar-J recipe does not use one audited joint scalar')
                return terms
            def record_actual_grad(*pargs,**kwargs):
                terms=live['terms']
                if terms['scalar_update'] or len(calls)>=2:
                    raise AssertionError('Unexpected extra gradient collection in native interaction')
                position=len(calls);expected_parameters=shared if position==0 else private
                expected_objective=terms['shared_objective'] if position==0 else terms['private_objective']
                actual_objective=pargs[0] if pargs else kwargs.get('outputs')
                actual_inputs=tuple(pargs[1] if len(pargs)>1 else kwargs['inputs'])
                if actual_objective is not expected_objective or [id(p) for p in actual_inputs]!=[id(p) for p in expected_parameters]:
                    raise AssertionError('Actual native autograd collection uses wrong objective or ordered parameter block')
                if any(p._version!=live['versions'][id(p)] for p in all_parameters) or report['actual_optimizer_transitions']!=before_transition:
                    raise AssertionError('Native parameter/optimizer transition occurred before both old-state collections')
                report['actual_reverse_collections_attempted'] += 1; write()
                gradients=original_grad(*pargs,**kwargs)
                report['actual_reverse_collections_completed'] += 1; write()
                for p,g in zip(actual_inputs,gradients):returned[id(p)]=g
                calls.append({'block':'shared' if position==0 else 'private','parameters':len(actual_inputs),
                    'objective_identity_checked':True,'extra_gradient_computation':False})
                return gradients
            def record_scalar_backward(tensor,*a,**kw):
                terms=live['terms']
                if not terms['scalar_update'] or tensor is not terms['shared_objective']:
                    raise AssertionError('Unexpected scalar backward objective')
                report['actual_reverse_collections_attempted'] += 1; write()
                result=original_backward(tensor,*a,**kw)
                report['actual_reverse_collections_completed'] += 1; write()
                return result
            def before_step(*a,**kw):
                terms=live['terms'];block_recipe=not terms['scalar_update']
                if len(calls)!=(2 if block_recipe else 0):raise AssertionError('Actual old-state collection count differs')
                if any(p._version!=live['versions'][id(p)] for p in all_parameters) or report['actual_optimizer_transitions']!=before_transition:
                    raise AssertionError('Source parameters moved before original Adam')
                if block_recipe:
                    for p in all_parameters:
                        g=returned[id(p)]
                        if (g is None)!=(p.grad is None) or (g is not None and not torch.equal(g,p.grad)):
                            raise AssertionError('Parameter.grad differs from ACTUAL returned native block gradient')
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in all_parameters) \
                        or not any(p.grad is not None and bool((p.grad!=0).any()) for p in all_parameters):
                    raise AssertionError('Actual assigned gradients must be finite and active')
                checked[0]=True
                return original_step(*a,**kw)
            current.session.objectives.build=build;torch.autograd.grad=record_actual_grad;torch.Tensor.backward=record_scalar_backward;optimizer.step=before_step
            try:
                value=current.session.train_update();no_stale_sidecar(current)
                if not checked[0] or report['actual_optimizer_transitions']!=before_transition+1:
                    raise AssertionError('Exactly one original Adam transition with checked actual assignment required')
                if value['work']['query_records']!=1024 or value['work']['native_BUDDY_forwards']!=4 or value['work']['captured_source_query_records']!=1024:
                    raise AssertionError('Original source query batch changed')
                if value['work']['reverse_collections']!=(2 if args.method in ('interaction_geometry_input','interaction_responsibility_private_pool')else 1):
                    raise AssertionError('Actual native reverse collection count differs')
                report['actual_call_assignment_checks'].append({'actual_autograd_calls':calls,'assigned_returned_tensors_exact':bool(calls),
                    'scalar_J_objective_identity_audited':not bool(calls),'actual_gradients_finite_and_active':True,
                    'one_original_Adam_transition':True,'extra_reverse_collections':0,'source_or_auxiliary_extra_forward_calls':0})
                if len(set(value['interaction_diagnostics']['four_route_query_signatures']))!=1 or current.session._depth:
                    raise AssertionError('Four original source route identities differ')
                if sum(value['interaction_diagnostics']['source_group_counts'].values())!=1024:
                    raise AssertionError('Native mixed-batch group population changed')
                components=value['component_gradients']
                if args.method!='interaction_responsibility_private_pool' and not any((r['L2']or 0)>0 for r in components['input_residual']):
                    raise AssertionError('Input correction received no live native objective gradient')
                if args.method=='interaction_input_conditionalJ' and not any((r['L2']or 0)>0 for r in components['readout']):
                    raise AssertionError('Conditional readout received no live native objective gradient')
                if args.method=='interaction_geometry_input' and not value['interaction_diagnostics']['contrast']['auxiliary']['active']:
                    raise AssertionError('Actual native endpoint contrast lacks support')
                return value
            finally:
                current.session.objectives.build=original_build;torch.autograd.grad=original_grad;torch.Tensor.backward=original_backward;optimizer.step=original_step
        adapter = construct(); adapter.assert_identity()
        n = adapter.session.native
        report.update(native_TRAIN_positive_records=int(len(n.positive)),graph_nodes=int(n.features.shape[0]),native_features=int(n.features.shape[1]),
            native_optimizer_options=[{k:v for k,v in g.items() if k!='params'} for g in n.optimizer.param_groups],stage='first_source_update')
        write(); report['first_update'] = train(adapter)
        if report['actual_optimizer_transitions'] != 1: raise AssertionError('First source optimizer count differs')
        report['stage']='first_full_VALID';write(); first = valid(adapter)
        report['stage']='save_source_checkpoint';write()
        checkpoint = output/'after_one_update.pt'; torch.save({'session':adapter.state(),'job':job},checkpoint)
        report['checkpoint']=pin(checkpoint)
        first_outputs = first['outputs'].clone(); first_mean = first.get('same_bank_mean_outputs',first['outputs']).clone()
        first_ids, first_labels = first['query_endpoints'].clone(),first['labels'].clone()
        first_metrics,first_loss=copy.deepcopy(first['metrics']),first['loss']
        first_mean_metrics,first_mean_loss=copy.deepcopy(first['same_bank_mean_metrics']),first['same_bank_mean_loss']
        report['source_and_method_metadata']=adapter.metadata()
        adapter.close(); del n, adapter, first; adapter=None; gc.collect()
        report['stage']='construct_fresh_source_and_restore';write(); adapter=construct()
        references={k:id(v) for k,v in adapter.session.native.source_parameter_references.items()}
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);adapter.load_state(saved['session'])
        if references != {k:id(v) for k,v in adapter.session.native.source_parameter_references.items()}:
            raise AssertionError('Fresh native Parameter identities changed during restore')
        no_stale_sidecar(adapter);exact(saved['session'],adapter.state())
        report['fresh_restored_all_native_optimizer_sampler_RNG_state_exact']=True
        report['stage']='fresh_restore_full_VALID';write(); replay=valid(adapter)
        report['restored_served_max_abs_difference']=float((replay['outputs']-first_outputs).abs().max())
        report['restored_served_bit_exact']=torch.equal(replay['outputs'],first_outputs)
        report['restored_mean_bit_exact']=torch.equal(replay.get('same_bank_mean_outputs',replay['outputs']),first_mean)
        if not report['restored_served_bit_exact'] or not report['restored_mean_bit_exact'] \
                or not torch.equal(replay['query_endpoints'],first_ids) or not torch.equal(replay['labels'],first_labels):
            raise AssertionError('Fresh restoration changed served/mean output or actual query roles')
        if replay['metrics']!=first_metrics or replay['loss']!=first_loss or replay['same_bank_mean_metrics']!=first_mean_metrics or replay['same_bank_mean_loss']!=first_mean_loss:
            raise AssertionError('Restored full native actual/mean metrics or loss changed')
        adapter.load_state(saved['session'])
        report['stage']='resumed_second_source_update';write();report['resumed_second_update']=train(adapter)
        report['stage']='final_full_VALID';write();final=valid(adapter)
        if report['actual_optimizer_transitions'] != 2 or adapter.session.native.updates != 2:
            raise AssertionError('Expected exactly two actual source optimizer transitions and restored clock2')
        expected_reverse=4 if args.method in ('interaction_geometry_input','interaction_responsibility_private_pool') else 2
        if report['native_model_forwards_attempted']!=32 or report['native_model_forwards_completed']!=32 or report['actual_reverse_collections_completed']!=expected_reverse:
            raise AssertionError('Bounded source forward/reverse counts differ')
        torch.save({'session':adapter.state(),'job':job},output/'after_two_updates.pt');report['final_checkpoint']=pin(output/'after_two_updates.pt')
        report.update(status='passed',stage='complete',native_update_clock=adapter.session.native.updates,
            source_native_Parameter_identity_preserved=True,complete_source_minibatch_preserved=True,
            final_served_finite=True,final_same_bank_mean_finite=bool(torch.isfinite(final.get('same_bank_mean_outputs',final['outputs'])).all()),
            full_VALID_positive_records=60084,full_VALID_negative_records=100000)
    except BaseException as exc:
        report.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(limit=14)})
    finally:
        if adapter is not None:
            try:adapter.close()
            except Exception as exc:report['close_error']=str(exc)
        report['memory_events_after']=Path('/sys/fs/cgroup/memory.events').read_text()
        report['limits']=['Exact BUDDY/Collab CPU two-update engineering check only','No completed full TRAIN epoch, convergence or accuracy comparison',
            'Authenticated native BUDDY seed0; NCN/NCNC and production nonzero-seed providers require separate receipts','No CUDA admission, pair-resource qualification, TEST scoring or final TEST wrapper reconstruction']
        write()
    print(json.dumps({k:report[k] for k in ('method','status','actual_optimizer_transitions','VALID_calls','native_constructions','elapsed_seconds')}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
