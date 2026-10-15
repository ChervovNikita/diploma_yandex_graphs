"""One fixed BUDDY method: two source-sized CPU updates, full VALID, fresh restore."""
from __future__ import annotations
import argparse
import copy
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
    parser.add_argument('--context-manifest', required=True); parser.add_argument('--root', required=True); parser.add_argument('--method', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); root = Path(args.root).resolve(); output = Path(args.output).resolve()
    package = Path(__file__).resolve().parent; expansion = package.parent; phase = expansion.parent
    if not all(path.is_relative_to(root) for path in (package,output,phase)) or output.exists():
        raise ValueError('Fresh own repository output required')
    output.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_AS, (64*1024**3,64*1024**3))
    started = time.perf_counter()
    report = {'schema':'native-BUDDY-two-transition-real-CPU-v1', 'status':'running','method':args.method,
        'device':'cpu','threads':2,'address_space_limit_bytes':64*1024**3,'source_joint_batch':1024,'native_training_reduction':'joint source BCE mean, no added class balancing',
        'actual_optimizer_transitions':0,'optimizer_attempts':0,'VALID_calls':0,'native_constructions':0,
        'TEST_accessed':False,'scientific_fit':False,'GPU_models_or_updates':0,'script':pin(__file__),
        'stage':'before_numerical_import','memory_events_before':Path('/sys/fs/cgroup/memory.events').read_text()}
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
        recipe = copy.deepcopy(case['recipe']); recipe['_repository_root'] = str(root); recipe['_seed'] = 0; recipe['seed'] = 0
        portable = args.method.startswith('portable_input_')
        method_package = expansion/'portable_private_input_v2' if portable else package
        callback = method_package/('link_callback.py' if portable else 'callback.py')
        settings = {'manifest_sha256':pin(method_package/'MANIFEST.json')['sha256']}
        if not portable: settings.update(context_manifest=str(Path(args.context_manifest).resolve()),context_manifest_sha256=pin(args.context_manifest)['sha256'])
        if portable: settings.update(rank_cap=8,frame_policy='disjoint_if_fit_else_separate_orthonormal_frames')
        job = {'schema':'native-full-epoch-fit-v1','id':'BUDDY-CPU-engineering/'+args.method,'cohort_id':'BUDDY-CPU-engineering/split0/seed0',
            'family':'link','procedure':args.method,'base_procedure':'gnnm_boundary_native','seed':0,'split':0,'body':0,
            'recipe':recipe,'data_root':case['data_root'],'device':'cpu','TEST_access':False,
            'callback_module':str(callback),'callback_sha256':pin(callback)['sha256'],'method_settings':settings,
            'policy':{'family':'link','track':'matched_extended','maximum_epochs':1000,'minimum_epochs':100,'patience':50,
                'metric':'hits@50','direction':'max','ties':'first','evaluate_TRAIN_each_epoch':False,'evaluation_mode':'VALID_only_disclosed_adaptation'}}
        policy_source = load(expansion/'link_policy_v1/policy.py','_qualify_BUDDY_policy')
        policy = policy_source.create_policy(job,root)
        source = load(callback,'_qualify_BUDDY_callback')
        report.update(source_smoke_case=case['id'],smoke_manifest=pin(manifest),callback=pin(callback),method_manifest=pin(method_package/'MANIFEST.json'),
            torch=str(torch.__version__),job=job,stage='construct_first_native_source')
        write()
        context_library=load(package/'session.py','_BUDDY_context_auditor')
        context_job=copy.deepcopy(job)
        context_job['method_settings'].update(context_manifest=str(Path(args.context_manifest).resolve()),context_manifest_sha256=pin(args.context_manifest)['sha256'])
        report['materialized_context_manifest']=pin(args.context_manifest)
        def attach_context_auditor(current):
            import types
            n=current.session.native
            context,context_pin=context_library.native_context(n,current.session,context_job)
            state={'active':False,'ids':None,'VALID_checks':0}
            original_next,original_hydrate=n._next_batch,n._hydrate
            original_update,original_valid,original_close=current.session.train_update,current.session.validate,current.session.close
            def next_batch(owner):
                ids=original_next()
                if state['active']:state['ids']=ids.cpu().clone()
                return ids
            def hydrate(owner,links):
                values=original_hydrate(links)
                if state['active']:
                    expected=context['TRAIN_batches'][n.updates]
                    if not torch.equal(state['ids'],expected['sampler_ids']) or not torch.equal(links.cpu(),expected['links']) or not torch.equal(n.labels[state['ids']],expected['labels']):
                        raise AssertionError('Portable BUDDY sampler/context mismatch')
                else:
                    expected=next((row for row in context['VALID_inputs'].values() if links.shape==row['links'].shape and torch.equal(links.cpu(),row['links'])),None)
                    if expected is None:raise AssertionError('Unexpected portable BUDDY VALID context')
                    state['VALID_checks']+=1
                if any(not torch.equal(value.detach().cpu(),bound) for value,bound in zip(values,expected['inputs'])):
                    raise AssertionError('Portable BUDDY source hydration/materialized input mismatch')
                return values
            def update(session):
                state['active']=True
                try:return original_update()
                finally:state['active']=False
            def validate(session):
                state['VALID_checks']=0;value=original_valid()
                if state['VALID_checks']!=2:raise AssertionError('Both full BUDDY VALID roles required')
                value['materialized_VALID_context_verified']=True;return value
            def close(session):
                n._next_batch,n._hydrate=original_next,original_hydrate;original_close()
            n._next_batch,n._hydrate=types.MethodType(next_batch,n),types.MethodType(hydrate,n)
            current.session.train_update=types.MethodType(update,current.session)
            current.session.validate=types.MethodType(validate,current.session)
            current.session.close=types.MethodType(close,current.session)
        def construct():
            report['native_construction_attempts'] = report.get('native_construction_attempts',0)+1
            write()
            current = source.create_adapter(torch,copy.deepcopy(job),policy,root)
            if portable:
                attach_context_auditor(current)
            report['native_constructions'] += 1
            if any(p.device.type != 'cpu' for p in current.model.parameters()): raise AssertionError('Non-CPU model parameter')
            optimizer = current.optimizers[0]; original = optimizer.step
            def step(*a,**kw):
                report['optimizer_attempts'] += 1
                value = original(*a,**kw)
                report['actual_optimizer_transitions'] += 1; write(); return value
            optimizer.step = step
            return current
        def valid(current):
            value = current.validate(); report['VALID_calls'] += 1
            if (value['complete_VALID_positive_records'],value['complete_VALID_negative_records']) != (60084,100000) \
                    or value.get('complete_source_VALID') is not True or value.get('TEST_accessed') is not False:
                raise AssertionError('Incomplete source VALID or TEST disclosure')
            if not torch.isfinite(value['outputs']).all(): raise AssertionError('Nonfinite served full VALID')
            if not torch.equal(value['query_endpoints'],torch.cat((current.session.native.valid['edge'],current.session.native.valid['edge_neg'])).cpu()):
                raise AssertionError('Source VALID endpoint identities/order changed')
            report.setdefault('VALID_diagnostics', []).append({'call':report['VALID_calls'],'metrics':value['metrics'],'loss':value['loss'],'same_bank_mean_metrics':value.get('same_bank_mean_metrics',value['metrics']),'same_bank_mean_loss':value.get('same_bank_mean_loss',value['loss']), 'materialized_context_verified':value.get('materialized_VALID_context_verified')})
            return value
        adapter = construct(); adapter.assert_identity()
        n = adapter.session.native
        report.update(native_TRAIN_positive_records=int(len(n.positive)),graph_nodes=int(n.features.shape[0]),native_features=int(n.features.shape[1]),
            native_optimizer_options=[{k:v for k,v in g.items() if k!='params'} for g in n.optimizer.param_groups],stage='first_source_update')
        write(); report['first_update'] = adapter.session.train_update()
        if report['actual_optimizer_transitions'] != 1: raise AssertionError('First source optimizer count differs')
        report['stage']='first_full_VALID';write(); first = valid(adapter)
        report['stage']='save_source_checkpoint';write()
        checkpoint = output/'after_one_update.pt'; torch.save({'session':adapter.state(),'job':job},checkpoint)
        report['checkpoint']=pin(checkpoint)
        first_outputs = first['outputs'].clone(); first_mean = first.get('same_bank_mean_outputs',first['outputs']).clone()
        first_ids, first_labels = first['query_endpoints'].clone(),first['labels'].clone()
        report['source_and_method_metadata']=adapter.metadata()
        adapter.close(); del n, adapter, first; adapter=None; gc.collect()
        report['stage']='construct_fresh_source_and_restore';write(); adapter=construct()
        references={k:id(v) for k,v in adapter.session.native.source_parameter_references.items()}
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);adapter.load_state(saved['session'])
        if references != {k:id(v) for k,v in adapter.session.native.source_parameter_references.items()}:
            raise AssertionError('Fresh native Parameter identities changed during restore')
        report['stage']='fresh_restore_full_VALID';write(); replay=valid(adapter)
        report['restored_served_max_abs_difference']=float((replay['outputs']-first_outputs).abs().max())
        report['restored_served_bit_exact']=torch.equal(replay['outputs'],first_outputs)
        report['restored_mean_bit_exact']=torch.equal(replay.get('same_bank_mean_outputs',replay['outputs']),first_mean)
        if not report['restored_served_bit_exact'] or not report['restored_mean_bit_exact'] \
                or not torch.equal(replay['query_endpoints'],first_ids) or not torch.equal(replay['labels'],first_labels):
            raise AssertionError('Fresh restoration changed served/mean output or actual query roles')
        adapter.load_state(saved['session'])
        report['stage']='resumed_second_source_update';write();report['resumed_second_update']=adapter.session.train_update()
        report['stage']='final_full_VALID';write();final=valid(adapter)
        if report['actual_optimizer_transitions'] != 2 or adapter.session.native.updates != 2:
            raise AssertionError('Expected exactly two actual source optimizer transitions and restored clock2')
        torch.save({'session':adapter.state(),'job':job},output/'after_two_updates.pt')
        report['final_checkpoint']=pin(output/'after_two_updates.pt')
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
            'NCN/NCNC remain separate packages; seed0 BUDDY context only','No CUDA admission, pair-resource qualification, TEST scoring or final TEST wrapper reconstruction']
        write()
    print(json.dumps({k:report[k] for k in ('method','status','actual_optimizer_transitions','VALID_calls','native_constructions','elapsed_seconds')}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
