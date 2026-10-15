"""One fixed NCN method: two source-sized CPU updates, full VALID, fresh restore."""
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
    resource.setrlimit(resource.RLIMIT_AS, (32*1024**3,32*1024**3))
    started = time.perf_counter()
    report = {'schema':'native-NCN-interaction-v2-two-transition-real-CPU-v1', 'status':'running','method':args.method,
        'device':'cpu','threads':2,'address_space_limit_bytes':32*1024**3,'source_positive_batch':65536,'source_negative_batch':65536,
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
        case = next(row for row in json.loads(manifest.read_text())['cases'] if row['id']=='collab-ncn/gnnm_boundary_native')
        recipe = copy.deepcopy(case['recipe']); recipe['_repository_root'] = str(root); recipe['_seed'] = 0; recipe['seed'] = 0
        if args.method not in ('interaction_geometry_input','interaction_responsibility_private_pool','interaction_input_meanJ','interaction_input_conditionalJ'):
            raise ValueError('Only the four frozen explicit NCN interaction recipes are admitted')
        method_package = expansion/'interaction_link_ncn_v2'
        callback = method_package/'callback.py'
        settings = {'manifest_sha256':pin(method_package/'MANIFEST.json')['sha256']}
        job = {'schema':'native-full-epoch-fit-v1','id':'NCN-interaction-v2-CPU-engineering/'+args.method,'cohort_id':'NCN-CPU-engineering/split0/seed0',
            'family':'link','procedure':args.method,'base_procedure':'gnnm_boundary_native','seed':0,'split':0,'body':0,
            'recipe':recipe,'data_root':case['data_root'],'device':'cpu','TEST_access':False,
            'callback_module':str(callback),'callback_sha256':pin(callback)['sha256'],'method_settings':settings,
            'policy':{'family':'link','track':'matched_extended','maximum_epochs':1000,'minimum_epochs':100,'patience':50,
                'metric':'hits@50','direction':'max','ties':'first','evaluate_TRAIN_each_epoch':False,'evaluation_mode':'VALID_only_disclosed_adaptation'}}
        policy_source = load(expansion/'link_policy_v1/policy.py','_qualify_NCN_policy')
        policy = policy_source.create_policy(job,root)
        source = load(callback,'_qualify_NCN_callback')
        report.update(source_smoke_case=case['id'],smoke_manifest=pin(manifest),callback=pin(callback),method_manifest=pin(method_package/'MANIFEST.json'),
            torch=str(torch.__version__),job=job,stage='construct_first_native_source')
        write()
        def construct():
            current = source.create_adapter(torch,copy.deepcopy(job),policy,root)
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
            sidecar=source.source().selected_mean_sidecar(current.session)
            if sidecar['source_updates']!=current.session.native.updates or not torch.equal(sidecar['actual_served_raw_logits'],value['outputs']) \
                    or not torch.equal(sidecar['same_bank_mean_raw_logits'],value['same_bank_mean_outputs']) or sidecar['extra_forward_calls']!=0:
                raise AssertionError('Same-call served/mean sidecar identities or source update clock differ')
            return value
        report['additional_old_state_gradient_collections'] = 0
        report['old_state_gradient_checks'] = []
        def exact(a,b,path='state'):
            if isinstance(a,torch.Tensor):
                if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a.cpu(),b.cpu()):
                    raise AssertionError('Exact restored tensor mismatch at '+path)
            elif isinstance(a,dict):
                if not isinstance(b,dict) or set(a)!=set(b): raise AssertionError('Restored keys differ at '+path)
                for key in a: exact(a[key],b[key],path+'.'+str(key))
            elif isinstance(a,(tuple,list)):
                if type(a)is not type(b) or len(a)!=len(b): raise AssertionError('Restored sequence differs at '+path)
                for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'['+str(i)+']')
            elif type(a)is not type(b) or a!=b: raise AssertionError('Restored scalar differs at '+path)
        def no_stale_sidecar(current):
            try: source.source().selected_mean_sidecar(current.session)
            except ValueError:return
            raise AssertionError('Sidecar survived TRAIN/restore without fresh current VALID')
        def train(current,probe_credit):
            no_stale_sidecar(current) if current.session._last_VALID is None else None
            original=current.session.objectives.build
            optimizer=current.optimizers[0];original_step=optimizer.step
            expected={};checked=[False]
            def build(*pargs,**kwargs):
                terms=original(*pargs,**kwargs)
                if probe_credit:
                    n=current.session.native;shared=tuple(n.source_parameter_references.values());private=tuple(n.core.private_parameters())
                    if terms['scalar_update']:
                        gradients=torch.autograd.grad(terms['shared_objective'],shared+private,retain_graph=True,allow_unused=True)
                        collections=1
                    else:
                        gs=torch.autograd.grad(terms['shared_objective'],shared,retain_graph=True,allow_unused=True)
                        gp=torch.autograd.grad(terms['private_objective'],private,retain_graph=True,allow_unused=True)
                        gradients=gs+gp;collections=2
                    report['additional_old_state_gradient_collections']+=collections
                    for p,g in zip(shared+private,gradients):expected[id(p)]=None if g is None else g.detach().clone()
                return terms
            def before_step(*a,**kw):
                if probe_credit:
                    n=current.session.native
                    for p in n.model.parameters():
                        g=expected[id(p)]
                        if (g is None)!=(p.grad is None) or (g is not None and not torch.equal(g,p.grad)):
                            raise AssertionError('Assigned interaction gradient differs from exact declared old-state credit')
                    checked[0]=True
                return original_step(*a,**kw)
            current.session.objectives.build=build;optimizer.step=before_step
            try:
                value=current.session.train_update();no_stale_sidecar(current)
                if value['work']['positive_query_records']!=65536 or value['work']['negative_query_records']!=65536:
                    raise AssertionError('Original source query batch changed')
                if value['work']['reverse_collections']!=(2 if args.method in ('interaction_geometry_input','interaction_responsibility_private_pool')else 1):
                    raise AssertionError('Actual native reverse collection count differs')
                if probe_credit and not checked[0]:raise AssertionError('Old-state credit probe did not reach native Adam')
                report['old_state_gradient_checks'].append({'probed':probe_credit,'assigned_gradients_exact':checked[0],
                    'source_or_auxiliary_extra_forward_calls':0,'optimizer_state_mutated_by_probe':False})
                components=value['component_gradients']
                if args.method!='interaction_responsibility_private_pool' and not any((r['L2']or 0)>0 for r in components['input_residual']):
                    raise AssertionError('Input correction received no live native objective gradient')
                if args.method=='interaction_input_conditionalJ' and not any((r['L2']or 0)>0 for r in components['readout']):
                    raise AssertionError('Conditional readout received no live native objective gradient')
                if args.method=='interaction_geometry_input' and not value['interaction_diagnostics']['contrast']['auxiliary']['active']:
                    raise AssertionError('Actual native endpoint contrast lacks support')
                return value
            finally:
                current.session.objectives.build=original;optimizer.step=original_step
        adapter = construct(); adapter.assert_identity()
        n = adapter.session.native
        report.update(native_TRAIN_positive_records=int(n.positive.shape[1]),graph_nodes=int(n.x.shape[0]),native_features=int(n.x.shape[1]),
            native_optimizer_options=[{k:v for k,v in g.items() if k!='params'} for g in n.optimizer.param_groups],stage='first_source_update')
        write(); report['first_update'] = train(adapter,probe_credit=True)
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
        no_stale_sidecar(adapter);exact(saved['session'],adapter.state())
        report['fresh_restored_all_native_optimizer_sampler_RNG_state_exact']=True
        report['stage']='fresh_restore_full_VALID';write(); replay=valid(adapter)
        report['restored_served_max_abs_difference']=float((replay['outputs']-first_outputs).abs().max())
        report['restored_served_bit_exact']=torch.equal(replay['outputs'],first_outputs)
        report['restored_mean_bit_exact']=torch.equal(replay.get('same_bank_mean_outputs',replay['outputs']),first_mean)
        if not report['restored_served_bit_exact'] or not report['restored_mean_bit_exact'] \
                or not torch.equal(replay['query_endpoints'],first_ids) or not torch.equal(replay['labels'],first_labels):
            raise AssertionError('Fresh restoration changed served/mean output or actual query roles')
        adapter.load_state(saved['session'])
        report['stage']='resumed_second_source_update';write();report['resumed_second_update']=train(adapter,probe_credit=False)
        report['stage']='final_full_VALID';write();final=valid(adapter)
        if report['actual_optimizer_transitions'] != 2 or adapter.session.native.updates != 2:
            raise AssertionError('Expected exactly two actual source optimizer transitions and restored clock2')
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
        report['limits']=['Exact NCN/Collab CPU two-update engineering check only','No completed full TRAIN epoch, convergence or accuracy comparison',
            'This interaction factory supports NCN only; NCNC/BUDDY unsupported','No CUDA admission, pair-resource qualification, TEST scoring or final TEST wrapper reconstruction']
        write()
    print(json.dumps({k:report[k] for k in ('method','status','actual_optimizer_transitions','VALID_calls','native_constructions','elapsed_seconds')}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
