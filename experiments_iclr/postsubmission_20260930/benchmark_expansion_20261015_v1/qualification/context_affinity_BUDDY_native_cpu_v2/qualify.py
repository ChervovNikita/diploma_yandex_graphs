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
    report = {'schema':'native-BUDDY-context-affinity-two-transition-real-CPU-v1', 'status':'running','method':args.method,
        'device':'cpu','threads':2,'address_space_limit_bytes':64*1024**3,'source_joint_batch':1024,'native_training_reduction':'joint source BCE mean',
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
        jobs = json.loads((package/'JOBS.json').read_text())
        if args.method not in jobs: raise ValueError('Only the two frozen BUDDY C3 methods are admitted')
        job = copy.deepcopy(jobs[args.method])
        if job['recipe']['_repository_root'] != str(root): raise ValueError('Frozen job repository differs')
        method_package = expansion/'context_affinity_BUDDY_port_v1'
        callback = method_package/'callback.py'
        if pin(method_package/'MANIFEST.json')['sha256'] != job['method_settings']['manifest_sha256']:
            raise ValueError('Frozen BUDDY C3 source differs')
        if pin(job['method_settings']['context_manifest'])['sha256'] != job['method_settings']['context_manifest_sha256']:
            raise ValueError('Frozen materialized native context differs')
        policy_source = load(expansion/'link_policy_v1/policy.py','_qualify_BUDDY_policy')
        policy = policy_source.create_policy(job,root)
        source = load(callback,'_qualify_BUDDY_callback')
        report.update(source_smoke_case=case['id'],smoke_manifest=pin(manifest),callback=pin(callback),method_manifest=pin(method_package/'MANIFEST.json'),
            torch=str(torch.__version__),job=job,stage='construct_first_native_source')
        write()
        def construct():
            report['native_construction_attempts'] = report.get('native_construction_attempts',0)+1; write()
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
            if not value.get('materialized_VALID_context_verified'):
                raise AssertionError('Both complete native BUDDY VALID hydration roles must authenticate')
            report.setdefault('VALID_diagnostics', []).append({'call':report['VALID_calls'], 'metrics':value['metrics'],
                'loss':value['loss'], 'materialized_context_verified':True, 'work':value['work']})
            write()
            return value
        report['additional_auxiliary_gradient_collections'] = 0
        report['ordinary_training_backward_collections'] = 0
        report['auxiliary_gradient_probes'] = []
        def train(current,probe_auxiliary):
            original = current.session.objective.context_auxiliary
            def probe(*pargs, **kwargs):
                extra = original(*pargs, **kwargs)
                if not probe_auxiliary:return extra
                n = current.session.native
                shared = tuple(n.source_parameter_references.values())
                private = tuple(n.core.private_parameters())
                output_ids = {id(p) for p in n.core.banks['site_2'].parameters()}
                gradients = torch.autograd.grad(extra['loss'], shared+private, retain_graph=True, allow_unused=True)
                report['additional_auxiliary_gradient_collections'] += 1
                if any(g is not None and not torch.isfinite(g).all() for g in gradients):
                    raise AssertionError('Nonfinite passive auxiliary gradient')
                def norm(rows):
                    return sum(float(g.square().sum()) for g in rows if g is not None)**.5
                count = len(shared)
                input_gradients = [g for p,g in zip(private, gradients[count:]) if id(p) not in output_ids]
                output_gradients = [g for p,g in zip(private, gradients[count:]) if id(p) in output_ids]
                row = {'shared_auxiliary_gradient_L2':norm(gradients[:count]),
                    'private_input_auxiliary_gradient_L2':norm(input_gradients),
                    'private_output_auxiliary_gradient_L2':norm(output_gradients),
                    'optimizer_state_mutated':False, 'source_or_auxiliary_extra_forwards':0,
                    'auxiliary_objects':int(kwargs['preheads'].shape[1]),
                    'source_reduction':kwargs['reduction']}
                report['auxiliary_gradient_probes'].append(row);write()
                if row['shared_auxiliary_gradient_L2']<=0 or row['private_input_auxiliary_gradient_L2']<=0 or row['private_output_auxiliary_gradient_L2']!=0:
                    raise AssertionError('Expected live shared/input-factor AUX path and zero direct output-factor path')
                return extra
            current.session.objective.context_auxiliary = probe
            try:
                value = current.session.train_update()
                details = value['context_affinity_diagnostics']
                if value['work']['query_records']!=1024 or value['work']['native_BUDDY_forwards']!=4 or value['work']['captured_native_task_query_records']!=1024:
                    raise AssertionError('Native F minibatch changed')
                if details['support'].get('distinct_target_anchors_at_1e_8',0)<=0 or details['support'].get('A_nonuniform_anchors_at_1e_8',0)<=0:
                    raise AssertionError('No distinct/nonuniform live context targets in actual source batch')
                if len(set(details['four_route_query_signatures']))!=1:
                    raise AssertionError('Actual native four-route queries differ')
                if not details['native_joint_BCE_parity_checked'] or sum(details['native_source_group_counts'].values()) != 1024:
                    raise AssertionError('Native mixed source population or joint BCE parity differs')
                if value['work']['reverse_collections']!=1 or value['work']['extra_source_forward_views']!=0:
                    raise AssertionError('One ordinary training reverse collection and no extra source forward required')
                report['ordinary_training_backward_collections'] += 1
                return value
            finally:
                current.session.objective.context_auxiliary = original
        def exact(a,b,path='state'):
            if isinstance(a,torch.Tensor):
                if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a.cpu(),b.cpu()):
                    raise AssertionError('Fresh exact restored tensor mismatch at '+path)
            elif isinstance(a,np.ndarray):
                if not isinstance(b,np.ndarray) or not np.array_equal(a,b): raise AssertionError('Restored NumPy state differs at '+path)
            elif isinstance(a,dict):
                if not isinstance(b,dict) or set(a)!=set(b):raise AssertionError('Restored keys differ at '+path)
                for key in a:exact(a[key],b[key],path+'.'+str(key))
            elif isinstance(a,(tuple,list)):
                if type(a)is not type(b) or len(a)!=len(b):raise AssertionError('Restored sequence differs at '+path)
                for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'['+str(i)+']')
            elif type(a)is not type(b) or a!=b:raise AssertionError('Restored scalar differs at '+path)
        adapter = construct(); adapter.assert_identity()
        n = adapter.session.native
        report.update(native_TRAIN_positive_records=int(len(n.positive)),graph_nodes=int(n.features.shape[0]),native_features=int(n.features.shape[1]),
            native_optimizer_options=[{k:v for k,v in g.items() if k!='params'} for g in n.optimizer.param_groups],stage='first_source_update')
        write(); report['first_update'] = train(adapter,probe_auxiliary=True)
        if report['actual_optimizer_transitions'] != 1: raise AssertionError('First source optimizer count differs')
        report['stage']='first_full_VALID';write(); first = valid(adapter)
        report['stage']='save_source_checkpoint';write()
        checkpoint = output/'after_one_update.pt'; torch.save({'session':adapter.state(),'job':job},checkpoint)
        report['checkpoint']=pin(checkpoint)
        first_outputs = first['outputs'].clone(); first_mean = first.get('same_bank_mean_outputs',first['outputs']).clone()
        first_ids, first_labels = first['query_endpoints'].clone(),first['labels'].clone()
        first_metrics, first_loss = copy.deepcopy(first['metrics']), first['loss']
        report['source_and_method_metadata']=adapter.metadata()
        adapter.close(); del n, adapter, first; adapter=None; gc.collect()
        report['stage']='construct_fresh_source_and_restore';write(); adapter=construct()
        references={k:id(v) for k,v in adapter.session.native.source_parameter_references.items()}
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);adapter.load_state(saved['session'])
        if references != {k:id(v) for k,v in adapter.session.native.source_parameter_references.items()}:
            raise AssertionError('Fresh native Parameter identities changed during restore')
        exact(saved['session'],adapter.state())
        report['fresh_restored_original_Adam_sampler_global_private_RNG_targets_exact']=True
        report['stage']='fresh_restore_full_VALID';write(); replay=valid(adapter)
        report['restored_served_max_abs_difference']=float((replay['outputs']-first_outputs).abs().max())
        report['restored_served_bit_exact']=torch.equal(replay['outputs'],first_outputs)
        report['restored_mean_bit_exact']=torch.equal(replay.get('same_bank_mean_outputs',replay['outputs']),first_mean)
        if not report['restored_served_bit_exact'] or not report['restored_mean_bit_exact'] \
                or not torch.equal(replay['query_endpoints'],first_ids) or not torch.equal(replay['labels'],first_labels):
            raise AssertionError('Fresh restoration changed served/mean output or actual query roles')
        if replay['metrics'] != first_metrics or replay['loss'] != first_loss: raise AssertionError('Restored native VALID metrics/loss differ')
        adapter.load_state(saved['session'])
        report['stage']='resumed_second_source_update';write();report['resumed_second_update']=train(adapter,probe_auxiliary=True)
        report['stage']='final_full_VALID';write();final=valid(adapter)
        if report['actual_optimizer_transitions'] != 2 or adapter.session.native.updates != 2:
            raise AssertionError('Expected exactly two actual source optimizer transitions and restored clock2')
        torch.save({'session':adapter.state(),'job':job},output/'after_two_updates.pt')
        report['final_checkpoint']=pin(output/'after_two_updates.pt')
        if report['additional_auxiliary_gradient_collections']!=2 or report['ordinary_training_backward_collections']!=2: raise AssertionError('Bounded reverse collection counts differ')
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
            'Native shared BN retained; private BN is a separate candidate; authenticated seed0 only','No CUDA admission, pair-resource qualification, TEST scoring or final TEST wrapper reconstruction']
        write()
    print(json.dumps({k:report[k] for k in ('method','status','actual_optimizer_transitions','VALID_calls','native_constructions','elapsed_seconds')}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
