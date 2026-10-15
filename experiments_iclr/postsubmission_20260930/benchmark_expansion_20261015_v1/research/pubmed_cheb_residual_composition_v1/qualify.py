"""Two full-graph engineering updates, three VALID calls and fresh restore.

One explicit variant/device per invocation. No fit, selector or TEST. Requires
root staging/review and outer literal-allocation guard before reading source.
"""
import argparse
import copy
import gc
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback


def require(ok, message):
    if not ok: raise ValueError(message)


def main():
    started = time.perf_counter()
    def timeout(signum, frame): raise TimeoutError('Bounded engineering invocation exceeded900seconds')
    def interrupted(signum, frame): raise InterruptedError('Engineering owner received signal'+str(signum))
    signal.signal(signal.SIGALRM, timeout); signal.signal(signal.SIGTERM, interrupted)
    signal.setitimer(signal.ITIMER_REAL,900)
    require(socket.gethostname() == 'anogena-2-0' and subprocess.check_output(
        ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'], text=True, timeout=10).splitlines() ==
        ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Dedicated allocation required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True); parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--variant', choices=('uniform','structured','permuted'), required=True)
    parser.add_argument('--device', choices=('cpu','cuda:0'), default='cpu'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); root = Path(args.project_root).resolve(); here = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    require(here.is_relative_to(root) and output.is_relative_to(root) and not output.exists(), 'Fresh contained engineering output required')
    require(Path(sys.prefix).resolve() == (root/'.venv').resolve(), 'Original virtualenv required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' if args.device == 'cpu' else
        os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','GPU-44039938-fd82-41d2-fefd-de71514e2fac'), 'Explicit device visibility required')
    spec = importlib.util.spec_from_file_location('_pubmed_cheb_qualifier_extension', here/'extension.py')
    ext = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ext; spec.loader.exec_module(ext)
    receipts = ext.source_bindings(root, args.manifest_sha256)
    binding = json.loads((here/'BINDING.json').read_text())
    for row in binding['data_pins']:
        path = Path(row['path']).resolve(); require(path.is_relative_to(root), 'Data receipt escaped project')
        actual = ext.pin(path)
        require(all(actual[k] == row[k] for k in ('bytes','sha256')), 'Exact existing provider/basis/roles changed')
    planned = ext.descriptor(root, 101, args.variant, args.manifest_sha256)
    sys.path.insert(0, str(ext.E/'runtime'))
    from run_fit import validate_job, OptimizerObserver
    from adapters import create_adapter
    from common import seed_all, global_rng, restore_rng, cpu_clone
    policy = validate_job(planned); job = copy.deepcopy(planned); job['device'] = args.device
    output.mkdir(parents=True)
    report = {'schema':'pubmed-cheb-residual-two-update-engineering-v1','status':'running','variant':args.variant,
        'device':args.device,'source_receipts':receipts,'script':ext.pin(__file__),'planned_job_hash':ext.digest(planned),
        'selector_calls':0,'scientific_fit':False,'TEST_accessed':False,'constructions_attempted':0,'constructions_completed':0,
        'native_forward_attempts_including_recomputation':0,'native_forwards_completed':0,
        'reverse_attempts_including_checks':0,'reverse_completed_including_checks':0,'VALID_completed':0,
        'optimizer_calls':{'attempts':[0],'completed':[0]},'bounds':{'constructions':2,'optimizer_transitions':2,
        'VALID_calls':3,'reverse_collections_including_checks':8,'native_forward_attempts_including_recomputation':256,
        'invocation_wall_seconds':900},'CPU_threads':2,'VALID_wall_seconds':[],'diagnostic_update_wall_seconds':[]}
    adapters, hooks, observer = [], [], None

    def write():
        if observer is not None: report['optimizer_calls'] = observer.state()
        report.update(wall_seconds=time.perf_counter()-started, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output/'RESULT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')

    try:
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key] = '2'
        import torch
        torch.set_num_threads(2); report['torch'] = str(torch.__version__)
        require(args.device != 'cpu' or not torch.cuda.is_initialized(), 'CPU qualifier must not initialize CUDA')
        if args.device == 'cuda:0': torch.cuda.reset_peak_memory_stats()

        def construct():
            report['constructions_attempted'] += 1; require(report['constructions_attempted'] <= 2, 'Construction cap exceeded')
            seed_all(torch, 101); a = create_adapter(torch, copy.deepcopy(job), policy, root); adapters.append(a)
            report['constructions_completed'] += 1
            require(sum(p.numel() for p in a.model.parameters()) == 600627, 'Exact total parameter count differs')
            def before(module, arguments):
                report['native_forward_attempts_including_recomputation'] += 1
                require(report['native_forward_attempts_including_recomputation'] <= 256, 'Native forward attempt cap exceeded')
            def after(module, arguments, result): report['native_forwards_completed'] += 1
            hooks.extend([a.model.native.register_forward_pre_hook(before), a.model.native.register_forward_hook(after)])
            return a

        def valid(a):
            require(report['VALID_completed'] < 3, 'VALID cap exceeded')
            before = time.perf_counter()
            v = a.validate()
            require(len(v['ids']) == 3943 and torch.bincount(v['labels'],minlength=3).tolist() == [72,1919,1952] and
                v['role'] == 'VALID' and v['TEST_accessed'] is False and torch.isfinite(v['outputs']).all().item(),
                'Complete original VALID support required')
            report['VALID_completed'] += 1; report['VALID_wall_seconds'].append(time.perf_counter()-before); return v

        def update(a, first):
            before = time.perf_counter()
            session, model = a.session, a.model
            shared = tuple(p for _, p in model.native.named_parameters()); private = tuple(model.private_parameters())
            old = [p.detach().clone() for p in shared+private]
            original_grad, original_step = torch.autograd.grad, session.optimizer.step
            calls, alternatives = [], []
            def reverse(loss, parameters, **kwargs):
                index = len(calls); expected = shared if index == 0 else private
                require(index < 2 and tuple(map(id,parameters)) == tuple(map(id,expected)), 'F/shared then W/private required')
                require(all(torch.equal(p,z) for p,z in zip(shared+private,old)), 'Old state changed before both gradients')
                def charged(pars, options):
                    report['reverse_attempts_including_checks'] += 1
                    require(report['reverse_attempts_including_checks'] <= 8, 'Reverse cap exceeded')
                    value = original_grad(loss, pars, **options)
                    report['reverse_completed_including_checks'] += 1; return value
                alternatives.append(charged(private if index == 0 else shared,
                    {'retain_graph':True,'create_graph':False,'allow_unused':True}))
                gradients = charged(parameters, kwargs); calls.append((float(loss.detach()),parameters,gradients)); return gradients
            def step(*args, **kwargs):
                require(len(calls) == 2 and all(torch.equal(p,z) for p,z in zip(shared+private,old)), 'One step after both collections required')
                for _, pars, grads in calls:
                    for p,g in zip(pars,grads): require((g is None and p.grad is None) or
                        (g is not None and p.grad is not None and torch.equal(g,p.grad)), 'Assigned block gradient differs')
                if first:
                    down, up = calls[1][2]
                    require(down is not None and down.count_nonzero().item() == 0 and up is not None and
                        up.count_nonzero().item() > 0, 'Zero-up first gradient pattern differs')
                return original_step(*args, **kwargs)
            torch.autograd.grad, session.optimizer.step = reverse, step
            try: value = session.train_update()
            finally: torch.autograd.grad, session.optimizer.step = original_grad, original_step
            require(len(calls) == 2 and calls[0][0] == value['native_unweighted_F'] and calls[1][0] == value['private_weighted_W'],
                'Actual block objective labels differ')
            differs = any(g is not None and h is not None and not torch.equal(g,h) for g,h in zip(calls[1][2],alternatives[0]))
            require(not first or differs == (args.variant != 'uniform'), 'Native fixture failed to identify uniform/weighted private credit')
            report['diagnostic_update_wall_seconds'].append(time.perf_counter()-before)
            return {**value,'correct_old_state_gradients':True,'private_F_vs_W_differ':differs}

        source = construct(); model, owner = source.model, source.session
        model.eval()
        with torch.no_grad():
            reference = model.native(owner.data)
            initial = torch.stack([model.run_member(m,owner.data).raw_logits for m in range(4)])
        require(torch.equal(initial,reference[None].expand_as(initial)), 'Initial native eval function differs')
        report['initial_native_eval_bit_exact'] = True
        initial_state = source.state(); report['assignment'] = initial_state['session']['method']['table']
        observer = OptimizerObserver(torch,source.optimizers)
        report['first_update'] = update(source,True); first_valid = valid(source)
        require(observer.state() == {'attempts':[1],'completed':[1]}, 'First Adam count differs')
        checkpoint = output/'ENGINEERING_after_one.pt'
        torch.save({'adapter':cpu_clone(torch,source.state()),'global_rng':cpu_clone(torch,global_rng(torch)),
            'optimizer_calls':observer.state(),'engineering_only_not_scientific_initialization':True},checkpoint)
        report['checkpoint'] = ext.pin(checkpoint); observer.close(); observer = None
        report['optimizer_calls'] = {'attempts':[1],'completed':[1]}
        fresh = construct(); saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        for field in ('method','table_record','weights'):
            bad = copy.deepcopy(saved['adapter'])
            if field == 'method': bad['session'][field]['seed'] = 202
            elif field == 'table_record': bad['session'][field]['metrics'][0] += 1
            else: bad['session'][field][0,0] += 1
            try: fresh.load_state(bad)
            except ValueError: pass
            else: raise AssertionError('Changed method/assignment state accepted')
        before = ext.load(ext.LEGACY).optimizer_contract(fresh.session.optimizer)
        fresh.load_state(saved['adapter']); replay = valid(fresh)
        after = ext.load(ext.LEGACY).optimizer_contract(fresh.session.optimizer)
        require(before == after and torch.equal(first_valid['outputs'],replay['outputs']) and
            torch.equal(first_valid['ids'],replay['ids']), 'Fresh restore optimizer contract/VALID differs')
        fresh.load_state(saved['adapter']); restore_rng(torch,saved['global_rng'])
        observer = OptimizerObserver(torch,fresh.optimizers,previous=saved['optimizer_calls'])
        report['second_update'] = update(fresh,False); valid(fresh)
        require(observer.state() == {'attempts':[2],'completed':[2]} and fresh.session.updates == 2, 'Total Adam/update count differs')
        report.update(status='passed',fresh_restore_VALID_bit_exact=True,optimizer_contract_unchanged=True,
            changed_state_rejected=True,source_metadata=fresh.metadata(),not_scientific_initialization=True)
        require(args.device != 'cpu' or not torch.cuda.is_initialized(), 'CPU execution initialized CUDA')
        if args.device == 'cuda:0':
            torch.cuda.synchronize(); report['GPU_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['GPU_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    except BaseException as error:
        report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=12)})
    finally:
        if 'torch' in locals() and args.device == 'cuda:0' and torch.cuda.is_initialized():
            report['GPU_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['GPU_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        write()
        if observer is not None: observer.close()
        for h in hooks: h.remove()
        for a in adapters:
            try: a.close()
            except Exception as error: report.setdefault('close_errors',[]).append(str(error)); report['status'] = 'failed'
        write(); gc.collect(); signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({k:report[k] for k in ('status','variant','device','optimizer_calls','VALID_completed')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__': raise SystemExit(main())
