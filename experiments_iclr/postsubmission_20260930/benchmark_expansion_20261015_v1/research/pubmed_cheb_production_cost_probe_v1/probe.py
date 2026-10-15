"""One cold model, two production TRAIN updates and two whole VALID calls.

Source preparation only until root admits an isolated allocation invocation.
No selector, TEST, diagnostic reverse, checkpoint clone or scientific fit.
"""
import argparse
import copy
import hashlib
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

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
PHASE = E.parent
COMPOSITION = E/'research/pubmed_cheb_residual_composition_v1'
COMPOSITION_SHA256 = '1c5d3669cec6681bbfbda045a9497805805c287ab3e03b4aca5aa65c0e3cf0dc'
VARIANTS = ('M1', 'G0', 'uniform', 'structured')


def require(ok, message):
    if not ok: raise ValueError(message)


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(path):
    name = '_pubmed_cost_' + pin(path)['sha256']
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return sys.modules[name]


def descriptor(root, variant):
    """Same original native builder and residual producer as the transfer driver."""
    require(variant in VARIANTS, 'One declared cost arm required')
    root = Path(root)
    spec = json.loads((HERE/'NATIVE_BUILDER.json').read_text())
    require(str(root) == spec['recipe']['_repository_root'], 'Exact allocation repository binding required')
    if variant in ('uniform', 'structured'):
        return load(COMPOSITION/'extension.py').descriptor(root, 101, variant, COMPOSITION_SHA256)
    sys.path.insert(0, str(E/'runtime'))
    jobs, _ = load(E/'runtime/run_pilot.py').build_jobs(spec)
    selected = [j for j in jobs if j['seed'] == 101 and
        (j['procedure'] == 'native_single' and j['body'] == 0 if variant == 'M1' else j['procedure'] == 'gnnm_boundary_native')]
    require(len(selected) == 1, 'Exactly one original seed101 M1/G0 descriptor required')
    return selected[0]


def source_bindings(root, expected):
    require(HERE.is_relative_to(root), 'Source must be inside the allocation repository')
    manifest = pin(HERE/'MANIFEST.json'); require(manifest['sha256'] == expected, 'Exact reviewed probe manifest required')
    receipts = [manifest]
    for base, rows, key in ((HERE, json.loads((HERE/'MANIFEST.json').read_text())['files'], 'path'),
            (PHASE, json.loads((HERE/'SOURCES.json').read_text())['files'], 'path_relative_to_phase')):
        for row in rows:
            path = (base/row[key]).resolve(); require(path.is_relative_to(base), 'Source path escaped')
            actual = pin(path); require(all(actual[k] == row[k] for k in ('bytes','sha256')), 'Source pin changed: '+str(path))
            receipts.append(actual)
    receipts += load(COMPOSITION/'extension.py').source_bindings(root, COMPOSITION_SHA256)
    return receipts


def main():
    started = time.perf_counter()
    require(socket.gethostname() == 'anogena-2-0' and subprocess.check_output(
        ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'], text=True, timeout=10).splitlines() ==
        ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Dedicated allocation required before project reads')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True); parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--device', choices=('cpu','cuda:0'), required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--root-resource-admitted', action='store_true')
    args = parser.parse_args(); root = Path(args.project_root).resolve(); output = Path(args.output).resolve()
    require(args.root_resource_admitted, 'Root must first admit this isolated resource use')
    require(HERE.is_relative_to(root) and output.is_relative_to(root) and not output.exists(), 'Fresh contained output required')
    require(Path(sys.prefix).resolve() == (root/'.venv').resolve(), 'Original allocation virtualenv required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' if args.device == 'cpu' else
        os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','GPU-44039938-fd82-41d2-fefd-de71514e2fac'), 'Explicit device visibility required')
    def stop(signum, frame): raise TimeoutError('Bounded cost probe interrupted: signal'+str(signum))
    signal.signal(signal.SIGALRM, stop); signal.signal(signal.SIGTERM, stop); signal.setitimer(signal.ITIMER_REAL, 900)
    receipts = source_bindings(root, args.manifest_sha256)
    binding = json.loads((COMPOSITION/'BINDING.json').read_text())
    for row in binding['data_pins']:
        path = Path(row['path']).resolve(); require(path.is_relative_to(root), 'Data pin escaped project')
        actual = pin(path); require(all(actual[k] == row[k] for k in ('bytes','sha256')), 'Exact existing data changed')
    planned = descriptor(root, args.variant)
    sys.path.insert(0, str(E/'runtime'))
    from run_fit import validate_job, OptimizerObserver
    from adapters import create_adapter
    from common import atomic_json, digest, seed_all
    policy = validate_job(planned); job = copy.deepcopy(planned); job['device'] = args.device
    output.mkdir(parents=True); atomic_json(output/'JOB.json', planned)
    reverse_cap = 4 if args.variant in ('uniform','structured') else 2
    counts = dict(native_forward_attempts=0, native_forwards_completed=0,
        backward_attempts=0, backward_completed=0, grad_attempts=0, grad_completed=0)
    report = {'schema':'pubmed-cheb-production-cost-probe-v1','status':'running','variant':args.variant,
        'device':args.device,'seed':101,'split':0,'source_receipts':receipts,'planned_job_sha256':digest(planned),
        'execution_job_sha256':digest(job),'constructions_attempted':0,'constructions_completed':0,
        'stages':[],'counts':counts,'optimizer_calls':{'attempts':[0],'completed':[0]},
        'bounds':{'constructors':1,'TRAIN_updates':2,'VALID_calls':2,'reverse_calls':reverse_cap,
            'native_forward_attempts_including_recomputation':64,'invocation_alarm_seconds':900},
        'CPU_threads':2,'root_resource_admitted':True,'selector_calls':0,'TEST_accessed':False,
        'diagnostic_reverse_calls':0,'scientific_fit':False,'scientific_initialization':False,
        'steady_state_established':False,'checkpoint_written':False}
    adapter, observer, hooks, original_grad, original_backward = None, None, [], None, None
    tensor_library = None

    def memory():
        result = {'own_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        if tensor_library is not None and args.device == 'cuda:0' and tensor_library.cuda.is_initialized():
            t = tensor_library
            result.update(cuda_peak_allocated_bytes=t.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=t.cuda.max_memory_reserved())
        return result

    def write():
        if observer is not None: report['optimizer_calls'] = observer.state()
        report.update(memory()); report['wall_seconds'] = time.perf_counter()-started
        atomic_json(output/'RESULT.json', report)

    try:
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key] = '2'
        import torch
        tensor_library = torch
        torch.set_num_threads(2); report['torch'] = str(torch.__version__)
        require(args.device != 'cpu' or not torch.cuda.is_initialized(), 'CPU probe must not initialize CUDA')
        def sync():
            if args.device == 'cuda:0': torch.cuda.synchronize()
        if args.device == 'cuda:0': sync(); torch.cuda.reset_peak_memory_stats()
        def snapshot():
            return {**counts, 'Adam_attempts':sum(observer.state()['attempts']) if observer else 0,
                'Adam_completed':sum(observer.state()['completed']) if observer else 0}
        def timed(kind, index, function):
            sync(); begin = time.perf_counter(); before = snapshot(); passed = False
            try:
                value = function(); sync(); passed = True; return value
            finally:
                after = snapshot()
                report['stages'].append({'kind':kind,'index':index,'completed':passed,
                    'wall_seconds':time.perf_counter()-begin,'CUDA_synchronized_on_success':args.device == 'cuda:0',
                    'work':{k:after[k]-before[k] for k in before}})
                write()
        seed_all(torch, 101); report['constructions_attempted'] = 1
        adapter = timed('constructor', 1, lambda:create_adapter(torch, copy.deepcopy(job), policy, root))
        report['constructions_completed'] = 1
        session = adapter.session
        require(session.data.num_nodes == 19717 and len(session.train_ids) == 11829 and
            torch.bincount(session.targets,minlength=3).tolist() == [3943]*3, 'Exact full native TRAIN required')
        for value, key in ((session.train_ids,'ids_sha256'),(session.targets,'labels_sha256')):
            require(hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest() ==
                binding['provider']['role_digests']['TRAIN'][key], 'TRAIN identity changed')
        expected_parameters = {'M1':592435,'G0':595995,'uniform':600627,'structured':600627}[args.variant]
        require(sum(p.numel() for p in adapter.model.parameters()) == expected_parameters, 'Model parameter scope differs')
        native = adapter.model if args.variant == 'M1' else adapter.model.native
        def pre(module, arguments):
            counts['native_forward_attempts'] += 1
            require(counts['native_forward_attempts'] <= 64, 'Native forward cap exceeded')
        def post(module, arguments, value): counts['native_forwards_completed'] += 1
        hooks = [native.register_forward_pre_hook(pre), native.register_forward_hook(post)]
        original_grad, original_backward = torch.autograd.grad, torch.autograd.backward
        def reverse(kind, function):
            def call(*values, **kwargs):
                counts[kind+'_attempts'] += 1
                require(counts['backward_attempts']+counts['grad_attempts'] <= reverse_cap, 'Production reverse cap exceeded')
                result = function(*values, **kwargs); counts[kind+'_completed'] += 1; return result
            return call
        torch.autograd.grad = reverse('grad', original_grad)
        torch.autograd.backward = reverse('backward', original_backward)
        observer = OptimizerObserver(torch, adapter.optimizers)
        for index in (1,2):
            update = timed('TRAIN_update', index, adapter.train_epoch)
            require(update['complete_native_TRAIN'] is True and update['TRAIN_records_traversed'] == 11829 and
                update['optimizer_maps'] == 1 and adapter.completed_epochs == index and session.updates == index and
                observer.state() == {'attempts':[index],'completed':[index]}, 'Exactly one complete native update required')
            report['stages'][-1]['native_update'] = update
            valid = timed('VALID', index, adapter.validate)
            require(valid['role'] == 'VALID' and valid['TEST_accessed'] is False and valid['complete_source_VALID'] is True and
                tuple(valid['outputs'].shape) == (3943,3) and torch.bincount(valid['labels'],minlength=3).tolist() == [72,1919,1952],
                'Whole original VALID required')
            for key, authority in (('ids','ids_sha256'),('labels','labels_sha256')):
                require(hashlib.sha256(valid[key].contiguous().numpy().tobytes()).hexdigest() ==
                    binding['provider']['role_digests']['VALID'][authority], 'VALID identity changed')
            report['stages'][-1]['VALID_records'] = len(valid['ids'])
            report['stages'][-1]['export_tensor_payload_bytes'] = sum(valid[k].numel()*valid[k].element_size()
                for k in ('outputs','ids','labels'))
            del valid
            write()
        require(counts['backward_completed']+counts['grad_completed'] == reverse_cap, 'Production reverse count differs')
        require(args.device != 'cpu' or not torch.cuda.is_initialized(), 'CPU execution initialized CUDA')
        def tensor_bytes(value):
            if torch.is_tensor(value): return value.numel()*value.element_size()
            if isinstance(value, dict): return sum(map(tensor_bytes,value.values()))
            if isinstance(value, (list,tuple)): return sum(map(tensor_bytes,value))
            return 0
        report['existing_state_tensor_bytes'] = {
            'model_parameters':sum(p.numel()*p.element_size() for p in adapter.model.parameters()),
            'model_buffers':sum(b.numel()*b.element_size() for b in adapter.model.buffers()),
            'Adam_state':tensor_bytes(session.optimizer.state),
            'scope':'Live tensors only; not serialized checkpoint size. No state export or clone performed.'}
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),
            'traceback':traceback.format_exc(limit=12)})
    finally:
        if original_grad is not None: torch.autograd.grad = original_grad
        if original_backward is not None: torch.autograd.backward = original_backward
        if observer is not None: report['optimizer_calls'] = observer.state(); observer.close(); observer = None
        for hook in hooks: hook.remove()
        if adapter is not None:
            try: adapter.close()
            except Exception as error: report['close_error'] = str(error); report['status'] = 'failed'
        write(); signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({k:report[k] for k in ('status','variant','device','optimizer_calls','counts')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__': raise SystemExit(main())
