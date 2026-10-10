"""Disabled representative whole-TRAIN integration entry; no quality scoring.

Run only through root's existing finite run_fit owner after a separate release.
This is three local-plus-global source checks, not fits, a search or a launcher.
All numerical imports and data loading occur after the new release guard.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import weakref

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
KINDS = ("baseline", "exchange", "separable")
CAP = 64 * 1024**3
OWNED_CAP = 68 * 1024**3
MIN_FREE = 72 * 1024**3
TRAIN_MANIFEST = {
    "path": "wikics_official_acquisition_root_20261007_v1/TRAIN_only_projection_20261007_v1/TRAIN_ONLY_MANIFEST.json",
    "sha256": "978b382d1f95af23512606ef6ac9530f31ee510efcf36efb35d059bd0ad7615e",
}
TRAIN_PAYLOAD_SHA = "8e24e740953dc59ea435e4650700264e47eebd7a5c227a2425e1fd7b2883accd"
RTOL, ATOL = 1e-5, 1e-6


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bound(row):
    path = (PHASE / row['path']).resolve(strict=True)
    require(path.is_relative_to(PHASE) and path.is_file() and sha(path) == row['sha256'],
            'Exact phase source/data binding required: ' + row['path'])
    if 'bytes' in row:
        require(path.stat().st_size == row['bytes'], 'Bound byte count changed')
    return path


def sealed(row):
    manifest = bound(row)
    for item in json.loads(manifest.read_text())['files']:
        path = (manifest.parent/item['path']).resolve(strict=True)
        require(path.is_relative_to(manifest.parent) and sha(path) == item['sha256']
                and path.stat().st_size == item['bytes'], 'Source manifest payload changed')
    return manifest.parent


def module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def streams_copy(streams):
    return [{key:value.clone() for key,value in row.items()} for row in streams]


def streams_equal(torch, left, right):
    return len(left) == len(right) and all(a.keys() == b.keys()
        and all(torch.equal(a[k], b[k]) for k in a) for a,b in zip(left,right))


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps({'shape': list(value.shape), 'dtype': str(value.dtype)}, sort_keys=True)
    return hashlib.sha256(header.encode() + value.numpy().tobytes()).hexdigest()


def state_signature(session):
    model = {k:tensor_sha(v) for k,v in session.model.state_dict().items()}
    state = session.optimizers[0].state_dict()
    moments = {str(k):{field:tensor_sha(value) if isinstance(value, session.torch.Tensor) else value
                       for field,value in row.items()} for k,row in state['state'].items()}
    return {'model': model, 'Adam_state': moments, 'Adam_groups': state['param_groups'], 'steps': session.steps}


def registration(session):
    named = dict(session.model.named_parameters())
    owned = [p for group in session.optimizers[0].param_groups for p in group['params']]
    require(len(session.optimizers) == 1 and len(owned) == len(named)
            and len(set(map(id, owned))) == len(owned)
            and set(map(id, owned)) == {id(p) for p in named.values()}
            and all(p.requires_grad for p in named.values()),
            'Fresh Adam must uniquely own every native/scorer/block Parameter')
    prefix = 'models.0.live_route_block.'
    added = {name:list(p.shape) for name,p in named.items() if name.startswith(prefix)}
    expected = {} if session.live_kind == 'baseline' else (
        {prefix+k:shape for k,shape in {'Q':[512,16], 'K':[512,16], 'V':[512,16], 'U':[16,512]}.items()}
        if session.live_kind == 'exchange' else {prefix+'B':[512,32], prefix+'C':[32,512]})
    require(added == expected, 'Only exact fixed additional block weights')
    scorers = {name:list(p.shape) for name,p in named.items()
               if '.parametrizations.att_' in name}
    scorer_expected = {'models.0.body.local_convs.%d.parametrizations.%s.original' % (layer, side): [4,1,1,512]
                       for layer in range(7) for side in ('att_src','att_dst')}
    require(scorers == scorer_expected, 'All fourteen native private scorer banks owned exactly once')
    scalars = sum(p.numel() for p in named.values())
    require(scalars == 7680788 + (0 if session.live_kind == 'baseline' else 32768),
            'Exact native/private scorer body plus fixed added block budget')
    block_hashes = {name:tensor_sha(named[name]) for name in added}
    require(all(bool(session.torch.isfinite(named[name]).all()) for name in added), 'Finite fresh additional weights')
    if session.live_kind != 'baseline':
        ending = 'U' if session.live_kind == 'exchange' else 'C'
        require(bool((named[prefix+ending] == 0).all()), 'Exact zero initial residual output map')
    require(not session.optimizers[0].state, 'Registration before fresh Adam history')
    return {'unique_parameters': len(named), 'scalars': scalars,
            'added_block_shapes': added, 'private_scorer_shapes': scorers,
            'initial_block_parameter_sha256': block_hashes,
            'fresh_Adam_empty': True, 'all_parameters_risk': 'plain own F'}


def compare(torch, expected, actual):
    require(len(expected) == len(actual) == 2, 'Complete logits/representation pair required')
    errors = []
    for a,b in zip(expected, actual):
        b = b.detach().cpu()
        require(tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype
                and bool(torch.isfinite(b).all()), 'Factual TRAIN output shape/type/finiteness')
        error = float((a-b).abs().max())
        require(torch.allclose(a,b,rtol=RTOL,atol=ATOL), 'Factual source replay mismatch: ' + str(error))
        errors.append(error)
    return errors


def factual(session, batch, train_mode):
    session.model.train(train_mode)
    with session.torch.no_grad():
        values = session.forward(batch)
    require(len(values) == 2 and tuple(values[0].shape) == (4,580,10)
            and tuple(values[1].shape) == (4,580,512)
            and all(v.dtype == session.torch.float32 and bool(session.torch.isfinite(v).all()) for v in values),
            'Factual complete TRAIN-row float32 logits/representations')
    return tuple(v.detach().cpu().clone() for v in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    require(sha(args.release) == args.release_sha256 and args.release.resolve().is_relative_to(PHASE),
            'Exact root qualification release')
    release = json.loads(args.release.read_text())
    require(release['schema'] == 'live-route-native-init-qualification-release-v3'
            and release['enabled'] is release['root_execution_authorized'] is release['source_review_approved'] is True,
            'New source/release is disabled or unreviewed')
    require(release['TEST_access'] is release['VALID_values_access'] is release['quality_scoring'] is False
            and release['fits_authorized'] is release['automatic_retry'] is False,
            'Whole-TRAIN engineering only; no development scoring or fits')
    require(release['kinds'] == list(KINDS) and release['seed'] == 6101
            and release['owned_GPU_allocator_cap_bytes'] == CAP
            and release['owned_tree_GPU_memory_cap_bytes'] == OWNED_CAP
            and release['minimum_fresh_GPU_free_bytes'] == MIN_FREE,
            'One fixed three-case64GiB allocator/68GiB owned-tree setup check')
    require(socket.gethostname() == 'peptide' and Path.cwd().resolve() == Path(release['runtime_repository'])
            and str(Path(sys.executable).absolute()) == release['runtime_python']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == release['physical_gpu_uuid'],
            'Declared77 repository/interpreter/physical visible singleton')
    require(os.environ.get('PYTHONPATH','') == '' and not os.environ.get('PYTHONHOME'), 'No runtime overlay')
    require(release['qualifier_program_sha256'] == sha(__file__)
            and release['qualification_manifest_sha256'] == sha(HERE/'MANIFEST.json'), 'Exact new qualifier source')
    sealed({'path': str((HERE/'MANIFEST.json').relative_to(PHASE)), 'sha256': release['qualification_manifest_sha256']})
    bindings = json.loads((HERE/'QUALIFICATION_BINDINGS.json').read_text())
    for key,row in bindings['release_bindings'].items():
        require(release[key] == row, 'Root release changed a fixed source/runtime binding: ' + key)
    for row in bindings['source_and_runtime_files']:
        bound(row)
    source_root = sealed(release['scientific_source_manifest'])
    context = json.loads(bound(release['runtime_context']).read_text())
    require(context['hostname'] == 'peptide' and context['repository'] == release['runtime_repository']
            and release['physical_gpu_uuid'] in context['physical_gpu_inventory']
            and release['runtime_python'] == context['python_executable']['path'], 'Exact existing77 provider context')
    interpreter = context['python_executable']
    require(str(Path(sys.executable).resolve()) == interpreter['resolved_path']
            and Path(sys.executable).resolve().stat().st_size == interpreter['bytes']
            and sha(Path(sys.executable).resolve()) == interpreter['sha256']
            and sys.prefix == context['runtime_prefix'], 'Exact existing77 interpreter bytes and prefix')
    bound(context['runtime_provider']); bound(context['runtime_environment_verification'])
    inventory = []
    for row in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total',
            '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines():
        uuid,name,memory = [item.strip() for item in row.split(',')]
        inventory.append({'uuid':uuid,'name':name,'memory_total_MiB':int(memory)})
    require([row['uuid'] for row in inventory] == context['physical_gpu_inventory']
            and all(row['name'] == context['physical_gpu_name']
                    and row['memory_total_MiB'] == context['physical_gpu_memory_total_MiB'] for row in inventory),
            'Exact authorized77 physical inventory before numerical import')
    for key in ('existing_run_fit_helper', 'existing_ownership_helper'):
        bound(release[key])
    external = json.loads(bound(release['external_owner_evidence']).read_text())
    require(external['root_execution_authorized'] is True
            and external['qualifier_program_sha256'] == sha(__file__)
            and external['existing_run_fit_helper'] == release['existing_run_fit_helper']
            and external['existing_ownership_helper'] == release['existing_ownership_helper']
            and external['owned_tree_GPU_memory_cap_bytes'] == OWNED_CAP
            and external['minimum_fresh_GPU_free_bytes'] == MIN_FREE
            and external['active_seconds'] == release['active_seconds']
            and external['external_hard_seconds'] == release['external_hard_seconds']
            and external['owned_tree_RSS_cap_bytes'] == release['owned_tree_RSS_cap_bytes']
            and external['own_fit_output_cap_bytes'] == release['own_fit_output_cap_bytes']
            and external['physical_gpu_uuid'] == release['physical_gpu_uuid']
            and external['output'] == release['output'], 'Existing finite owner must bound this exact child')
    require(type(release['active_seconds']) is int and 0 < release['active_seconds'] <= 1200,
            'Root-bound finite setup duration, no automatic retries')
    require(type(release['external_hard_seconds']) is int
            and release['active_seconds'] < release['external_hard_seconds'] <= 1500
            and type(release['owned_tree_RSS_cap_bytes']) is int and release['owned_tree_RSS_cap_bytes'] > 0
            and type(release['own_fit_output_cap_bytes']) is int and release['own_fit_output_cap_bytes'] > 0,
            'Finite external duration/RSS/output bounds required')
    require(release['TRAIN_only_manifest'] == TRAIN_MANIFEST, 'Use the existing exact TRAIN-only projection')
    authority = json.loads(bound(TRAIN_MANIFEST).read_text())
    require(authority['available']['sha256'] == TRAIN_PAYLOAD_SHA
            and authority['qualified_reader_deserializes_VALID_TEST_values'] is False,
            'No TRAIN/DEVELOPMENT fallback or TEST container')
    bound(authority['available'])
    output = (PHASE/release['output']).resolve()
    require(output.is_relative_to(PHASE) and output.parent.is_dir() and not output.exists(), 'Fresh root-owned output')
    output.mkdir()
    began = time.monotonic()
    result = {'schema':'live-route-native-init-qualification-v3', 'complete':False, 'quality_scoring':False,
              'VALID_values_access':False, 'TEST_access':False, 'scientific_fits':0,
              'source_manifest':release['scientific_source_manifest'], 'rows':[], 'costs':[],
              'counts':{'constructor_attempts':0,'constructor_completions':0,'TRAIN_update_attempts':0,
                        'TRAIN_update_completions':0,'Adam_attempts':0,'Adam_completions':0,
                        'backward_attempts':0,'backward_completions':0,
                        'save_attempts':0,'save_completions':0,'restore_attempts':0,'restore_completions':0}}
    session = None
    def flush():
        result['elapsed_seconds'] = time.monotonic()-began
        result['current_actual_native_work'] = copy.deepcopy(session.live_work) if session is not None else None
        result['current_actual_live_view_order'] = list(session.live_last_update_events) if session is not None else None
        atomic(output/'PROGRESS.json', result)
    def timed(label, operation):
        require(time.monotonic()-began < release['active_seconds'], 'Finite qualification duration exhausted')
        row = {'operation':label, 'completed':False}; result['costs'].append(row); flush()
        started = time.monotonic()
        try:
            value = operation(); torch.cuda.synchronize(0); row['completed'] = True
            return value
        finally:
            row.update(seconds=time.monotonic()-started, peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(0),
                       peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(0),
                       cumulative_process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            flush()
    def construct(facade):
        result['counts']['constructor_attempts'] += 1
        value = facade.Session('wikics','be_unit',6101,'cuda:0',polynormer=bound(release['native']))
        result['counts']['constructor_completions'] += 1
        return value
    try:
        import numpy as np
        import torch
        import torch_geometric
        import torch_scatter
        import torch_sparse
        modules = {'numpy':np,'torch':torch,'torch_geometric':torch_geometric,
                   'torch_scatter':torch_scatter,'torch_sparse':torch_sparse}
        provider = module('_live_existing77_provider_verifier', bound(release['provider_verifier']))
        provider.verify_runtime_provider(context, modules, sha)
        versions = {'torch':str(torch.__version__), 'numpy':np.__version__, 'PyG':torch_geometric.__version__,
                    'torch_scatter':torch_scatter.__version__, 'torch_sparse':torch_sparse.__version__, 'CUDA':torch.version.cuda}
        require(versions == context['runtime_versions'], 'Exact existing77 operator versions')
        require(torch.cuda.device_count() == 1, 'One root-owned visible GPU')
        total = torch.cuda.get_device_properties(0).total_memory
        require(total >= CAP, 'Fresh64GiB joint-autograd cap; old32GiB not inherited')
        torch.cuda.set_per_process_memory_fraction(CAP/total,0)
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
        public = module('portable', bound(release['public']))
        source = module('_live_native_init_source_v3', source_root/'__init__.py', True)
        loader = module('_live_existing_TRAIN_loader', bound(release['TRAIN_loader']))
        data, loaded_authority = timed('existing_TRAIN_only_loader', lambda: loader.load_data(torch, {'data_manifest':TRAIN_MANIFEST}, True))
        require(loaded_authority == authority, 'Loader must use the exact TRAIN-only authority')
        batch = {'x':data['x'],'edge_index':data['edge_index'],'ids':data['train_ids']}
        labels = data['train_y']
        require(tuple(labels.shape) == (580,), 'All580 TRAIN targets; no masked subset')
        result.update(full_nodes=11701, ordered_edges=442907, TRAIN_targets=580, data_authority=TRAIN_MANIFEST,
                      data_payload_sha256=TRAIN_PAYLOAD_SHA, tolerance={'rtol':RTOL,'atol':ATOL},
                      GPU_allocator_cap_bytes=CAP, owned_tree_GPU_memory_cap_bytes=OWNED_CAP,
                      minimum_fresh_GPU_free_bytes=MIN_FREE, runtime_context=release['runtime_context'])
        initial_eval = initial_train = initial_native = initial_receipt = initial_streams = initial_endpoint = None
        for kind in KINDS:
            case = {'kind':kind, 'complete':False}
            result['rows'].append(case)
            torch.cuda.reset_peak_memory_stats(0)
            facade = source.adapted_public(public,kind)
            session = timed(kind+'_fresh_constructor',lambda:construct(facade))
            reg = registration(session)
            require(session.steps == 0 and session.independent_native_start['before_original_Adam'] is True,
                    'Native scorer initializer and block precede the fresh original Adam')
            streams = streams_copy(session.streams)
            for member, stream in enumerate(streams):
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(6101+1009*member+300001); torch.cuda.manual_seed(6101+1009*member+300001)
                    require(torch.equal(stream['cpu'],torch.get_rng_state())
                            and torch.equal(stream['cuda'],torch.cuda.get_rng_state(0)), 'Original per-member dropout bytes')
            native = {k:tensor_sha(v) for k,v in session.model.models[0].body.state_dict().items()}
            receipt = {k:v for k,v in session.independent_native_start.items() if k != 'elapsed_seconds'}
            ev = timed(kind+'_initial_factual_local',lambda:factual(session,batch,False))
            require(streams_equal(torch, streams, session.streams), 'Eval must not advance native dropout streams')
            tr = timed(kind+'_initial_stochastic_parity',lambda:factual(session,batch,True))
            endpoint = streams_copy(session.streams)
            session.streams = streams_copy(streams)  # Parity is charged but does not consume update's two views.
            parity = None
            if kind == 'baseline':
                initial_eval,initial_train,initial_native,initial_receipt = ev,tr,native,receipt
                initial_streams,initial_endpoint = streams,endpoint
            else:
                require(native == initial_native and receipt == initial_receipt
                        and streams_equal(torch,streams,initial_streams)
                        and streams_equal(torch,endpoint,initial_endpoint), 'Exact same-source native start/RNG endpoint')
                parity = {'eval':compare(torch,initial_eval,ev), 'stochastic':compare(torch,initial_train,tr)}
            def perform_update(global_mode, expected_step):
                require(session.model.models[0].body._global is global_mode, 'Fixed native training mode')
                work_before = copy.deepcopy(session.live_work)
                torch.cuda.reset_peak_memory_stats(0)
                optimizer = session.optimizers[0]; original_step = optimizer.step; gradients = {}
                expected_order = ['zero_grad_completed','A_forward_attempt','A_forward_completed',
                    'A_backward_attempt','A_backward_completed','A_graph_references_released',
                    'B_forward_attempt','B_forward_completed','B_backward_attempt','B_backward_completed',
                    'B_graph_references_released','Adam_attempt_after_both_backwards']
                def observe_live_event(event):
                    if event in ('A_backward_attempt','B_backward_attempt'):
                        result['counts']['backward_attempts'] += 1
                    elif event in ('A_backward_completed','B_backward_completed'):
                        result['counts']['backward_completions'] += 1
                    flush()
                session.live_update_observer = observe_live_event
                def observed_step(*args,**kwargs):
                    result['counts']['Adam_attempts'] += 1
                    require(session.live_last_update_events == expected_order
                            and session.live_work['backward_attempts']-work_before['backward_attempts'] == 2
                            and session.live_work['backward_completions']-work_before['backward_completions'] == 2,
                            'One original Adam only after sequential A/B half-loss backwards and releases')
                    active = {name:p for name,p in session.model.named_parameters() if p.grad is not None}
                    require(active and all(bool(torch.isfinite(p.grad).all()) for p in active.values()), 'Actual finite full-joint own gradients')
                    global_names = [name for name,_ in session.model.named_parameters()
                                    if name.startswith('models.0.body.global_attn.') or name.startswith('models.0.body.pred_global.')]
                    local_head_names = [name for name,_ in session.model.named_parameters()
                                        if name.startswith('models.0.body.pred_local.')]
                    require(all(name in active for name in global_names) if global_mode
                            else not any(name in active for name in global_names), 'Native global gradients follow actual mode')
                    require(not any(name in active for name in local_head_names) if global_mode
                            else all(name in active for name in local_head_names), 'Only the actual native head receives gradients')
                    block = {name:p for name,p in active.items() if name.startswith('models.0.live_route_block.')}
                    require(set(block) == set(reg['added_block_shapes']), 'Every added weight participates in joint autograd')
                    if kind != 'baseline':
                        ending = 'U' if kind == 'exchange' else 'C'
                        require(bool(block['models.0.live_route_block.'+ending].grad.abs().max() > 0), 'Residual output map has a genuine task cotangent')
                    gradients.update(finite=True, active_parameter_tensors=len(active), block_gradient_names=list(block),
                                     block_max_abs_gradients={name:float(p.grad.abs().max()) for name,p in block.items()},
                                     native_global_parameter_gradients_present=global_mode,
                                     one_optimizer_after_two_live_views=True, sequential_view_graphs=True,
                                     same_scalar_F_in_real_arithmetic=True, bitwise_two_view_trajectory_equivalence=False,
                                     no_old_disjoint_replay=True)
                    value = original_step(*args,**kwargs)
                    torch.cuda.synchronize(0)
                    result['counts']['Adam_completions'] += 1
                    return value
                optimizer.step = observed_step
                result['counts']['TRAIN_update_attempts'] += 1
                try:
                    returned = timed(kind+('_global' if global_mode else '_local')+'_two_view_full_TRAIN_update',
                                     lambda:session.train_step(batch,labels))
                    result['counts']['TRAIN_update_completions'] += 1
                finally:
                    optimizer.step = original_step
                    session.live_update_observer = None
                require(session.steps == expected_step and all(bool(torch.isfinite(v)) for v in returned.values()),
                        'One genuine original two-view F update in this mode')
                scalars = {name:float(value.detach().cpu()) for name,value in returned.items()}
                delta = {k:[a-b for a,b in zip(v,work_before[k])] if isinstance(v,list) else v-work_before[k]
                         for k,v in session.live_work.items()}
                require(delta == {'joint_forward_attempts':2,'joint_forward_completions':2,'stem_completions':8,
                    'global_completions':8 if global_mode else 0,'local_head_completions':0 if global_mode else 8,
                    'global_head_completions':8 if global_mode else 0,
                    'block_completions':0 if kind == 'baseline' else 2,'local_conv_completions':[8]*7,
                    'backward_attempts':2,'backward_completions':2},
                    'Actual two complete native dropout views in the fixed mode')
                require(session.live_last_update_events == expected_order+['Adam_completed'], 'Actual completed live-view update order')
                return {'mode':'global' if global_mode else 'local','steps_after':session.steps,
                        'actual_native_work':delta,'gradients':gradients,'TRAIN_objective_scalars':scalars,
                        'actual_live_view_order':list(session.live_last_update_events),
                        'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(0),
                        'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(0),
                        'includes_existing_native_Adam_and_allocations':True}
            case['local_TRAIN_update'] = perform_update(False,1)
            local_signature = state_signature(session)
            local = timed(kind+'_updated_local_factual',lambda:factual(session,batch,False))
            local_streams = streams_copy(session.streams)
            local_path,global_path = output/(kind+'_local_state.pt'),output/(kind+'_global_state.pt')
            def save(path, epoch):
                result['counts']['save_attempts'] += 1
                session.save_training_state(path,epoch=epoch)
                result['counts']['save_completions'] += 1
            timed(kind+'_source_bound_local_save',lambda:save(local_path,1))
            continuation = timed(kind+'_TRAIN_RNG_continuation_reference',lambda:factual(session,batch,True))
            continuation_streams = streams_copy(session.streams)
            session.model.set_global(True)
            case['global_TRAIN_update'] = perform_update(True,2)
            global_signature = state_signature(session)
            global_output = timed(kind+'_native_global_forward',lambda:factual(session,batch,False))
            global_streams = streams_copy(session.streams)
            timed(kind+'_source_bound_global_save',lambda:save(global_path,2))
            global_continuation = timed(kind+'_global_TRAIN_RNG_reference',lambda:factual(session,batch,True))
            global_continuation_streams = streams_copy(session.streams)
            work_original = copy.deepcopy(session.live_work)
            for hook in session.live_work_hooks: hook.remove()
            # The update helper has dropped its optimizer/bound-step/GPU scalar
            # locals. Removed hooks no longer close over the old Session.
            original_refs = [weakref.ref(session), weakref.ref(session.model), weakref.ref(session.optimizers[0])]
            session = None; gc.collect(); torch.cuda.empty_cache()
            require(all(reference() is None for reference in original_refs), 'Original Session/model/Adam released before fresh constructor')
            torch.cuda.synchronize(0)
            allocation_after_original_release = torch.cuda.memory_allocated(0)
            session = timed(kind+'_fresh_restore_constructor',lambda:construct(facade))
            registration(session)
            def restore(path, expected_epoch, expected_signature):
                result['counts']['restore_attempts'] += 1
                epoch = session.restore_training_state(path)
                result['counts']['restore_completions'] += 1
                require(epoch == expected_epoch and state_signature(session) == expected_signature, 'Exact trained model/Adam/steps restored')
            timed(kind+'_local_restore',lambda:restore(local_path,1,local_signature))
            require(session.model.models[0].body._global is False and streams_equal(torch,session.streams,local_streams), 'Local mode and member RNG restored')
            local_error = compare(torch,local,timed(kind+'_local_factual_replay',lambda:factual(session,batch,False)))
            continuation_error = compare(torch,continuation,timed(kind+'_TRAIN_RNG_replay',lambda:factual(session,batch,True)))
            require(streams_equal(torch,session.streams,continuation_streams), 'Factual next TRAIN dropout endpoints match')
            timed(kind+'_global_restore',lambda:restore(global_path,2,global_signature))
            require(session.model.models[0].body._global is True and streams_equal(torch,session.streams,global_streams), 'Global flag and RNG state restored')
            global_error = compare(torch,global_output,timed(kind+'_global_factual_replay',lambda:factual(session,batch,False)))
            global_continuation_error = compare(torch,global_continuation,
                timed(kind+'_global_TRAIN_RNG_replay',lambda:factual(session,batch,True)))
            require(streams_equal(torch,session.streams,global_continuation_streams), 'Global next TRAIN dropout endpoints match')
            require(torch.cuda.max_memory_reserved(0) <= CAP, 'Actual joint source exceeds the new64GiB cap')
            case.update({'kind':kind,'complete':True,'registration':reg,'initializer':receipt,
                'initial_parity':parity,
                'local_restore_error':local_error,'TRAIN_RNG_restore_error':continuation_error,'global_restore_error':global_error,
                'original_actual_work':work_original,'fresh_restore_actual_work':copy.deepcopy(session.live_work),
                'CUDA_allocated_after_original_release_bytes':allocation_after_original_release,
                'original_Session_model_Adam_weakrefs_absent':True,
                'global_TRAIN_RNG_restore_error':global_continuation_error,
                'local_state_sha256':sha(local_path),'global_state_sha256':sha(global_path),
                'global_TRAIN_backward_qualified':True,'quality_scores_read':False})
            flush()
            for hook in session.live_work_hooks: hook.remove()
            session = None; gc.collect(); torch.cuda.empty_cache()
        require(result['counts'] == {'constructor_attempts':6,'constructor_completions':6,
                'TRAIN_update_attempts':6,'TRAIN_update_completions':6,'Adam_attempts':6,'Adam_completions':6,
                'backward_attempts':12,'backward_completions':12,
                'save_attempts':6,'save_completions':6,'restore_attempts':6,'restore_completions':6},
                'Exact finite three-case constructor/update/save/restore work')
        work_rows = [work for row in result['rows'] for work in
                     (row['original_actual_work'],row['fresh_restore_actual_work'])]
        total_work = {key:[sum(work[key][i] for work in work_rows) for i in range(7)]
                      if isinstance(work_rows[0][key],list) else sum(work[key] for work in work_rows)
                      for key in work_rows[0]}
        result['aggregate_actual_native_work'] = total_work
        require(total_work == {'joint_forward_attempts':42,'joint_forward_completions':42,'stem_completions':168,
                'global_completions':72,'local_head_completions':96,'global_head_completions':72,
                'block_completions':28,'local_conv_completions':[168]*7,
                'backward_attempts':12,'backward_completions':12}, 'Exact actual finite whole-TRAIN forward/backward work')
        result['complete'] = True; flush(); atomic(output/'QUALIFIED.json',result)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), complete=False, automatic_retry=False)
        flush(); atomic(output/'FAILURE.json',result)
        raise


if __name__ == '__main__':
    main()
