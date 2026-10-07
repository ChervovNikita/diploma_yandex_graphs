"""Four fresh real MolHIV O/I/P/G CUDA representative fixtures; not a scientific fit."""
import argparse, ast, copy, gc, hashlib, importlib.util, json, os, platform, resource, signal, socket, sys, time
from pathlib import Path

INTERFACE_SHA = 'aa55302f80629971819469e29b6b8c39636825ea2e01ca6c18f019a85de88b0a'
INTERFACE_PROGRAM_SHA = 'b2ae8b0e7241b2deae11fb90887012b1d1ba29557e39ddc1f069364dbae8acae'
GEOMETRY_SHA = '88d573cc103f2bb8a1f7ce5c3e0ef5bc1cf8df2f300a7a1314b4d5eb79d4c623'
TRAIN_SHA = 'b85a5de6b3279056781fcee7f67f9d6b8a6966d54b1ee812a5ccc0d232fe368c'
VALID_SHA = '23e57ea40d0148a431c0d6e34ff64679bde8b7ff326a1cb3fe8ac5c9d5dfbd01'
POLICIES = ('O', 'I', 'P', 'G')
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
ALLOCATION_SHA = 'e45c7c74a4e47864867275b41818d4b3a8f2ccf169288d18e0580b8deb4b43dd'
ALLOCATION_PROGRAM_SHA = '5fb90f0cc6e40acc0308b0a851258fc0f9904cbb933f021d786e7c6e9649c473'
GROUPS = ('shared_own', 'internal_private', 'boundary_own')

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''): digest.update(block)
    return digest.hexdigest()

def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

def interrupted(signum, frame):
    raise TimeoutError('Owned CUDA worker received termination signal ' + str(signum))

def joint_selector(public_root):
    """Use the original selector statements, without its scientific epoch loop."""
    tree = ast.parse((public_root / 'train.py').read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    body = next(node for node in main.body if isinstance(node, ast.Try)).body
    loop = next(node for node in body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'epoch')
    selector = []
    for node in loop.body:
        if not isinstance(node, ast.If): continue
        names = {child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)}
        if 'best' in names or 'best_local' in names or ast.unparse(node.test) == 'session.model.independent': selector.append(node)
    require(len(selector) == 3, 'Original public joint/local/member selector structure required')
    return compile(ast.Module(body=selector, type_ignores=[]), str(public_root / 'train.py'), 'exec')

def observe_update(session, batch, labels, torch):
    """Observe O/I/P/G CUDA target sets, finite supplied gradients and the actual Adam.

    No reference update/gradient addition or storage-pointer assertion. O's
    internal gradient is the source own-plus-alignment sum; I directly collects J+A.
    """
    raw = session.session; partition = session.control.partition; optimizer = raw.optimizers[0]
    policy = session.identity['policy']
    if policy == 'O':
        expected = [('own/all', partition['all']), ('.05A/internal', partition['internal_private'])]
    elif policy == 'I':
        own_rows = [(name, parameter) for name, parameter in partition['all']
                    if partition['roles'][name] != 'internal_private']
        expected = [('own/shared,boundary', own_rows), ('J+.05A/internal', partition['internal_private'])]
    elif policy == 'P':
        expected = [('own/shared', partition['shared_own']), ('J/boundary', partition['boundary_own']),
                    ('J+.05A/internal', partition['internal_private'])]
    elif policy == 'G':
        expected = [('J/all', partition['all']), ('.05A/internal', partition['internal_private'])]
    else:
        raise ValueError('This CUDA fixture exercises only exact O/I/P/G controls')
    before = {name: parameter.detach().clone() for name, parameter in partition['all']}
    require(not optimizer.state and raw.steps == 0, 'Fresh empty native Adam required')
    collections = []; supplied = {}; calls = {'forwards': 0, 'Adam': 0}
    original_grad, original_step, original_forward = torch.autograd.grad, optimizer.step, raw.forward

    def old_state():
        require(calls['Adam'] == 0 and all(torch.equal(before[name], parameter.detach())
                for name, parameter in partition['all']), 'Reverse collection must observe the same old parameters')

    def summaries(rows, values):
        mapping = {id(parameter): gradient for (_, parameter), gradient in zip(rows, values)}
        result = {}
        for group in GROUPS:
            selected = [(name, mapping[id(parameter)]) for name, parameter in partition[group] if id(parameter) in mapping]
            require(all(value is None or torch.isfinite(value).all() for _, value in selected), 'Nonfinite observed gradient')
            result[group] = {'tensors': len(selected), 'None_names': [name for name, value in selected if value is None],
                'finite_non_None_tensors': sum(value is not None for _, value in selected),
                'nonzero_tensors': sum(value is not None and bool(torch.count_nonzero(value)) for _, value in selected)}
        return result

    def grad(*args, **kwargs):
        index = len(collections); require(index < len(expected), 'Unexpected reverse collection')
        old_state(); loss_role, rows = expected[index]
        require(tuple(id(value) for value in args[1]) == tuple(id(parameter) for _, parameter in rows),
                'Actual policy reverse target set differs')
        values = original_grad(*args, **kwargs)
        collections.append({'collection_index0': index, 'source_loss_and_target': loss_role,
            'parameter_names': [name for name, _ in rows], 'gradient_groups': summaries(rows, values),
            'same_old_parameters_before_Adam': True})
        return values

    def forward(*args, **kwargs):
        calls['forwards'] += 1; return original_forward(*args, **kwargs)

    def step(*args, **kwargs):
        require(len(collections) == len(expected) and calls['forwards'] == 2, 'Complete policy reverse/view work before Adam required')
        old_state()
        values = [parameter.grad for _, parameter in partition['all']]
        require(all(parameter.is_cuda for _, parameter in partition['all']) and all(value is None or (value.is_cuda and value.dtype == torch.float32) for value in values), 'Actual CUDA parameters and supplied gradients required')
        require(all(value is None or value.shape == parameter.shape for (_, parameter), value in zip(partition['all'], values)),
                'Actual supplied gradient shape differs')
        supplied.update(summaries(partition['all'], values))
        require(all(supplied[group]['finite_non_None_tensors'] > 0 for group in GROUPS), 'Every real group needs active finite supplied gradients')
        calls['Adam'] += 1; return original_step(*args, **kwargs)

    torch.autograd.grad, optimizer.step, raw.forward = grad, step, forward
    try: result = session.train_step(batch, labels)
    finally:
        torch.autograd.grad, optimizer.step = original_grad, original_step
        del raw.__dict__['forward']
    reverse_calls = len(expected)
    require(calls == {'forwards': 2, 'Adam': 1} and len(collections) == reverse_calls and raw.steps == 1, 'Observed update counts differ')
    require(session.control.counters == {'two_view_updates': 1, 'Session_forward_calls': 2, 'member_forwards': 8,
        'autograd_grad_calls': reverse_calls, 'Adam_calls': 1, 'Adam_steps': 1}, 'Actual O/I/P/G method counters differ')
    raw.core['selection'].finite_state(raw.model, raw.optimizers)
    changes = {}
    for group in GROUPS:
        rows = partition[group]
        changed = sum(not torch.equal(before[name], parameter.detach()) for name, parameter in rows)
        require(changed > 0, 'No actual parameter progress in group ' + group)
        changes[group] = {'parameter_tensors': len(rows), 'parameter_elements': sum(parameter.numel() for _, parameter in rows), 'changed_tensors': changed}
    require(all(float(state['step']) == 1 for state in optimizer.state.values()), 'Native Adam states must record one transition')
    require(len(optimizer.state) == sum(value['finite_non_None_tensors'] for value in supplied.values()), 'Native Adam state/active gradient catalog differs')
    require(all(torch.isfinite(value).all() for value in result.values() if isinstance(value, torch.Tensor)), 'Nonfinite real TRAIN objective')
    return {'policy': policy, 'observed_calls': calls, 'autograd_grad_calls': reverse_calls,
        'observed_collections': collections, 'supplied_gradient': supplied, 'group_changes': changes,
        'Adam_initialized_parameter_states': len(optimizer.state), 'Adam_state_steps_all_one': True,
        'all_collections_at_same_old_parameters': True, 'finite_TRAIN_objectives_gradients_model_Adam': True, 'actual_CUDA_parameters_and_gradients': True,
        'supplied_gradient_storage_pointer_check_used': False, 'gradient_addition_reference_computed': False,
        'reference_update_computed': False, 'tiny_numerical_parity_test': False, 'cost_padding': False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('public-root', 'adapter-root', 'interface-root', 'allocation-root', 'train', 'valid', 'geometry', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic(); os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False); cases = []; stage = 'sealed_sources'; current_policy = None
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        require(sys.platform == 'linux', 'Linux runtime/RSS units required for this receipt')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU, 'Exact single physical GPU exposure required')
        require(all(os.environ.get(name) == '2' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'Two-thread CPU environment required')
        require(sha(args.interface_root / 'train_steering.py') == INTERFACE_PROGRAM_SHA, 'Exact published complete interface required')
        interface = load(args.interface_root / 'train_steering.py', '_representative_complete_interface')
        interface._verify(args.interface_root, INTERFACE_SHA)
        interface._verify(args.allocation_root, ALLOCATION_SHA)
        require(sha(args.allocation_root / 'train.py') == ALLOCATION_PROGRAM_SHA, 'Exact sealed allocation interface required')
        allocation = load(args.allocation_root / 'train.py', '_representative_allocation_interface')
        public, ops, adapter = allocation.sources(args.public_root, args.adapter_root, args.interface_root)
        require(sha(args.geometry) == GEOMETRY_SHA, 'Existing complete TRAIN geometry scan required')
        geometry = json.loads(args.geometry.read_text()); selected = geometry['global_max_nodes']
        require(geometry['complete'] and geometry['batches_scanned'] == 77400 and not geometry['label_based_selection'], 'Original label-free batch selection required')
        require((selected['seed'], selected['epoch'], selected['batch_index_zero_based'], selected['graphs'], selected['nodes'], selected['edges']) == (6101, 68, 166, 128, 3948, 8404), 'Fixed maximum-node source batch required')
        stage = 'CUDA_dependencies_and_exact_roles'
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One visible CUDA device required')
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        # Cache the unchanged evaluator imports before any seeded fresh Session.
        from ogb.graphproppred import Evaluator as GraphEvaluator
        from ogb.linkproppred import Evaluator as LinkEvaluator
        require(torch.cuda.is_initialized() and torch.cuda.current_device() == 0, 'Actual CUDA context required')
        train, valid, origin = ops.load_train_valid('molhiv', args.train, args.valid)
        require((origin['train_npz_sha256'], origin['valid_npz_sha256']) == (TRAIN_SHA, VALID_SHA), 'Existing exact portable MolHIV roles required')
        data = ops.load_train_valid.__globals__['_data']()
        order = torch.randperm(len(train['ids']), generator=torch.Generator().manual_seed(6101 + 19709 + 1000 * 68))
        positions = order[166 * 128:167 * 128]
        require(positions.tolist() == selected['positions'] and train['ids'][positions].tolist() == selected['official_graph_ids'], 'Exact original batch positions/graph IDs required')
        selector = joint_selector(args.public_root); initial_hash = catalog_hash = None
        for policy in POLICIES:
            current_policy = policy; stage = 'fresh_' + policy
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0); began = time.monotonic()
            directory = args.output / policy; directory.mkdir()
            identity = allocation.control_identity(policy, 'be_init', .5)
            config = copy.deepcopy(public.recipe('molhiv')); config['arms'] = [identity['control_id']]; config['allocation_control'] = identity
            raw = public.Session('molhiv', 'be_init', 6101, 'cuda:0')
            session = allocation.PolicySession(raw, adapter, identity, config)
            require(raw.device.type == 'cuda' and raw.cuda_index == 0 and len(raw.optimizers) == 1, 'Fresh CUDA shared BE bank required')
            require(all(parameter.is_cuda and parameter.dtype == torch.float32 for parameter in raw.model.parameters()), 'Original float32 CUDA parameters required')
            partition = session.control.partition
            catalog = [{'name': name, 'shape': list(parameter.shape), 'elements': parameter.numel(), 'dtype': str(parameter.dtype), 'device': str(parameter.device), 'role': partition['roles'][name]} for name, parameter in partition['all']]
            observed_catalog_hash = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
            digest = hashlib.sha256()
            for name, value in raw.model.state_dict().items(): digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            for stream in raw.streams:
                digest.update(stream['cpu'].numpy().tobytes()); digest.update(stream['cuda'].cpu().numpy().tobytes())
            digest.update(torch.get_rng_state().numpy().tobytes()); digest.update(torch.cuda.get_rng_state(0).cpu().numpy().tobytes())
            observed_initial_hash = digest.hexdigest()
            if initial_hash is None:
                initial_hash, catalog_hash = observed_initial_hash, observed_catalog_hash
                write(args.output / 'PARAMETER_CATALOG.json', {'actual_catalog': catalog, 'catalog_sha256': catalog_hash, 'exhaustive_permissions_passed': True})
            require(observed_initial_hash == initial_hash and observed_catalog_hash == catalog_hash, 'Fresh matched initialization/streams/catalog required')
            batch, labels = data.molecular_batch(train, positions, raw.device)
            require((batch['graph'].num_graphs, batch['graph'].num_nodes, batch['graph'].edge_index.shape[1]) == (128, 3948, 8404), 'Actual selected real batch geometry differs')
            stage = 'real_two_view_update_' + policy; update_began = time.monotonic()
            update = observe_update(session, batch, labels, torch); torch.cuda.synchronize(0)
            update_seconds = time.monotonic() - update_began
            write(directory / 'UPDATE_CHECK.json', update); del batch, labels
            stage = 'complete_VALID_' + policy; evaluate_began = time.monotonic()
            valid_calls = session.work['external_forward_calls']; valid_members = session.work['external_member_forwards']
            metric, per = ops.evaluate(session, train, valid)  # Selector values remain inside resource-only artifacts.
            torch.cuda.synchronize(0); evaluate_seconds = time.monotonic() - evaluate_began
            session.work['complete_VALID_evaluations'] += 1
            session.work['complete_VALID_forward_calls'] += session.work['external_forward_calls'] - valid_calls
            session.work['complete_VALID_member_forwards'] += session.work['external_member_forwards'] - valid_members
            session.work['complete_VALID_dispatch_seconds'] += evaluate_seconds
            require(session.work['complete_VALID_forward_calls'] == 33
                    and session.work['complete_VALID_member_forwards'] == 132, 'Actual complete VALID work differs')
            run = {'resource_only': True, 'scientific_fit': False, 'scientific_epoch_claimed': False,
                'task': 'molhiv', 'method_identity': identity['control_id'], 'underlying_constructor_arm': 'be_init',
                'allocation_control': session.metadata(), 'lambda0_5_interface_fixture_only': True,
                'data': origin, 'source_batch_epoch': 68, 'source_batch_index_zero_based': 166, 'TEST_scoring': False}
            def snapshot(value, epoch, score, members, annotated_run):
                saved = ops.joint_snapshot(value, epoch, score, members, annotated_run)
                saved.update(allocation_control=value.metadata(), method_identity=identity['control_id']); return saved
            from types import SimpleNamespace
            ns = dict(session=session, args=SimpleNamespace(task='molhiv', output=directory), torch=torch,
                ordinary_independent=False, metric=metric, per=per, epoch=1, best=-float('inf'), best_local=-float('inf'),
                own_best=[-float('inf')] * 4, own_local=[-float('inf')] * 4, run=run, config=config, joint_snapshot=snapshot)
            stage = 'original_joint_selector_reload_' + policy; checkpoint_began = time.monotonic()
            exec(selector, ns); selected_hash = sha(directory / 'selected.pt')
            ns['epoch'] = 2; exec(selector, ns)
            require(sha(directory / 'selected.pt') == selected_hash, 'Original strict cached VALID tie must keep first checkpoint')
            saved = torch.load(directory / 'selected.pt', map_location='cpu', weights_only=False)
            require(saved['run']['method_identity'] == identity['control_id'] and not saved.get('evaluation_only', False), 'Labelled original joint snapshot required')
            raw.model.load_state_dict(saved['model'])
            require(len(saved['optimizers']) == 1, 'Original single Adam snapshot required')
            raw.optimizers[0].load_state_dict(saved['optimizers'][0]); raw.streams = saved['streams']
            require(all(state['exp_avg'].is_cuda and state['exp_avg_sq'].is_cuda for state in raw.optimizers[0].state.values()), 'Native Adam moments must restore to CUDA')
            require(all(stream['cpu'].device.type == 'cpu' and stream['cuda'].device.type == 'cpu' for stream in raw.streams), 'CPU/CUDA RNG byte tensors remain CPU resident for original RNG APIs')
            raw.core['selection'].finite_state(raw.model, raw.optimizers)
            torch.cuda.synchronize(0); checkpoint_seconds = time.monotonic() - checkpoint_began
            stage = 'complete_reload_serving_' + policy; serving_began = time.monotonic(); served = batches = 0
            raw.model.eval()
            with torch.no_grad():
                for batch, labels in ops.validation_batches('molhiv', train, valid, raw.device):
                    logits, _ = session.forward(batch); raw.core['selection'].finite_predictions(logits, session.serving(logits))
                    served += logits.shape[1]; batches += 1
            torch.cuda.synchronize(0)
            require(served == 4113 and batches == 33 and torch.cuda.is_initialized(), 'Complete finite CUDA reload serving required')
            measured = {'policy': policy, 'constructor_arm': 'be_init', 'seed': 6101, 'lambda': identity['lambda'],
                'lambda0_5_interface_fixture_only': True, 'strength_adopted': False,
                'method_identity': identity['control_id'], 'allocation_control': session.metadata(),
                'source_batch_epoch': 68, 'source_batch_index_zero_based': 166, 'TRAIN_graphs': 128, 'TRAIN_nodes': 3948, 'TRAIN_edges': 8404,
                'fresh_model_updates': 1, 'members': 4, 'own_views': 2, 'reverse_sweeps': allocation.method.REVERSE_CALLS[policy], 'Adam_transitions': 1,
                'actual_catalog_sha256': catalog_hash, 'matched_fresh_initial_state_and_streams': True, 'update': update,
                'complete_VALID_graphs': 4113, 'VALID_batches': 33, 'reload_served_graphs': served,
                'original_public_joint_selector_snapshot_used': True, 'strict_cached_VALID_tie_keeps_first_checkpoint': True,
                'selected_file_sha256': selected_hash, 'quality_values_closed': True,
                'peak_CUDA_allocated_bytes': int(torch.cuda.max_memory_allocated(0)), 'peak_CUDA_reserved_bytes': int(torch.cuda.max_memory_reserved(0)),
                'actual_CUDA_parameters_gradients_and_Adam': True, 'Adam_moments_restored_CUDA': True, 'RNG_byte_states_restored_CPU_for_CUDA_API': True,
                'seconds': time.monotonic() - began, 'phase_seconds': {'update': update_seconds, 'complete_VALID': evaluate_seconds,
                    'selector_save_reload': checkpoint_seconds, 'reload_serving': time.monotonic() - serving_began},
                'artifact_storage_bytes': sum(path.stat().st_size for path in directory.iterdir() if path.is_file())}
            cases.append(measured); write(directory / 'CASE_CHECK.json', measured)
            write(args.output / 'PROGRESS.json', {'completed_policies': [row['policy'] for row in cases], 'complete': False, 'quality_values_closed': True})
            del raw, session, partition, saved, ns, batch, labels, logits, metric, per; gc.collect(); torch.cuda.empty_cache()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        runtime = {'hostname': socket.gethostname(), 'python': platform.python_version(), 'executable': str(Path(sys.executable).resolve()),
            'torch': str(torch.__version__), 'numpy': sys.modules['numpy'].__version__,
            'pyg': sys.modules['torch_geometric'].__version__, 'ogb': sys.modules['ogb'].__version__,
            'torch_build_cuda': torch.version.cuda, 'device': 'cuda:0', 'cuda_initialized': torch.cuda.is_initialized(),
            'visible_device_count': torch.cuda.device_count(), 'physical_gpu_uuid': GPU,
            'TF32_matmul': torch.backends.cuda.matmul.allow_tf32, 'TF32_cudnn': torch.backends.cudnn.allow_tf32,
            'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads()}
        candidate = {'schema': 'allocation-OIPG-MolHIV-GPU-representative-candidate-v1', 'complete': True, 'task': 'molhiv',
            'public_manifest_sha256': interface.PUBLIC_SHA, 'adapter_manifest_sha256': interface.ADAPTER_SHA,
            'complete_interface_manifest_sha256': INTERFACE_SHA, 'allocation_controls_manifest_sha256': ALLOCATION_SHA,
            'allocation_interface_program_sha256': ALLOCATION_PROGRAM_SHA, 'worker_program_sha256': sha(__file__),
            'data': {'train_npz_sha256': TRAIN_SHA, 'valid_npz_sha256': VALID_SHA}, 'geometry_sha256': GEOMETRY_SHA,
            'runtime': runtime, 'cases': cases, 'inclusive_worker_seconds': time.monotonic() - started,
            'worker_CPU_user_seconds': usage.ru_utime, 'worker_CPU_system_seconds': usage.ru_stime,
            'peak_GPU_bytes': max(max(row['peak_CUDA_allocated_bytes'], row['peak_CUDA_reserved_bytes']) for row in cases),
            'peak_worker_RSS_bytes': usage.ru_maxrss * 1024, 'RSS_units': 'Linux ru_maxrss KiB converted to bytes',
            'max_node_batch_qualified': True, 'global_max_edge_batch_qualified': False, 'unqualified_max_edge_excess': 26,
            'scientific_fit': False, 'full_horizon_shortened': False, 'quality_values_closed': True, 'strength_adopted': False,
            'resource_weights_used_as_fit_start': False, 'TEST_scoring': False, 'automatic_retry': False, 'GPU_execution': True,
            'resource_supervisor_or_detector_interaction': False, 'other_jobs_modified': False}
        write(args.output / 'CANDIDATE.json', candidate)
        print(json.dumps({'complete': True, 'policies': len(cases), 'quality_values_closed': True, 'GPU_execution': True}))
    except BaseException as error:
        write(args.output / 'FAILURE.json', {'complete': False, 'stage': stage, 'policy': current_policy,
            'error_type': type(error).__name__, 'error': str(error), 'completed_cases': cases,
            'seconds': time.monotonic() - started, 'quality_values_closed': True, 'automatic_retry': False, 'TEST_scoring': False})
        raise

if __name__ == '__main__': main()
