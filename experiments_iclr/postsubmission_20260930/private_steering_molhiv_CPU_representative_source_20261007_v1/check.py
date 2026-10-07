"""Three fresh real-data MolHIV private-path CPU fixtures; never a scientific fit."""
import argparse, ast, copy, gc, hashlib, importlib.util, json, os, platform, resource, signal, socket, sys, time
from pathlib import Path

INTERFACE_SHA = 'aa55302f80629971819469e29b6b8c39636825ea2e01ca6c18f019a85de88b0a'
INTERFACE_PROGRAM_SHA = 'b2ae8b0e7241b2deae11fb90887012b1d1ba29557e39ddc1f069364dbae8acae'
GEOMETRY_SHA = '88d573cc103f2bb8a1f7ce5c3e0ef5bc1cf8df2f300a7a1314b4d5eb79d4c623'
TRAIN_SHA = 'b85a5de6b3279056781fcee7f67f9d6b8a6966d54b1ee812a5ccc0d232fe368c'
VALID_SHA = '23e57ea40d0148a431c0d6e34ff64679bde8b7ff326a1cb3fe8ac5c9d5dfbd01'
MODES = ('alignment_private', 'hidden_private', 'gncl_private')
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
    raise TimeoutError('Owned CPU worker received termination signal ' + str(signum))

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
    """Observe actual calls/returned gradients/Adam input; no reference update."""
    raw = session.session; partition = session.steering.partition; optimizer = raw.optimizers[0]
    before = {name: parameter.detach().clone() for name, parameter in partition['all']}
    require(not optimizer.state and raw.steps == 0, 'Fresh empty native Adam required')
    collections = []; supplied = {}; calls = {'forwards': 0, 'Adam': 0}
    original_grad, original_step, original_forward = torch.autograd.grad, optimizer.step, raw.forward

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
        index = len(collections); require(index < 2 and calls['Adam'] == 0, 'Exactly two old-state gradient collections required')
        rows = partition['all'] if index == 0 else partition['internal_private']
        require(tuple(id(value) for value in args[1]) == tuple(id(parameter) for _, parameter in rows), 'Actual gradient parameter catalog differs')
        values = original_grad(*args, **kwargs)
        collections.append((rows, values, summaries(rows, values))); return values

    def forward(*args, **kwargs):
        calls['forwards'] += 1; return original_forward(*args, **kwargs)

    def step(*args, **kwargs):
        require(len(collections) == 2 and calls['Adam'] == 0 and calls['forwards'] == 2, 'Two views/two gradients before one Adam required')
        require(all(torch.equal(before[name], parameter.detach()) for name, parameter in partition['all']), 'Parameters moved before both gradient collections finished')
        own = {id(parameter): value for (_, parameter), value in zip(*collections[0][:2])}
        private = {id(parameter): value for (_, parameter), value in zip(*collections[1][:2])}
        for _, parameter in partition['all']:
            expected = private[id(parameter)] if id(parameter) in private else own[id(parameter)]
            actual = parameter.grad
            require((actual is None and expected is None) or (actual is not None and expected is not None
                    and actual.shape == expected.shape and actual.data_ptr() == expected.data_ptr()), 'Supplied gradient does not reference the correct observed collection')
        supplied.update(summaries(partition['all'], [parameter.grad for _, parameter in partition['all']]))
        calls['Adam'] += 1; return original_step(*args, **kwargs)

    torch.autograd.grad, optimizer.step, raw.forward = grad, step, forward
    try: result = session.train_step(batch, labels)
    finally:
        torch.autograd.grad, optimizer.step = original_grad, original_step
        del raw.__dict__['forward']
    require(calls == {'forwards': 2, 'Adam': 1} and len(collections) == 2 and raw.steps == 1, 'Observed update counts differ')
    require(session.steering.counters == {'two_view_updates': 1, 'member_forwards': 8, 'autograd_grad_calls': 2, 'Adam_steps': 1}, 'Actual adapter counters differ')
    raw.core['selection'].finite_state(raw.model, raw.optimizers)
    changes = {}
    for group in GROUPS:
        rows = partition[group]
        changed = sum(not torch.equal(before[name], parameter.detach()) for name, parameter in rows)
        require(changed > 0, 'No actual parameter progress in group ' + group)
        changes[group] = {'parameter_tensors': len(rows), 'parameter_elements': sum(parameter.numel() for _, parameter in rows), 'changed_tensors': changed}
    require(all(float(state['step']) == 1 for state in optimizer.state.values()), 'Native Adam state must record one transition')
    require(len(optimizer.state) == sum(value['finite_non_None_tensors'] for value in supplied.values()), 'Native Adam state/active gradient catalog differs')
    require(all(value is None or torch.isfinite(value).all() for value in result.values() if isinstance(value, torch.Tensor)), 'Nonfinite real TRAIN objective')
    return {'observed_calls': calls, 'autograd_grad_calls': 2, 'mean_own_gradient': collections[0][2],
        'internal_private_gradient': collections[1][2], 'supplied_gradient': supplied, 'group_changes': changes,
        'Adam_initialized_parameter_states': len(optimizer.state), 'Adam_state_steps_all_one': True,
        'both_collections_at_old_parameters': True, 'supplied_gradient_source_storage_matches': True,
        'finite_TRAIN_objectives_gradients_model_Adam': True}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('public-root', 'adapter-root', 'interface-root', 'train', 'valid', 'geometry', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic(); os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False); cases = []; stage = 'sealed_sources'; current_mode = None
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        require(sys.platform == 'linux', 'Linux runtime/RSS units required for this receipt')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment required')
        require(all(os.environ.get(name) == '2' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'Two-thread CPU environment required')
        require(sha(args.interface_root / 'train_steering.py') == INTERFACE_PROGRAM_SHA, 'Exact published complete interface required')
        interface = load(args.interface_root / 'train_steering.py', '_representative_complete_interface')
        interface._verify(args.interface_root, INTERFACE_SHA)
        public, ops, adapter = interface._sources(args.public_root, args.adapter_root)
        require(sha(args.geometry) == GEOMETRY_SHA, 'Existing complete TRAIN geometry scan required')
        geometry = json.loads(args.geometry.read_text()); selected = geometry['global_max_nodes']
        require(geometry['complete'] and geometry['batches_scanned'] == 77400 and not geometry['label_based_selection'], 'Original label-free batch selection required')
        require((selected['seed'], selected['epoch'], selected['batch_index_zero_based'], selected['graphs'], selected['nodes'], selected['edges']) == (6101, 68, 166, 128, 3948, 8404), 'Fixed maximum-node source batch required')
        stage = 'CPU_dependencies_and_exact_roles'
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        # Cache the unchanged evaluator imports before any seeded fresh Session.
        from ogb.graphproppred import Evaluator as GraphEvaluator
        from ogb.linkproppred import Evaluator as LinkEvaluator
        require(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized')
        train, valid, origin = ops.load_train_valid('molhiv', args.train, args.valid)
        require((origin['train_npz_sha256'], origin['valid_npz_sha256']) == (TRAIN_SHA, VALID_SHA), 'Existing exact portable MolHIV roles required')
        data = ops.load_train_valid.__globals__['_data']()
        order = torch.randperm(len(train['ids']), generator=torch.Generator().manual_seed(6101 + 19709 + 1000 * 68))
        positions = order[166 * 128:167 * 128]
        require(positions.tolist() == selected['positions'] and train['ids'][positions].tolist() == selected['official_graph_ids'], 'Exact original batch positions/graph IDs required')
        selector = joint_selector(args.public_root); initial_hash = catalog_hash = None
        for mode in MODES:
            current_mode = mode; stage = 'fresh_' + mode; began = time.monotonic()
            directory = args.output / mode; directory.mkdir()
            identity = interface.control_identity(mode, 'be_init', .5 if mode == 'gncl_private' else None)
            config = copy.deepcopy(public.recipe('molhiv')); config['arms'] = [identity['control_id']]; config['private_steering_control'] = identity
            raw = public.Session('molhiv', 'be_init', 6101, 'cpu')
            session = interface.SteeringSession(raw, adapter, identity, config)
            require(raw.device.type == 'cpu' and raw.cuda_index is None and len(raw.optimizers) == 1, 'Fresh CPU shared BE bank required')
            partition = session.steering.partition
            catalog = [{'name': name, 'shape': list(parameter.shape), 'elements': parameter.numel(), 'dtype': str(parameter.dtype), 'device': str(parameter.device), 'role': partition['roles'][name]} for name, parameter in partition['all']]
            observed_catalog_hash = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
            digest = hashlib.sha256()
            for name, value in raw.model.state_dict().items(): digest.update(name.encode()); digest.update(value.detach().contiguous().numpy().tobytes())
            for stream in raw.streams: digest.update(stream['cpu'].numpy().tobytes())
            digest.update(torch.get_rng_state().numpy().tobytes()); observed_initial_hash = digest.hexdigest()
            if initial_hash is None:
                initial_hash, catalog_hash = observed_initial_hash, observed_catalog_hash
                write(args.output / 'PARAMETER_CATALOG.json', {'actual_catalog': catalog, 'catalog_sha256': catalog_hash, 'exhaustive_permissions_passed': True})
            require(observed_initial_hash == initial_hash and observed_catalog_hash == catalog_hash, 'Fresh matched initialization/streams/catalog required')
            batch, labels = data.molecular_batch(train, positions, raw.device)
            require((batch['graph'].num_graphs, batch['graph'].num_nodes, batch['graph'].edge_index.shape[1]) == (128, 3948, 8404), 'Actual selected real batch geometry differs')
            stage = 'real_two_view_update_' + mode; update_began = time.monotonic()
            update = observe_update(session, batch, labels, torch); update_seconds = time.monotonic() - update_began
            write(directory / 'UPDATE_CHECK.json', update); del batch, labels
            stage = 'complete_VALID_' + mode; evaluate_began = time.monotonic()
            metric, per = ops.evaluate(session, train, valid)  # Selector values remain inside resource-only artifacts.
            evaluate_seconds = time.monotonic() - evaluate_began
            run = {'resource_only': True, 'scientific_fit': False, 'scientific_epoch_claimed': False,
                'task': 'molhiv', 'method_identity': identity['control_id'], 'underlying_constructor_arm': 'be_init',
                'private_steering_control': session.metadata(), 'lambda0_5_interface_fixture_only': mode == 'gncl_private',
                'data': origin, 'source_batch_epoch': 68, 'source_batch_index_zero_based': 166, 'TEST_scoring': False}
            def snapshot(value, epoch, score, members, annotated_run):
                saved = ops.joint_snapshot(value, epoch, score, members, annotated_run)
                saved.update(private_steering_control=value.metadata(), method_identity=identity['control_id']); return saved
            from types import SimpleNamespace
            ns = dict(session=session, args=SimpleNamespace(task='molhiv', output=directory), torch=torch,
                ordinary_independent=False, metric=metric, per=per, epoch=1, best=-float('inf'), best_local=-float('inf'),
                own_best=[-float('inf')] * 4, own_local=[-float('inf')] * 4, run=run, config=config, joint_snapshot=snapshot)
            stage = 'original_joint_selector_reload_' + mode; checkpoint_began = time.monotonic()
            exec(selector, ns); selected_hash = sha(directory / 'selected.pt')
            ns['epoch'] = 2; exec(selector, ns)
            require(sha(directory / 'selected.pt') == selected_hash, 'Original strict cached VALID tie must keep first checkpoint')
            saved = torch.load(directory / 'selected.pt', map_location='cpu', weights_only=False)
            require(saved['run']['method_identity'] == identity['control_id'] and not saved.get('evaluation_only', False), 'Labelled original joint snapshot required')
            raw.model.load_state_dict(saved['model'])
            require(len(saved['optimizers']) == 1, 'Original single Adam snapshot required')
            raw.optimizers[0].load_state_dict(saved['optimizers'][0]); raw.streams = saved['streams']
            raw.core['selection'].finite_state(raw.model, raw.optimizers)
            checkpoint_seconds = time.monotonic() - checkpoint_began
            stage = 'complete_reload_serving_' + mode; serving_began = time.monotonic(); served = batches = 0
            raw.model.eval()
            with torch.no_grad():
                for batch, labels in ops.validation_batches('molhiv', train, valid, raw.device):
                    logits, _ = session.forward(batch); raw.core['selection'].finite_predictions(logits, session.serving(logits))
                    served += logits.shape[1]; batches += 1
            require(served == 4113 and batches == 33 and not torch.cuda.is_initialized(), 'Complete finite CPU reload serving required')
            measured = {'mode': mode, 'constructor_arm': 'be_init', 'seed': 6101, 'lambda': identity['lambda'],
                'lambda0_5_interface_fixture_only': mode == 'gncl_private', 'strength_adopted': False,
                'source_batch_epoch': 68, 'source_batch_index_zero_based': 166, 'TRAIN_graphs': 128, 'TRAIN_nodes': 3948, 'TRAIN_edges': 8404,
                'fresh_model_updates': 1, 'members': 4, 'own_views': 2, 'reverse_sweeps': 2, 'Adam_transitions': 1,
                'actual_catalog_sha256': catalog_hash, 'matched_fresh_initial_state_and_streams': True, 'update': update,
                'complete_VALID_graphs': 4113, 'VALID_batches': 33, 'reload_served_graphs': served,
                'original_public_joint_selector_snapshot_used': True, 'strict_cached_VALID_tie_keeps_first_checkpoint': True,
                'selected_file_sha256': selected_hash, 'quality_values_closed': True,
                'seconds': time.monotonic() - began, 'phase_seconds': {'update': update_seconds, 'complete_VALID': evaluate_seconds,
                    'selector_save_reload': checkpoint_seconds, 'reload_serving': time.monotonic() - serving_began},
                'artifact_storage_bytes': sum(path.stat().st_size for path in directory.iterdir() if path.is_file())}
            cases.append(measured); write(directory / 'CASE_CHECK.json', measured)
            write(args.output / 'PROGRESS.json', {'completed_modes': [row['mode'] for row in cases], 'complete': False, 'quality_values_closed': True})
            del raw, session, partition, saved, ns, batch, labels, logits, metric, per; gc.collect()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        runtime = {'hostname': socket.gethostname(), 'python': platform.python_version(), 'executable': str(Path(sys.executable).resolve()),
            'torch': str(torch.__version__), 'numpy': sys.modules['numpy'].__version__,
            'pyg': sys.modules['torch_geometric'].__version__, 'ogb': sys.modules['ogb'].__version__,
            'torch_build_cuda': torch.version.cuda, 'device': 'cpu', 'cuda_initialized': False,
            'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads()}
        candidate = {'schema': 'private-steering-MolHIV-CPU-representative-candidate-v1', 'complete': True, 'task': 'molhiv',
            'public_manifest_sha256': interface.PUBLIC_SHA, 'adapter_manifest_sha256': interface.ADAPTER_SHA,
            'complete_interface_manifest_sha256': INTERFACE_SHA, 'worker_program_sha256': sha(__file__),
            'data': {'train_npz_sha256': TRAIN_SHA, 'valid_npz_sha256': VALID_SHA}, 'geometry_sha256': GEOMETRY_SHA,
            'runtime': runtime, 'cases': cases, 'inclusive_worker_seconds': time.monotonic() - started,
            'worker_CPU_user_seconds': usage.ru_utime, 'worker_CPU_system_seconds': usage.ru_stime,
            'peak_worker_RSS_bytes': usage.ru_maxrss * 1024, 'RSS_units': 'Linux ru_maxrss KiB converted to bytes',
            'max_node_batch_qualified': True, 'global_max_edge_batch_qualified': False, 'unqualified_max_edge_excess': 26,
            'scientific_fit': False, 'full_horizon_shortened': False, 'quality_values_closed': True, 'strength_adopted': False,
            'resource_weights_used_as_fit_start': False, 'TEST_scoring': False, 'automatic_retry': False, 'GPU_execution': False,
            'resource_supervisor_or_detector_interaction': False, 'other_jobs_modified': False}
        write(args.output / 'CANDIDATE.json', candidate)
        print(json.dumps({'complete': True, 'modes': len(cases), 'quality_values_closed': True, 'GPU_execution': False}))
    except BaseException as error:
        write(args.output / 'FAILURE.json', {'complete': False, 'stage': stage, 'mode': current_mode,
            'error_type': type(error).__name__, 'error': str(error), 'completed_cases': cases,
            'seconds': time.monotonic() - started, 'quality_values_closed': True, 'automatic_retry': False, 'TEST_scoring': False})
        raise

if __name__ == '__main__': main()
