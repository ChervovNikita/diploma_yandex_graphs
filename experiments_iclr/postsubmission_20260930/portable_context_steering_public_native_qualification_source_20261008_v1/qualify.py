"""Bounded public-context native engineering check; TRAIN-only, discarded updates."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
CONTEXT = PHASE / 'portable_context_steering_public_interface_20261008_v1'
PUBLIC = PHASE / 'portable_internal_be_public_interface_20261007_v2'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
METHODS = ('shared_common', 'shared_route', 'shared_route_permuted',
           'single_common_context', 'independent4_route_context')
SEED = 901337


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inside(relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Phase-relative custody required')
    path = (PHASE / relative).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Project research repository custody required')
    return path


def bound(row):
    path = inside(row['path'])
    require(path.is_file() and sha(path) == row['sha256'], 'Exact input/source bytes changed')
    return path


def verify_manifest(root, digest):
    require(sha(root/'MANIFEST.json') == digest, 'Exact source manifest required')
    for row in read(root/'MANIFEST.json')['files']:
        path = root / row['path']
        require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Sealed source component changed')


def admit(path, digest):
    require(sha(path) == digest, 'Exact separate root engineering release required')
    cfg = read(path)
    require(cfg.get('schema') == 'public-context-full-native-qualifier-release-v1'
        and cfg.get('enabled') is True
        and cfg.get('root_engineering_execution_authorized') is True
        and cfg.get('source_review_approved') is True
        and cfg.get('exact_one_GPU_route_verified') is True
        and cfg.get('external_supervision_confirmed') is True
        and cfg.get('VALID_labels_access') is False and cfg.get('TEST_access') is False
        and cfg.get('scientific_fit') is False and cfg.get('automatic_retry') is False,
        'Engineering release disabled pending root source/data/runtime/resource admission')
    require(cfg['methods'] == list(METHODS) and cfg['seed'] == SEED, 'Fixed engineering coverage, not a grid')
    verify_manifest(HERE, cfg['source_manifest_sha256'])
    pins = read(HERE/'SOURCE_BINDINGS.json')
    verify_manifest(CONTEXT, pins['public_context_manifest_sha256'])
    verify_manifest(PUBLIC, pins['public_dependency_manifest_sha256'])
    for row in pins['files']:
        require(bound(row).stat().st_size == row['bytes'], 'Bound component size changed')
    require(socket.gethostname() == 'anogena-2-0'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == GPU,
        'Only the authorized one-GPU allocation and sole physical UUID')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, timeout=10).splitlines()
    require(inventory == [GPU], 'Actual authorized sole-GPU inventory')
    free_mib = int(subprocess.check_output(['nvidia-smi', '--id='+GPU,
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
    require(cfg['owned_GPU_memory_cap_bytes'] == 20*1024**3
        and cfg['minimum_fresh_free_GPU_bytes'] == 24*1024**3
        and free_mib*1024**2 >= cfg['minimum_fresh_free_GPU_bytes'],
        'Prospective20GiB owned allocator cap and24GiB fresh shared-host admission')
    require(cfg['external_active_seconds'] == 1800 and cfg['external_cleanup_seconds'] == 10
        and cfg['external_hard_seconds'] == 1810, 'Finite external active/cleanup/hard envelope')
    require(set(cfg['inputs']) == {'train', 'polynormer'}, 'Exact TRAIN/native inputs only; caller targets are reconstructed without VALID/TEST')
    for row in cfg['inputs'].values():
        bound(row)
    return cfg


def tensor_tree(value, torch):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_tree(item, torch)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_tree(item, torch)


def tree_digest(value, torch):
    """Exact state fingerprint for engineering restoration, not scientific parity."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key, child in item.items():
                digest.update(repr(key).encode()); visit(child)
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def gradients(session, global_mode):
    """Declare actual native stage inactivity, rather than hide missing gradients."""
    torch = session.torch
    expected_inactive = ('pred_local.',) if global_mode else ('ln.', 'global_attn.', 'pred_global.')
    active, inactive = [], []
    for name, parameter in session.model.named_parameters():
        require('.body.' in name, 'Expected exact WikiBackbone parameter naming')
        native_name = name.split('.body.', 1)[1]
        absent = native_name.startswith(expected_inactive)
        if absent:
            require(parameter.grad is None, 'Declared inactive native stage has a gradient: '+name)
            inactive.append(name)
        else:
            require(parameter.grad is not None and torch.isfinite(parameter.grad).all(),
                    'Missing/nonfinite declared active gradient: '+name)
            active.append(name)
    require(active and inactive, 'Complete declared active/inactive native parameter accounting')
    return dict(all_declared_active_gradients_present_and_finite=True,
        active_parameter_names=active, inactive_parameter_names=inactive,
        inactive_native_roots=list(expected_inactive))


def serve_train(session, batch):
    torch = session.torch
    session.model.eval()
    with torch.no_grad():
        logits, representation = session.forward(batch)
        pooled = session.serving(logits)
        session.core['selection'].finite_predictions(logits, pooled)
        require(tuple(logits.shape) == (session.model.members, 580, 10)
            and tuple(pooled.shape) == (580, 10) and torch.isfinite(representation).all(),
            'Complete finite original TRAIN-only serving')
        require(torch.allclose(pooled.sum(-1), torch.ones(580, device=session.device), atol=1e-5, rtol=1e-5),
                'Original mean-probability serving simplex')
    return pooled.detach()


def local_transition_check(session, facade, policy, dispatch):
    """Use engineering snapshots, without choosing any checkpoint by a score."""
    torch = session.torch
    expected_model = tree_digest(session.model.state_dict(), torch)
    expected_opt = tree_digest([opt.state_dict() for opt in session.optimizers], torch)
    expected_streams = tree_digest(session.streams, torch)
    if policy.own_selected_four:
        states = { 'own_local_'+str(m)+'.pt': session._cpu_tree({
            'model': body.state_dict(), 'optimizer': session.optimizers[m].state_dict()})
            for m, body in enumerate(session.model.models)}
    else:
        states = {'selected_local.pt': session._cpu_tree({
            'model': session.model.state_dict(),
            'optimizers': [opt.state_dict() for opt in session.optimizers]})}
    # Explicitly disturb only this discarded engineering model/history, then
    # verify full restoration before another native forward/update.
    with torch.no_grad():
        next(session.model.parameters()).add_(1.)
        first_moment = next(value for opt in session.optimizers for state in opt.state.values()
                            for key, value in state.items() if key == 'exp_avg')
        first_moment.add_(1.)
    require(tree_digest(session.model.state_dict(), torch) != expected_model
        and tree_digest([opt.state_dict() for opt in session.optimizers], torch) != expected_opt,
        'Engineering restoration challenge actually changed model and Adam history')
    names = []
    def load(name):
        names.append(name); return states[name]
    dispatch.local_transition(session, load, policy, facade)
    require(names == list(states)
        and tree_digest(session.model.state_dict(), torch) == expected_model
        and tree_digest([opt.state_dict() for opt in session.optimizers], torch) == expected_opt
        and tree_digest(session.streams, torch) == expected_streams
        and all(body.body._global for body in session.model.models),
        'Native own/joint model+Adam restore, live streams and local/global transition')
    return dict(requested_checkpoint_names=names, model_and_Adam_restored_exactly=True,
        end_local_member_streams_unchanged=True, native_global_mode_enabled=True,
        checkpoint_selection_performed=False)


def qualify_method(public, data, cli, method, cfg, train, prepared_targets, output):
    import torch
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    session, facade, policy, preparation = cli.make_session(public, method, SEED,
        'cuda:0', bound(cfg['inputs']['polynormer']), train, prepared_targets)
    cli.install_replay(session, diagnostics=True)
    cli.install_cost_counters(session)
    cpu_panel = torch.linspace(0, 579, steps=512, device='cpu').long()
    require(torch.equal(facade.index.cpu(), cpu_panel), 'Exact CUDA panel matches frozen original CPU panel')
    require(tuple(train['x'].shape) == (11701, 300) and tuple(train['edge_index'].shape) == (2, 442907)
        and len(train['y']) == 580, 'Complete original representative graph and TRAIN labels')
    preparation_seconds = time.monotonic()-began
    rows, transition = [], None
    for global_mode in (False, True):
        require(all(body.body._global == global_mode for body in session.model.models), 'Original native stage dispatch')
        torch.cuda.synchronize(); started = time.monotonic()
        batches = data.train_batches('wikics', train, 101 if global_mode else 1, SEED, session.device)
        batch, labels = next(batches)
        require(next(batches, None) is None and len(labels) == 580, 'Complete original single TRAIN batch')
        before = dict(session.execution_totals)
        result = session.train_step(batch, labels)
        torch.cuda.synchronize(); update_seconds = time.monotonic()-started
        require(all(torch.isfinite(value).all() for value in result.values()), 'Finite two-own-view loss outputs')
        declared_gradients = gradients(session, global_mode)
        delta = {key: value-before[key] for key, value in session.execution_totals.items()}
        members = session.model.members
        require(delta == dict(shadow_member_forwards=2*members, replay_member_forwards=2*members,
            output_cotangent_collections=1, member_reverse_collections=2*members,
            optimizer_bank_updates=len(session.optimizers), exact_member_RNG_endpoint_checks=1),
            'Complete original two-view gradients, all VJPs before original Adam bank steps')
        prediction = serve_train(session, batch); inference_calls = members
        snapshot = output/(method+('_global.pt' if global_mode else '_local.pt'))
        session.save_training_state(snapshot, epoch=0)
        saved = torch.load(snapshot, map_location='cpu', weights_only=False)
        require(all(value.device.type == 'cpu' for value in tensor_tree(saved, torch)), 'CPU-only original state snapshot')
        require(all(value.device.type == 'cpu' and value.dtype == torch.uint8
            for stream in saved['streams'] for value in stream.values()), 'Original CPU-byte member RNG states')
        model_digest = tree_digest(session.model.state_dict(), torch)
        opt_digest = tree_digest([opt.state_dict() for opt in session.optimizers], torch)
        stream_digest = tree_digest(session.streams, torch)
        require(session.restore_training_state(snapshot) == 0
            and tree_digest(session.model.state_dict(), torch) == model_digest
            and tree_digest([opt.state_dict() for opt in session.optimizers], torch) == opt_digest
            and tree_digest(session.streams, torch) == stream_digest,
            'Original CUDA model, Adam and RNG state reload')
        require(all(body.body._global == global_mode for body in session.model.models)
            and all(parameter.device == session.device for parameter in session.model.parameters()),
            'Restored original CUDA parameters and local/global flags')
        moments = [value for opt in session.optimizers for state in opt.state.values()
                   for key, value in state.items() if key in ('exp_avg', 'exp_avg_sq')]
        require(moments and all(value.device == session.device and torch.isfinite(value).all()
                for value in moments), 'All native CUDA Adam moments finite after reload')
        restored_prediction = serve_train(session, batch); inference_calls += members
        # Record arithmetic variation without a microscopic parity rejection.
        reload_difference = float((prediction-restored_prediction).abs().max())
        if not global_mode:
            transition = local_transition_check(session, facade, policy, cli)
        torch.cuda.synchronize()
        rows.append(dict(global_mode=global_mode, updates=1, train_objects=580, own_views=2,
            complete_graph_nodes=11701, complete_graph_edges=442907, checkpoint_selection_performed=False,
            gradients=declared_gradients, execution_accounting=delta,
            TRAIN_member_inference_calls=inference_calls, exact_CPU_CUDA_panel_identity=True,
            original_state_reload_passed=True, reload_probability_max_abs_difference=reload_difference,
            reload_difference_is_blocking=False, replay_diagnostics=session.last_replay_diagnostics,
            peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_GPU_bytes=torch.cuda.max_memory_reserved(),
            snapshot_sha256=sha(snapshot), full_TRAIN_update_seconds=update_seconds,
            update_serving_reload_transition_seconds=time.monotonic()-started))
        del prediction, restored_prediction, saved
    own_bank = None
    if policy.own_selected_four:
        states = {'own_best_'+str(m)+'.pt': session._cpu_tree({
            'model': body.state_dict(), 'global': bool(m % 2), 'epoch': 0})
            for m, body in enumerate(session.model.models)}
        names = []
        def load(name):
            names.append(name); return states[name]
        cli.restore_own_best_bank(session, load, policy, facade)
        require([body.body._global for body in session.model.models] == [False, True, False, True]
            and names == list(states), 'Original per-member selected-stage flag dispatch')
        serve_train(session, batch)
        own_bank = dict(engineering_fixture_flags=[False, True, False, True],
            own_model_identity_preserved=True, TRAIN_member_inference_calls=4,
            checkpoint_selection_performed=False, evaluation_only=True)
    require(session.steps == 2 and facade.alignment_source_calls == 2 and facade.residual_source_calls == 0,
            'Exactly two discarded native updates, fixed context alignment only')
    require(torch.cuda.max_memory_reserved() < cfg['owned_GPU_memory_cap_bytes'], 'Measured owned allocator cap')
    expected_inference = 4*session.model.members + (4 if policy.own_selected_four else 0)
    require(session.public_cost['training_member_calls_attempted'] == session.public_cost['training_member_calls_returned'] == 8*session.model.members
        and session.public_cost['development_member_calls_attempted'] == session.public_cost['development_member_calls_returned'] == expected_inference
        and session.public_cost['update_calls_attempted'] == session.public_cost['update_calls_returned'] == 2,
        'Actual public API counters cover all discarded updates and TRAIN-only serving')
    record = dict(method=method, underlying_arm=session.arm, policy=policy.__dict__, rows=rows,
        local_global_transition=transition, own_bank_dispatch=own_bank,
        target_TV_common=preparation['route_common_mean_target_TV'],
        target_TV_permuted=preparation['route_permuted_mean_target_TV'],
        preparation_and_initialization_seconds=preparation_seconds,
        execution_totals=dict(session.execution_totals), public_API_counters=dict(session.public_cost),
        public_eval_counter_semantics='The development_phase counter names TRAIN-only eval calls in this engineering harness; no VALID labels are loaded.',
        seconds=time.monotonic()-began,
        model_quality_established=False, scientific_fit=False)
    del session, facade, preparation, batch, labels
    gc.collect(); torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    # Controller temporarily blocks these while capturing the owned PID/start ticks.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    cfg = admit(args.release, args.release_sha256)
    output = inside(cfg['output']); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); rows = []
    try:
        sys.path.insert(0, str(CONTEXT))
        import importlib.util
        spec = importlib.util.spec_from_file_location('_public_context_engineering_cli', CONTEXT/'train.py')
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        public = cli.load_public_interface(PUBLIC)
        import data_interface as data
        # Match the actual public main's provider preload and runtime initialization.
        from ogb.graphproppred import Evaluator as GraphEvaluator
        from ogb.linkproppred import Evaluator as LinkEvaluator
        torch, device = cli.initialize_runtime('cuda:0')
        require(str(torch.__version__) == '2.1.2+cu118' and torch.cuda.device_count() == 1,
                'Existing measured numeric runtime and sole mapped GPU')
        torch.cuda.set_per_process_memory_fraction(cfg['owned_GPU_memory_cap_bytes']/torch.cuda.get_device_properties(0).total_memory, 0)
        providers = {name: importlib.metadata.version(name) for name in
                     ('numpy', 'torch-geometric', 'ogb', 'torch-scatter', 'torch-sparse')}
        train = data._data().load_npz(bound(cfg['inputs']['train']), ('x','edge_index','ids','y'))
        data._data().check_projection('wikics', train, None, {'split_index': 0})
        # Use the public reconstruction API: no private target archive or preprocessing receipt.
        target_started = time.monotonic()
        prepared_targets = cli.prepare_targets(train, torch, device)
        target_seconds = time.monotonic()-target_started
        import numpy as np
        np.savez(output/'ENGINEERING_TRAIN_TARGETS.npz', **prepared_targets.arrays)
        (output/'PUBLIC_TARGET_PREPARATION.json').write_text(json.dumps(
            dict(prepared_targets.metadata, target_preparation_seconds=target_seconds,
                 TRAIN_payload_sha256=cfg['inputs']['train']['sha256'],
                 target_archive_sha256=sha(output/'ENGINEERING_TRAIN_TARGETS.npz')),
            indent=2, sort_keys=True, allow_nan=False)+'\n')
        for method in METHODS:
            rows.append(qualify_method(public, data, cli, method, cfg, train, prepared_targets, output))
            (output/'PROGRESS.json').write_text(json.dumps(dict(rows=rows, engineering_only=True), indent=2)+'\n')
        require(len(rows) == 5 and sum(len(row['rows']) for row in rows) == 10, 'Whole fixed native engineering coverage')
        result = dict(schema='public-context-full-native-qualification-v1', complete=True,
            scientific_fit=False, discarded_training=True, discarded_native_updates=10,
            discarded_training_member_forwards=136, discarded_member_reverse_collections=68,
            TRAIN_member_inference_calls=72, VALID_labels_access=False, TEST_access=False,
            checkpoint_selection_performed=False, automatic_retry=False,
            methodological_novelty_or_model_quality_established=False,
            bitwise_author_parity_claimed=False, rows=rows, providers=providers,
            source_manifest_sha256=cfg['source_manifest_sha256'],
            release_sha256=args.release_sha256, physical_gpu_uuid=GPU,
            TRAIN_payload_sha256=cfg['inputs']['train']['sha256'],
            public_context_manifest_sha256=read(HERE/'SOURCE_BINDINGS.json')['public_context_manifest_sha256'],
            caller_TRAIN_targets_reconstructed=True, prepared_TRAIN_targets=prepared_targets.metadata,
            target_preparation_seconds=target_seconds,
            PUBLIC_API_training_member_calls=sum(row['public_API_counters']['training_member_calls_returned'] for row in rows),
            PUBLIC_API_TRAIN_only_eval_calls=sum(row['public_API_counters']['development_member_calls_returned'] for row in rows),
            seconds=time.monotonic()-started)
        (output/'QUALIFIED.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    except BaseException as error:
        (output/'FAILURE.json').write_text(json.dumps(dict(complete=False, engineering_only=True,
            rows=rows, scientific_fit=False, error_type=type(error).__name__, error=str(error),
            automatic_retry=False, seconds=time.monotonic()-started), indent=2)+'\n')
        raise
    print(json.dumps(dict(engineering_complete=True, discarded_updates=10, output=str(output))))


if __name__ == '__main__':
    main()
