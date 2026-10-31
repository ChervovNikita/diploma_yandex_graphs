"""Source-only native E point/fixed-noise/Gaussian acquisition integration.

Root supplies exact full inputs, new action admission and finite resource owner.
No import-time numerical code, automatic launch, reader calibration or verdict.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS = (7301, 7403, 7507)
SPECS = (
    ('E_point4', 'point', 4, 1), ('E_fixed_noise4', 'fixed_noise', 4, 1),
    ('E_Rank1_q4', 'learned_q', 4, 1), ('E_point_M1', 'point', 1, 1),
    ('E_point_I4', 'point', 4, 4), ('E_Rank1_q_M1', 'learned_q', 1, 1),
    ('E_Rank1_q_I4', 'learned_q', 4, 4))
FIXED = dict(depth=2, hidden=128, dropout=.2, learning_rate=.001, max_updates=1000,
             patience_updates=300, rank=16, member_seed_stride=1000003,
             factor_seed_offset=2000003, dropout_seed_offset=3000007, block_seed_offset=4000037,
             stalk_dim=4, baseline_policy='fresh_required', stage='F', K=None, k_selection_receipt=None)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def file_receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def checked(item):
    require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'}
            and file_receipt(item['path']) == item, 'Exact immutable source/data/state receipt')
    return Path(item['path']).resolve()


def write(path, value):
    require(not path.exists(), 'Fresh terminal record; retain every attempt')
    temp = path.with_suffix(path.suffix + '.tmp'); require(not temp.exists(), 'Retain partial writes')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def route():
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd() == R, 'Authorized host/repository before project reads')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines() == [UUID],
            'Sole authorized GPU allocation')


def admission(spec, root_guard, action):
    route()
    require(callable(root_guard), 'New explicit root admission callback')
    source = Path(__file__).resolve(); files = [source, source.with_name('posterior.py'), source.with_name('SOURCE_BINDINGS.json'), source.with_name('INTERFACE.json')]
    proof = root_guard(copy.deepcopy(spec), action=action, source_paths=[str(p) for p in files])
    require(isinstance(proof, dict) and proof.get('admitted') is True and proof.get('action') == action
            and proof.get('source_sha256') == {str(p): sha(p) for p in files}
            and proof.get('original_runtime_checked') is True and proof.get('full_input_custody_checked') is True
            and proof.get('finite_external_owner_bound') is True and proof.get('TEST_access') is False
            and proof.get('current_Physics_Penn_access') is False
            and type(proof.get('external_owner_wall_seconds')) is int and proof['external_owner_wall_seconds'] > 0,
            'Actual new source/runtime/full-input/finite-owner admission; never an inherited PASS')
    cfg = spec['native_cfg']; limits = cfg['family_limits']
    require(set(limits) == {'wall_seconds', 'cuda_bytes', 'rss_bytes'}
            and all(type(v) is int and v > 0 for v in limits.values()) and proof.get('limits') == limits,
            'Root chooses and binds finite wall/CUDA/RSS limits separately')
    require(cfg['seeds'] == list(SEEDS) and cfg['device'] == 'cuda:0'
            and Path(cfg['native_repo']).resolve() == R
            and all(cfg.get(key) == value for key, value in FIXED.items()), 'Unchanged native scientific recipe')
    if action == 'E_Rank1_BNN_native_acquisition':
        require(proof.get('new_full_input_readiness_complete') is True,
                'Root verifies actual new full-input readiness before scientific acquisition')
    return proof


def construct(spec, proof, output, training_only):
    here = Path(__file__).resolve().parent
    bindings = json.loads((here / 'SOURCE_BINDINGS.json').read_text())
    cfg = copy.deepcopy(spec['native_cfg']); sources = cfg['source_receipts']
    require(set(sources) == set(bindings['native_dependencies']), 'All unchanged original dependencies')
    for key, item in sources.items():
        checked(item)
        require(item['bytes'] == bindings['native_dependencies'][key]['bytes']
                and item['sha256'] == bindings['native_dependencies'][key]['sha256'], 'Exact original dependency: ' + key)
    for key, item in cfg['safe_roles'].items():
        require(key in ('train', 'valid') and checked(item) == Path(cfg[key + '_npz']).resolve(), 'Safe original role custody')
        require(item['bytes'] == bindings['original_safe_roles'][key]['bytes']
                and item['sha256'] == bindings['original_safe_roles'][key]['sha256'], 'Original paired full inputs/masks unchanged')
    require(set(cfg['safe_roles']) == {'train', 'valid'}, 'TRAIN/VALID only, never TEST')
    require(checked(sources['common_routes']) == Path(cfg['common_routes']).resolve()
            and checked(sources['factors']) == Path(cfg['factors']).resolve(), 'Original wrapper/factor paths')
    proposal = Path(spec['proposal_directory']).resolve()
    for name, item in bindings['proposal_files'].items():
        actual = file_receipt(proposal / name)
        require(actual['bytes'] == item['bytes'] and actual['sha256'] == item['sha256'], 'Frozen proposal preserved')
    owner = load('native_rank1_original_owner', checked(sources['training_family']))
    # Original constructor invokes its module's validator before imports/data.
    # This hook consumes the actual new proof; no old five-arm admission is forged.
    def validate(actual_cfg, qualification_only=False):
        require(actual_cfg == cfg and qualification_only is training_only, 'One exact newly admitted construction')
        return dict(source_sha256={item['path']: item['sha256'] for item in sources.values()},
                    safe_role_sha256={key: item['sha256'] for key, item in cfg['safe_roles'].items()},
                    root_runtime_admission=copy.deepcopy(proof), new_rank1_action=proof['action'])
    owner.validate_config = validate
    output.mkdir()
    family = owner.Family(cfg, output, hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(), training_only=training_only)
    require(family.edge_metadata == dict(nodes=11701, edges=442907, self_edges=11701), 'Actual complete original WikiCS support')
    posterior = load('native_rank1_posterior', here / 'posterior.py')
    delta = load('native_rank1_frozen_delta', proposal / 'acquisition_delta.py')
    posterior.bind(family, delta); family._operation = 'endpoint'
    return owner, family, posterior


def guard(family, started):
    limits = family.cfg['family_limits']; t = family.torch
    family._rank1_peak_cuda = max(getattr(family, '_rank1_peak_cuda', 0), t.cuda.max_memory_allocated(family.device))
    require(time.monotonic() - started < limits['wall_seconds']
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024) <= limits['rss_bytes']
            and t.cuda.max_memory_allocated(family.device) <= limits['cuda_bytes'], 'Actual finite root-chosen limits')


def gaussian_make(family, posterior, seed, member, members, law):
    family._rank1_law = law
    model, optimizer, dropout = family.make(seed, member, 'baseline', True, members, 'shared' if members == 4 else 'single')
    rows = family._rank1_rows
    require(not optimizer.state, 'Fresh native optimizer; filtering discards no moments or updates')
    if law == 'fixed_noise':
        for module in rows.sites.values():
            module.r_log_std.requires_grad_(False); module.s_log_std.requires_grad_(False)
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p.requires_grad]
        if members == 4: model.check_optimizer_ownership(optimizer)
    rows.move_centres(family.device)
    rng = posterior.generators(family.torch, family.device, seed, member, members)
    trainable = [p for p in model.parameters() if p.requires_grad]
    owned = [p for group in optimizer.param_groups for p in group['params']]
    require(len(owned) == len({id(p) for p in owned}) and {id(p) for p in owned} == {id(p) for p in trainable}, 'All trainable native means/scales exactly once')
    require(not list(model.named_buffers()), 'External immutable prior constants, never model buffers')
    count = sum(p.numel() for p in model.parameters()); expected = 253306 if members == 4 else 229558
    require(count == expected and rows.checkpoint()['factor_coordinates_per_route'] == 3958, 'Actual proposed native/fast/scale counts')
    require(sum(p.numel() for p in trainable) == (expected if law == 'learned_q' else expected - 3958 * members), 'Fixed scales excluded from AdamW')
    return model, optimizer, dropout, rows, rng


def sampled_update(family, model, optimizer, dropout, rows, factor_rng):
    t = family.torch; members = rows.members; model.train(); optimizer.zero_grad()
    draws = {m: rows.draw(m, factor_rng[m]) for m in range(members)}
    with rows.effective(draws):
        logits, hidden = family.training_outputs(model, dropout, members == 4)
    targets = family.data.y[family.train_ids] if members == 4 else family.data.y[family.data.train_mask]
    require(logits.shape == (members, 580, 10), 'Every full580TRAIN native route')
    ce = [family.F.cross_entropy(logits[m], targets, reduction='mean') for m in range(members)]
    KL = [rows.KL(m) for m in range(members)]
    loss = rows.delta.acquisition_loss(t, ce, KL)
    require(bool(t.isfinite(loss)) and bool(t.isfinite(logits).all()), 'Finite unchanged CE+conditional-KL/580 objective')
    loss.backward()
    require(all(p.grad is None or bool(t.isfinite(p.grad).all()) for p in model.parameters()), 'Finite native/mean/scale gradients')
    scales = [p for name, p in model.named_parameters() if name.endswith(('.r_log_std', '.s_log_std'))]
    result = dict(objective=float(loss.detach()), mean_sampled_own_CE=sum(float(v.detach()) for v in ce) / members,
                  conditional_KL=[float(v.detach()) for v in KL], mean_KL_per580=sum(float(v.detach()) for v in KL) / (members * 580),
                  factor_draws=members, TRAIN_routes=members, backward_calls=1, optimizer_steps=1,
                  trainable_scale_tensors=sum(p.requires_grad for p in scales),
                  trainable_scale_tensors_with_gradient=sum(p.requires_grad and p.grad is not None for p in scales))
    optimizer.step(); rows.mean_scope_closed()
    del logits, hidden, draws, ce, KL, loss
    return result


def state_value(model, optimizer, dropout, rows, factor_rng, step, selection=None):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), streams=dropout,
                factor_rng=[g.get_state() for g in factor_rng], posterior=rows.checkpoint(),
                step=step, selection=selection, selector='original_mean_factor_probability_mean')


def restore(family, model, optimizer, rows, factor_rng, checkpoint):
    state = family.torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
    rows.restore_constants(state['posterior'])
    require(len(state['factor_rng']) == len(factor_rng) == rows.members, 'Exact selected factor generator roster')
    for generator, value in zip(factor_rng, state['factor_rng']): generator.set_state(value)
    require(len(state['streams']) == rows.members, 'Exact native dropout stream roster')
    model.eval(); rows.mean_scope_closed()
    return state


def gaussian_fit(family, posterior, folder, seed, member, members, law, started):
    folder.mkdir(); family.progress(member, 0); guard(family, started)
    t = family.torch; before = time.monotonic()
    model, optimizer, dropout, rows, factor_rng = gaussian_make(family, posterior, seed, member, members, law)
    checkpoint = folder / 'selected.pt'; best, stale = -1., 0
    with (folder / 'trace.jsonl').open('x') as trace:
        for step in range(1, 1001):
            guard(family, started); update = sampled_update(family, model, optimizer, dropout, rows, factor_rng)
            model.eval(); rng_before = posterior.rng_hashes(factor_rng)
            with t.no_grad(): selection = family.metrics(family.logits(model, dropout, members == 4, family.valid_ids))
            require(posterior.rng_hashes(factor_rng) == rng_before, 'Mean selector consumes no factor RNG')
            improved = selection['accuracy'] > best
            if improved:
                best, stale, selected_step = selection['accuracy'], 0, step
                t.save(state_value(model, optimizer, dropout, rows, factor_rng, step, selection), checkpoint)
            else: stale += 1
            trace.write(json.dumps(dict(step=step, update=update, mean_factor_VALID=selection,
                                        strict_improvement=improved, stale_updates=stale), allow_nan=False) + '\n'); trace.flush()
            family.progress(member, step)
            if stale >= 300: break
    family.synchronize(); acquisition_seconds = time.monotonic() - before; before = time.monotonic()
    state = restore(family, model, optimizer, rows, factor_rng, checkpoint)
    require(state['step'] == selected_step, 'Strict selected state restoration')
    family.synchronize(); restore_seconds = time.monotonic() - before; before = time.monotonic()
    rng_before = posterior.rng_hashes(factor_rng)
    with t.no_grad():
        logits = family.logits(model, state['streams'], members == 4, family.valid_ids)
        restored = family.metrics(logits); logits = logits.cpu()
    require(posterior.rng_hashes(factor_rng) == rng_before, 'Primary mean serving consumes no factor RNG')
    family.synchronize()
    result = dict(member=member, factorized=True, factor_law=law, native_bodies_fitted=1, optimizer_acquisition_bundles=1,
                  learning_objective='mean own sampled CE plus average conditional Gaussian KL/580',
                  architecture='shared' if members == 4 else 'single', selected_step=selected_step, completed_updates=step,
                  selection=state['selection'], restored=restored, selected_state_receipt=file_receipt(checkpoint),
                  posterior=rows.summary(), factor_rng_selected=posterior.rng_hashes(factor_rng),
                  costs=dict(acquisition_seconds=acquisition_seconds, selected_restore_seconds=restore_seconds,
                             selected_forward_and_metrics_seconds=time.monotonic() - before,
                             parameters=sum(p.numel() for p in model.parameters()),
                             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                             checkpoint_bytes=checkpoint.stat().st_size,
                             cuda_peak_allocated_bytes=t.cuda.max_memory_allocated(family.device),
                             lifetime_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                  operation_counts=dict(updates=step, backward_calls=step, optimizer_steps=step, factor_draws=members * step,
                                        TRAIN_routes=members * step, mean_selector_VALID_routes=members * step, selected_primary_routes=members),
                  mean_selector=True, sampled_selector=False, factor_means_mutated_to_install_draw=False)
    write(folder / 'RESULT.json', result)
    family._fit_model = family._single_adapter = family._rank1_rows = None; family.completed_units += 1
    del model, optimizer, state, rows, factor_rng, dropout
    t.cuda.empty_cache(); guard(family, started)
    return logits, result


def secondary(family, posterior, bank, law, seed, members, bundles, folder, started):
    """Exactly12 raw selected eval routes; no selection, dropout or calibration."""
    t = family.torch; models = []; outputs = []; component = []
    generator = t.Generator(device=family.device).manual_seed(seed + 6000011)
    for fit in bank['fits']:
        guard(family, started); count = 1 if bundles == 4 else members
        model, optimizer, dropout, rows, rng = gaussian_make(family, posterior, seed, fit['member'], count, law)
        state = restore(family, model, optimizer, rows, rng, checked(fit['selected_state_receipt']))
        models.append((model, rows, state['streams'])); del optimizer, state, rng, dropout
    for block in range(3):
        for position in range(4):
            guard(family, started)
            route_id = position if members == 4 else 0
            model, rows, dropout = models[position if bundles == 4 else 0]
            active = 0 if bundles == 4 or members == 1 else position
            body = model.body if rows.members == 4 else model
            with t.no_grad(): draw = rows.draw(active, generator)
            with rows.effective({active: draw}), family.factors.member_context(body, active), t.no_grad():
                full = body(family.batch['graph'], family.batch['x'])
                logits = full[family.valid_ids]
                require(logits.shape == (5274, 10) and bool(t.isfinite(logits).all()), 'Full raw sampled selected route')
                outputs.append(logits.cpu()); component.append(route_id)
            del draw, full, logits
    require(len(outputs) == 12, 'All12 sampled forwards retained, no best-block choice')
    raw = t.stack(outputs); probabilities = raw.softmax(-1)
    labels = t.tensor(component, dtype=t.long)
    integrated_components = t.stack([probabilities[labels == m].mean(0) for m in range(members)])
    path = folder / 'selected_secondary_sampled_VALID.npz'
    family.np.savez(path, ids=family.valid_ids.cpu().numpy(), y=family.valid_y.cpu().numpy(),
                    draw_raw_logits=raw.numpy(), draw_probability=probabilities.numpy(),
                    draw_component=family.np.asarray(component, dtype=family.np.int64),
                    component_integrated_probabilities=integrated_components.numpy(),
                    all12_probability_mean=probabilities.mean(0).numpy(),
                    fixed_four_forward_block_probability_means=probabilities.reshape(3, 4, 5274, 10).mean(1).numpy())
    for model, rows, dropout in models: rows.mean_scope_closed()
    family._fit_model = family._single_adapter = family._rank1_rows = None
    del models, outputs, raw, probabilities, labels, integrated_components
    t.cuda.empty_cache(); guard(family, started)
    return dict(archive=file_receipt(path), native_selected_routes=12, RNG_seed=seed + 6000011,
                primary_component_means_archive=bank['primary_archive'],
                component_sequence=component, raw_only=True, blocks=3, best_block_selection=False,
                primary=False, calibration=False, native_dropout=False, model_updates=0)


def owner(spec, root_guard, readiness):
    action = 'E_Rank1_BNN_full_input_readiness' if readiness else 'E_Rank1_BNN_native_acquisition'
    proof = admission(spec, root_guard, action)
    output = Path(spec['output']).resolve(); require(not output.exists() and output.is_relative_to(R), 'One fresh authorized output')
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'Dedicated finite external process owner')
    previous = signal.getsignal(signal.SIGALRM)
    def timeout(signum, frame): raise TimeoutError('Finite root-chosen action cap; no retry or survivor continuation')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(spec['native_cfg']['family_limits']['wall_seconds'])
    output.mkdir(); started = time.monotonic(); family = None
    banks = [] if readiness else [dict(seed=s, procedure=p, factor_law=law, members=m, fits=[], status='UNSTARTED_RETAINED')
                                 for s in SEEDS for p, law, m, n in SPECS]
    try:
        _, family, posterior = construct(spec, proof, output / 'native_constructor', readiness)
        original_progress = family.progress
        def progress(member, update):
            original_progress(member, update); guard(family, started)
        family.progress = progress
        if readiness:
            model, optimizer, dropout, rows, rng = gaussian_make(family, posterior, SEEDS[0], 0, 4, 'learned_q')
            update = sampled_update(family, model, optimizer, dropout, rows, rng); guard(family, started)
            checkpoint = output / 'DISPOSABLE_FULL_INPUT_READINESS.pt'
            family.torch.save(state_value(model, optimizer, dropout, rows, rng, 1), checkpoint)
            state = restore(family, model, optimizer, rows, rng, checkpoint)
            with family.torch.no_grad(): mean = family.logits(model, state['streams'], True, family.train_ids)
            require(mean.shape == (4, 580, 10) and bool(family.torch.isfinite(mean).all()), 'Actual full-input selected-schema mean restoration')
            value = dict(complete=True, action=action, scientific_fit=False, disposable_optimizer_updates=1,
                         full_graph=family.edge_metadata, TRAIN_rows=580, VALID_deserialized=False, TEST_access=False,
                         procedure='E_Rank1_q4', seed=SEEDS[0], update=update, posterior=rows.summary(),
                         strict_restore_receipt=file_receipt(checkpoint), factor_rng_restored=posterior.rng_hashes(rng),
                         numerical_parity_claim=False, capability_or_quality_claim=False, resources_admitted_by_result=False,
                         current_Physics_Penn_access=False, automatic_acquisition=False)
        else:
            for seed in SEEDS:
                for procedure, law, members, bundles in SPECS:
                    guard(family, started); family._rank1_law = law; family._operation = 'endpoint'
                    family.current.update(arm=procedure, seed=seed)
                    folder = output / (procedure + '_seed' + str(seed)); folder.mkdir()
                    bank = next(b for b in banks if b['seed'] == seed and b['procedure'] == procedure)
                    bank['status'] = 'RUNNING'; logits = []
                    for member in range(bundles):
                        count = 1 if bundles == 4 else members
                        if law == 'point':
                            # Original method, objective, native RNG and selector remain unchanged.
                            selected, fit = family.fit(folder / ('member' + str(member)), seed, member, 'baseline', True,
                                                        count, 'own_only', False, 'single' if count == 1 else 'shared')
                        else:
                            selected, fit = gaussian_fit(family, posterior, folder / ('member' + str(member)), seed, member, count, law, started)
                        logits.append(selected); bank['fits'].append(fit)
                    raw = family.torch.cat(logits); probabilities = raw.softmax(-1); pool = probabilities.mean(0)
                    y = family.valid_y.cpu(); archive = folder / 'selected_VALID.npz'
                    family.np.savez(archive, ids=family.valid_ids.cpu().numpy(), y=y.numpy(), raw_logits=raw.numpy(),
                                    probability_mean=pool.numpy(), member_errors=probabilities.argmax(-1).ne(y).numpy(),
                                    pooled_errors=pool.argmax(-1).ne(y).numpy())
                    bank.update(status='PRIMARY_SELECTED_COMPLETE', primary_archive=file_receipt(archive), primary_mean_factor_metrics=family.metrics(raw),
                                primary_selected_native_routes=members, optimizer_fits=bundles, primary_serving='mean_factor_probability_mean')
                    if law != 'point': bank['secondary'] = secondary(family, posterior, bank, law, seed, members, bundles, folder, started)
                    bank['status'] = 'SELECTED_COMPLETE'
                    write(folder / 'RESULT.json', bank); del raw, logits, probabilities, pool, y
                    guard(family, started)
            require(len(banks) == 21 and family.completed_units == 39 and sum(len(b['fits']) for b in banks) == 39
                    and all(b['status'] == 'SELECTED_COMPLETE' for b in banks)
                    and sum(b['primary_selected_native_routes'] for b in banks) == 66
                    and sum(b.get('secondary', {}).get('native_selected_routes', 0) for b in banks) == 144,
                    'Entire exact21-bank/39-fit/39-state representative roster')
            TRAIN_routes = sum(f['operation_counts'].get('TRAIN_routes', f['operation_counts'].get('train_fullgraph_native_trajectories', 0)) for b in banks for f in b['fits'])
            selector_routes = sum(f['operation_counts'].get('mean_selector_VALID_routes', f['operation_counts'].get('valid_selection_fullgraph_native_trajectories', 0)) for b in banks for f in b['fits'])
            updates = sum(f['completed_updates'] for b in banks for f in b['fits'])
            require(updates <= 39000 and TRAIN_routes <= 66000 and selector_routes <= 66000, 'Exact prospective work ceilings')
            value = dict(complete=True, action=action, banks=21, optimizer_fits=39, selected_states=39, primary_archives=21,
                         actual_optimizer_updates=updates, actual_TRAIN_routes=TRAIN_routes, actual_mean_selector_VALID_routes=selector_routes,
                         selected_primary_routes=66, secondary_sampled_routes=144, records=banks,
                         required_future_calibration_attempts=105, requested_future_scalar_updates=52500,
                         calibration_performed=False, calibration_recipe='Original5-fold11709 CPUfloat64 Adam500 fixed endpoint; no final refit',
                         original_point_method_reused=True, original_sealed_sources_modified=False,
                         no_novelty_or_capability_verdict=True, no_automatic_reader_or_fit=True, TEST_access=False,
                         current_Physics_Penn_access=False)
        for source_path, digest in proof['source_sha256'].items(): require(sha(source_path) == digest, 'Action source/interface/bindings remain fixed')
        for item in spec['native_cfg']['source_receipts'].values(): checked(item)
        for item in spec['native_cfg']['safe_roles'].values(): checked(item)
        bindings = json.loads(Path(__file__).with_name('SOURCE_BINDINGS.json').read_text())
        for name, item in bindings['proposal_files'].items():
            actual = file_receipt(Path(spec['proposal_directory']) / name)
            require(actual['bytes'] == item['bytes'] and actual['sha256'] == item['sha256'], 'Proposal remains immutable')
        for bank in banks:
            for fit in bank['fits']: checked(fit['selected_state_receipt'])
        guard(family, started)
        value.update(admission=proof, seconds=time.monotonic() - started,
                     peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     peak_CUDA_allocated_bytes=family._rank1_peak_cuda)
        write(output / ('FULL_INPUT_READINESS_COMPLETE.json' if readiness else 'COMPLETE_RANK1_FAMILY.json'), value)
        return value
    except BaseException as error:
        for bank in banks:
            if bank['status'] in ('RUNNING', 'PRIMARY_SELECTED_COMPLETE'): bank['status'] = 'FAILED_RETAINED'
        write(output / 'FAILURE.json', dict(complete=False, action=action, type=type(error).__name__, message=str(error),
              traceback=traceback.format_exc(), records=banks, completed_fits=0 if family is None else family.completed_units,
              current=None if family is None else family.current,
              retry=False, survivor_mean=False, automatic_continuation=False, seconds=time.monotonic() - started))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


def full_input_readiness(spec, root_guard):
    return owner(spec, root_guard, True)


def acquire(spec, root_guard):
    return owner(spec, root_guard, False)


if __name__ == '__main__':
    raise SystemExit('Inactive source: root supplies exact spec/new admission callback and finite external owner.')
