"""Disabled full-input WikiCS native/corrector pilot cell; no owner or queue.

Original native single Session.train_step/evaluate/local_transition are reused.
The sealed label-only core is unchanged. No numerical imports occur at import.
"""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
CONDITIONS = ('label_only4_detached', 'multihead_single4',
              'shared_backbone_untied_correctors4', 'native_single', 'one_path')
SEEDS = (6101, 6203, 6307)
WORK_PER_UPDATE = {
    'label_only4_detached': dict(predictions=4, heads=4, backwards=4, Adam=1),
    'multihead_single4': dict(predictions=1, heads=4, backwards=1, Adam=1),
    'shared_backbone_untied_correctors4': dict(predictions=4, heads=4, backwards=4, Adam=4),
    'one_path': dict(predictions=1, heads=1, backwards=1, Adam=1),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def _helpers():
    def load(filename, name):
        spec = importlib.util.spec_from_file_location(name, HERE / filename)
        value = importlib.util.module_from_spec(spec); sys.modules[name] = value
        spec.loader.exec_module(value)
        return value
    return load('common.py', '_label_native_integration_common'), load('capture.py', '_label_native_capture')


def _core_seeds(seed):
    return dict(initializer_seed=seed + 1700001, mask_seed=seed + 1900001)


def _control_binding(condition, supplied):
    if condition in ('label_only4_detached', 'native_single'):
        require(supplied is None, 'Candidate/plain native use only their fixed original source')
        return None
    fixed = json.loads((HERE / 'CONTROL_CORE_BINDING.json').read_text())['source_binding']
    require(supplied is None or supplied == fixed, 'No alternative control source or operator grid')
    return fixed


def _new_corrector(native, train, core_module, public_root, common, condition, control_binding):
    old_rng = common.native_rng_state(native)
    config = native.config['model']
    arguments = dict(nodes=len(train['x']), feature_width=config['hidden_channels'] * config['heads'],
        classes=config['out_channels'], edge_index=train['edge_index'],
        train_ids=train['ids'], train_labels=train['y'], device=str(native.device), later_execution_authorized=True)
    if condition == 'label_only4_detached':
        require(control_binding is None, 'Candidate uses only the unchanged sealed core')
        core = core_module.make_core(**arguments, public_root=public_root, **_core_seeds(native.seed))
    else:
        require(condition in ('multihead_single4', 'shared_backbone_untied_correctors4', 'one_path'),
                'Single native backbone cannot supply ordinary_independent4')
        controls = common.control_module(control_binding)
        heads = WORK_PER_UPDATE[condition]['heads']
        core = controls.make_shared_controls(**arguments, mode=condition,
            initializer_seeds=tuple(_core_seeds(native.seed)['initializer_seed'] + 1009 * head
                                    for head in range(heads)),
            mask_seed=_core_seeds(native.seed)['mask_seed'],
            context_identity=common.context_identity(native, train),
            backbone_ids=(common.backbone_id(native),), selection_policy=common.selection_policy(condition),
            core_root=common.PHASE / common.CORE_NAME)
    common.require_same_native_rng(native, old_rng, 'isolated corrector construction')
    require(core.allowed_count == 580 and core.query_count == 290,
            'Original full WikiCS TRAIN role and unchanged half-mask core required')
    require(core.members == WORK_PER_UPDATE[condition]['predictions'], 'Explicit prediction/head-bank API')
    return core


def _capture_metadata(native, parameter_epoch, common, logical_steps=None):
    return dict(backbone_id=common.backbone_id(native), source_parameter_epoch=parameter_epoch,
        native_logical_steps_before_update=native.steps if logical_steps is None else logical_steps,
        global_mode=bool(native.model.models[0].body._global))


def _train_core(core, H, base, mask, condition, native_capture):
    if condition == 'label_only4_detached':
        return core.train_corrector_step(H, base, mask)
    return core.train_corrector_step(H, base, mask, native_capture=native_capture)


def _serve_core(core, H, base, ids, condition, native_capture):
    if condition == 'label_only4_detached':
        return core.serve(H, base, ids)
    # The outer integration owns one coherent selected-model/optimizer snapshot.
    # This is never the controls' ordinary independently-selected GNN4 API.
    return core.serve(H, base, ids, native_capture=native_capture)


def _learned_corrector_state(core):
    return {name: parameter.detach().cpu().clone()
            for name, parameter in core.named_parameters(remove_duplicate=False)}


def _restore_learned_corrector(core, saved, native):
    """Restore only learned coordinates, never overwrite immutable role buffers."""
    named = dict(core.named_parameters(remove_duplicate=False))
    require(set(named) == set(saved['parameters']), 'Exact corrector learned-role inventory')
    with native.torch.no_grad():
        for name, parameter in named.items():
            value = saved['parameters'][name]
            require(value.shape == parameter.shape and value.dtype == parameter.dtype,
                    'Exact saved corrector parameter shape/dtype')
            parameter.copy_(value.to(parameter.device))
    core.optimizer.load_state_dict(saved['optimizer'])
    core._check_roles(); core._finite_state()
    # Logical work/call counters and private mask RNG remain LIVE. No exact
    # optimizer/iterator/RNG resume or selected-state training is offered.


def _snapshot(native, core, driver, epoch, native_metric, pool_metric, members, run, kind):
    require(core._pending_mask is None and not core._failed_update, 'Completed coherent update required')
    state = driver.joint_snapshot(native, epoch, native_metric, [native_metric], run)
    state['checkpoint_kind'] = 'Native state at coherent ' + kind + ' epoch'
    return native._cpu_tree(dict(schema='label-only-native-single-coherent-state-v1',
        epoch=epoch, kind=kind, global_mode=bool(native.model.models[0].body._global),
        native_logical_steps=native.steps,
        native=state, corrector=dict(parameters=_learned_corrector_state(core),
            optimizer=core.optimizer.state_dict(), mask_rng_state=core.mask_generator.get_state(),
            logical_steps=core.steps, descriptor=core.descriptor()),
        native_own_VALID=native_metric, ensemble_VALID=pool_metric, member_VALID=members,
        run=run, exact_resume_supported=False))


def _restore_native_own_local(native, core, state, run, common):
    require(state['schema'] == 'label-only-native-single-coherent-state-v1'
            and state['kind'] == 'native_own_local' and state['global_mode'] is False
            and 1 <= state['epoch'] <= 100 and state['run'] == run,
            'Own-native-local selected coherent source/role/epoch state required')
    live_rng = common.native_rng_state(native)
    live_mask_rng = core.mask_generator.get_state().clone()
    old_core_steps = core.steps; old_native_steps = native.steps
    native.core['selection'].local_transition(native.model, native.optimizers,
        lambda filename: state['native'], ordinary_independent=False)
    _restore_learned_corrector(core, state['corrector'], native)
    common.require_same_native_rng(native, live_rng, 'native-own-local coherent restore')
    require(native.torch.equal(core.mask_generator.get_state(), live_mask_rng)
            and core.steps == old_core_steps and native.steps == old_native_steps,
            'End-local mask stream and logical work counters must remain live')
    return state['epoch']


def _evaluate(native, core, capture, driver, train, valid, common, condition, parameter_epoch):
    capture.begin('VALID', 1)
    # This is the unmodified original complete native evaluation. Its metric
    # chooses the native local checkpoint; correctors cannot change that metric.
    native_metric, native_per = driver.evaluate(native, train, valid)
    H, base = capture.finish()
    old_rng = common.native_rng_state(native)
    ids = valid['ids'].to(native.device)
    prediction = _serve_core(core, H, base, ids, condition,
                             _capture_metadata(native, parameter_epoch, common))
    common.require_same_native_rng(native, old_rng, 'detached full-context corrector VALID serving')
    torch = native.torch; truth = valid['y'].to(native.device)
    bank = prediction['member_logits']; pooled = prediction['served_probabilities']
    require(bank.shape == (core.members, 5274, 10) and pooled.shape == (5274, 10),
            'Complete original development population for every corrected route')
    native.core['selection'].finite_predictions(bank, pooled)
    metric = float((pooled.argmax(-1) == truth).float().mean())
    members = [float((logits.argmax(-1) == truth).float().mean()) for logits in bank]
    lp = torch.nn.functional.log_softmax(bank, dim=-1)
    true_lp = lp.gather(-1, truth[None, :, None].expand(core.members, -1, 1)).squeeze(-1)
    pool_lp = torch.logsumexp(true_lp, dim=0) - math.log(core.members)
    brier = ((pooled - torch.nn.functional.one_hot(truth, 10)) ** 2).sum(-1).mean()
    stats = dict(served_accuracy=metric, served_NLL=float(-pool_lp.mean()), Brier=float(brier),
                 member_accuracy=members, member_NLL=[float(-row.mean()) for row in true_lp],
                 native_own_accuracy=native_metric, native_member_accuracy=native_per)
    require(all(math.isfinite(value) for value in
                [metric, stats['served_NLL'], stats['Brier']] + members + stats['member_NLL']),
            'Finite complete development readouts required')
    return native_metric, metric, members, stats


def _native_reference(driver, train_path, valid_path, output, polynormer, seed, device, common):
    """Delegate plain control to the original full driver, without core creation."""
    original_write = driver.json_write
    identity = common.integration_identity()
    def annotate(path, value):
        if isinstance(value, dict):
            value = dict(value, pilot_condition='native_single', integration_source=identity,
                         label_corrector_instantiated=False)
        original_write(path, value)
    driver.json_write = annotate
    old_argv = sys.argv
    try:
        sys.argv = ['label-only-native-reference', '--task', 'wikics', '--arm', 'single',
            '--seed', str(seed), '--device', str(device), '--train', str(train_path),
            '--valid', str(valid_path), '--output', str(output), '--polynormer', str(polynormer)]
        driver.main()
    finally:
        sys.argv = old_argv
    return dict(output=str(output), condition='native_single', TEST_scoring=False)


def run_complete(*, train, valid, output, polynormer, seed=6101,
                 condition='label_only4_detached', device='cpu', control_core_binding=None,
                 later_execution_authorized=False):
    """One fresh full-input cell. Default refuses; no campaign/owner/retry."""
    require(later_execution_authorized is True, 'Disabled source-only integration; later separate adoption required')
    require(condition in CONDITIONS and type(seed) is int and seed in SEEDS,
            'One fixed declared pilot condition and seed')
    control_core_binding = _control_binding(condition, control_core_binding)
    output = Path(output)
    require(not output.exists(), 'Fresh output required; no resume/overwrite')
    common, capture_module = _helpers()
    public, data, driver, core_module, public_root = common.dependencies()
    if condition == 'native_single':
        return _native_reference(driver, train, valid, output, polynormer, seed, device, common)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); trace = []; native = core = capture = None
    work = dict(native_update_attempts=0, native_update_completions=0, corrector_update_completions=0,
                complete_VALID_events=0, native_local_restorations=0,
                native_RNG_isolation_checks=0, chosen_native_local_epoch=None)
    try:
        # Same evaluator imports BEFORE Session seeding as the original driver.
        from ogb.graphproppred import Evaluator as GraphEvaluator
        from ogb.linkproppred import Evaluator as LinkEvaluator
        train_data, valid_data, origin = data.load_train_valid('wikics', train, valid)
        native = public.Session('wikics', 'single', seed, device, polynormer)
        core = _new_corrector(native, train_data, core_module, public_root, common,
                              condition, control_core_binding)
        capture = capture_module.NativeSingleCapture(native, 11701, 512, 10)
        run = dict(task='wikics', condition=condition, seed=seed, device=str(native.device),
            data=origin, native_recipe=native.config, source=common.integration_identity(),
            native_core=native.core_provenance, native_source=native.native_provenance,
            corrector_seeds=_core_seeds(seed), corrector_training_capture='View B of the original2 own-CE forwards',
            corrector_initializer_seeds=tuple(_core_seeds(seed)['initializer_seed'] + 1009 * head
                for head in range(WORK_PER_UPDATE[condition]['heads']))
                if condition != 'label_only4_detached' else (_core_seeds(seed)['initializer_seed'],),
            control_core_binding=copy.deepcopy(control_core_binding),
            context_identity=common.context_identity(native, train_data),
            backbone_id=common.backbone_id(native),
            corrector_work_contract=WORK_PER_UPDATE[condition],
            update_order='Native original two-view CE/Adam, then detached old-view-B corrector CE/Adam',
            native_local_selector='Native own complete VALID strict-first maximum',
            final_selector='Ensemble complete VALID strict-first maximum of coherent epochs',
            source_only_runtime_qualification_performed_in_packet=False,
            author_execution_or_bitwise_trajectory_parity_claimed=False,
            torch=str(native.torch.__version__), numpy=native.np.__version__, CUDA=native.torch.version.cuda,
            TEST_scoring=False, automatic_retry=False, exact_resume_supported=False)
        json_write(output / 'RUN.json', run)
        best_pool = best_native = best_native_local = -float('inf')
        epochs = native.config['training']['epochs']; local_epochs = native.config['training']['local_epochs']
        require(epochs == 1100 and local_epochs == 100, 'Unchanged full original horizon required')
        parameter_epoch = 0
        for epoch in range(1, epochs + 1):
            if epoch == local_epochs + 1:
                saved = native.torch.load(output / 'NATIVE_OWN_LOCAL_COHERENT.pt',
                                         map_location=native.device, weights_only=False)
                work['chosen_native_local_epoch'] = _restore_native_own_local(native, core, saved, run, common)
                work['native_local_restorations'] += 1
                parameter_epoch = work['chosen_native_local_epoch']
            updates = 0; last_native = last_core = None
            for batch, labels in data.train_batches('wikics', train_data, epoch, seed, native.device):
                old_rng = common.native_rng_state(native)
                mask = core.draw_common_query_mask()  # Before mask-independent feature forwards.
                common.require_same_native_rng(native, old_rng, 'private common TRAIN query mask')
                work['native_RNG_isolation_checks'] += 1
                capture.begin('TRAIN', 2)
                native_capture = _capture_metadata(native, parameter_epoch, common)
                work['native_update_attempts'] += 1
                last_native = native.train_step(batch, labels)  # Unmodified native CE/views/Adam.
                work['native_update_completions'] += 1
                H, base = capture.finish()  # Existing second forward, no native replay.
                old_rng = common.native_rng_state(native)
                last_core = _train_core(core, H, base, mask, condition, native_capture)
                common.require_same_native_rng(native, old_rng, 'detached old-view-B corrector update')
                work['native_RNG_isolation_checks'] += 1
                work['corrector_update_completions'] += 1
                parameter_epoch = epoch
                updates += 1
                del H, base
            require(updates == 1, 'Complete original full TRAIN epoch required')
            native_metric, pool_metric, members, stats = _evaluate(native, core, capture, driver,
                train_data, valid_data, common, condition, parameter_epoch)
            work['complete_VALID_events'] += 1; work['native_RNG_isolation_checks'] += 1
            if pool_metric > best_pool:
                best_pool = pool_metric
                native.torch.save(_snapshot(native, core, driver, epoch, native_metric,
                    pool_metric, members, run, 'ensemble_selected'), output / 'selected.pt')
            if native_metric > best_native:
                best_native = native_metric
                native.torch.save(_snapshot(native, core, driver, epoch, native_metric,
                    pool_metric, members, run, 'native_own_final'), output / 'NATIVE_OWN_BEST_COHERENT.pt')
            if epoch <= local_epochs and native_metric > best_native_local:
                best_native_local = native_metric
                native.torch.save(_snapshot(native, core, driver, epoch, native_metric,
                    pool_metric, members, run, 'native_own_local'), output / 'NATIVE_OWN_LOCAL_COHERENT.pt')
            trace.append(dict(epoch=epoch, native_own_VALID=native_metric, ensemble_VALID=pool_metric,
                readouts=stats, TRAIN=dict(native_own_mean=float(last_native['own_mean']),
                    native_auxiliary=float(last_native['auxiliary']), corrector=last_core),
                old_view_B_native_capture=native_capture,
                global_mode=bool(native.model.models[0].body._global), seconds=time.monotonic() - started))
            json_write(output / 'VALID_TRACE.json', trace)
            json_write(output / 'PROGRESS.json', dict(complete=False, epoch=epoch, total_epochs=epochs,
                work=work, capture_counts=capture.counts, core_counts=core.counters))
        expected = WORK_PER_UPDATE[condition]
        require(native.steps == core.steps == 1100
                and work['native_update_attempts'] == work['native_update_completions']
                == work['corrector_update_completions'] == work['complete_VALID_events'] == 1100
                and work['native_local_restorations'] == 1
                and capture.counts['TRAIN_head_captures'] == 2200
                and capture.counts['VALID_head_captures'] == 1100
                and core.counters['route_backwards'] == expected['backwards'] * 1100
                and core.counters['training_route_forwards'] == expected['heads'] * 1100
                and core.counters['serving_route_forwards'] == expected['heads'] * 1100
                and core.counters['corrector_Adam_steps'] == expected['Adam'] * 1100
                and core.counters['mask_draws'] == 1100,
                'Complete fixed native/corrector/selector work required')
        json_write(output / 'COMPLETE.json', dict(complete=True, run=run, epochs=1100,
            native_steps=native.steps, corrector_steps=core.steps, work=work,
            capture_counts=capture.counts, corrector=core.descriptor(),
            selected_sha256=common.sha(output / 'selected.pt'),
            native_own_best_sha256=common.sha(output / 'NATIVE_OWN_BEST_COHERENT.pt'),
            native_own_local_sha256=common.sha(output / 'NATIVE_OWN_LOCAL_COHERENT.pt'),
            seconds=time.monotonic() - started, TEST_scoring=False,
            primary_mechanistic_conditions_source_integrated=True,
            full_representative_comparator_panel_implemented=False, exact_resume_supported=False))
        return dict(output=str(output), condition=condition, TEST_scoring=False)
    except BaseException as error:
        json_write(output / 'FAILURE.json', dict(complete=False, error_type=type(error).__name__,
            error=str(error), complete_epochs=len(trace), work=work,
            native_steps=native.steps if native else 0, core_steps=core.steps if core else 0,
            capture_counts=dict(capture.counts) if capture else None,
            core_counts=dict(core.counters) if core else None,
            seconds=time.monotonic() - started, automatic_retry=False, TEST_scoring=False))
        raise
    finally:
        if capture is not None:
            capture.close()


def reconstruct_selected(*, state, train, valid, polynormer, device='cpu',
                         later_execution_authorized=False):
    """Rebuild a trusted coherent candidate state for serving only, no resume."""
    require(later_execution_authorized is True, 'Disabled selected-state adapter; later separate adoption required')
    common, capture_module = _helpers()
    public, data, driver, core_module, public_root = common.dependencies()
    require(state['schema'] == 'label-only-native-single-coherent-state-v1'
            and state['kind'] == 'ensemble_selected' and state['run']['source'] == common.integration_identity(),
            'Exact unchanged coherent ensemble-selected source state required')
    train_data, valid_data, origin = data.load_train_valid('wikics', train, valid)
    require(origin == state['run']['data'], 'Exact original complete role payloads')
    seed = state['run']['seed']
    native = public.Session('wikics', 'single', seed, device, polynormer)
    condition = state['run']['condition']
    core = _new_corrector(native, train_data, core_module, public_root, common,
                          condition, state['run']['control_core_binding'])
    require(state['run']['native_core'] == native.core_provenance
            and state['run']['native_source'] == native.native_provenance
            and state['run']['native_recipe'] == native.config
            and type(state['global_mode']) is bool, 'Exact native source/recipe/selected stage')
    native.model.load_state_dict(state['native']['model'], strict=True)
    native.model.set_global(state['global_mode']); native.model.eval()
    native.streams = common.clone_streams(state['native']['streams'])
    _restore_learned_corrector(core, state['corrector'], native)
    core.eval()
    capture = capture_module.NativeSingleCapture(native, 11701, 512, 10)

    class ServingOnly:
        def serve(self):
            torch = native.torch
            with torch.no_grad():
                batch = dict(x=train_data['x'].to(native.device),
                    edge_index=train_data['edge_index'].to(native.device), ids=valid_data['ids'].to(native.device))
                capture.begin('SERVE', 1)
                native.forward(batch)
                H, base = capture.finish()
                old_rng = common.native_rng_state(native)
                prediction = _serve_core(core, H, base, batch['ids'], condition,
                    _capture_metadata(native, state['epoch'], common,
                                      logical_steps=state['native_logical_steps']))
                common.require_same_native_rng(native, old_rng, 'coherent selected corrector serving')
                return prediction

        def close(self):
            capture.close()

        def train_step(self, *args, **kwargs):
            raise RuntimeError('Coherent selected adapter is serving-only; no exact resume')

    return ServingOnly()
