"""Four discarded complete real-WikiCS updates; structural CUDA engineering only."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

from audit import RouteAudit, same_tree
from owned import HERE, OWNER_ID, birth, require, sha, validate_release, write

EPOCHS = (1, 2, 101, 102)
ATOL = RTOL = 2e-5


def _screen():
    binding = json.loads((HERE / 'SCREEN_BINDING.json').read_text())
    root = (HERE.parent / binding['directory']).resolve(strict=True)
    require(root.is_relative_to(HERE.parent) and sha(root / 'MANIFEST.json') == binding['manifest_sha256'],
            'Exactly reviewed four-bank source manifest')
    for row in json.loads((root / 'MANIFEST.json').read_text())['files']:
        path = (root / row['path']).resolve(strict=True)
        require(path.is_relative_to(root) and sha(path) == row['sha256']
                and path.stat().st_size == row['bytes'], 'Changed four-bank source payload')
    require(sha(root / 'screen.py') == binding['program_sha256'], 'Exact four-bank executor')
    spec = importlib.util.spec_from_file_location('_label_four_bank_cuda_engineering_screen', root / 'screen.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _roles(session):
    return {arm: {name: session.common.tensor_digest(value)
                  for name, value in bank.named_buffers()}
            for arm, bank in session.banks.items()}


def _copied_state(session, state):
    torch = session.native.torch
    same_tree(torch, session.native.model.state_dict(), state['native']['model'], 'native.model')
    same_tree(torch, [optimizer.state_dict() for optimizer in session.native.optimizers],
              state['native']['optimizers'], 'native.Adam')
    for arm, bank in session.banks.items():
        same_tree(torch, session.engine._learned_corrector_state(bank),
                  state['banks'][arm]['parameters'], arm + '.learned')
        same_tree(torch, bank.optimizer.state_dict(), state['banks'][arm]['optimizer'], arm + '.Adam')


def _adam_steps(value):
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            if key == 'step':
                result.append(float(item))
            else:
                result.extend(_adam_steps(item))
        return result
    if isinstance(value, (list, tuple)):
        return [step for item in value for step in _adam_steps(item)]
    return []


def _engineering_state(state):
    require(state['kind'] == 'engineering_snapshot' and state['selector_performed'] is False
            and state['snapshot_purpose'] == 'engineering_qualification'
            and state['stats'] is None and state['native']['selected_VALID'] is None
            and state['native']['member_VALID'] is None,
            'Predetermined unscored engineering state; no invented selection or metric')


def run_fixed(*, owner, later_execution_authorized=False):
    """Actual worker callable; numerical imports occur only after owner/release gates."""
    release = owner['release']
    phase = validate_release(release, later_execution_authorized=later_execution_authorized)
    output = phase / release['output_relative']
    require(owner['owner_id'] == OWNER_ID and owner['worker_PID'] == os.getpid()
            and owner['worker_start_ticks'] == birth(os.getpid())
            and os.getppid() == owner['parent_PID'] and birth(os.getppid()) == owner['parent_start_ticks']
            and os.getpgrp() == owner['worker_process_group'] == os.getpid()
            and Path(owner['output']) == output and output.is_dir(),
            'Only the newly owned normal-host worker may execute')
    started = time.monotonic(); session = reconstructed = audit = replay_audit = None
    work = dict(native_complete_updates=0, all_four_bank_complete_updates=0,
        local_complete_updates=0, global_complete_updates=0, full_development_ID_serving_events=0,
        reconstruction_serving_events=0, detached_leaf_and_native_gradient_checks=0,
        corrector_native_RNG_checks=0, engineered_own_local_restores=0, forbidden_call_guards=0)
    events = []; serving_custody = []; checks = {}; origin = None; error = None; passed = False; roles_loaded = False
    try:
        screen = _screen()
        # Reuse the unchanged public loader and validators. It deserializes and
        # validates VALID y. No truth is scored, supplied to models/objectives,
        # indexed by this qualifier, or used to select a checkpoint.
        data = screen._dependencies()[5]
        train, valid, origin = data.load_train_valid('wikics', phase / release['roles']['train']['relative_path'],
                                                   phase / release['roles']['valid']['relative_path'])
        roles_loaded = True
        heldout_ids = valid['ids']; del valid
        session = screen.make_session(train_data=train, seed=release['native_seed'], device='cuda:0',
            polynormer=phase / release['polynormer']['relative_path'],
            purpose='engineering_qualification', later_execution_authorized=True)
        native = session.native; torch = native.torch
        require(torch.cuda.device_count() == 1 and native.cuda_index == 0
                and torch.cuda.get_device_properties(0).total_memory >= 75 * 1024 ** 3,
                'Authorized sole full-memory CUDA GPU required')
        torch.cuda.reset_peak_memory_stats(0)
        require(tuple(session.banks) == screen.ARMS and len(train['ids']) == 580
                and len(heldout_ids) == 5274, 'Complete four banks and original role populations')
        bank_roles = _roles(session)
        heads = {arm: screen.EXPECTED[arm][1] for arm in screen.ARMS}
        audit = RouteAudit(torch, session.banks, heads, screen.CONDITIONS)

        original_update = session.update_banks
        def audited_update(H, base, masks, *, native_capture=None, verify_native_gradients=False):
            require(verify_native_gradients is True and H.shape == (11701,512)
                    and base.shape == (11701,10) and not H.requires_grad and not base.requires_grad,
                    'Existing full view B and observed native gradient guard required')
            # Real captured values, unchanged. Requiring leaf gradients tests
            # each core's explicit detach boundary without another native call.
            H_leaf = H.detach().requires_grad_(True)
            base_leaf = base.detach().requires_grad_(True)
            before = session.common.native_rng_state(native)
            audit.begin('TRAIN', masks=masks)
            result = original_update(H_leaf, base_leaf, masks, native_capture=native_capture,
                                     verify_native_gradients=True)
            require(H_leaf.grad is None and base_leaf.grad is None, 'All banks detach H and base logits')
            session.common.require_same_native_rng(native, before, 'engineering detached all-bank executor')
            audit.finish()
            work['detached_leaf_and_native_gradient_checks'] += 1
            work['corrector_native_RNG_checks'] += 1
            return result
        session.update_banks = audited_update

        def serve(current, probe, *, replay=False):
            original = current.serve_banks
            def audited_serve(H, base, ids, *, native_capture=None):
                before = current.common.native_rng_state(current.native)
                probe.begin('SERVE', ids=ids)
                result = original(H, base, ids, native_capture=native_capture)
                current.common.require_same_native_rng(current.native, before, 'engineering all-bank serving')
                probe.finish(); work['corrector_native_RNG_checks'] += 1
                return result
            current.serve_banks = audited_serve
            try:
                predictions = current.serve_ids(heldout_ids)
            finally:
                current.serve_banks = original
            for arm, prediction in predictions.items():
                require(prediction['member_logits'].shape == (screen.EXPECTED[arm][0],5274,10)
                        and prediction['served_probabilities'].shape == (5274,10)
                        and bool(torch.isfinite(prediction['member_logits']).all())
                        and bool(torch.isfinite(prediction['served_probabilities']).all()),
                        'Finite full-population serving without a truth argument')
                if screen.CONDITIONS[arm] != 'label_only4_detached':
                    require(current.banks[arm].last_native_capture == current.metadata(),
                            'Actual control serving records the same coherent source parameter/logical state')
            serving_custody.append(dict(purpose=current.purpose, reconstruction=replay,
                                         capture=current.metadata(), serving_IDs=5274))
            key = 'reconstruction_serving_events' if replay else 'full_development_ID_serving_events'
            work[key] += 1
            return predictions

        try:
            session.evaluate(None)
        except ValueError:
            work['forbidden_call_guards'] += 1
        else:
            raise ValueError('Engineering purpose must refuse VALID evaluation before truth access')

        local_state = None
        for epoch in EPOCHS:
            if epoch == 101:
                torch.cuda.synchronize(0)
                serve(session, audit)
                live_rng = session.common.native_rng_state(native)
                live = {arm: dict(mask=bank.mask_generator.get_state().clone(), steps=bank.steps,
                                 counters=copy.deepcopy(bank.counters)) for arm, bank in session.banks.items()}
                require(native.steps == 2 and all(bank.steps == 2 for bank in session.banks.values()),
                        'Two discarded local updates were actually completed')
                require(any(not torch.equal(value.cpu(), local_state['native']['model'][name])
                            for name, value in native.model.state_dict().items()),
                        'Native restoration must exercise a changed learned state')
                require(_adam_steps([opt.state_dict() for opt in native.optimizers]) !=
                        _adam_steps(local_state['native']['optimizers']), 'Native Adam restore must be nonvacuous')
                for arm, bank in session.banks.items():
                    require(any(not torch.equal(value, local_state['banks'][arm]['parameters'][name])
                            for name, value in session.engine._learned_corrector_state(bank).items())
                        and _adam_steps(bank.optimizer.state_dict()) !=
                            _adam_steps(local_state['banks'][arm]['optimizer']),
                        'Every learned bank and its Adam must differ from the predetermined snapshot')
                require(session.restore_native_own_local(local_state) == 1,
                        'Predetermined own-native-local epoch1 restore')
                _copied_state(session, local_state)
                session.common.require_same_native_rng(native, live_rng, 'engineering coherent own-local restore')
                require(_roles(session) == bank_roles and native.steps == 2
                        and session.parameter_epoch == 1 and native.model.models[0].body._global is True,
                        'Immutable graph/TRAIN roles and actual epoch versus logical work custody')
                for arm, bank in session.banks.items():
                    require(torch.equal(bank.mask_generator.get_state(), live[arm]['mask'])
                            and bank.steps == live[arm]['steps'] and bank.counters == live[arm]['counters'],
                            'Every end-local mask stream and work counter remains live')
                work['engineered_own_local_restores'] += 1
                checks['own_local_exact_learned_and_Adam_restore_with_live_streams'] = True
                del live, live_rng
            masks = session.draw_common_masks()  # Before the original two native forwards.
            require(all(mask.draw_id == native.steps + 1 for mask in masks.values()),
                    'Mask draw IDs retain the live cumulative schedule')
            updates = 0
            for batch, labels in data.train_batches('wikics', train, epoch, release['native_seed'], native.device):
                require(updates == 0 and native.steps < 4, 'Predetermined four-update work ceiling')
                require(len(labels) == 580 and len(batch['ids']) == 580, 'All580 native own-CE labels')
                result = session.train_step(batch, labels, epoch=epoch, masks=masks, verify_native_gradients=True)
                require(result['old_view_B_native_capture']['global_mode'] == (epoch >= 101),
                        'Both original active-head capture paths exercised')
                events.append(dict(engineered_epoch=epoch, native_logical_steps=native.steps,
                    actual_parameter_epoch=session.parameter_epoch,
                    old_view_B_capture=result['old_view_B_native_capture'],
                    TRAIN_labels=580, common_Q=290, mask_draw_id=masks[screen.ARMS[0]].draw_id, view_count=2))
                updates += 1
                del result, batch, labels
            require(updates == 1, 'Exactly one complete full-input native update per event')
            work['native_complete_updates'] += 1; work['all_four_bank_complete_updates'] += 1
            work['global_complete_updates' if epoch >= 101 else 'local_complete_updates'] += 1
            torch.cuda.synchronize(0)
            require(max(torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0))
                    <= release['max_own_peak_cuda_bytes'],
                    'Fixed own-process CUDA peak allocation bound')
            if epoch == 1:
                local_state = session.snapshot(epoch=1, kind='engineering_snapshot', native_metric=None, stats=None)
                _engineering_state(local_state)
            write(output / 'PROGRESS.json', dict(work=work, events=events, source_work=session.work,
                capture_counts=session.capture.counts,
                bank_counts={arm: dict(bank.counters) for arm, bank in session.banks.items()},
                scientific_accuracy_endpoints=False, seconds=time.monotonic() - started))

        final_state = session.snapshot(epoch=102, kind='engineering_snapshot', native_metric=None, stats=None)
        _engineering_state(final_state)
        predictions = serve(session, audit)
        live_rng = session.common.native_rng_state(native)
        # The fresh native constructor seeds its own ambient RNG. This bounded
        # custody context preserves the finished original session's ambient state.
        with torch.random.fork_rng(devices=native.cuda_devices):
            try:
                reconstructed = screen.reconstruct_for_serving(state=final_state, train_data=train,
                    polynormer=phase / release['polynormer']['relative_path'], device='cuda:0',
                    later_execution_authorized=True)
                _copied_state(reconstructed, final_state)
                require(_roles(reconstructed) == bank_roles
                        and reconstructed.parameter_epoch == 102
                        and reconstructed.restored_logical_steps == 4,
                        'Exact full source/role/coherent engineering reconstruction custody')
                replay_audit = RouteAudit(torch, reconstructed.banks, heads, screen.CONDITIONS)
                replay = serve(reconstructed, replay_audit, replay=True)
                for arm in screen.ARMS:
                    require(torch.allclose(predictions[arm]['member_logits'], replay[arm]['member_logits'],
                                           atol=ATOL, rtol=RTOL)
                            and torch.allclose(predictions[arm]['served_probabilities'],
                                              replay[arm]['served_probabilities'], atol=ATOL, rtol=RTOL),
                            'Full-ID reconstruction prediction consistency at the declared float32 tolerance')
                try:
                    reconstructed.train_step(None, None, epoch=103)
                except ValueError:
                    work['forbidden_call_guards'] += 1
                else:
                    raise ValueError('Reconstruction must refuse training before model input')
            finally:
                random.setstate(live_rng['python']); native.np.random.set_state(live_rng['numpy'])
        session.common.require_same_native_rng(native, live_rng, 'engineering reconstruction custody context')
        _copied_state(session, final_state)
        require(_roles(session) == bank_roles, 'Original graph/TRAIN buffers remain unchanged')
        checks['serving_only_coherent_reconstruction_and_prediction_consistency'] = True
        require(native.steps == 4 and session.work['native_update_attempts'] ==
                session.work['native_update_completions'] == session.work['all_bank_update_completions'] == 4
                and session.work['common_Q_checks'] == 4 and session.work['native_local_restorations'] == 1
                and session.work['complete_VALID_events'] == 0
                and session.capture.counts['TRAIN_head_captures'] == 8
                and session.capture.counts['VALID_head_captures'] == 0
                and session.capture.counts['SERVE_head_captures'] == 2
                and reconstructed.capture.counts['SERVE_head_captures'] == 1
                and work['detached_leaf_and_native_gradient_checks'] == 4
                and work['corrector_native_RNG_checks'] == 7
                and len(audit.events) == 6 and len(replay_audit.events) == 1
                and work['forbidden_call_guards'] == 2, 'Complete fixed engineering work and no scoring events')
        for arm, bank in session.banks.items():
            _, head_count, backwards, adam = screen.EXPECTED[arm]
            require(bank.steps == bank.counters['mask_draws'] == 4
                    and bank.counters['training_route_forwards'] == 4 * head_count
                    and bank.counters['route_backwards'] == 4 * backwards
                    and bank.counters['corrector_Adam_steps'] == 4 * adam
                    and bank.counters['serving_route_forwards'] == 2 * head_count,
                    'All complete required bank work: ' + arm)
        torch.cuda.synchronize(0)
        require(max(torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0))
                <= release['max_own_peak_cuda_bytes'],
                'Whole qualifier including reconstruction stays within own CUDA bound')
        checks.update(common_Q_exclusion_and_real_zero_context=True,
            full_input_local_and_global_capture=True, detached_H_base_and_native_gradients=True,
            native_RNG_isolation=True, all_required_bank_work=True,
            no_VALID_truth_scoring_model_input_objective_or_selector=True)
        passed = True
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        raise
    finally:
        # Retain actual attempted/completed source counters even after partial failure.
        if session is not None:
            work['native_complete_updates'] = session.work['native_update_completions']
            work['all_four_bank_complete_updates'] = session.work['all_bank_update_completions']
        result = dict(schema='label-four-bank-full-input-cuda-engineering-result-v1',
            engineering_passed=passed, owner_id=OWNER_ID, error=error, work=work, events=events,
            serving_custody_events=serving_custody, checks=checks,
            source_work=dict(session.work) if session else None,
            capture_counts=dict(session.capture.counts) if session else None,
            bank_counts={arm: dict(bank.counters) for arm, bank in session.banks.items()} if session else None,
            real_graph_route_audits=audit.events if audit else [],
            reconstruction_route_audits=replay_audit.events if replay_audit else [],
            full_input_origin=origin, source_binding=json.loads((HERE / 'SCREEN_BINDING.json').read_text()),
            observed_metadata_sha256={key: sha(phase / release[key]['relative_path'])
                                      for key in ('projection_manifest','available_manifest')},
            runtime=dict(torch=str(session.native.torch.__version__), numpy=session.native.np.__version__,
                         CUDA=session.native.torch.version.cuda,
                         observed_GPU_name=session.native.torch.cuda.get_device_name(0),
                         observed_GPU_total_bytes=session.native.torch.cuda.get_device_properties(0).total_memory,
                         native_and_bank_source_recipe=session.run) if session else None,
            own_peak_cuda_allocated_bytes=session.native.torch.cuda.max_memory_allocated(0) if session else None,
            own_peak_cuda_reserved_bytes=session.native.torch.cuda.max_memory_reserved(0) if session else None,
            seconds=time.monotonic() - started, fixed_engineered_epochs=list(EPOCHS),
            predetermined_restore_epoch=1, selector_performed=False, snapshot_metrics=None,
            VALID_truth_deserialized_validated_by_unchanged_public_loader=roles_loaded,
            VALID_truth_scoring=False, VALID_truth_model_input=False, VALID_truth_objective=False,
            scientific_accuracy_endpoints=False, scientific_launch_admitted=False,
            complete_1100_schedule_qualified=False, scientific_resume_supported=False,
            reconstruction_float32_atol=ATOL, reconstruction_float32_rtol=RTOL,
            checkpoints_written=False, discarded_engineering_work=True, automatic_retry=False)
        try:
            write(output / 'ENGINEERING_RESULT.json', result)
        finally:
            if replay_audit is not None: replay_audit.close()
            if reconstructed is not None: reconstructed.close()
            if audit is not None: audit.close()
            if session is not None: session.close()
    return dict(engineering_passed=True, output=str(output), scientific_launch_admitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    arguments = parser.parse_args()
    # Parent atomically fills the freshly created worker's exact PID/birth tuple.
    deadline = time.monotonic() + 5
    while True:
        owner = json.loads(arguments.worker.read_text())
        if owner.get('worker_PID') == os.getpid():
            break
        require(time.monotonic() < deadline, 'Owned parent did not finish worker registration')
        time.sleep(.01)
    run_fixed(owner=owner, later_execution_authorized=True)


if __name__ == '__main__':
    main()
