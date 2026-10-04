#!/usr/bin/env python3
"""Fresh six-fit Pubmed native cohort; TRAIN/VALID only, separately root released."""
import argparse
import json
from pathlib import Path
import time
from common import HERE, EXECUTION, require, sha, write, utc, gate, capture, validation, setup_runtime, save_tensor, clone, inventory, start_monitor


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release, cohort = gate(args.root_release, args.release_sha256, 'fit_cohort')
    output = EXECUTION / 'fit_cohort/run01'
    require(not output.exists(), 'Fresh cohort output required; no retry/resume')
    output.mkdir(parents=True)
    progress = {'stage': 'fit_cohort', 'native_epochs_started': 0, 'native_epochs_completed': 0,
                'Adam_calls_started': 0, 'Adam_calls_completed': 0, 'complete_VALID_serves': 0, 'completed_fits': [],
                'current_fit': None, 'current_epoch': 0, 'current_phase': 'setup'}
    stopped = observer = sample = None
    try:
        torch, np, bodies, evaluate_mrr, eval_mrr, inspection, x, train, valid, negative, identities = setup_runtime(cohort)
        stopped, observer, peaks, sample = start_monitor(output, cohort['stages']['fit_cohort']['caps'], progress, torch)
        write(output / 'AVAILABLE_INSPECTION.json', inspection)
        config = {'schema': 'pubmed-native-predictive-config-v1', 'cohort': cohort, 'input_files': identities,
                  'author_commit': 'c447cbff4c493b60d14b6544c3c39d3b9c5ddff0', 'root_release_sha256': args.release_sha256,
                  'source_manifest_sha256': release['source_manifest_sha256'], 'feature_authority': release['feature_authority'],
                  'negative_pool_authority': release['negative_pool_authority'], 'prerequisites': release['prerequisites'],
                  'feature_equivalence_receipt_sha256': release['feature_equivalence_receipt_sha256'],
                  'qualified_source_binding_sha256': sha(HERE / 'SOURCE_BINDING.json'), 'initialized_from': 'Fresh native constructor/seed/reset cadence only'}
        write(output / 'CONFIG.json', config)
        config_sha = sha(output / 'CONFIG.json')
        fit_freezes = []
        for fit in cohort['fits']:
            model_name, seed, seed_id = fit['model'], fit['seed'], fit['literal_seed_id']
            directory = output / fit['fit_id']
            directory.mkdir()
            progress.update(current_fit=fit, current_epoch=0, current_phase='fresh_native_factory')
            write(output / 'PROGRESS.json', progress)
            # This is the only initialization of this scientific fit. Never restore
            # an engineering file or reseed/reset the model after this factory.
            unit = bodies.candidate_factory(model_name, seed, x, train)
            model, predictor, optimizer, ux, data, train_device = unit
            fit_started_updates = progress['Adam_calls_completed']
            def before_step(optimizer, args, kwargs):
                progress['Adam_calls_started'] += 1
            def after_step(optimizer, args, kwargs):
                progress['Adam_calls_completed'] += 1
            optimizer.register_step_pre_hook(before_step)
            optimizer.register_step_post_hook(after_step)
            best_valid, selected_score, selected_epoch, kill = 0.0, None, None, 0
            epoch, stop_reason = 0, 'native_max_epochs'
            selected_state = directory / 'selected_state.pt'
            selected_scores = directory / 'selected_VALID_scores.pt'
            with (directory / 'VALID_HISTORY.jsonl').open('x') as history:
                for epoch in range(1, 10000):
                    progress.update(current_epoch=epoch, current_phase='native_train')
                    progress['native_epochs_started'] += 1
                    write(output / 'PROGRESS.json', progress)
                    if model_name == 'SAGE':
                        loss = bodies.candidate_sage_train(model, predictor, train_device, ux, optimizer, 1024)
                    else:
                        loss = bodies.candidate_ncnc_train(model, predictor, data, {'train': {'edge': train}}, optimizer, 1024, True, [], None)
                    progress['native_epochs_completed'] += 1
                    expected_updates = epoch * (37 if model_name == 'SAGE' else 36)
                    require(progress['Adam_calls_completed'] - fit_started_updates == expected_updates, 'Native epoch/tail update coverage differs')
                    write(output / 'PROGRESS.json', progress)
                    if epoch % 5:
                        continue
                    progress['current_phase'] = 'complete_native_VALID'
                    before = capture(unit, torch, np)
                    pos, neg = validation(bodies, model_name, unit, valid, negative, torch)
                    result = evaluate_mrr(None, pos, neg)
                    per_query = clone(eval_mrr(pos, neg), torch)
                    score = result['MRR']  # native evaluate_mrr already rounds to4 decimals
                    after = capture(unit, torch, np)
                    progress['complete_VALID_serves'] += 1
                    improves_selection = selected_score is None or score > selected_score
                    if score > best_valid:
                        best_valid, kill = score, 0
                    else:
                        kill += 1
                    row = {'fit_id': fit['fit_id'], 'model': model_name, 'seed': seed, 'literal_seed_id': seed_id,
                           'epoch': epoch, 'loss_last_native_epoch': float(loss), 'metrics': result,
                           'raw_VALID_MRR': float(per_query['mrr_list'].mean()), 'selection_improved': improves_selection,
                           'kill_count': kill, 'completed_Adam_updates': expected_updates}
                    history.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                    history.flush()
                    if improves_selection:
                        selected_score, selected_epoch = score, epoch
                        identity = {'fit_id': fit['fit_id'], 'model': model_name, 'seed': seed, 'literal_seed_id': seed_id,
                                    'selected_epoch': epoch, 'selected_VALID_MRR': score, 'config_sha256': config_sha,
                                    'source_manifest_sha256': release['source_manifest_sha256']}
                        semantics = 'native sigmoid probabilities' if model_name == 'SAGE' else 'native unbounded logits'
                        save_tensor(selected_scores, {'schema': 'pubmed-native-selected-VALID-scores-v1', **identity,
                                    'positive_scores': pos, 'negative_scores': neg, 'score_semantics': semantics,
                                    'native_metrics_rounded4': result, 'native_per_query_metrics': per_query,
                                    'input_files': identities, 'row_order': 'Exact released VALID positive/pool row and candidate order',
                                    'native_evaluator_sha256': cohort['metric']['evaluator_sha256']}, torch)
                        save_tensor(selected_state, {'schema': 'pubmed-native-selected-full-state-v1', **identity,
                                    'state_before_VALID': before, 'state_after_VALID': after,
                                    'selected_VALID_scores_sha256': sha(selected_scores), 'input_files': identities,
                                    'completed_Adam_updates': expected_updates, 'selector_state': {'best_valid': best_valid, 'kill_count': kill},
                                    'initialization_source': 'Fresh native factory; no engineering continuation state'}, torch)
                    write(output / 'PROGRESS.json', progress)
                    del before, after, pos, neg, per_query
                    if kill > 10:
                        stop_reason = 'native_kill_count_gt10'
                        break
            require(selected_epoch is not None and progress['Adam_calls_started'] == progress['Adam_calls_completed'], 'Selected fit state or completed update accounting absent')
            freeze = {'schema': 'pubmed-native-fit-freeze-v1', 'status': 'COMPLETE', **fit,
                      'selected_epoch': selected_epoch, 'selected_VALID_MRR': selected_score, 'last_training_epoch': epoch,
                      'stop_reason': stop_reason, 'last_kill_count': kill, 'native_Adam_updates': epoch * (37 if model_name == 'SAGE' else 36),
                      'selected_state_sha256': sha(selected_state), 'selected_VALID_scores_sha256': sha(selected_scores),
                      'VALID_history_sha256': sha(directory / 'VALID_HISTORY.jsonl'), 'config_sha256': config_sha,
                      'input_files': identities, 'model_reset_after_fit': False, 'selected_state_replay_status': 'PENDING_SEPARATE_ROOT_RELEASE'}
            write(directory / 'FIT_FREEZE.json', freeze)
            fit_freezes.append(freeze)
            progress['completed_fits'].append(fit['fit_id'])
            # Release Python references only; no allocator peak reset/cache change.
            del unit, model, predictor, optimizer, ux, data, train_device
        torch.cuda.synchronize(0)
        stopped.set()
        observer.join()
        sample()
        freeze = {'schema': 'pubmed-native-cohort-freeze-v1', 'status': 'COMPLETE', 'UTC': utc(), 'fits': fit_freezes,
                  'cohort_sha256': sha(HERE / 'COHORT.json'), 'source_manifest_sha256': release['source_manifest_sha256'],
                  'root_release_sha256': args.release_sha256, 'config_sha256': config_sha,
                  'progress': progress, 'inclusive_child_wall_seconds': time.monotonic() - started,
                  'scientific_claim': 'TRAIN/VALID baseline experiment only; no TEST, accuracy, novelty, or transfer claim',
                  'selected_state_replay_status': 'PENDING_SEPARATE_ROOT_RELEASE'}
        write(output / 'COHORT_FREEZE.json', freeze)
    except BaseException as error:
        write(output / 'FAILURE.json', {'status': 'FAILED', 'type': type(error).__name__, 'condition': str(error),
              'progress': progress, 'inclusive_child_wall_seconds': time.monotonic() - started, 'automatic_retry': False})
        raise
    finally:
        if stopped is not None:
            stopped.set()
            observer.join()
        write(output / 'FINAL_CUSTODY.json', {'files': inventory(output), 'stage': 'fit_cohort', 'completed': (output / 'COHORT_FREEZE.json').exists()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
