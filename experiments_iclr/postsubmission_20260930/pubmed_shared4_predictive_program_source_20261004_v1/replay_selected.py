#!/usr/bin/env python3
"""Verify the six owned scientific selected states; no training or reselection."""
import argparse
import json
from pathlib import Path
import time
from common import (HERE, EXECUTION, require, sha, gate, write, utc, capture,
                    restore, validation, setup_runtime, inventory, start_monitor,
                    extension, fresh_unit, equal, Progress)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release, cohort = gate(args.root_release, args.release_sha256, 'replay_selected')
    output = EXECUTION / 'replay_selected/run01'
    require(not output.exists(), 'Fresh replay output required; no retry')
    output.mkdir(parents=True)
    progress = Progress(output, 'replay_selected')
    stopped = observer = sample = None
    reports = []
    try:
        torch, np, bodies, evaluate_mrr, eval_mrr, inspection, x, train, valid, negative, identities = setup_runtime(cohort)
        bridge, prototype, graph_ops, _ = extension(bodies, torch)
        stopped, observer, peaks, sample = start_monitor(output, cohort['stages']['replay_selected']['caps'], progress, torch)
        write(output / 'AVAILABLE_INSPECTION.json', inspection)
        fit_output = EXECUTION / 'fit_cohort/run01'
        freeze = json.loads((fit_output / 'COHORT_FREEZE.json').read_text())
        require(freeze.get('schema') == 'pubmed-shared4-cohort-freeze-v1' and freeze.get('status') == 'COMPLETE'
                and freeze['cohort_sha256'] == sha(HERE / 'COHORT.json')
                and freeze['source_manifest_sha256'] == release['source_manifest_sha256'], 'Own frozen scientific cohort identity differs')
        require([row['fit_id'] for row in freeze['fits']] == [row['fit_id'] for row in cohort['fits']]
                and len(freeze['paired_train_stream_evidence']) == 3, 'Complete six-fit/three-pair cohort differs')
        custody = json.loads((fit_output / 'FINAL_CUSTODY.json').read_text())
        require(custody.get('completed') is True and custody.get('stage') == 'fit_cohort', 'Fit custody incomplete')
        for row in custody['files']:
            path = fit_output / row['path']
            require(path.resolve().is_relative_to(fit_output) and not path.is_symlink() and sha(path) == row['sha256']
                    and path.stat().st_size == row['bytes'], 'Owned scientific fit artifact differs')
        for fit, selected in zip(cohort['fits'], freeze['fits']):
            progress.update(current_fit=fit, current_phase='owned_selected_state_restore')
            directory = fit_output / fit['fit_id']
            require(selected == json.loads((directory / 'FIT_FREEZE.json').read_text())
                    and selected.get('schema') == 'pubmed-shared4-fit-freeze-v1'
                    and selected.get('status') == 'COMPLETE', 'Own per-fit freeze differs')
            for key, value in fit.items():
                require(selected.get(key) == value, 'Frozen arm/seed identity differs: ' + key)
            state_path = directory / 'selected_state.pt'
            score_path = directory / 'selected_VALID_scores.pt'
            require(sha(state_path) == selected['selected_state_sha256'] and sha(score_path) == selected['selected_VALID_scores_sha256'], 'Selected artifact hashes differ')
            state = torch.load(state_path, map_location='cpu', weights_only=True)
            scores = torch.load(score_path, map_location='cpu', weights_only=True)
            require(state.get('schema') == 'pubmed-shared4-selected-full-state-v1'
                    and scores.get('schema') == 'pubmed-shared4-selected-VALID-scores-v1', 'Selected state/score schema differs')
            require(scores.get('native_evaluator_sha256') == cohort['metric']['evaluator_sha256']
                    and scores.get('score_semantics') == 'Mean of four native unbounded raw logits', 'Native mean-logit metric provenance differs')
            identity_keys = tuple(fit) + ('selected_epoch', 'selected_VALID_MRR', 'config_sha256', 'source_manifest_sha256')
            for key in identity_keys:
                require(state[key] == scores[key] and (key not in selected or state[key] == selected[key]), 'Selected identity differs: ' + key)
            require(state['input_files'] == scores['input_files'] == selected['input_files'] == identities
                    and state['selected_VALID_scores_sha256'] == sha(score_path), 'Input/score provenance differs')
            require(state['source_manifest_sha256'] == release['source_manifest_sha256']
                    and state['config_sha256'] == freeze['config_sha256'] == sha(fit_output / 'CONFIG.json')
                    and state['initialization_sha256'] == selected['initialization_sha256'] == sha(directory / 'INITIALIZATION.json'), 'Selected source/config/initialization differs')
            for key, value in fit.items():
                require(state[key] == value, 'Selected fixed arm/seed differs: ' + key)
            require(state['selected_epoch'] % 5 == 0 and state['completed_Adam_updates'] == state['selected_epoch'] * 36,
                    'Selected native epoch/update identity differs')
            # Fresh verifier reconstruction is immediately overwritten by its own
            # scientifically selected full state. No scientific fit is continued.
            unit = fresh_unit(bridge, prototype, graph_ops, bodies, fit, x, train)
            def forbid_optimizer_step(optimizer, args, kwargs):
                progress.add(Adam_started=1)
                require(False, 'No optimizer step admitted during selected-state replay')
            handle = unit[2].register_step_pre_hook(forbid_optimizer_step)
            restore(unit, state['state_after_VALID'], torch, np)
            equal(state['state_after_VALID'], capture(unit, torch, np), torch, fit['fit_id'] + '/exact_serialized_poststate_restore')
            restore(unit, state['state_before_VALID'], torch, np)
            equal(state['state_before_VALID'], capture(unit, torch, np), torch, fit['fit_id'] + '/exact_serialized_prestate_restore')
            progress.update(current_phase='complete_native_VALID')
            pos, neg = validation(bodies, 'NCNC', unit, valid, negative, torch)
            progress.add(complete_VALID_serves=1)
            tolerance = cohort['replay_rule']
            maximum = 0.0
            for label, actual, expected in (('positive', pos, scores['positive_scores']), ('negative', neg, scores['negative_scores'])):
                require(actual.shape == expected.shape and actual.dtype == expected.dtype and bool(torch.isfinite(expected).all()), 'Saved VALID score schema differs')
                delta = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
                allowed = tolerance['atol'] + tolerance['rtol'] * expected.to(torch.float64).abs()
                require(bool((delta <= allowed).all()), fit['fit_id'] + '/' + label + ': fixed replay tolerance exceeded')
                maximum = max(maximum, float(delta.max()))
            equal(scores['native_per_query_metrics'], eval_mrr(pos, neg), torch, fit['fit_id'] + '/exact_all_per_query_metrics')
            require(scores['native_metrics_rounded4'] == evaluate_mrr(None, pos, neg), 'Rounded native evaluator differs')
            require(scores['native_metrics_rounded4']['MRR'] == state['selected_VALID_MRR'], 'Selected metric differs')
            equal(state['state_after_VALID'], capture(unit, torch, np), torch, fit['fit_id'] + '/exact_full_postserve_state_and_RNG')
            require(progress.snapshot()['Adam_started'] == progress.snapshot()['Adam_completed'] == 0, 'Replay optimizer update observed')
            report = {**fit, 'selected_epoch': state['selected_epoch'], 'status': 'PASS', 'state_restore_tolerance': 0,
                      'full_postserve_state_and_RNG_tolerance': 0, 'all_per_query_and_rounded_metric_tolerance': 0,
                      'scores_atol': tolerance['atol'], 'scores_rtol': tolerance['rtol'], 'observed_max_absolute_score_difference': maximum,
                      'selected_state_sha256': sha(state_path), 'selected_VALID_scores_sha256': sha(score_path), 'optimizer_updates': 0}
            reports.append(report)
            progress.completed('completed_replays', fit['fit_id'])
            write(output / 'REPLAY_PROGRESS.json', {'reports': reports, 'progress': progress.snapshot()})
            handle.remove()
            del unit, state, scores, pos, neg
        require(len(reports) == 6 and progress.snapshot()['complete_VALID_serves'] == 6, 'Complete six selected replays required')
        torch.cuda.synchronize(0)
        stopped.set()
        observer.join()
        sample()
        write(output / 'REPLAY.json', {'schema': 'pubmed-shared4-selected-serialized-replay-v1', 'status': 'PASS', 'UTC': utc(),
              'reports': reports, 'fit_cohort_freeze_sha256': release['fit_cohort_freeze_sha256'],
              'source_manifest_sha256': release['source_manifest_sha256'], 'root_release_sha256': args.release_sha256,
              'inclusive_child_wall_seconds': time.monotonic() - started, 'optimizer_updates': 0, 'progress': progress.snapshot(),
              'rule': cohort['replay_rule'], 'scientific_accuracy_or_novelty_claim': False, 'TEST_supported': False})
    except BaseException as error:
        write(output / 'FAILURE.json', {'status': 'FAILED', 'type': type(error).__name__, 'condition': str(error),
              'progress': progress.snapshot(), 'reports': reports, 'inclusive_child_wall_seconds': time.monotonic() - started, 'automatic_retry': False})
        raise
    finally:
        if stopped is not None:
            stopped.set()
            observer.join()
        write(output / 'FINAL_CUSTODY.json', {'files': inventory(output), 'stage': 'replay_selected', 'completed': (output / 'REPLAY.json').exists()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
