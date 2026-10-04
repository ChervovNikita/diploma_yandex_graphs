#!/usr/bin/env python3
"""Verify owned serialized scientific selected states; no training or reselection."""
import argparse
import json
from pathlib import Path
import time
from common import HERE, EXECUTION, require, sha, gate, write, utc, capture, restore, validation, setup_runtime, inventory, start_monitor


def equal(left, right, torch, label):
    require(type(left) is type(right), label + ': type differs')
    if isinstance(left, torch.Tensor):
        require(left.shape == right.shape and left.dtype == right.dtype and left.layout == right.layout and torch.equal(left, right), label + ': tensor differs')
    elif isinstance(left, dict):
        require(left.keys() == right.keys(), label + ': keys differ')
        for key in left:
            equal(left[key], right[key], torch, label + '/' + str(key))
    elif isinstance(left, (list, tuple)):
        require(len(left) == len(right), label + ': length differs')
        for index, (a, b) in enumerate(zip(left, right)):
            equal(a, b, torch, label + '/' + str(index))
    else:
        require(left == right, label + ': value differs')


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
    progress = {'stage': 'replay_selected', 'optimizer_updates': 0, 'completed_replays': [], 'current_fit': None}
    stopped = observer = sample = None
    reports = []
    try:
        torch, np, bodies, evaluate_mrr, eval_mrr, inspection, x, train, valid, negative, identities = setup_runtime(cohort)
        stopped, observer, peaks, sample = start_monitor(output, cohort['stages']['replay_selected']['caps'], progress, torch)
        write(output / 'AVAILABLE_INSPECTION.json', inspection)
        fit_output = EXECUTION / 'fit_cohort/run01'
        freeze = json.loads((fit_output / 'COHORT_FREEZE.json').read_text())
        require([row['fit_id'] for row in freeze['fits']] == [row['fit_id'] for row in cohort['fits']], 'Complete frozen cohort differs')
        custody = json.loads((fit_output / 'FINAL_CUSTODY.json').read_text())
        require(custody.get('completed') is True, 'Fit custody incomplete')
        for row in custody['files']:
            path = fit_output / row['path']
            require(path.resolve().is_relative_to(fit_output) and not path.is_symlink() and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Owned fit artifact differs')
        for fit, selected in zip(cohort['fits'], freeze['fits']):
            progress['current_fit'] = fit
            directory = fit_output / fit['fit_id']
            state_path = directory / 'selected_state.pt'
            score_path = directory / 'selected_VALID_scores.pt'
            require(sha(state_path) == selected['selected_state_sha256'] and sha(score_path) == selected['selected_VALID_scores_sha256'], 'Selected artifact hashes differ')
            state = torch.load(state_path, map_location='cpu', weights_only=True)
            scores = torch.load(score_path, map_location='cpu', weights_only=True)
            require(state.get('schema') == 'pubmed-native-selected-full-state-v1' and scores.get('schema') == 'pubmed-native-selected-VALID-scores-v1', 'Selected state/score schema differs')
            require(scores.get('native_evaluator_sha256') == cohort['metric']['evaluator_sha256'], 'Native metric provenance differs')
            identity_keys = ('fit_id', 'model', 'seed', 'literal_seed_id', 'selected_epoch', 'selected_VALID_MRR', 'config_sha256', 'source_manifest_sha256')
            for key in identity_keys:
                require(state[key] == scores[key] and (key not in selected or state[key] == selected[key]), 'Selected identity differs: ' + key)
            require(state['input_files'] == scores['input_files'] == identities and state['selected_VALID_scores_sha256'] == sha(score_path), 'Input/score provenance differs')
            require(state['source_manifest_sha256'] == release['source_manifest_sha256'] and state['config_sha256'] == sha(fit_output / 'CONFIG.json'), 'Selected source/config differs')
            require(state['literal_seed_id'] == fit['literal_seed_id'] and state['seed'] == fit['seed'] and state['model'] == fit['model'], 'Literal paired seed/model differs')
            # A fresh reconstruction belongs only to this verifier. It is immediately
            # overwritten by the owned selected state; no fit is continued or reset.
            unit = bodies.candidate_factory(fit['model'], fit['seed'], x, train)
            restore(unit, state['state_after_VALID'], torch, np)
            equal(state['state_after_VALID'], capture(unit, torch, np), torch, fit['fit_id'] + '/exact_serialized_poststate_restore')
            restore(unit, state['state_before_VALID'], torch, np)
            equal(state['state_before_VALID'], capture(unit, torch, np), torch, fit['fit_id'] + '/exact_serialized_prestate_restore')
            pos, neg = validation(bodies, fit['model'], unit, valid, negative, torch)
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
            report = {**fit, 'selected_epoch': state['selected_epoch'], 'status': 'PASS', 'state_restore_tolerance': 0,
                      'full_postserve_state_and_RNG_tolerance': 0, 'all_per_query_and_rounded_metric_tolerance': 0,
                      'scores_atol': tolerance['atol'], 'scores_rtol': tolerance['rtol'], 'observed_max_absolute_score_difference': maximum,
                      'selected_state_sha256': sha(state_path), 'selected_VALID_scores_sha256': sha(score_path), 'optimizer_updates': 0}
            reports.append(report)
            progress['completed_replays'].append(fit['fit_id'])
            write(output / 'REPLAY_PROGRESS.json', {'reports': reports, 'progress': progress})
            del unit, state, scores, pos, neg
        torch.cuda.synchronize(0)
        stopped.set()
        observer.join()
        sample()
        write(output / 'REPLAY.json', {'schema': 'pubmed-native-selected-serialized-replay-v1', 'status': 'PASS', 'UTC': utc(),
              'reports': reports, 'fit_cohort_freeze_sha256': release['fit_cohort_freeze_sha256'],
              'source_manifest_sha256': release['source_manifest_sha256'], 'root_release_sha256': args.release_sha256,
              'inclusive_child_wall_seconds': time.monotonic() - started, 'optimizer_updates': 0,
              'rule': cohort['replay_rule'], 'scientific_accuracy_or_novelty_claim': False})
    except BaseException as error:
        write(output / 'FAILURE.json', {'status': 'FAILED', 'type': type(error).__name__, 'condition': str(error),
              'progress': progress, 'reports': reports, 'inclusive_child_wall_seconds': time.monotonic() - started, 'automatic_retry': False})
        raise
    finally:
        if stopped is not None:
            stopped.set()
            observer.join()
        write(output / 'FINAL_CUSTODY.json', {'files': inventory(output), 'stage': 'replay_selected', 'completed': (output / 'REPLAY.json').exists()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
