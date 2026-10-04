"""Read the entire frozen J/F pair only after authenticated normal closure.

Stdlib only; this program does not load models, queries or datasets. It reports
all planned diagnostics and no inferential interval from the single seed.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import time


REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
EXECUTION = PHASE / 'graph_ncNC_structural_pattern_execution_root_20261004_v5'
HERE = Path(__file__).resolve().parent
DRIVER_SHA = '9fc539b8f92d7d4e883224c3b0aa85ae583b64c70f2a701a4a648fd818aa32a1'
FAMILY = 'ncnc-structural-pattern-pair-20261003-v1'
ROUTES = ('own', 'crossed_cyclic_1', 'crossed_cyclic_2', 'crossed_cyclic_3', 'pooled_clamped_weights')
STRATA = ('all', 'has_synthetic_removal', 'no_synthetic_removal', 'cn0', 'has_common')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def confined(path, root):
    path = Path(path)
    require(path.is_absolute() and path.resolve().is_relative_to(root), 'Path outside admitted root')
    require(path.resolve() == path and not path.is_symlink(), 'Canonical nonsymlink path required')
    return path


def read_pin(pin, root, expected=None):
    path = confined(pin['path'], root)
    if expected is not None:
        require(path == expected, 'Unexpected fixed artifact path')
    raw = path.read_bytes()
    require(len(raw) <= 8_000_000 and len(raw) == pin['bytes'] and sha(raw) == pin['sha256'],
            'Pinned artifact bytes differ')
    return decode(raw)


def source_custody(manifest_raw):
    for pin in decode(manifest_raw)['files']:
        path = HERE / pin['path']
        require(path.resolve().parent == HERE and not path.is_symlink(), 'Reader source path escaped')
        raw = path.read_bytes()
        require(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'Reader source payload changed')


def finite_tree(value):
    if type(value) is float:
        require(math.isfinite(value), 'Nonfinite result value')
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def rate(value):
    require(type(value) in (float, int) and 0 <= value <= 1, 'Invalid Hits50/fraction')
    return value


def mean_or_none(total, count):
    return total / count if count else None


def aggregate(batches, label):
    rows = [b[label] for b in batches]
    require(all(r['queries'] == 65536 and set(r['strata']) == set(STRATA) for r in rows),
            'Full native diagnostic query/stratum coverage differs')
    sums = {name: sum(r[name] for r in rows) for name in
            ('queries', 'active_queries', 'residual_slots', 'synthetic_removed_observed_slots',
             'source_unobserved_slots', 'component_entropy_query_sum',
             'between_component_spread_query_sum', 'active_responsibility_entropy_sum',
             'active_responsibility_max_sum')}
    require(sums['queries'] == 1114112 and 0 <= sums['active_queries'] <= sums['queries']
            and sums['synthetic_removed_observed_slots'] + sums['source_unobserved_slots'] == sums['residual_slots'],
            'Diagnostic denominator/observed-slot accounting differs')
    strata = {}
    for name in STRATA:
        count = sum(r['strata'][name]['queries'] for r in rows)
        strata[name] = {'queries': count, **{
            key.removesuffix('_query_sum'): mean_or_none(sum(r['strata'][name][key] for r in rows), count)
            for key in ('joint_query_sum', 'factorial_query_sum',
                        'joint_minus_factorial_query_sum', 'between_component_spread_query_sum')}}
    require(strata['all']['queries'] == sums['queries'], 'All-query denominator differs')
    bits = {}
    for bit in ('0', '1'):
        count = sum(r['bit_marginal'][bit]['slots'] for r in rows)
        bits[bit] = {'slots': count, **{
            key.removesuffix('_sum'): mean_or_none(sum(r['bit_marginal'][bit][key] for r in rows), count)
            for key in ('nll_sum', 'brier_sum', 'predicted_observation_probability_sum')}}
    require(bits['0']['slots'] + bits['1']['slots'] == sums['residual_slots'], 'Bit denominator differs')
    return {'totals': sums, 'strata': strata, 'bit_marginal': bits,
            'mean_component_entropy_per_query_including_empty': mean_or_none(sums['component_entropy_query_sum'], sums['queries']),
            'mean_component_spread_per_query_including_empty': mean_or_none(sums['between_component_spread_query_sum'], sums['queries']),
            'mean_responsibility_entropy_per_active_query': mean_or_none(sums['active_responsibility_entropy_sum'], sums['active_queries']),
            'mean_max_responsibility_per_active_query': mean_or_none(sums['active_responsibility_max_sum'], sums['active_queries']),
            'normalized_native_q_vs_stable_sigmoid_max_abs': max(r['normalized_native_q_vs_stable_sigmoid_max_abs'] for r in rows)}


def assemble(pair, closure, terminal):
    finite_tree(pair); finite_tree(closure); finite_tree(terminal)
    require(pair['schema'] == 'ncnc-pattern-pair-results-v1' and pair['identity'] == closure['identity'],
            'Entire pair identity/schema differs')
    require(set(pair['selections']) == set(pair['representation_diagnostics']) == {'J', 'F'}
            and pair['all_100_native_streams_RNG_and_actual_supports_match'] is True
            and pair['test_file_opened'] is False, 'Incomplete/unmatched pair or TEST access')
    require(pair['scope'] == 'single_seed_validation_selected_development_pilot_not_confirmatory'
            and pair['source_pattern_scope'] == 'TRAIN_observation_incidence_not_latent_link_truth'
            and pair['stronger_claim_requires_covariance_aware_competent_single'] is True,
            'Scientific scope differs')
    summaries = {}
    for arm in ('J', 'F'):
        selection = pair['selections'][arm]; detail = pair['representation_diagnostics'][arm]
        valid = detail['VALID']; batches = detail['mask_event_batches']
        require(type(selection['order']) is int and 1 <= selection['order'] <= 100
                and detail['selected_epoch'] == selection['order']
                and valid['served_hits50'] == selection['hits50'], 'Selection/replay differs')
        rate(selection['hits50'])
        require(len(batches) == 17 and len(valid['member_hits50']) == 4
                and len(valid['positive_coincident_error_fractions']) == 6
                and len(valid['member_balanced_BCE']) == 4
                and valid['positive_queries'] == 60084 and valid['negative_queries'] == 100000
                and set(valid['fixed_bank_counterfactual_routes']) == set(ROUTES)
                and detail['new_optimization_or_selection'] is False
                and detail['auxiliary_scorer_backward'] is False
                and valid['not_a_selector'] is True and valid['graph'] == 'complete_TRAIN_only'
                and valid['serving_pool'] == 'mean_raw_logits', 'Fixed diagnostic coverage differs')
        require(detail['mask_seed'] == 2026100307 and len(detail['support_digests']) == 34,
                'Diagnostic seed/positive-negative support coverage differs')
        require(valid['fixed_bank_counterfactual_routes']['own']['served_hits50'] == valid['served_hits50'],
                'Own route differs from primary served score')
        for value in valid['member_hits50'] + valid['positive_coincident_error_fractions'] + [valid['positive_oracle_union']]:
            rate(value)
        for route in ROUTES:
            rate(valid['fixed_bank_counterfactual_routes'][route]['served_hits50'])
        require(all(len(b['native_member_main_loss']) == 4 for b in batches), 'Missing member losses')
        summaries[arm] = {'selection': selection, 'VALID': valid,
                          'positive_mask_events': aggregate(batches, 'positive'),
                          'negative_mask_events': aggregate(batches, 'negative'),
                          'mean_TRAIN_mask_member_main_loss': [sum(b['native_member_main_loss'][m] for b in batches) / 17 for m in range(4)],
                          'source_pattern_mask_wall_seconds': detail['source_pattern_mask_wall_seconds']}
    require(pair['representation_diagnostics']['J']['TRAIN_stream'] == pair['representation_diagnostics']['F']['TRAIN_stream']
            and pair['representation_diagnostics']['J']['support_digests'] == pair['representation_diagnostics']['F']['support_digests']
            and pair['representation_diagnostics']['J']['source_teacher'] == pair['representation_diagnostics']['F']['source_teacher'],
            'Diagnostic realized streams/supports differ')
    delta = summaries['J']['selection']['hits50'] - summaries['F']['selection']['hits50']
    require(delta == pair['selected_served_VALID_Hits50_J_minus_F'], 'Primary arithmetic differs')
    return {'schema': 'ncnc-complete-pair-reader-report-v1', 'identity': pair['identity'],
            'arms': summaries, 'primary_J_minus_F_Hits50_pp': 100 * delta,
            'primary_interpretation': 'Positive single-pair development screen; prospective replication and heldout confirmation still required' if delta > 0 else
                'No quality support for this representative fixed J/F setting; mechanism diagnostics do not rescue the endpoint',
            'closed_pair_raw_results_retained': True, 'all_planned_diagnostics_retained': True,
            'original_paper_scores_changed': False, 'TEST_opened': False,
            'seed_count': 1, 'confidence_intervals_or_significance_claims': False,
            'pattern_targets_are_TRAIN_observation_not_latent_truth': True,
            'preclosure_costs': closure['preclosure_bound_stage_accounting'],
            'closure_accounting': closure['inclusive_accounting'],
            'normal_supervision': terminal,
            'cost_scope': closure['cost_scope'],
            'unknown_or_unlisted_costs_not_certified': True,
            'nested_driver_and_supervisor_wall_times_not_additive': True,
            'manuscript_acceptance_or_general_superiority_established': False}


def render(report):
    lines = ['# Complete fixed J/F development pair', '',
             'The experiment tests whether learning a whole observed connection pattern helps link ranking beyond learning each connection separately. Architecture, training labels, native masks and main loss are matched.', '',
             'One seed and one validation-selected time split provide a screening result. These results are not heldout confirmation or a comparison proving superiority over independent ensembles.', '',
             '| Arm | Selected epoch | Validation Hits@50 (%) |', '| --- | ---: | ---: |']
    for arm in ('J', 'F'):
        s = report['arms'][arm]['selection']
        lines.append(f"| {arm} | {s['order']} | {100 * s['hits50']:.4f} |")
    lines.extend(['', f"Joint minus individual-incidence supervision: **{report['primary_J_minus_F_Hits50_pp']:+.4f} percentage points**.",
                  report['primary_interpretation'], '', '## Fixed routing diagnostics', '',
                  'Crossed routes keep recipient features and decoders fixed while giving them another member’s completion weights. Pooled weights average the clamped completion bank before each recipient decoder. Every route uses mean raw target logits. These are fixed-bank diagnostics, not newly trained methods or checkpoint selectors.', '',
                  '| Route | J Hits@50 (%) | F Hits@50 (%) |', '| --- | ---: | ---: | ---: |'])
    for route in ROUTES:
        vals = [report['arms'][a]['VALID']['fixed_bank_counterfactual_routes'][route]['served_hits50'] * 100 for a in ('J', 'F')]
        lines.append(f'| {route} | {vals[0]:.4f} | {vals[1]:.4f} |')
    lines.extend(['', '## Pattern and member evidence', '',
                  'ANALYSIS.json retains every fixed stratum, marginal observation NLL/Brier statistic, normalized joint/factorial pattern loss, component spread and responsibility statistic, and every member score/error-overlap diagnostic. Denominators are explicit. No mask events or queries are treated as independent graphs.', '',
                  'The labels describe connections observed in TRAIN. Zero means unobserved in TRAIN. New mask events reuse that source teacher and do not establish latent-link posterior accuracy. Target BCE is measured on the fixed balanced validation query pool and does not certify population calibration.', '',
                  'Better joint reconstruction alone is insufficient. A dependence interpretation also needs useful target quality, matched marginal competence, capable structural-single controls and prospective replication.', '',
                  '## Work and interpretation limits', '',
                  'Both scientific fits completed 100 epochs and 1,700 optimizer updates, with 100 validation checkpoint candidates each. Together, the fitted selections, replays and diagnostics account for 206 full validation traversals. The machine-readable report retains bound stage accounting, closure accounting and normal-supervision envelopes. Nested driver/supervisor durations overlap and are not summed. Unlisted interruptions and terminal-write tails remain outside certified costs.', '',
                  'There is no confidence interval from this one seed, no heldout evaluation, no new state-of-the-art claim and no manuscript acceptance claim.', ''])
    return '\n'.join(lines)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True)
    args = parser.parse_args(); started = time.monotonic()
    release_path = confined(args.root_release, EXECUTION)
    release_raw = release_path.read_bytes(); release = decode(release_raw)
    require(release['schema'] == 'ncnc-complete-pair-reader-root-release-v1' and release['execution_enabled'] is True,
            'Explicit post-closure root release required')
    require(Path.cwd() == REPO and socket.gethostname() == 'peptide', 'Authorized repository/host required')
    manifest_raw = (HERE / 'MANIFEST.json').read_bytes()
    require(sha(manifest_raw) == release['reader_manifest_sha256'], 'Reader source manifest differs')
    source_custody(manifest_raw)
    terminal = read_pin(release['closure_normal_terminal'], EXECUTION,
                        EXECUTION / 'supervision/close/run01/SUPERVISOR_TERMINAL.json')
    require(terminal['schema'] == 'ncnc-pattern-normal-diagnostics-closure-supervision-terminal-v1'
            and terminal['status'] == 'COMPLETE' and terminal['stage'] == 'close'
            and terminal['child_exit_code'] == 0 and terminal['child_signal'] is None
            and terminal['cap_violation'] is None and terminal['complete_stage_custody_checks_passed'] is True
            and terminal['complete_driver_peak_capture_checks_passed'] is True and terminal['TEST_opened'] is False,
            'Normal complete closure is missing')
    require(terminal['closure_receipt'] == release['closure_receipt'], 'Normal-terminal closure binding differs')
    closure = read_pin(release['closure_receipt'], EXECUTION, EXECUTION / 'close/run01/CLOSURE.json')
    require(closure['schema'] == 'ncnc-pattern-pair-closure-v1' and closure['status'] == 'CLOSED'
            and closure['identity']['driver_manifest_sha256'] == DRIVER_SHA and closure['identity']['family_id'] == FAMILY
            and closure['unique_scientific_fits'] == 2 and closure['scientific_optimizer_steps'] == 3400
            and closure['scientific_selector_candidates'] == 200
            and closure['complete_scientific_VALID_evaluations_including_replay_and_diagnostics'] == 206
            and closure['matched_pair_streams_RNG_and_supports'] is True and closure['TEST_supported'] is False,
            'Fixed complete pair closure differs')
    pair_pin = dict(closure['pair_results'], path=str(EXECUTION / 'close/run01/PAIR_RESULTS.json'))
    require(closure['pair_results']['path'] == 'PAIR_RESULTS.json'
            and terminal['owned_artifact_receipts']['PAIR_RESULTS.json'] == pair_pin,
            'Entire results payload binding differs')
    pair = read_pin(pair_pin, EXECUTION, EXECUTION / 'close/run01/PAIR_RESULTS.json')
    report = assemble(pair, closure, terminal)
    # Recheck all pinned bytes after assembly and before publishing any score.
    require(release_path.read_bytes() == release_raw, 'Root release changed')
    require((HERE / 'MANIFEST.json').read_bytes() == manifest_raw, 'Reader manifest changed')
    source_custody(manifest_raw)
    for pin in (release['closure_normal_terminal'], release['closure_receipt'], pair_pin):
        read_pin(pin, EXECUTION)
    output = confined(release['output_directory'], EXECUTION)
    require(output == EXECUTION / 'complete_pair_analysis/run01', 'Unexpected analysis output')
    output.mkdir(parents=True, exist_ok=False)
    report.update(UTC=datetime.now(timezone.utc).isoformat(), root_release_sha256=sha(release_raw),
                  reader_manifest_sha256=sha(manifest_raw), reader_observed_wall_seconds=time.monotonic() - started,
                  input_receipts=[release['closure_normal_terminal'], release['closure_receipt'], pair_pin],
                  final_write_tail_measured=False)
    for name, value in [('ANALYSIS.json', json.dumps(report, indent=2, allow_nan=False) + '\n'),
                        ('REPORT.md', render(report)), ('COMPLETE_PAIR_RESULTS.json', json.dumps(pair, indent=2, allow_nan=False) + '\n')]:
        with (output / name).open('x') as handle:
            handle.write(value)
    print('COMPLETE_PAIR_ANALYSIS_WRITTEN ' + str(output), flush=True)


if __name__ == '__main__':
    main()
