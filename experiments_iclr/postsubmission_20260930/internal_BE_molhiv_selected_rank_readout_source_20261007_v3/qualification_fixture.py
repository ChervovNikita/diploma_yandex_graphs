"""Synthetic source fixture only; no release, roles, checkpoints, serving or scientific qualification.

Root may execute separately after authorizing the intended numerical environment.
This packet preparation only parses the fixture; no fixture execution is implied.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile


def main():
    import numpy as np
    from protocol import paired
    from rank_repair import (PAIR_ID_SCHEMA, competence, freeze_baseline, load_raw,
                             ordered_pair_identity, probability_diagnostics, transitions, validate_cohort)
    checks = []

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def close(a, b):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

    def raw(labels, members, ids=None):
        member = np.asarray(members, dtype=np.float32)
        return {'ids': np.asarray(ids if ids is not None else range(len(labels)), dtype=np.int64),
                'labels': np.asarray(labels, dtype=np.float32), 'member_logits': member,
                'pool_logits': member.mean(0, dtype=np.float32)}

    tied = probability_diagnostics(np.asarray([0, 0], np.float32), np.asarray([1, 0], np.float32), np)
    check(close(tied['average_precision'], .5) and close(tied['PR_AUC_trapezoidal'], .75), 'AP and trapezoidal PR differ under one tied score group')
    check(close(tied['Brier'], .25) and close(tied['BCE'], math.log(2)) and tied['ECE_equal_width_10'] == 0, 'balanced tied Brier/BCE/reliability')
    check(tied['tied_score_groups'] == 1 and tied['distinct_raw_logit_score_groups'] == 1, 'PR ties grouped together')
    rare = probability_diagnostics(np.zeros(4, np.float32), np.asarray([1, 0, 0, 0], np.float32), np)
    check(close(rare['average_precision'], .25) and close(rare['PR_AUC_trapezoidal'], .625), 'prevalence-specific AP and labeled trapezoid convention')
    check(close(rare['ECE_equal_width_10'], .25) and close(rare['mean_probability_minus_prevalence'], .25), 'fixed probability bins use positive-event frequencies')
    extreme = probability_diagnostics(np.asarray([1000, -1000], np.float32), np.asarray([1, 0], np.float32), np)
    check(extreme['Brier'] == 0 and extreme['BCE'] == 0 and extreme['reliability_equal_width_10'][-1]['count'] == 1, 'stable sigmoid, finite extreme scores and inclusive final bin')
    perfect = probability_diagnostics(np.asarray([2, -2], np.float32), np.asarray([1, 0], np.float32), np)
    reversed_rank = probability_diagnostics(np.asarray([-2, 2], np.float32), np.asarray([1, 0], np.float32), np)
    check(perfect['average_precision'] == 1 and perfect['PR_AUC_trapezoidal'] == 1 and close(reversed_rank['PR_AUC_trapezoidal'], .25), 'perfect and inverted grouped PR')

    with tempfile.TemporaryDirectory(prefix='mol18_v3_synthetic_fixture_',
                                     dir=Path(__file__).resolve().parent.parent) as folder:
        folder = Path(folder)
        base = raw([1, 1, 0, 0], [[0, 0, 1, 1]] * 4, ids=[41, 31, 29, 17])
        receipt = freeze_baseline(base, folder / 'O.npz')
        with np.load(folder / 'O.npz', allow_pickle=False) as archive:
            cohort = {key: archive[key].copy() for key in archive.files}
        manual = hashlib.sha256((PAIR_ID_SCHEMA + '\0').encode('ascii'))
        manual.update(struct.pack('<qq', 2, 2))
        for positive in (41, 31):
            for negative in (29, 17):
                manual.update(struct.pack('<qq', positive, negative))
        check(receipt['ordered_pair_identity_sha256'] == manual.hexdigest(), 'ordered pair hash uses positive-major original-ID pairs')
        check(receipt['pairs'] == 4 and receipt['O_same_pair_strictly_inverted_by_all_members'] == 4, 'O common-inversion mask excludes ties')
        candidate = raw([1, 1, 0, 0], [[3, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], ids=[41, 31, 29, 17])
        result = transitions(base, candidate, cohort, np)
        attribution = result['O_common_inversion_attribution']
        check(attribution['strict_coverage_acquired_count'] == 2 and attribution['strict_coverage_acquired_but_pool_tie'] == 2,
              'strict acquired coverage can yield only pool ties')
        check(attribution['strict_coverage_acquired_not_fully_served'] == 2 and attribution['no_strict_coverage_pool_loss'] == 2,
              'pooling loss and partial tie credit separated')
        check(close(result['all_pairs']['net_AUC_change_within_cohort'], .25) and
              close(result['O_common_inversion']['net_AUC_contribution_full_population'], .25), 'net credit with exact half-tie and global denominator')
        check(result['member_pair_order_comparison']['members'][0]['strict_loss_to_win'] == 2 and
              result['member_pair_order_comparison']['members'][0]['strict_loss_to_tie'] == 2, 'same-slot acquired member pair ranks recorded')
        bad = {key: value.copy() for key, value in cohort.items()}
        bad['positive_ids'] = bad['positive_ids'][::-1]
        try:
            validate_cohort(base, bad, np)
        except ValueError:
            check(True, 'changed O pair identity rejected')
        else:
            check(False, 'changed O pair identity rejected')

        # No member changes ordering: a relative positive scale nevertheless repairs a pool disagreement.
        mixed = raw([1, 0], [[2, 0], [0, 1], [0, 1], [0, 1]])
        scaled = raw([1, 0], [[4, 0], [0, 1], [0, 1], [0, 1]])
        freeze_baseline(mixed, folder / 'mixed.npz')
        with np.load(folder / 'mixed.npz', allow_pickle=False) as archive:
            mixed_cohort = {key: archive[key].copy() for key in archive.files}
        scale_result = transitions(mixed, scaled, mixed_cohort, np)
        check(scale_result['member_pair_order_comparison']['all_same_slot_full_pair_orders_preserved'] and
              scale_result['all_pairs']['O_to_I']['loss_to_win'] == 1, 'relative scale repair distinguished from acquired member ordering')
        check(scale_result['O_common_inversion_attribution']['pairs'] == 0 and
              scale_result['O_common_inversion_attribution']['strict_coverage_acquired_count_rate_within_O_cohort'] is None,
              'empty fixed O cohort is unavailable, not zero-rate evidence')
        check(close(competence(candidate, np)['pool_pair_AUC'] - competence(base, np)['pool_pair_AUC'], .25), 'full pair competence agrees with transition identity')

        # Raw archive validator is exercised using a wholly artificial full-sized population.
        complete = raw([1] + [0] * 4112, np.zeros((4, 4113), np.float32))
        np.savez(folder / 'complete.npz', **complete)
        check(load_raw(folder / 'complete.npz', np)['ids'].shape == (4113,), 'complete synthetic role shape accepted')
        complete['member_logits'][0, 0] = np.nan
        np.savez(folder / 'nonfinite.npz', **complete)
        try:
            load_raw(folder / 'nonfinite.npz', np)
        except ValueError:
            check(True, 'nonfinite prediction population rejected without sampling')
        else:
            check(False, 'nonfinite prediction population rejected without sampling')
    check(paired([0., 0., 0.])['exploratory_paired_t95_df2'] == [0., 0.] and
          not paired([0., None, 0.])['available'], 'seed-only three-pair summary, missing pair unavailable')
    print(json.dumps({'schema': 'mol18-v3-synthetic-source-fixture-v1', 'checks': checks,
                      'scientific_runtime_qualified': False, 'serving_calls': 0,
                      'roles_checkpoints_or_live_outputs_accessed': False}, indent=2))


if __name__ == '__main__':
    main()
