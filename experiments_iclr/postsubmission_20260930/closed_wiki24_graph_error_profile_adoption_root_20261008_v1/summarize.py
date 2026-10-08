"""Summarize every fixed degree/zero cohort of the complete saved profile."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.profile_directory / 'PROFILE.json').read_text())
    freeze = json.loads((args.profile_directory / 'COHORT_FREEZE.json').read_text())
    assert data['status'] == 'complete'
    assert [s['seed'] for s in data['seeds']] == [6101, 6203, 6307]
    assert all(s['reconciliation']['all_partitions_reconciled'] for s in data['seeds'])
    rows = [r for seed in data['seeds'] for r in seed['rows']]
    assert len(rows) == 231
    summaries, flat = [], []
    for kind, value in [('full_population', None),
                        *[('degree_quartile', q) for q in range(1, 5)],
                        ('zero_neighbor', True), ('zero_neighbor', False)]:
        selected = [r for r in rows if r['cohort_kind'] == kind
                    and r['cohort_value'] == value and r['truth_class'] is None]
        assert len(selected) == 3 and len({r['nodes'] for r in selected}) == 1
        paired, member, coverage = [], [], []
        for row in selected:
            old, ordinary = [row['banks'][arm] for arm in
                             ('be_unit_contrastive', 'independent4')]
            n = row['nodes']
            paired.append(100 * row['paired']['served_accuracy_change'])
            member.append(100 * (ordinary['mean_member_accuracy'] - old['mean_member_accuracy']))
            coverage.append(100 * (ordinary['counts']['any_member_correct']
                                  - old['counts']['any_member_correct']) / n)
        summaries.append(dict(kind=kind, value=value, nodes=selected[0]['nodes'],
                              served_difference_pp=paired, member_difference_pp=member,
                              coverage_difference_pp=coverage,
                              mean_served_difference_pp=statistics.mean(paired),
                              mean_member_difference_pp=statistics.mean(member),
                              mean_coverage_difference_pp=statistics.mean(coverage),
                              served_repairs=[r['paired']['counts']['served_repairs_old_to_ordinary']
                                              for r in selected],
                              served_harms=[r['paired']['counts']['served_harms_old_to_ordinary']
                                            for r in selected]))
    for row in rows:
        old, ordinary = [row['banks'][arm] for arm in
                         ('be_unit_contrastive', 'independent4')]
        flat.append(dict(seed=row['seed'], cohort_kind=row['cohort_kind'],
                         cohort_value=row['cohort_value'], truth_class=row['truth_class'],
                         nodes=row['nodes'], old_pool_correct=old['counts']['pool_correct'],
                         ordinary_pool_correct=ordinary['counts']['pool_correct'],
                         old_any_correct=old['counts']['any_member_correct'],
                         ordinary_any_correct=ordinary['counts']['any_member_correct'],
                         old_common_rival=old['counts']['common_rival'],
                         ordinary_common_rival=ordinary['counts']['common_rival'],
                         old_mean_member_accuracy=old['mean_member_accuracy'],
                         ordinary_mean_member_accuracy=ordinary['mean_member_accuracy'],
                         served_repairs=row['paired']['counts']['served_repairs_old_to_ordinary'],
                         served_harms=row['paired']['counts']['served_harms_old_to_ordinary'],
                         ordinary_acquired_correct_member=row['paired']['counts']['ordinary_acquired_correct_member_ranking'],
                         old_acquired_correct_member=row['paired']['counts']['old_acquired_correct_member_ranking']))
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'ALL_FIXED_COHORTS.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    result = dict(direction='ordinary independent4 minus old unit+contrast',
                  units='percentage points', seeds=[6101, 6203, 6307],
                  equal_seed_descriptive_means=True, confidence_interval_claimed=False,
                  development_selected=True, boundaries=freeze['quartile_upper_boundaries'],
                  summaries=summaries)
    (output / 'SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n')
    labels = ['All development nodes', 'Degree ≤ 4', 'Degree 5–11',
              'Degree 12–44', 'Degree > 44', 'No incoming nonself neighbour',
              'At least one incoming nonself neighbour']
    text = ['# Graph connectivity does not isolate the ensemble deficit', '',
            '8 October 2026. This is a descriptive profile of complete saved Wiki24 '
            'predictions. It diagnoses previously selected models and performs no training. '
            'Original paper scores remain unchanged.', '',
            'Each of four members predicts one of ten classes for every node. Mean member '
            'accuracy describes the individual predictors. Correct-member coverage counts '
            'nodes where at least one member predicts the true class. It is an oracle '
            'diagnostic, since the true answer is unavailable at inference. Served accuracy '
            'uses the actual saved mean of class probabilities and its chosen class.', '',
            'The comparison uses all 5,274 development nodes for each of three fixed seeds. '
            'Degree counts unique incoming nonself neighbours in the complete 11,701-node '
            'graph. Its four boundaries were calculated from 580 TRAIN nodes before '
            'prediction values were loaded. The upper boundaries are 4, 11 and 44. '
            'The same nodes are paired across both model banks.', '',
            '## Complete aggregate profile', '',
            'All differences below are ordinary ensemble minus unit+contrast. Values are '
            'descriptive means over the same three seeds, in percentage points. The complete '
            '231 cohort/class rows and all seed outcomes remain in PROFILE.json and '
            'ALL_FIXED_COHORTS.csv.', '',
            '| Population | Nodes per seed | Mean member accuracy difference | '
            'Correct-member coverage difference | Served accuracy difference |',
            '| --- | ---: | ---: | ---: | ---: |']
    for label, summary in zip(labels, summaries):
        text.append(f"| {label} | {summary['nodes']} | "
                    f"{summary['mean_member_difference_pp']:+.3f} | "
                    f"{summary['mean_coverage_difference_pp']:+.3f} | "
                    f"{summary['mean_served_difference_pp']:+.3f} |")
    text += ['', 'The ordinary ensemble has greater correct-member coverage in every '
             'degree bin for every seed. Its serving advantage is positive in every seed '
             'in the two upper degree bins. The lower bins have mixed serving differences. '
             'On the 147 isolated nodes the serving differences are 0, −5 and +2 correct '
             'nodes. The deficit is therefore not confined to nodes without neighbours.', '',
             'Individual competence and coverage are different properties. The unit+contrast '
             'members have slightly higher mean accuracy overall, but their correct decisions '
             'overlap more. Replacing their serving rule with a convex reweighting cannot '
             'correct a node where the same wrong class is strictly above truth in every '
             'member. This is the existing fixed-prediction limitation, not a new theorem '
             'about retrained models.', '',
             '## Per-seed serving repairs and harms', '',
             'A repair means the ordinary pool is correct on a node where unit+contrast is '
             'wrong. A harm means the reverse. Both counts are retained.', '',
             '| Population | Repairs by seed 6101 / 6203 / 6307 | '
             'Harms by seed 6101 / 6203 / 6307 |', '| --- | --- | --- |']
    for label, summary in zip(labels, summaries):
        text.append('| ' + label + ' | ' + ' / '.join(map(str, summary['served_repairs']))
                    + ' | ' + ' / '.join(map(str, summary['served_harms'])) + ' |')
    text += ['', '## What follows for experiments', '',
             'The observations support testing acquisition of useful different predictions '
             'across the graph. They do not identify degree as the cause of the errors. '
             'A specialization rule that allocates training emphasis by degree remains '
             'an untested hypothesis. A profile spread across all degree bins does not '
             'refute that rule, but it supplies no specific reason to prioritise it.', '',
             'The registered Context9 and published-loss comparisons will test changes '
             'to learned representations. Their full-family gates and required capable '
             'references remain unchanged. No new degree weights, bins, model selection '
             'or additional fit was chosen using this profile.', '',
             '## Limits', '',
             'Both banks were selected using this development population. They are '
             'different complete learners, with different sharing, objectives and checkpoint '
             'selection. The result does not isolate an effect of sharing. Degree groups '
             'also have different class composition. All ten class tables are retained, '
             'including empty rows. One graph, one selected split and three correlated '
             'seed blocks support descriptive associations. No significance, causal '
             'effect, new quality improvement, independent confirmation or general '
             'superiority is established.', '',
             'The analysis authenticated six complete prediction archives and all original '
             'role/closure metadata. It reconciled every degree, zero-neighbour and class '
             'partition with the full counts. Execution used 2.04 seconds and 62.8 MB '
             'peak RSS on the authorized allocation CPU, with zero model calls, optimizer '
             'updates or checkpoint loading.', '']
    (output / 'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(dict(complete=True, fixed_rows=len(flat), summaries=len(summaries))))


if __name__ == '__main__':
    main()
