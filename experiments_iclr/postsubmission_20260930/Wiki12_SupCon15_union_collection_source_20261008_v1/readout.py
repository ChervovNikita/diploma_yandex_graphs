"""Reuse Wiki12/Wiki24 arithmetic; add alignment-frozen canonical-loss pairs."""
from pathlib import Path

SEEDS = (6101, 6203, 6307)
CONDITIONS = ('plain', 'alignment_only', 'residual_only', 'combined', 'supcon_eq2')


def configure(gate, pins, output, collection):
    analysis = gate.module(gate.bound(pins['reuse']['legacy_analyse']), '_union15_existing_wiki12_analysis')
    contract = analysis.contracts(pins)
    analysis.SEEDS, analysis.CONDITIONS = SEEDS, CONDITIONS
    # The prospectively matched comparator is S-A. S-P is a root-requested contextual contrast.
    analysis.CONTRASTS = list(analysis.CONTRASTS) + [
        ('SupCon-A', {'supcon_eq2': 1, 'alignment_only': -1}),
        ('SupCon-P', {'supcon_eq2': 1, 'plain': -1})]
    analysis.LIMITS = dict(analysis.LIMITS,
        cohorts='Original same-seed plain for Wiki12 and SupCon-P; original alignment-only for SupCon-A; all frozen before candidate inspection; overlapping cohorts are not additive.',
        history='Original failed owner closure and resource wait remain immutable; Wiki12 result custody is the explicit old10 + never-started2 union.',
        canonical_control='Published canonical SupCon Eq2 adapted to the same two own views, original512 panel and native prediction embedding; no projector or cross-member contrast. This is a class-loss comparison, not proof of specialization.',
        selection='Joint strict-first pooled selection and actual saved stage/epoch/member streams for every sharedM4 cell; selected development is reused.',
        replication='Three optimizer seed blocks on one graph; no node/member independence, unused TEST evidence, novelty, confirmation or general superiority claim.',
        prospective_primary_SupCon_reference='All three original alignment_only states; no combined substitution or favorable anchor selection.')
    base_summary, base_pair = analysis.summary, analysis.pair
    def summary(np, arrays, mask, original):
        row = base_summary(np, arrays, mask, original)
        row['pool_minus_mean_member_accuracy'] = (row['served_accuracy'] - row['mean_member_accuracy']
            if row['served_accuracy'] is not None else None)
        return row
    def pair(np, before, after, cohorts, seed, label, original):
        source = 'plain'
        if label == 'SupCon-A':
            frozen = collection['alignment_cohorts'][str(seed)]
            if not frozen.get('available'):
                return dict(seed=seed, comparison=label, available=False,
                    reason='All original alignment-only cohorts must be frozen before canonical predictions')
            cohorts = original.load(np, Path(output) / 'raw' / ('alignment_cohorts_' + str(seed) + '.npz'), frozen['archive'])
            source = 'alignment_only'
        result = base_pair(np, before, after, cohorts, seed, label, original)
        old, new = original.error_masks(np, before), original.error_masks(np, after)
        for row in result['cohorts']:
            mask = cohorts[row['cohort']]
            gained = ~old['any_member_correct'] & new['any_member_correct']
            lost = old['any_member_correct'] & ~new['any_member_correct']
            row['coverage'] = dict(gained_objects=int(gained[mask].sum()), lost_objects=int(lost[mask].sum()),
                net_objects=int(gained[mask].sum()) - int(lost[mask].sum()),
                definition='At least one original logits.argmax member predicts truth; no oracle serving rule is introduced.')
            common = mask & old['common_competitor']
            row['baseline_common_error_repair'] = dict(eligible_objects=int(common.sum()),
                coverage_acquired=int((common & new['any_member_correct']).sum()),
                served_correct_after=int((common & new['pool_correct']).sum()),
                served_repaired=int((common & ~old['pool_correct'] & new['pool_correct']).sum()),
                introduced_served_errors=int((common & old['pool_correct'] & ~new['pool_correct']).sum()),
                still_all_members_wrong=int((common & new['all_member_wrong']).sum()),
                common_competitor_cleared=int((common & ~new['common_competitor']).sum()))
            for member, change in enumerate(row['member_changes']):
                change['correctness_acquired'] = int((mask & ~old['member_correct'][member] & new['member_correct'][member]).sum())
                change['new_errors_introduced'] = int((mask & old['member_correct'][member] & ~new['member_correct'][member]).sum())
            row['cohort_source_condition'] = source
        result['cohort_source_condition'] = source
        return result
    analysis.summary, analysis.pair = summary, pair
    analysis.METRICS = tuple(analysis.METRICS) + ('pool_minus_mean_member_accuracy',)
    return analysis, contract


def run(gate, pins, output, collection):
    gate.require(collection.get('whole_Wiki12_union_and_SupCon3_closed_before_opening') is True
                 and collection.get('owners_and_all15_children_terminal_before_opening') is True,
                 'Whole union + canonical family custody required')
    cells = collection['cells']
    gate.require(len(cells) == 15 and {(row['seed'], row['condition']) for row in cells}
                 == {(seed, condition) for seed in SEEDS for condition in CONDITIONS}
                 and all(row['family_status'] == 'complete' for row in cells), 'Entire fixed15 roster; no fit exclusion')
    if any(row['condition'] == 'supcon_eq2' and row['collection_status'] == 'complete' for row in cells):
        gate.require(all(collection['alignment_cohorts'][str(seed)].get('available') is True
                         and collection['alignment_cohorts'][str(seed)].get('frozen_before_SupCon_prediction_inspection') is True
                         for seed in SEEDS), 'Entire original alignment3 cohort freeze must precede canonical predictions')
    analysis, contract = configure(gate, pins, output, collection)
    analysis.run(output, collection, pins)
    paired_path = Path(output) / 'compact' / 'PAIRED_ERRORS.json'
    paired = gate.read(paired_path)
    for row in paired['paired']:
        if row['comparison'] == 'SupCon-A' and not row['available']:
            row['reason'] = 'Complete original alignment-only/canonical pair and the entire alignment3 cohort freeze required'
            row['cohort_source_condition'] = 'alignment_only'
    contract.write(paired_path, paired)
    report = Path(output) / 'compact' / 'REPORT.md'
    text = report.read_text().replace('# Wiki12 exploratory component attribution', '# Wiki12 union and canonical SupCon exploratory comparison')
    text = text.replace('plain-frozen overlapping cohorts', 'plain-frozen Wiki12/SupCon-P cohorts and alignment-frozen SupCon-A cohorts')
    report.write_text(text + '\nSupCon-A is the prospective matched loss contrast; SupCon-P is contextual. Original old-owner failure and costs remain retained. A canonical-loss gain is not a sharing, diversity or novelty claim.\n')
