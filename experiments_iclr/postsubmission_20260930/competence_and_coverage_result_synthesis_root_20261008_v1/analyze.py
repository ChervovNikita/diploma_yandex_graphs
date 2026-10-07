"""Derive selected-state decision limits from the complete retained Wiki24 counts."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
P = HERE.parent
INPUT = P / 'Wiki24_selected_analysis_after_closure_execution_root_20261007_v1/predictions/compact/PER_SEED.json'


def main():
    source = json.loads(INPUT.read_text())
    records = []
    for seed in source['seeds']:
        cells = {c['arm']: c for c in seed['cells']}
        selected = {}
        for arm in ('single', 'independent4', 'be_unit', 'be_unit_contrastive', 'be_init', 'be_init_contrastive'):
            cell = cells[arm]
            assert cell['family_status'] == cell['collection_status'] == 'complete'
            population = cell['full_population']
            assert population['nodes'] == 5274
            counts = population['counts']
            assert counts['pool_correct'] == counts['any_member_correct'] - counts['pool_harm'] + counts['pool_rescue']
            selected[arm] = dict(
                objects=population['nodes'],
                pool_correct=counts['pool_correct'],
                mean_member_accuracy=population['mean_member_accuracy'],
                any_member_correct=counts['any_member_correct'],
                all_member_wrong=counts['all_member_wrong'],
                strict_common_rival=counts['common_competitor'],
                common_rival_and_pool_correct=counts['common_competitor_and_pool_correct'],
                pool_harm=counts['pool_harm'], pool_rescue=counts['pool_rescue'],
                served_accuracy=population['served_accuracy'],
                served_nll=population['served_nll'])
        candidate, ordinary = selected['be_unit_contrastive'], selected['independent4']
        records.append(dict(seed=seed['seed'], cells=selected,
            candidate_oracle_member_selection_accuracy=candidate['any_member_correct'] / 5274,
            candidate_exact_arithmetic_convex_pool_accuracy_upper_bound=1-candidate['strict_common_rival'] / 5274,
            candidate_oracle_selection_minus_ordinary_served_correct_nodes=candidate['any_member_correct']-ordinary['pool_correct'],
            candidate_mean_member_minus_ordinary_mean_member=candidate['mean_member_accuracy']-ordinary['mean_member_accuracy'],
            candidate_all_wrong_equal_common_rival=candidate['all_member_wrong']==candidate['strict_common_rival']))
    assert [r['seed'] for r in records] == [6101, 6203, 6307]
    mean = lambda arm, key: sum(r['cells'][arm][key] for r in records) / 3
    candidate = 'be_unit_contrastive'; ordinary = 'independent4'; unit = 'be_unit'
    result = dict(schema='selected-state-competence-coverage-synthesis-v1',
        input=dict(path=str(INPUT.relative_to(P)), sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest()),
        seeds=records,
        equal_seed_means=dict(
            candidate_accuracy=mean(candidate, 'served_accuracy'),
            ordinary_accuracy=mean(ordinary, 'served_accuracy'),
            candidate_member_accuracy=mean(candidate, 'mean_member_accuracy'),
            ordinary_member_accuracy=mean(ordinary, 'mean_member_accuracy'),
            candidate_oracle_member_selection_accuracy=mean(candidate, 'any_member_correct') / 5274,
            candidate_exact_arithmetic_convex_pool_accuracy_upper_bound=1-mean(candidate, 'strict_common_rival') / 5274,
            ordinary_minus_candidate_any_correct_nodes=mean(ordinary, 'any_member_correct')-mean(candidate, 'any_member_correct'),
            ordinary_minus_candidate_pool_harm_nodes=mean(ordinary, 'pool_harm')-mean(candidate, 'pool_harm'),
            candidate_minus_unit_accuracy=mean(candidate, 'served_accuracy')-mean(unit, 'served_accuracy'),
            candidate_minus_unit_member_accuracy=mean(candidate, 'mean_member_accuracy')-mean(unit, 'mean_member_accuracy')),
        all_seeds_oracle_member_selection_below_ordinary_pool=all(r['candidate_oracle_selection_minus_ordinary_served_correct_nodes']<0 for r in records),
        limits=['Complete previously selected development states on one graph/split, not independent TEST evidence.',
                'Oracle selection uses true labels only for an upper bound and is not a usable prediction rule.',
                'Strict common-rival bound applies to convex combinations of existing class probabilities in exact arithmetic.',
                'It does not bound a new classifier, nonlinear hidden-state aggregation or retraining.',
                'Member competence is mean individual accuracy. Coverage is existence of at least one correct member.',
                'Seed-node repetitions and members are not independent experimental units.'],
        new_model_calls=0, new_fits=0, original_paper_scores_changed=False,
        frozen_running_families_changed=False)
    out=HERE/'RESULT.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(result['equal_seed_means'], indent=2))


if __name__ == '__main__':
    main()
