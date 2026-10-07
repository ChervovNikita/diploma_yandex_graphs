"""Logical counterexamples only: artificial probabilities, no scientific result."""
import json
from pathlib import Path
import numpy as np
from adapter import all_common_classes,diagnose


def fixture(probabilities):
    probabilities=np.asarray(probabilities,dtype=np.float64)
    assert probabilities.shape==(4,10) and (probabilities>0).all()
    assert np.allclose(probabilities.sum(axis=1),1.,atol=1e-12,rtol=1e-12)
    return dict(valid_ids=np.array([314],dtype=np.int64),truth=np.array([0],dtype=np.int64),
        member_logits=np.log(probabilities)[:,None,:],
        member_prediction=probabilities.argmax(axis=1)[:,None],
        pool_prediction=np.array([probabilities.mean(axis=0).argmax()],dtype=np.int64))


def main():
    records=[]
    def check(name,baseline,candidate,expected):
        before=fixture(baseline);after=fixture(candidate)
        result=diagnose(np,before,after,all_common_classes(np,before['member_logits'],before['truth']))
        row=result['COMMON_obstruction_population']
        for key,value in expected.items():
            assert row[key]==value,(name,key,row[key],value)
        records.append(dict(name=name,artificial_COMMON_probabilities=baseline,
            artificial_candidate_probabilities=candidate,expected=expected,diagnostic=result,
            scientific_result=False))
    base_one=[.2,.4,.12]+[.04]*7
    base_two=[.08,.36,.36,.02]+[.03]*6
    new_rival=[.25,.10,.44]+[.03]*7
    check('all_old_rivals_beaten_but_new_rival_blocks',[base_one]*4,[new_rival]*4,
        dict(one_same_member_strictly_beats_ALL_old_classes=1,
            same_member_ALL_but_no_candidate_member_correct=1,
            same_member_ALL_with_a_new_common_competitor=1,served_repairs=0))
    first=[.32,.40,.08]+[.2/7]*7
    second=[.32,.08,.40]+[.2/7]*7
    check('different_members_repair_pool_without_one_member_beating_all',
        [base_two]*4,[first,second,first,second],
        dict(one_same_member_strictly_beats_ALL_old_classes=0,
            every_old_class_reversed_by_some_potentially_different_member=1,
            any_candidate_member_actually_correct=0,served_repairs=1,
            served_repair_with_no_candidate_member_correct=1))
    first_weak=[.25,.60,.02]+[.13/7]*7
    second_weak=[.25,.02,.60]+[.13/7]*7
    check('per_class_reversal_opportunity_is_insufficient_for_pool_repair',
        [base_two]*4,[first_weak,second_weak,first_weak,second_weak],
        dict(one_same_member_strictly_beats_ALL_old_classes=0,
            every_old_class_reversed_by_some_potentially_different_member=1,served_repairs=0))
    tied=[.3,.3]+[.05]*8
    check('native_argmax_tie_correctness_is_not_strict_reversal',[base_one]*4,[tied]*4,
        dict(one_same_member_strictly_beats_ALL_old_classes=0,
            every_old_class_reversed_by_some_potentially_different_member=0,
            any_candidate_member_actually_correct=1,served_repairs=1))
    result=dict(schema='synthetic_all_competitor_logical_counterexamples-v1',cases=records,
        synthetic_only=True,scientific_results=0,model_or_data_access=False,
        model_forward_calls=0,training_updates=0,
        purpose='Validate quantifier order, insufficiency/new rivals, different-member pool rescue and tie semantics; not an experiment or evidence of method quality.')
    (Path(__file__).parent/'SYNTHETIC_COUNTEREXAMPLES.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(logical_counterexamples_passed=len(records),scientific_results=0)))


if __name__=='__main__':
    main()
