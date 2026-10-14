"""Meaningful CPU-only identity/pairing checks; no Torch or model execution."""
import json
import random
from .planning import authenticate, contrast_plan, match_endpoint_support, query_keys, raw_responsibilities


def run():
    before = random.getstate()
    edges = [(1,2), (1,3), (1,4), (5,6), (2,4), (7,8)]
    labels = [1,1,0,0,0,1]
    keys = query_keys(edges)
    assert keys == query_keys([(v,u) for u,v in edges])
    signature = authenticate(keys, labels, 'TRAIN')
    assert signature != authenticate(list(reversed(keys)), list(reversed(labels)), 'TRAIN')
    for role in ('VALID', 'TEST'):
        try:
            authenticate(keys, labels, role)
        except ValueError:
            pass
        else:
            raise AssertionError('Non-TRAIN objective rows were admitted')
    for mode in ('endpoint', 'global', 'cosine'):
        plan = contrast_plan(keys, labels, 123, mode, 6, 2)
        assert plan == contrast_plan(keys, labels, 123, mode, 6, 2)
        assert len(plan['anchors']) <= 6
        for anchor, alternatives in zip(plan['anchors'], plan['alternatives']):
            assert len(alternatives) <= 2 and len(alternatives) == len(set(alternatives))
            for other in alternatives:
                assert keys[anchor] != keys[other] and labels[anchor] != labels[other]
                if mode == 'endpoint':
                    assert set(keys[anchor][:2]) & set(keys[other][:2])
        if mode == 'endpoint':
            assert 3 not in plan['anchors'] and 5 not in plan['anchors']
            assert 1 not in plan['alternatives'][plan['anchors'].index(0)]
            assert plan['new_graph_or_query_pairs_created'] == 0
    # Contradictory labels on the same semantic pair cannot become negatives.
    duplicate_keys = query_keys([(1,2), (2,1)])
    assert contrast_plan(duplicate_keys, [1,0], 123)['anchors'] == []
    relation_keys = query_keys([(1,2), (1,3)], [0,1])
    assert contrast_plan(relation_keys, [1,0], 123)['anchors'] == []
    reference = contrast_plan(keys, labels, 123, 'endpoint', 6, 2)
    for mode in ('cosine', 'global'):
        control = match_endpoint_support(reference, contrast_plan(keys, labels, 123, mode, 6, 2))
        assert control['anchors'] == reference['anchors']
        if mode == 'global':
            assert list(map(len, control['alternatives'])) == list(map(len, reference['alternatives']))
    for mode in ('endpoint', 'query'):
        weights = raw_responsibilities(keys, 123, mode)
        assert weights == raw_responsibilities(keys, 123, mode)
        assert all(value > .25 for row in weights for value in row)
        assert len({tuple(row) for row in weights}) == 4
    weights = raw_responsibilities(duplicate_keys, 123, 'endpoint')
    assert all(row[0] == row[1] for row in weights)
    assert random.getstate() == before
    return {'query_canonicalization': 'passed', 'alignment_order_authentication': 'passed',
        'TRAIN_role_enforcement': 'passed', 'bounded_existing_query_alternatives': 'passed',
        'same_query_and_relation_collision_exclusion': 'passed',
        'no_all_observed_edges_positive_class': 'passed', 'zero_coverage_disclosed': 'passed',
        'control_anchor_coverage_and_negative_count_matching': 'passed',
        'fixed_positive_endpoint_and_IID_weights': 'passed', 'global_Python_RNG_unchanged': True,
        'Torch_imported': False, 'model_or_fit_execution': False}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
