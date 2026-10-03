"""Independent scalar references and hostile closure cases, no real study data."""
import copy
import math
import torch
import analysis


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def log_normalizer(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def main():
    torch.set_num_threads(1)
    checks = []
    symmetric = torch.tensor([[[2.0, 0.0]], [[-2.0, 0.0]]], dtype=torch.float64)
    target = torch.tensor([0])
    first = analysis.decomposition(torch, symmetric, target)
    check(abs(first['pool_NLL_FP64'] - math.log(2)) < 1e-13, 'Independent binary pooled CE')
    check(abs(first['mean_ambiguity'] - math.log(math.cosh(1))) < 1e-13,
          'Independent binary ambiguity')
    second = analysis.decomposition(torch, symmetric, torch.tensor([1]))
    check(first['mean_ambiguity'] == second['mean_ambiguity'], 'Ambiguity must be label-independent')
    scaled = analysis.decomposition(torch, 2 * symmetric, target)
    check(scaled['pool_NLL_FP64'] == first['pool_NLL_FP64']
          and scaled['mean_ambiguity'] > first['mean_ambiguity'], 'Separation can preserve pooled predictor')
    checks.append('binary_scalar_identity_and_label_independence; centered_scale_counterexample')

    values = [
        [[1, 2, -1, 0], [0, 1, 3, -2], [2, -1, 0, 1]],
        [[2, -1, 1, 0], [1, 0, -1, 2], [-1, 0, 2, 1]],
        [[0, 1, 2, -1], [-1, 3, 0, 2], [1, 2, -1, 0]],
        [[-1, 0, 1, 2], [2, -1, 1, 0], [0, 1, 3, -2]],
    ]
    labels = [0, 2, 3]
    tensor = torch.tensor(values, dtype=torch.float64)
    measured = analysis.decomposition(torch, tensor, torch.tensor(labels))
    member_ce = sum(log_normalizer(values[m][n]) - values[m][n][labels[n]]
                    for m in range(4) for n in range(3)) / 12
    pooled_values = [[sum(values[m][n][c] for m in range(4)) / 4 for c in range(4)] for n in range(3)]
    pooled_ce = sum(log_normalizer(pooled_values[n]) - pooled_values[n][labels[n]] for n in range(3)) / 3
    check(abs(measured['mean_member_NLL'] - member_ce) < 1e-13
          and abs(measured['pool_NLL_FP64'] - pooled_ce) < 1e-13, 'Independent multiclass scalar CE')
    single = analysis.decomposition(torch, tensor[:1], torch.tensor(labels))
    check(single['mean_ambiguity'] == 0 and single['mean_KL_pool_to_member'] == 0, 'Native M1 zero ambiguity')
    checks.append('four_member_multiclass_python_scalar_reference; native_one_member_identity')

    # Each member picks a different false class; the common second choice wins
    # after pooling. Member-oracle coverage is not a pooling upper bound.
    rescue = torch.tensor([[[0.9, 1.0, -1.0]], [[0.9, -1.0, 1.0]]])
    structure = analysis.error_structure(torch, rescue, torch.tensor([0]))
    check(structure['every_member_wrong_fraction'] == 1
          and structure['any_member_correct_fraction'] == 0
          and structure['pooled_error_fraction'] == 0, 'Pooling can beat member correctness coverage')
    checks.append('different_member_errors_can_pool_to_correct_class')

    valid = dict(rows=[dict(seed=seed, arm=arm, status='selected', selected_state_replay=True,
                           final_labels_closed=True) for seed in analysis.SEEDS for arm in analysis.ARMS],
                 summary=dict(status='complete_development_summary'), final_labels_closed=True,
                 original_inputs_verified_unchanged=True,
                 admission=dict(study_freeze_sha256=analysis.FROZEN_SHA))
    check(len(analysis.closed_family(valid)) == 35, 'Complete family admitted')
    partial = copy.deepcopy(valid); partial['rows'].pop()
    failed = copy.deepcopy(valid); failed['rows'][0]['status'] = 'failed'
    duplicate = copy.deepcopy(valid); duplicate['rows'][0] = duplicate['rows'][1]
    opened = copy.deepcopy(valid); opened['final_labels_closed'] = False
    changed = copy.deepcopy(valid); changed['admission']['study_freeze_sha256'] = 'wrong'
    for case in (partial, failed, duplicate, opened, changed):
        try:
            analysis.closed_family(case)
        except ValueError:
            pass
        else:
            raise AssertionError('Ineligible family accepted')
    analysis.selection_check([dict(epoch=1, validation_NLL=1.0), dict(epoch=2, validation_NLL=1.0)],
                             dict(epoch=2, validation_NLL=1.0))
    try:
        analysis.selection_check([dict(epoch=1, validation_NLL=1.0), dict(epoch=2, validation_NLL=1.0)],
                                 dict(epoch=1, validation_NLL=1.0))
    except ValueError:
        pass
    else:
        raise AssertionError('Earlier tied state accepted')
    checks.append('complete_denominator_failure_duplicate_heldout_freeze_guards; latest_tie_selection')
    import json
    print(json.dumps(dict(status='PASS', checks=checks, synthetic_CPU_only=True,
                          real_labels_or_outputs_opened=False, optimizer_steps=0,
                          new_theorem_claim=False)))


if __name__ == '__main__':
    main()
