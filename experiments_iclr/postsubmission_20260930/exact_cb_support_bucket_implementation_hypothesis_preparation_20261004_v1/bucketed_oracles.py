"""Unexecuted fabricated CPU QA for the separate exact-CB bucket hypothesis."""
from itertools import product
from math import log
import torch
from conditional_loss import TrainPatternLabels, require, training_pattern_losses
from conditional_loss_bucketed import (bucketed_training_pattern_losses,
                                      _masked_log_esp_reachable, _support_bucket_plan)
from oracles import subsets, subset_oracle, near, eta_fixture


FIXTURES = (
    (((0, 0), (1, 1), (3, 4), (4, 3), (5, 6), (6, 7), (7, 8), (8, 5),
      (5, 6), (6, 7), (7, 8), (8, 5)),
     ((0, 0), (1, 0), (1, 3), (3, 1), (2, 2), (2, 2), (2, 2), (2, 2),
      (3, 4), (4, 5), (5, 6), (6, 3))),
    (((6, 7), (7, 8), (8, 6), (6, 7), (7, 8), (8, 6), (0, 0)),
     ((3, 3), (3, 3), (3, 3), (3, 4), (4, 5), (5, 3), (0, 0))),
    (((0, 0), (5, 9), (9, 5), (16, 17), (17, 16), (3, 4), (4, 3)),
     ((0, 0), (2, 3), (3, 2), (2, 3), (3, 2), (1, 1), (1, 1))),
    (((9, 16), (10, 15), (11, 14), (12, 13), (13, 12), (14, 11), (15, 10), (16, 9)),
     ((2, 14), (8, 2), (2, 12), (10, 2), (2, 10), (12, 2), (2, 8), (14, 2))),
)


def fixture(support, counts, dtype, reverse):
    rows, bits = [], []
    for side in (0, 1):
        rs, bs = [], []
        for query, (n, k) in enumerate(zip(support, counts)):
            pattern = [float(index < k[side]) for index in range(n[side])]
            if reverse:
                pattern.reverse()
            rs.extend([query] * n[side]); bs.extend(pattern)
        rows.append(torch.tensor(rs, dtype=torch.long))
        bits.append(torch.tensor(bs, dtype=dtype))
    return rows, torch.cat(bits)


def exhaustive_reference(left, right, rows, bits, support):
    """Independent finite subsets and analytic inclusion/responsibility gradients."""
    queries, members = len(support), left.shape[0]
    values = {name: [] for name in ('J_K', 'J_K_sep', 'W_K', 'member_nll', 'rho')}
    gradients = {name: [torch.zeros_like(left, dtype=torch.float64), torch.zeros_like(right, dtype=torch.float64)]
                 for name in ('J_K', 'J_K_sep', 'W_K')}
    for query in range(queries):
        masks = [row == query for row in rows]
        labels = [bits[:len(rows[0])][masks[0]], bits[len(rows[0]):][masks[1]]]
        side = [subset_oracle(eta[:, mask], z)[1:] for eta, mask, z in zip((left, right), masks, labels)]
        ll, lr = side[0][0], side[1][0]
        total = ll + lr
        rho = torch.softmax(-total, dim=0)
        denominator = max(sum(support[query]), 1)
        values['J_K'].append((-torch.logsumexp(-total, dim=0) + log(members)) / denominator)
        values['J_K_sep'].append((-torch.logsumexp(-ll, dim=0) - torch.logsumexp(-lr, dim=0)
                                  + 2 * log(members)) / denominator)
        values['W_K'].append(total.mean() / denominator)
        values['member_nll'].append(total); values['rho'].append(rho)
        weights = {'J_K': (rho, rho),
                   'J_K_sep': (torch.softmax(-ll, dim=0), torch.softmax(-lr, dim=0)),
                   'W_K': (torch.full_like(rho, 1 / members),) * 2}
        for name in gradients:
            for s in (0, 1):
                gradients[name][s][:, masks[s]] = weights[name][s][:, None] * side[s][1] / (denominator * queries)
    return {name: torch.stack(items, dim=1 if name in ('member_nll', 'rho') else 0)
            for name, items in values.items()}, gradients


def check_plan(descriptors):
    """Check ownership, bounds and launch inequality without reimplementing plan."""
    plan, assignment = _support_bucket_plan(descriptors)
    flattened = [index for _, _, source in plan for index in source]
    require(sorted(flattened) == list(range(len(descriptors))), 'Plan must cover each exact group once')
    require(len(assignment) == len(descriptors), 'Plan assignment length differs')
    direct_loops = sum(n for n, r in descriptors if r >= 2)
    bucket_loops = sum(width for r, width, _ in plan if r >= 2)
    require(bucket_loops <= direct_loops, 'Adaptive recurrence slot loops increased')
    for group, (r, width, sources) in enumerate(plan):
        for source in sources:
            n, degree = descriptors[source]
            require(assignment[source] == group and degree == r, 'Group degree/assignment changed')
            if r:
                require(n <= width < 2 * n and n >= 2 * r, 'Real slots truncated or padding bound exceeded')
            else:
                require(width == 0, 'Constant group must have no gathered logits')
        if r >= 2 and len(sources) > 1:
            require(width < sum(descriptors[source][0] for source in sources), 'Merged genuine bucket has no slot-loop saving')


def qualify_bucketed(check_cap, ledger):
    """A future gated one-attempt CPU runner may call this; none is supplied here."""
    reports = ledger['comparison_reports']
    cases = ledger['bucketed_case_counts'] = {'ragged_analytic_cases': 0, 'masked_ESP_cases': 0,
                                             'zero_cases': 0, 'plan_cases': 0}
    plan_inputs = (
        [[0, 0], [1, 0], [2, 0]],
        [[3, 1], [4, 1], [5, 2], [6, 2], [7, 2], [8, 2]],
        [[5, 2], [9, 3], [16, 2], [17, 3]],
        [[n, r] for n in range(1, 65) for r in range(min(n // 2, 4) + 1)],
        [[n, 2] for n in range(33, 65)],
    )
    for descriptors in plan_inputs:
        check_plan(descriptors); cases['plan_cases'] += 1; check_cap()
    for dtype, members, reverse, large, fixture_index in product(
            (torch.float32, torch.float64), (1, 2, 4), (False, True), (False, True), range(len(FIXTURES))):
        support, counts = FIXTURES[fixture_index]
        rows, bits = fixture(support, counts, dtype, reverse)
        left = eta_fixture(members, len(rows[0]), dtype, large).requires_grad_()
        right = (eta_fixture(members, len(rows[1]), dtype, large) * .5).requires_grad_()
        labels = TrainPatternLabels(bits, 'complete_TRAIN_observation_membership')
        actual = bucketed_training_pattern_losses(left, right, *rows, labels, len(support))
        direct = training_pattern_losses(left, right, *rows, labels, len(support))
        expected, gradients = exhaustive_reference(left, right, rows, bits, support)
        tag = f'bucket/{dtype}/{members}/{reverse}/{large}/{fixture_index}'
        require(actual['support_counts'] == [[n[s] for n in support] for s in (0, 1)]
                and actual['teacher_counts'] == [[k[s] for k in counts] for s in (0, 1)]
                and actual['denominator'] == [max(sum(n), 1) for n in support]
                and actual['teacher_counts_for_serving'] is False, 'Original ragged output metadata changed')
        for name in expected:
            near(actual[name], expected[name], dtype, tag + '/' + name + '/independent_subsets', reports)
            near(actual[name], direct[name], dtype, tag + '/' + name + '/direct_source_parity', reports)
        for name in gradients:
            grads = torch.autograd.grad(actual[name].mean(), (left, right), retain_graph=True)
            old_grads = torch.autograd.grad(direct[name].mean(), (left, right), retain_graph=True)
            for side in (0, 1):
                near(grads[side], gradients[name][side], dtype, tag + '/' + name + '/analytic_gradient_' + str(side), reports)
                near(grads[side], old_grads[side], dtype, tag + '/' + name + '/direct_gradient_' + str(side), reports)
                for query in range(len(support)):
                    if counts[query][side] in (0, support[query][side]):
                        require(bool((grads[side][:, rows[side] == query] == 0).all()), 'Extreme actual-slot gradient not exactly zero')
        shifts = torch.arange(members, dtype=dtype)[:, None] * 64 + torch.arange(len(support), dtype=dtype)[None, :] * 32
        shifted_left = (left.detach() + shifts[:, rows[0]]).requires_grad_()
        shifted_right = (right.detach() - 2 * shifts[:, rows[1]]).requires_grad_()
        shifted = bucketed_training_pattern_losses(shifted_left, shifted_right, *rows, labels, len(support))
        for name in expected:
            near(shifted[name], actual[name], dtype, tag + '/' + name + '/dyadic_common_shift', reports)
        for name in gradients:
            changed = torch.autograd.grad(shifted[name].mean(), (shifted_left, shifted_right), retain_graph=True)
            original = torch.autograd.grad(actual[name].mean(), (left, right), retain_graph=True)
            for side in (0, 1):
                near(changed[side], original[side], dtype, tag + '/' + name + '/shift_gradient_' + str(side), reports)
        cases['ragged_analytic_cases'] += 1; check_cap()
    for dtype, members, degree in product((torch.float32, torch.float64), (1, 2, 4), (2, 3)):
        lengths = [2 * degree + j for j in range(4)]
        width = 1 << (max(lengths) - 1).bit_length()
        raw = ((torch.arange(members * 4 * width, dtype=dtype).reshape(members, 4, width) % 19) - 9) / 8
        padded = raw.requires_grad_()
        real = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
        value = _masked_log_esp_reachable(padded, real, degree)
        gradient = torch.autograd.grad(value.sum(), padded)[0]
        require(bool((gradient.masked_select(~real[None, :, :].expand_as(gradient)) == 0).all()),
                'Padding logit gradients must be exact zero')
        tag = f'masked_ESP/{dtype}/{members}/{degree}'
        for query, length in enumerate(lengths):
            bits = torch.tensor([float(i < degree) for i in range(length)], dtype=dtype)
            normalizer, _, inclusion_minus_bits = subset_oracle(raw[:, query, :length], bits)
            near(value[:, query], normalizer, dtype, tag + '/subset_normalizer_' + str(query), reports)
            near(gradient[:, query, :length], inclusion_minus_bits + bits[None, :], dtype,
                 tag + '/analytic_real_inclusion_' + str(query), reports)
        extended = torch.cat((raw.detach(), torch.full_like(raw, 17)), dim=-1).requires_grad_()
        extended_real = torch.arange(2 * width)[None, :] < torch.tensor(lengths)[:, None]
        extended_value = _masked_log_esp_reachable(extended, extended_real, degree)
        extended_gradient = torch.autograd.grad(extended_value.sum(), extended)[0]
        near(extended_value, value, dtype, tag + '/tail_extension_normalizer', reports)
        near(extended_gradient[..., :width], gradient, dtype, tag + '/tail_extension_real_gradient', reports)
        require(bool((extended_gradient[..., width:] == 0).all()), 'Extended padding gradients not exact zero')
        cases['masked_ESP_cases'] += 1; check_cap()
    for dtype, mode in product((torch.float32, torch.float64), ('empty', 'all_zero', 'all_one')):
        support = ((0, 0), (5, 7), (9, 3), (1, 1)) if mode != 'empty' else ((0, 0),) * 4
        counts = support if mode == 'all_one' else ((0, 0),) * len(support)
        rows, bits = fixture(support, counts, dtype, False)
        logits = [torch.full((4, len(row)), torch.finfo(dtype).max / 2, dtype=dtype, requires_grad=True) for row in rows]
        actual = bucketed_training_pattern_losses(*logits, *rows,
            TrainPatternLabels(bits, 'complete_TRAIN_observation_membership'), len(support))
        for name in ('J_K', 'J_K_sep', 'W_K'):
            require(bool((actual[name] == 0).all()), 'Empty/extreme loss must be exact zero')
            require(all(bool((g == 0).all()) for g in torch.autograd.grad(actual[name].sum(), logits, retain_graph=True)),
                    'Empty/extreme gradients must be exact zero')
        require(bool((actual['member_nll'] == 0).all()) and bool((actual['rho'] == .25).all()), 'Constant member state changed')
        cases['zero_cases'] += 1; check_cap()
    require(cases == {'ragged_analytic_cases': 96, 'masked_ESP_cases': 12, 'zero_cases': 6, 'plan_cases': 5},
            'Declared bucket-hypothesis QA coverage differs')
    return {'cases': cases, 'CPU_QA_result_not_source_preparation': True,
            'native_neural_or_full65536_resource_qualified': False,
            'methodological_or_inference_efficiency_novelty_claim': False}
