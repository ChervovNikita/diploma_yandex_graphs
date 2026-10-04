"""Bounded fabricated ragged/float32 assurance; no native data or training."""
from itertools import product
from math import log, log1p, comb
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from conditional_loss import require, TrainPatternLabels, training_pattern_losses
from conditional_single import ConditionalSingleHead, visible_context, WIDTH
from oracles import subsets, subset_oracle, near, eta_fixture

SUPPORT = ((0, 0), (3, 0), (0, 4), (3, 2), (4, 3), (5, 4), (5, 4), (1, 1), (6, 3))
COUNTS = ((0, 0), (1, 0), (0, 3), (2, 1), (2, 2), (4, 1), (1, 3), (1, 0), (3, 1))
QUERIES = len(SUPPORT)


def fixture(dtype, reverse=False):
    rows, nodes, bits = [], [], []
    for side in (0, 1):
        rs, ns, bs = [], [], []
        for q, (sizes, counts) in enumerate(zip(SUPPORT, COUNTS)):
            n, k = sizes[side], counts[side]
            pattern = [float(i < k) for i in range(n)]
            if reverse:
                pattern.reverse()
            rs.extend([q] * n)
            ns.extend(range((20 if side == 0 else 64) + 3 * q,
                            (20 if side == 0 else 64) + 3 * q + n))
            bs.extend(pattern)
        rows.append(torch.tensor(rs, dtype=torch.long))
        nodes.append(torch.tensor(ns, dtype=torch.long))
        bits.append(torch.tensor(bs, dtype=dtype))
    return rows, nodes, torch.cat(bits)


def reference_ragged(left, right, rows, bits):
    """Exhaustive subset probabilities/analytic logit gradients, not grouped ESP."""
    values = {name: [] for name in ('J_K', 'J_K_sep', 'W_K', 'member_nll', 'rho')}
    grads = {name: [torch.zeros_like(left, dtype=torch.float64), torch.zeros_like(right, dtype=torch.float64)]
             for name in ('J_K', 'J_K_sep', 'W_K')}
    members = left.shape[0]
    for q in range(QUERIES):
        masks = [row == q for row in rows]
        labels = [bits[:len(rows[0])][masks[0]], bits[len(rows[0]):][masks[1]]]
        pair = [subset_oracle(eta[:, mask], z)[1:] for eta, mask, z in zip((left, right), masks, labels)]
        ll, lr = pair[0][0], pair[1][0]
        total = ll + lr
        rho = torch.softmax(-total, dim=0)
        d = max(sum(SUPPORT[q]), 1)
        values['J_K'].append((-torch.logsumexp(-total, dim=0) + log(members)) / d)
        values['J_K_sep'].append((-torch.logsumexp(-ll, dim=0) - torch.logsumexp(-lr, dim=0) + 2 * log(members)) / d)
        values['W_K'].append(total.mean() / d)
        values['member_nll'].append(total)
        values['rho'].append(rho)
        weights = {'J_K': (rho, rho), 'J_K_sep': (torch.softmax(-ll, dim=0), torch.softmax(-lr, dim=0)),
                   'W_K': (torch.full_like(rho, 1 / members),) * 2}
        for name in grads:
            for side in (0, 1):
                grads[name][side][:, masks[side]] = weights[name][side][:, None] * pair[side][1] / (d * QUERIES)
    return {name: torch.stack(items, dim=1 if name in ('member_nll', 'rho') else 0)
            for name, items in values.items()}, grads


def reference_visible(h, queries, rows, nodes, unary):
    """Independent direct CSR slices/means and per-slot feature construction."""
    slot_parts, candidate_parts, qparts = [], [], []
    support_nodes = [[node[row == q] for q in range(QUERIES)] for row, node in zip(rows, nodes)]
    for side in (0, 1):
        for q in range(QUERIES):
            positions = torch.nonzero(rows[side] == q, as_tuple=False).flatten()
            for position in positions.tolist():
                u = unary[side][position] - unary[side][positions[0]]
                candidate = h[nodes[side][position]]
                slot_parts.append(torch.cat((u[None], h[queries[q, side]], h[queries[q, 1 - side]], candidate)))
                candidate_parts.append(candidate)
    for q in range(QUERIES):
        means = [h[ids].mean(dim=0) if len(ids) else h[:0].sum(dim=0) for ids in (support_nodes[0][q], support_nodes[1][q])]
        qparts.append(torch.cat((h[queries[q, 0]] * h[queries[q, 1]], means[0], means[1],
                                h.new_tensor([log1p(len(support_nodes[0][q])), log1p(len(support_nodes[1][q]))]))))
    return SimpleNamespace(slot=torch.stack(slot_parts), query=torch.stack(qparts), candidate=torch.stack(candidate_parts))


def reference_single(head, context, rows, bits, inputs):
    """Chain-rule BCE with prefix recomputed from explicit prior selected sets.

    This does not call training_nll or mutate running prefix/count summaries.
    Its matrix-affine head is explicit rather than the candidate nn.Sequential.
    """
    output, trace = [], []
    zero = sum(p.reshape(-1)[:0].sum() for p in head.parameters()) + sum(x.reshape(-1)[:0].sum() for x in inputs)
    for q in range(QUERIES):
        addresses = [torch.nonzero(row == q, as_tuple=False).flatten().tolist() for row in rows]
        addresses[1] = [index + len(rows[0]) for index in addresses[1]]
        k = [sum(int(bits[index]) for index in ids) for ids in addresses]
        d, terms = max(sum(len(ids) for ids in addresses), 1), []
        for side in (0, 1):
            for pos, address in enumerate(addresses[side]):
                previous = [addresses[0][:pos] if side == 0 else addresses[0],
                            [] if side == 0 else addresses[1][:pos]]
                selected = [[index for index in ids if int(bits[index]) == 1] for ids in previous]
                prefix = [context.candidate[ids].mean(dim=0) if ids else context.candidate[:0].sum(dim=0) for ids in selected]
                unseen = [len(addresses[0]) - pos if side == 0 else 0,
                          len(addresses[1]) if side == 0 else len(addresses[1]) - pos]
                remaining = [k[s] - len(selected[s]) for s in (0, 1)]
                n = [len(ids) for ids in addresses]
                progress = context.slot.new_tensor([float(side == 0), float(side == 1),
                    unseen[0] / max(n[0], 1), unseen[1] / max(n[1], 1),
                    remaining[0] / max(n[0], 1), remaining[1] / max(n[1], 1),
                    len(selected[0]) / max(n[0], 1), len(selected[1]) / max(n[1], 1)])
                features = torch.cat((context.slot[address], context.query[q], prefix[0], prefix[1], progress))
                pre = head.network[0].weight @ features + head.network[0].bias
                hidden = torch.clamp_min(pre, 0)
                score = torch.dot(head.network[2].weight.flatten(), hidden) + head.network[2].bias[0]
                forced = remaining[side] in (0, unseen[side])
                bit = int(bits[address])
                require(not forced or bit == int(remaining[side] == unseen[side]), 'Reference forced bit differs')
                terms.append(score * 0 if forced else F.binary_cross_entropy_with_logits(score, score.new_tensor(float(bit))))
                trace.append((q, features, pre, hidden, score, bit, forced, d))
        output.append(((torch.stack(terms).sum() if terms else zero) + zero) / d)
    return torch.stack(output), trace


def manual_head_gradients(head, trace, query):
    """Full independent affine/ReLU/BCE parameter derivatives for one pattern."""
    result = [torch.zeros_like(p) for p in head.parameters()]
    for q, features, pre, hidden, score, bit, forced, denominator in trace:
        if q != query or forced:
            continue
        require(bool((pre.detach().abs() > .25).all()), 'Fabricated reference must stay away from ReLU kinks')
        delta = (torch.sigmoid(score.detach()) - bit) / denominator
        first = delta * head.network[2].weight.detach().flatten() * (pre.detach() > 0)
        result[0] += first[:, None] * features.detach()[None, :]
        result[1] += first
        result[2] += delta * hidden.detach()[None, :]
        result[3] += delta
    return result


def assign_head(head, variant):
    with torch.no_grad():
        w1, b1, w2, b2 = tuple(head.parameters())
        w1.copy_((torch.arange(w1.numel(), dtype=w1.dtype).reshape_as(w1) % 7 - 3) / 4096)
        b1.copy_(torch.full_like(b1, 2) if variant == 0 else torch.where(torch.arange(len(b1)) % 2 == 0, 2., -2.).to(b1))
        w2.copy_((torch.arange(w2.numel(), dtype=w2.dtype).reshape_as(w2) % 5 - 2) / 128 + 1 / 64)
        b2.fill_(1 / 16)


def qualify_ragged(check_cap, ledger):
    reports = ledger['comparison_reports']
    cases = ledger['ragged_case_counts'] = {'grouped_ragged_cases': 0, 'grouped_zero_cases': 0, 'single_pattern_cases': 0,
        'single_normalization_groups': 0, 'endpoint_range_rejections': 0, 'single_shift_cases': 0,
        'single_all_query_mean_gradient_cases': 0}
    for dtype, members, reverse, large in product((torch.float32, torch.float64), (1, 2, 4), (False, True), (False, True)):
        rows, nodes, bits = fixture(dtype, reverse)
        left = eta_fixture(members, len(rows[0]), dtype, large).requires_grad_()
        right = (eta_fixture(members, len(rows[1]), dtype, large) * .5).requires_grad_()
        labels = TrainPatternLabels(bits, 'complete_TRAIN_observation_membership')
        actual = training_pattern_losses(left, right, *rows, labels, QUERIES)
        expected, expected_grads = reference_ragged(left, right, rows, bits)
        tag = f'grouped/{dtype}/{members}/{reverse}/{large}'
        require(actual['support_counts'] == [[pair[s] for pair in SUPPORT] for s in (0, 1)]
                and actual['teacher_counts'] == [[pair[s] for pair in COUNTS] for s in (0, 1)]
                and actual['denominator'] == [max(sum(pair), 1) for pair in SUPPORT], 'Grouped ragged metadata differs')
        for name in expected:
            near(actual[name], expected[name], dtype, tag + '/' + name, reports)
        for name in expected_grads:
            grads = torch.autograd.grad(actual[name].mean(), (left, right), retain_graph=True)
            for side in (0, 1):
                near(grads[side], expected_grads[name][side], dtype, tag + '/' + name + '/analytic_gradient_' + str(side), reports)
                for q in range(QUERIES):
                    if COUNTS[q][side] in (0, SUPPORT[q][side]):
                        require(bool((grads[side][:, rows[side] == q] == 0).all()), 'Grouped extreme gradient not exact zero')
        shifts = torch.arange(members, dtype=dtype)[:, None] * 64 + torch.arange(QUERIES, dtype=dtype)[None, :] * 32
        shifted_left = (left.detach() + shifts[:, rows[0]]).requires_grad_()
        shifted_right = (right.detach() - 2 * shifts[:, rows[1]]).requires_grad_()
        shifted = training_pattern_losses(shifted_left, shifted_right, *rows, labels, QUERIES)
        for name in ('J_K', 'J_K_sep', 'W_K', 'member_nll', 'rho'):
            near(shifted[name], actual[name], dtype, tag + '/' + name + '/dyadic_member_query_shift', reports)
        for name in ('J_K', 'J_K_sep', 'W_K'):
            for side, (a, b) in enumerate(zip(torch.autograd.grad(shifted[name].mean(), (shifted_left, shifted_right), retain_graph=True),
                                             torch.autograd.grad(actual[name].mean(), (left, right), retain_graph=True))):
                near(a, b, dtype, tag + '/' + name + '/shift_gradient_' + str(side), reports)
        cases['grouped_ragged_cases'] += 1
        check_cap()
    for dtype in (torch.float32, torch.float64):
        for mode in ('empty', 'all_zero_extreme', 'all_one_extreme'):
            rows, _, fixture_bits = fixture(dtype)
            if mode == 'empty':
                rows = [torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)]
            logits = [torch.full((4, len(row)), torch.finfo(dtype).max / 2, dtype=dtype, requires_grad=True) for row in rows]
            bits = torch.full((sum(len(row) for row in rows),), float(mode == 'all_one_extreme'), dtype=dtype)
            actual = training_pattern_losses(*logits, *rows, TrainPatternLabels(bits, 'complete_TRAIN_observation_membership'), QUERIES)
            for name in ('J_K', 'J_K_sep', 'W_K'):
                require(bool((actual[name] == 0).all()), 'Grouped empty/extreme exact zero fails')
                require(all(bool((g == 0).all()) for g in torch.autograd.grad(actual[name].sum(), logits, retain_graph=True)),
                        'Grouped empty/extreme exact zero gradient fails')
            require(bool((actual['member_nll'] == 0).all()) and bool((actual['rho'] == .25).all()),
                    'Grouped empty/extreme member state differs')
            cases['grouped_zero_cases'] += 1
            check_cap()
    for dtype in (torch.float32, torch.float64):
        rows, nodes, base_bits = fixture(dtype)
        queries = torch.arange(2 * QUERIES, dtype=torch.long).reshape(QUERIES, 2)
        neighbors = SimpleNamespace(left=(rows[0], nodes[0]), right=(rows[1], nodes[1]), queries=QUERIES)
        raw_h = ((torch.arange(96 * WIDTH, dtype=dtype).reshape(96, WIDTH) % 37) - 18) / 64
        raw_unary = [eta_fixture(1, len(row), dtype).flatten() for row in rows]
        for endpoint, value in product((0, 1), (-1, len(raw_h))):
            invalid = queries.clone(); invalid[0, endpoint] = value
            caught = False
            try:
                visible_context(raw_h, invalid, neighbors, *raw_unary)
            except ValueError:
                caught = True
            require(caught, 'Negative/out-of-range endpoint was not rejected by interface ValueError')
            cases['endpoint_range_rejections'] += 1
        for variant in (0, 1):
            head = ConditionalSingleHead().to(dtype=dtype)
            reference_head = ConditionalSingleHead().double()
            assign_head(head, variant); assign_head(reference_head, variant)
            h = raw_h.clone().requires_grad_()
            unary = [value.clone().requires_grad_() for value in raw_unary]
            h64 = raw_h.double().requires_grad_()
            unary64 = [value.double().requires_grad_() for value in raw_unary]
            for target_query in range(QUERIES):
                context = visible_context(h, queries, neighbors, *unary)
                ref_context = reference_visible(h64, queries, rows, nodes, unary64)
                for name in ('slot', 'query', 'candidate'):
                    near(getattr(context, name), getattr(ref_context, name), dtype,
                         f'single/{dtype}/{variant}/visible/' + name, reports)
                addresses = [torch.nonzero(row == target_query, as_tuple=False).flatten() for row in rows]
                probabilities = []
                for zl, zr in product(subsets(SUPPORT[target_query][0], COUNTS[target_query][0]),
                                      subsets(SUPPORT[target_query][1], COUNTS[target_query][1])):
                    bits = base_bits.clone()
                    bits[addresses[0]] = zl.to(dtype)
                    bits[len(rows[0]) + addresses[1]] = zr.to(dtype)
                    labels = TrainPatternLabels(bits, 'complete_TRAIN_observation_membership')
                    actual = head.training_nll(context, labels)
                    expected, trace = reference_single(reference_head, ref_context, rows, bits.double(), (h64, *unary64))
                    tag = f'single/{dtype}/{variant}/{target_query}/{cases["single_pattern_cases"]}'
                    near(actual, expected, dtype, tag + '/independent_all_query_per_pattern_NLL', reports)
                    parameters = tuple(head.parameters())
                    ref_parameters = tuple(reference_head.parameters())
                    grads = torch.autograd.grad(actual[target_query], (*parameters, h, *unary), retain_graph=True)
                    refs = torch.autograd.grad(expected[target_query], (*ref_parameters, h64, *unary64), retain_graph=True)
                    for number, (a, b) in enumerate(zip(grads, refs)):
                        near(a, b, dtype, tag + '/per_pattern_parameter_or_input_gradient_' + str(number), reports)
                    manual = manual_head_gradients(reference_head, trace, target_query)
                    for number, (a, b) in enumerate(zip(refs[:4], manual)):
                        near(a, b, torch.float64, tag + '/independent_manual_head_gradient_' + str(number), reports)
                    if target_query == 0 and len(probabilities) == 0:
                        for number, (a, b) in enumerate(zip(torch.autograd.grad(actual.mean(), (*parameters, h, *unary), retain_graph=True),
                                                          torch.autograd.grad(expected.mean(), (*ref_parameters, h64, *unary64), retain_graph=True))):
                            near(a, b, dtype, tag + '/all_query_mean_gradient_' + str(number), reports)
                        cases['single_all_query_mean_gradient_cases'] += 1
                    if all(COUNTS[target_query][s] in (0, SUPPORT[target_query][s]) for s in (0, 1)):
                        require(float(actual[target_query]) == 0 and all(bool((g == 0).all()) for g in grads), 'Forced/empty actual interface loss or gradients not exact zero')
                    probabilities.append(torch.exp(-actual[target_query] * max(sum(SUPPORT[target_query]), 1)))
                    cases['single_pattern_cases'] += 1
                    check_cap()
                mass = torch.stack(probabilities).sum()
                near(mass, torch.ones((), dtype=torch.float64), dtype, tag + '/ragged_supported_mass', reports)
                for number, grad in enumerate(torch.autograd.grad(mass, (*tuple(head.parameters()), h, *unary))):
                    near(grad, torch.zeros_like(grad), dtype, tag + '/ragged_mass_gradient_' + str(number), reports)
                cases['single_normalization_groups'] += 1
                check_cap()
            # Valid actual visible interface preserves per-query/side common shifts.
            context = visible_context(h, queries, neighbors, *unary)
            labels = TrainPatternLabels(base_bits, 'complete_TRAIN_observation_membership')
            base = head.training_nll(context, labels)
            shifts = torch.arange(QUERIES, dtype=dtype) * 32 + 512
            shifted = [(unary[0].detach() + shifts[rows[0]]).requires_grad_(),
                       (unary[1].detach() - shifts[rows[1]]).requires_grad_()]
            shifted_context = visible_context(h, queries, neighbors, *shifted)
            changed = head.training_nll(shifted_context, labels)
            near(changed, base, dtype, f'single/{dtype}/{variant}/actual_interface_shift_NLL', reports)
            for number, (a, b) in enumerate(zip(torch.autograd.grad(changed.mean(), (*tuple(head.parameters()), h, *shifted), retain_graph=True),
                                              torch.autograd.grad(base.mean(), (*tuple(head.parameters()), h, *unary), retain_graph=True))):
                near(a, b, dtype, f'single/{dtype}/{variant}/actual_interface_shift_gradient_' + str(number), reports)
            cases['single_shift_cases'] += 1
            check_cap()
    expected_cases = {'grouped_ragged_cases': 24, 'grouped_zero_cases': 6, 'single_pattern_cases': 532, 'single_normalization_groups': 36,
                      'endpoint_range_rejections': 8, 'single_shift_cases': 4, 'single_all_query_mean_gradient_cases': 4}
    require(cases == expected_cases, 'Supplemental bounded coverage differs')
    return {'cases': cases, 'all_actual_S_K_visible_and_progress_features_independently_reconstructed': True,
            'per_pattern_all_head_parameter_h_and_unary_gradients_checked': True,
            'full_manual_affine_ReLU_BCE_head_gradient_reference': True,
            'native_schedule_or_full65536_feasibility_qualified': False, 'law_oracles_only': True,
            'execution_result_not_preparation_claim': True}
