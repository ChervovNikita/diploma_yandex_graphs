"""Detached scalar reports, with no extra reverse pass or signal threshold."""
def norms(value):
    import torch
    value = value.detach()
    if value.numel() == 0:
        return dict(elements=0, finite=True, nonzero_elements=0, L1=0., L2=0., max_abs=0.)
    finite = bool(torch.isfinite(value).all())
    if not finite:
        raise RuntimeError('Nonfinite diagnostic tensor')
    working = value.to(torch.float64)
    return dict(elements=value.numel(), finite=finite, nonzero_elements=int((value != 0).sum()),
                L1=float(working.abs().sum()), L2=float(working.square().sum().sqrt()),
                max_abs=float(working.abs().max()))


def magnitudes(value):
    result = norms(value)
    result.update(min=float(value.detach().min()) if value.numel() else None,
                  max=float(value.detach().max()) if value.numel() else None)
    return result


def compare_tensors(actual, reference, atol, rtol):
    """Bucket versus direct under the unchanged native fixed elementwise rule."""
    import torch
    if actual.shape != reference.shape or actual.dtype != reference.dtype or actual.device != reference.device:
        raise RuntimeError('Native comparison shape/dtype/device differs')
    if actual.dtype != torch.float32:
        raise RuntimeError('Original native float32 working dtype required')
    difference = agreement(actual, reference, atol, rtol)
    allowance = atol + rtol * reference.detach().abs()
    ratio = (actual.detach() - reference.detach()).abs() / allowance
    return dict(direct=norms(reference), bucket=norms(actual), difference=difference,
                maximum_fraction_of_allowed_error=float(ratio.max()) if ratio.numel() else 0.,
                fixed_rule_agrees=difference['fixed_rule_agrees'], dtype=str(actual.dtype))


def compare_parameters(named, actual, reference, rule):
    if len(actual) != len(named) or len(reference) != len(named):
        raise RuntimeError('Named model gradient coverage differs')
    reports = []
    for (name, parameter), a, b in zip(named, actual, reference):
        if a is None or b is None:
            value = dict(status='BOTH_UNUSED' if a is None and b is None else 'REACHABILITY_MISMATCH',
                         direct_unused=b is None, bucket_unused=a is None, fixed_rule_agrees=a is None and b is None)
        else:
            value = dict(status='COMPARED', **compare_tensors(a, b, **rule))
        reports.append(dict(name=name, shape=list(parameter.shape), parameter_elements=parameter.numel(), **value))
    return dict(named_parameters=reports, parameter_count=sum(p.numel() for _, p in named),
                all_named_parameters_included=True, fixed_rule_agrees=all(r['fixed_rule_agrees'] for r in reports))


def compare_slots(record, actual, reference, rule):
    import torch
    reports = {}
    offset = 0
    for side, rows in enumerate(record['rows']):
        size = len(rows)
        a, b = actual[:, offset:offset+size], reference[:, offset:offset+size]
        forced = torch.minimum(record['k'][:, side], record['n'][:, side] - record['k'][:, side])[rows] == 0
        forced_zero = bool((a[:, forced] == 0).all()) and bool((b[:, forced] == 0).all())
        reports['left' if side == 0 else 'right'] = dict(actual_slots=size, member_slot_elements=a.numel(),
            all_actual_slots_included=True, comparison=compare_tensors(a, b, **rule),
            forced_r0_slots=int(forced.sum()), exact_zero_forced_gradients=forced_zero)
        offset += size
    return dict(sides=reports, fixed_rule_agrees=all(r['comparison']['fixed_rule_agrees'] and r['exact_zero_forced_gradients']
                                                   for r in reports.values()))


def agreement(a, b, atol, rtol):
    import torch
    delta = (a-b).detach()
    result = norms(delta)
    allowance = atol + rtol*b.detach().abs()
    result.update(atol=atol, rtol=rtol, violations=int((delta.abs() > allowance).sum()),
                  fixed_rule_agrees=bool((delta.abs() <= allowance).all()))
    return result


def block_name(name):
    if name.startswith('encoder.'):
        return 'encoder'
    family = name.split('.')[1]
    if family == 'ptlin':
        return 'decoder.ptlin_unused_fixed_pt'
    if family == 'beta':
        return 'decoder.beta'
    kind = 'factors' if name.endswith(('.r', '.s')) else 'shared_or_private_norm'
    return 'decoder.'+family+'.'+kind


def parameter_reports(named, gradients):
    import torch
    blocks = {}
    for index, (name, parameter) in enumerate(named):
        blocks.setdefault(block_name(name), []).append((index, name, parameter))
    output = {}
    for block, rows in blocks.items():
        vectors = {}
        reports = {}
        for objective, values in gradients.items():
            reachable, unused, pieces = [], [], []
            for index, name, parameter in rows:
                gradient = values[index]
                (unused if gradient is None else reachable).append(name)
                pieces.append(torch.zeros_like(parameter).flatten() if gradient is None else gradient.detach().flatten())
            vector = torch.cat(pieces)
            vectors[objective] = vector
            reports[objective] = dict(norms(vector), reachable_parameters=reachable, unused_parameters=unused)
        target = vectors['base'].to(torch.float64)
        comparisons = {}
        for objective in ('J_K', 'J_K_sep'):
            aux = vectors[objective].to(torch.float64)
            tn = reports['base']['L2']; an = reports[objective]['L2']
            ratios = {key: reports[objective][key]/reports['base'][key] if reports['base'][key] else None
                      for key in ('L1','L2','max_abs')}
            comparisons[objective] = dict(auxiliary_to_target_ratios=ratios,
                ratio_defined={key:reports['base'][key]>0 for key in ratios}, cosine=float((target*aux).sum())/(tn*an) if tn and an else None,
                cosine_defined=tn > 0 and an > 0,
                undefined_reason=None if tn and an else 'zero_target_or_auxiliary_L2')
        output[block] = dict(parameters=[name for _, name, _ in rows], objectives=reports,
                            auxiliary_to_target=comparisons,
                            shared_minus_separated=norms(vectors['J_K']-vectors['J_K_sep']))
    return output


def slot_reports(record, shared_gradient, separated_gradient, atol, rtol):
    import torch
    result = {}
    n = record['n']; k = record['k']; r = torch.minimum(k, n-k)
    coupled = (r > 0).all(1)
    categories = torch.where(r == 0, 0, torch.where(r == 1, 1, 2))
    offset = 0
    for side, rows in enumerate(record['rows']):
        size = len(rows)
        gs = shared_gradient[:, offset:offset+size]
        gp = separated_gradient[:, offset:offset+size]
        eta = record['detail']['t'][:, offset:offset+size]
        masks = {'forced_r0': r[rows, side] == 0,
                 'categorical_r1': r[rows, side] == 1,
                 'genuine_r_gt1': r[rows, side] > 1,
                 'both_variable': coupled[rows],
                 'forced_counterpart': ~coupled[rows]}
        masks.update({'joint_%d_%d'%(l, h): (categories[rows, 0] == l) & (categories[rows, 1] == h)
                      for l in range(3) for h in range(3)})
        strata = {}
        for name, mask in masks.items():
            left, right = gs[:, mask], gp[:, mask]
            comparison = agreement(left, right, atol, rtol)
            strata[name] = dict(slot_count=int(mask.sum()), member_slot_elements=left.numel(),
                                logits=magnitudes(eta[:, mask]), J_K=norms(left), J_K_sep=norms(right),
                                shared_minus_separated=comparison)
            if name == 'forced_r0':
                require_zero = strata[name]['J_K']['nonzero_elements'] == strata[name]['J_K_sep']['nonzero_elements'] == 0
                strata[name]['exact_zero_forced_gradients'] = require_zero
                if not require_zero:
                    raise RuntimeError('Forced conditional slot gradient is not exact zero')
            if name == 'forced_counterpart' and not comparison['fixed_rule_agrees']:
                raise RuntimeError('Shared/separated forced-counterpart fixed-rule disagreement')
        result['left' if side == 0 else 'right'] = strata
        offset += size
    return result
