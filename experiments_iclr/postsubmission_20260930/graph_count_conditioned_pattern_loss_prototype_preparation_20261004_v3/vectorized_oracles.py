"""Disabled fabricated vectorized-single equivalence and two-sided CB QA source."""
from itertools import product
from math import log
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import torch
from conditional_loss import require, TrainPatternLabels, training_pattern_losses
from conditional_single import ConditionalSingleHead, visible_context, HEAD_INPUT, HEAD_PARAMETERS, WIDTH
from oracles import near, subset_oracle, eta_fixture
from ragged_oracles import assign_head, manual_head_gradients

SUPPORT = ((0,0),(0,5),(6,0),(5,6),(6,5),(6,6),(4,7),(7,4),(1,1),(8,8),(3,3),(4,4))
COUNTS = ((0,0),(0,2),(3,0),(2,3),(1,2),(4,3),(2,6),(6,2),(1,0),(0,8),(1,2),(0,0))
QUERIES = len(SUPPORT)


def fixture(dtype, reverse=False, mode='regular'):
    rows, nodes, parts = [], [], []
    for side in (0,1):
        rs, ns, bs = [], [], []
        for q, (support, count) in enumerate(zip(SUPPORT, COUNTS)):
            n = 0 if mode == 'empty' else support[side]
            k = count[side] if mode == 'regular' else (n if mode == 'all_one' else 0)
            pattern = [float(i < k) for i in range(n)]
            if reverse:
                pattern.reverse()
            rs.extend([q] * n)
            ns.extend(range((40 if side == 0 else 200) + 10*q,
                            (40 if side == 0 else 200) + 10*q + n))
            bs.extend(pattern)
        rows.append(torch.tensor(rs, dtype=torch.long))
        nodes.append(torch.tensor(ns, dtype=torch.long))
        parts.append(torch.tensor(bs, dtype=dtype))
    return rows, nodes, torch.cat(parts)


def sealed_sequential_single():
    """Load only the exact sealed core-v2 comparator source after the CPU gate.

    SOURCE_BINDING authenticates this whole source before any numerical import.
    Its conditional_loss import resolves to the v2-byte-identical current core.
    No predecessor runner is launched and no predecessor artifact is modified.
    """
    source = Path(__file__).resolve().parent.parent / 'graph_count_conditioned_pattern_loss_prototype_preparation_20261004_v2/conditional_single.py'
    name = 'sealed_core_v2_sequential_single_reference'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def independent_features(head, context, rows, bits):
    """Explicit past selected sets and full manual affine/ReLU head trace.

    No production cumsum/layout/prefix helper is used. All teacher membership
    positions are reconstructed by query matching and prior address slices.
    """
    by_address, trace = {}, []
    for q in range(QUERIES):
        addresses = [torch.nonzero(row == q, as_tuple=False).flatten().tolist() for row in rows]
        addresses[1] = [i + len(rows[0]) for i in addresses[1]]
        sizes = [len(ids) for ids in addresses]
        counts = [sum(int(bits[i]) for i in ids) for ids in addresses]
        denominator = max(sum(sizes), 1)
        for side in (0,1):
            for position, address in enumerate(addresses[side]):
                past = (addresses[0][:position], []) if side == 0 else (addresses[0], addresses[1][:position])
                selected = [[i for i in ids if int(bits[i])] for ids in past]
                prefix = [context.candidate[ids].mean(0) if ids else context.candidate[:0].sum(0) for ids in selected]
                unseen = (sizes[0]-position, sizes[1]) if side == 0 else (0, sizes[1]-position)
                remaining = [counts[s]-len(selected[s]) for s in (0,1)]
                progress = context.slot.new_tensor([float(side == 0), float(side == 1),
                    unseen[0]/max(sizes[0],1), unseen[1]/max(sizes[1],1),
                    remaining[0]/max(sizes[0],1), remaining[1]/max(sizes[1],1),
                    len(selected[0])/max(sizes[0],1), len(selected[1])/max(sizes[1],1)])
                feature = torch.cat((context.slot[address], context.query[q], *prefix, progress))
                pre = head.network[0].weight @ feature + head.network[0].bias
                hidden = torch.clamp_min(pre, 0)
                score = torch.dot(head.network[2].weight.flatten(), hidden) + head.network[2].bias[0]
                bit = int(bits[address])
                forced = remaining[side] in (0, unseen[side])
                require(not forced or bit == int(remaining[side] == unseen[side]), 'Independent forced bit/count differs')
                by_address[address] = feature
                trace.append((q, feature, pre, hidden, score, bit, forced, denominator))
    features = torch.stack([by_address[i] for i in range(len(bits))]) if len(bits) else context.slot.new_zeros((0, HEAD_INPUT))
    return features, trace


def independent_cb(left, right, rows, bits):
    """Exhaustive genuine subsets on both sides with analytic mixture gradients."""
    names = ('J_K','J_K_sep','W_K','member_nll','rho')
    values = {name: [] for name in names}
    grads = {name: [torch.zeros_like(left,dtype=torch.float64), torch.zeros_like(right,dtype=torch.float64)]
             for name in names[:3]}
    members = left.shape[0]
    for q in range(QUERIES):
        masks = [row == q for row in rows]
        teacher = [bits[:len(rows[0])][masks[0]], bits[len(rows[0]):][masks[1]]]
        laws = [subset_oracle(eta[:,mask],z)[1:] for eta,mask,z in zip((left,right),masks,teacher)]
        ll, lr = laws[0][0], laws[1][0]
        total = ll + lr
        rho = torch.softmax(-total,0)
        denominator = max(sum(int(mask.sum()) for mask in masks),1)
        values['J_K'].append((-torch.logsumexp(-total,0)+log(members))/denominator)
        values['J_K_sep'].append((-torch.logsumexp(-ll,0)-torch.logsumexp(-lr,0)+2*log(members))/denominator)
        values['W_K'].append(total.mean()/denominator)
        values['member_nll'].append(total)
        values['rho'].append(rho)
        weights = {'J_K':(rho,rho),'J_K_sep':(torch.softmax(-ll,0),torch.softmax(-lr,0)),
                   'W_K':(torch.full_like(rho,1/members),)*2}
        for name in grads:
            for side in (0,1):
                grads[name][side][:,masks[side]] = weights[name][side][:,None] * laws[side][1] / (denominator*QUERIES)
    return {name:torch.stack(part,dim=1 if name in ('member_nll','rho') else 0) for name,part in values.items()}, grads


def qualify_vectorized(check_cap, ledger):
    reports = ledger['comparison_reports']
    cases = ledger['vectorized_case_counts'] = {'two_sided_grouped_cases':0,'two_sided_grouped_zero_cases':0,
        'single_equivalence_cases':0,'single_per_query_gradient_cases':0,'single_manual_head_cases':0,
        'single_zero_cases':0,'single_one_batch_all_slot_evaluation_cases':0}
    for dtype,members,reverse,large in product((torch.float32,torch.float64),(1,2,4),(False,True),(False,True)):
        rows,_,bits = fixture(dtype,reverse)
        left = eta_fixture(members,len(rows[0]),dtype,large).requires_grad_()
        right = (eta_fixture(members,len(rows[1]),dtype,large)*.5).requires_grad_()
        actual = training_pattern_losses(left,right,*rows,TrainPatternLabels(bits,'complete_TRAIN_observation_membership'),QUERIES)
        expected, gradients = independent_cb(left,right,rows,bits)
        tag = f'two_sided/{dtype}/{members}/{reverse}/{large}'
        require(actual['support_counts']==[[sizes[s] for sizes in SUPPORT] for s in (0,1)]
                and actual['teacher_counts']==[[k[s] for k in COUNTS] for s in (0,1)]
                and actual['denominator']==[max(sum(sizes),1) for sizes in SUPPORT], 'Two-sided ragged ownership differs')
        for name in expected:
            near(actual[name],expected[name],dtype,tag+'/'+name,reports)
        for name in gradients:
            observed = torch.autograd.grad(actual[name].mean(),(left,right),retain_graph=True)
            for side in (0,1):
                near(observed[side],gradients[name][side],dtype,tag+'/'+name+'/all_query_gradient_'+str(side),reports)
                if not large:
                    require(bool((observed[side] != 0).any()), 'Small genuine two-sided gradient fixture is vacuous')
        cases['two_sided_grouped_cases'] += 1
        check_cap()
    for dtype,mode in product((torch.float32,torch.float64),('empty','all_zero','all_one')):
        rows,_,bits = fixture(dtype,mode=mode)
        logits = [torch.full((4,len(row)),torch.finfo(dtype).max/2,dtype=dtype,requires_grad=True) for row in rows]
        actual = training_pattern_losses(*logits,*rows,TrainPatternLabels(bits,'complete_TRAIN_observation_membership'),QUERIES)
        for name in ('J_K','J_K_sep','W_K'):
            require(bool((actual[name] == 0).all()), 'Two-sided extreme loss not exact zero')
            require(all(bool((g == 0).all()) for g in torch.autograd.grad(actual[name].mean(),logits,retain_graph=True)),
                    'Two-sided extreme gradient not exact zero')
        cases['two_sided_grouped_zero_cases'] += 1
        check_cap()
    sequential = sealed_sequential_single()
    for dtype,variant,reverse in product((torch.float32,torch.float64),(0,1),(False,True)):
        rows,nodes,bits = fixture(dtype,reverse)
        raw_h = ((torch.arange(400*WIDTH,dtype=dtype).reshape(400,WIDTH)%23)-11)/128
        raw_unary = [((torch.arange(len(row),dtype=dtype)%9)-4)/8 for row in rows]
        h = raw_h.clone().requires_grad_(); unary = [u.clone().requires_grad_() for u in raw_unary]
        ref_h = raw_h.clone().requires_grad_(); ref_unary = [u.clone().requires_grad_() for u in raw_unary]
        queries = torch.stack((torch.arange(QUERIES),torch.arange(QUERIES)+QUERIES),1)
        neighbors = SimpleNamespace(queries=QUERIES,left=(rows[0],nodes[0]),right=(rows[1],nodes[1]))
        head = ConditionalSingleHead().to(dtype);assign_head(head,variant)
        reference = sequential.ConditionalSingleHead().to(dtype);reference.load_state_dict(head.state_dict())
        require(sum(p.numel() for p in head.parameters())==sum(p.numel() for p in reference.parameters())==HEAD_PARAMETERS,
                'Batched head parameter count changed')
        context = visible_context(h,queries,neighbors,*unary)
        ref_context = sequential.visible_context(ref_h,queries,neighbors,*ref_unary)
        tag = f'vectorized/{dtype}/{variant}/{reverse}'
        for name in ('slot','query','candidate'):
            near(getattr(context,name),getattr(ref_context,name),dtype,tag+'/visible_'+name,reports)
        seen = []
        handle = head.network[0].register_forward_pre_hook(lambda module,args:seen.append(args[0]))
        try:
            actual = head.training_nll(context,TrainPatternLabels(bits,'complete_TRAIN_observation_membership'))
        finally:
            handle.remove()
        expected = reference.training_nll(ref_context,TrainPatternLabels(bits,'complete_TRAIN_observation_membership'))
        require(len(seen)==1 and seen[0].shape==(len(bits),HEAD_INPUT),'All slots must evaluate exactly one unchanged head batch')
        features,trace = independent_features(reference,ref_context,rows,bits)
        near(seen[0],features,dtype,tag+'/all_past_only_prefix_progress_features',reports)
        near(actual,expected,dtype,tag+'/all_query_NLL',reports)
        parameters = tuple(head.parameters()); ref_parameters = tuple(reference.parameters())
        targets = (*parameters,h,*unary); refs = (*ref_parameters,ref_h,*ref_unary)
        mean_grads = torch.autograd.grad(actual.mean(),targets,retain_graph=True)
        ref_mean_grads = torch.autograd.grad(expected.mean(),refs,retain_graph=True)
        for number,(a,b) in enumerate(zip(mean_grads,ref_mean_grads)):
            near(a,b,dtype,tag+'/all_query_mean_gradient_'+str(number),reports)
            require(bool((a != 0).any()) and bool((b != 0).any()),'Head/h/both-unary gradient fixture must be nonzero')
        for q in range(QUERIES):
            grads = torch.autograd.grad(actual[q],targets,retain_graph=True)
            ref_grads = torch.autograd.grad(expected[q],refs,retain_graph=True)
            for number,(a,b) in enumerate(zip(grads,ref_grads)):
                near(a,b,dtype,tag+'/query_'+str(q)+'/parameter_h_unary_gradient_'+str(number),reports)
            manual = manual_head_gradients(reference,trace,q)
            for number,(a,b) in enumerate(zip(grads[:4],manual)):
                near(a,b,dtype,tag+'/query_'+str(q)+'/full_manual_head_gradient_'+str(number),reports)
            if all(COUNTS[q][s] in (0,SUPPORT[q][s]) for s in (0,1)):
                require(float(actual[q])==0 and all(bool((g == 0).all()) for g in grads),'Empty/forced query gradients must be exact zero')
            cases['single_per_query_gradient_cases'] += 1
            cases['single_manual_head_cases'] += 1
        cases['single_equivalence_cases'] += 1
        cases['single_one_batch_all_slot_evaluation_cases'] += 1
        check_cap()
    for dtype,variant,mode in product((torch.float32,torch.float64),(0,1),('empty','all_zero','all_one')):
        rows,nodes,bits = fixture(dtype,mode=mode)
        h = ((torch.arange(400*WIDTH,dtype=dtype).reshape(400,WIDTH)%23)/128).requires_grad_()
        unary = [torch.zeros(len(row),dtype=dtype,requires_grad=True) for row in rows]
        queries = torch.stack((torch.arange(QUERIES),torch.arange(QUERIES)+QUERIES),1)
        neighbors = SimpleNamespace(queries=QUERIES,left=(rows[0],nodes[0]),right=(rows[1],nodes[1]))
        head = ConditionalSingleHead().to(dtype);assign_head(head,variant)
        seen = []
        handle = head.network[0].register_forward_pre_hook(lambda module,args:seen.append(args[0].shape))
        try:
            result = head.training_nll(visible_context(h,queries,neighbors,*unary),TrainPatternLabels(bits,'complete_TRAIN_observation_membership'))
        finally:
            handle.remove()
        require(seen==[torch.Size((len(bits),HEAD_INPUT))],'Forced/empty head-row evaluation must remain present')
        require(bool((result == 0).all()),'Fully forced single NLL must be exact zero')
        grads = torch.autograd.grad(result.mean(),(*tuple(head.parameters()),h,*unary))
        require(all(bool((g == 0).all()) for g in grads),'Fully forced head/h/unary gradients must be exact zero')
        cases['single_zero_cases'] += 1
        cases['single_one_batch_all_slot_evaluation_cases'] += 1
        check_cap()
    expected_cases = {'two_sided_grouped_cases':24,'two_sided_grouped_zero_cases':6,'single_equivalence_cases':8,
        'single_per_query_gradient_cases':96,'single_manual_head_cases':96,'single_zero_cases':12,
        'single_one_batch_all_slot_evaluation_cases':20}
    require(cases==expected_cases,'Vectorized successor QA coverage differs')
    return {'cases':cases,'exact_sealed_sequential_single_comparator_used':True,
            'all_head_parameter_h_and_both_unary_mean_gradient_fixtures_nonzero':True,
            'genuine_right_both_and_mixed_categorical_subset_coverage':True,
            'native_full_batch_resource_or_float32_cuda_qualification':False,'source_preparation_is_QA_PASS':False}
