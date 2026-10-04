"""Fabricated CPU exhaustive law/gradient oracles; no data or scientific fits."""
from itertools import combinations, product
from math import log, comb
from types import SimpleNamespace
import torch
from conditional_loss import (require, log_elementary_symmetric, conditional_member_nll,
                              TrainPatternLabels, training_pattern_losses)
from conditional_single import (ConditionalSingleHead, visible_context, SingleVisibleContext,
                                WIDTH, SLOT_FEATURES, QUERY_FEATURES, HEAD_INPUT, HEAD_PARAMETERS)


def subsets(slots, count):
    rows = []
    for chosen in combinations(range(slots), count):
        rows.append([float(index in chosen) for index in range(slots)])
    return torch.tensor(rows, dtype=torch.float64).reshape(-1, slots) if slots else torch.empty((1,0),dtype=torch.float64)


def subset_oracle(eta, bits):
    # Independent exhaustive sum, not a recurrence or complement implementation.
    patterns = subsets(eta.shape[1], int(bits.sum(dtype=torch.int64)))
    values = eta.to(torch.float64) @ patterns.t()
    normalizer = torch.logsumexp(values, dim=1)
    probabilities = torch.softmax(values, dim=1)
    marginals = probabilities @ patterns
    nll = normalizer - (eta.to(torch.float64) * bits.to(torch.float64)[None,:]).sum(dim=1)
    return normalizer, nll, marginals - bits.to(torch.float64)[None,:]


def near(actual, expected, dtype, label, reports):
    # Predetermined CPU oracle tolerances, not an amendment of native science.
    atol,rtol = (1e-10,1e-9) if dtype == torch.float64 else (1.52587890625e-5,1.52587890625e-5)
    require(actual.shape == expected.shape, label + ': shape differs')
    a,b = actual.detach().to(torch.float64),expected.detach().to(torch.float64)
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), label + ': nonfinite')
    delta = (a-b).abs()
    allowed = atol + rtol*b.abs()
    reports.append({'label':label,'dtype':str(dtype),'atol':atol,'rtol':rtol,
                    'elements':a.numel(),'maximum_absolute_difference':float(delta.max()) if a.numel() else 0.,
                    'maximum_fraction_of_allowed_error':float((delta/allowed).max()) if a.numel() else 0.})
    require(bool((delta <= allowed).all()), label + ': fixed CPU oracle tolerance exceeded')


def eta_fixture(members, slots, dtype, large=False):
    positions = torch.arange(slots,dtype=dtype) - (slots-1)/2
    slopes = torch.arange(1,members+1,dtype=dtype)[:,None] * .375
    signs = torch.tensor([1 if m%2==0 else -1 for m in range(members)],dtype=dtype)[:,None]
    return slopes * signs * positions[None,:] * (512 if large else 1)


def qualify(check_cap, ledger):
    reports = ledger['comparison_reports']
    cases = ledger['case_counts']
    for dtype in (torch.float64,torch.float32):
        for slots in range(7):
            for large in (False,True):
                for count in range(slots+1):
                    for bits64 in subsets(slots,count):
                        bits = bits64.to(dtype)
                        eta = eta_fixture(4,slots,dtype,large).requires_grad_()
                        normalizer,expected,gradient = subset_oracle(eta,bits)
                        actual = conditional_member_nll(eta,bits)
                        actual_gradient = torch.autograd.grad(actual.sum(),eta)[0]
                        prefix = f'CB/{dtype}/{slots}/{count}/{large}/{cases["ESP_pattern_cases"]}'
                        near(actual,expected,dtype,prefix+'/NLL',reports)
                        near(actual_gradient,gradient,dtype,prefix+'/analytic_marginal_gradient',reports)
                        near(actual_gradient.sum(dim=1),torch.zeros(4,dtype=torch.float64),dtype,prefix+'/zero_sum_gradient',reports)
                        shifted_eta = (eta.detach() + torch.tensor([900.,-700.,23.,-19.],dtype=dtype)[:,None]).requires_grad_()
                        shifted_loss = conditional_member_nll(shifted_eta,bits)
                        near(shifted_loss,actual,dtype,prefix+'/member_common_shift_NLL',reports)
                        near(torch.autograd.grad(shifted_loss.sum(),shifted_eta)[0],actual_gradient,dtype,prefix+'/member_common_shift_gradient',reports)
                        esp_eta = eta.detach().clone().requires_grad_()
                        esp = log_elementary_symmetric(esp_eta,count)
                        near(esp,normalizer,dtype,prefix+'/ESP_exhaustive',reports)
                        inclusion = gradient + bits64[None,:]
                        near(torch.autograd.grad(esp.sum(),esp_eta)[0],inclusion,dtype,prefix+'/ESP_gradient',reports)
                        if count in (0,slots):
                            require(bool((actual == 0).all()) and bool((actual_gradient == 0).all()), prefix+'/extreme exact zero fails')
                        cases['ESP_pattern_cases'] += 1
                        check_cap()
        # Exact extreme zeros even when summing finite float32 inputs would overflow.
        extreme = torch.full((4,6),torch.finfo(dtype).max/2,dtype=dtype,requires_grad=True)
        for bit in (0.,1.):
            loss = conditional_member_nll(extreme,torch.full((6,),bit,dtype=dtype))
            grad = torch.autograd.grad(loss.sum(),extreme)[0]
            require(bool((loss == 0).all()) and bool((grad == 0).all()), 'Extreme zero must not depend on overflowing sum')
    # Exact differentiability beyond first gradients: ESP Hessian = subset covariance.
    eta = eta_fixture(1,4,torch.float64).flatten().requires_grad_()
    patterns = subsets(4,2)
    weights = torch.softmax(patterns @ eta.detach(),dim=0)
    marginal = weights @ patterns
    covariance = patterns.t() @ (weights[:,None]*patterns) - marginal[:,None]*marginal[None,:]
    hessian = torch.autograd.functional.hessian(lambda x:log_elementary_symmetric(x[None,:],2).sum(),eta)
    near(hessian,covariance,torch.float64,'ESP/subset_covariance_Hessian',reports)
    for slots in range(7):
        for count in range(slots+1):
            eta = torch.zeros((4,slots),dtype=torch.float64,requires_grad=True)
            bits = torch.tensor([float(i<count) for i in range(slots)],dtype=torch.float64)
            actual = conditional_member_nll(eta,bits)
            near(actual,torch.full((4,),log(comb(slots,count)),dtype=torch.float64),torch.float64,'uniform/log_binomial_NLL',reports)
            gradient = torch.autograd.grad(actual.sum(),eta)[0]
            expected = (torch.full((4,slots),count/max(slots,1),dtype=torch.float64)-bits[None,:])
            near(gradient,expected,torch.float64,'uniform/analytic_inclusion_gradient',reports)
    for members in (1,2,4):
        for nl,nr in product(range(4),repeat=2):
            for zl_tuple,zr_tuple in product(product((0.,1.),repeat=nl),product((0.,1.),repeat=nr)):
                zl,zr = torch.tensor(zl_tuple,dtype=torch.float64),torch.tensor(zr_tuple,dtype=torch.float64)
                left = eta_fixture(members,nl,torch.float64).requires_grad_()
                right = (eta_fixture(members,nr,torch.float64)*.73).requires_grad_()
                labels = TrainPatternLabels(torch.cat((zl,zr)),'complete_TRAIN_observation_membership')
                values = training_pattern_losses(left,right,torch.zeros(nl,dtype=torch.long),torch.zeros(nr,dtype=torch.long),labels,1)
                _,ll,gl = subset_oracle(left,zl)
                _,lr,gr = subset_oracle(right,zr)
                total = ll+lr
                rho = torch.softmax(-total,dim=0)
                denom = max(nl+nr,1)
                expected = {'J_K':(-torch.logsumexp(-total,dim=0)+log(members))/denom,
                            'J_K_sep':(-torch.logsumexp(-ll,dim=0)-torch.logsumexp(-lr,dim=0)+2*log(members))/denom,
                            'W_K':total.mean()/denom}
                weights = {'J_K':(rho,rho),'J_K_sep':(torch.softmax(-ll,dim=0),torch.softmax(-lr,dim=0)),
                           'W_K':(torch.full((members,),1/members,dtype=torch.float64),)*2}
                for name in expected:
                    near(values[name][0],expected[name],torch.float64,f'{name}/exhaustive/{cases["mixture_pattern_cases"]}',reports)
                    ga,gb = torch.autograd.grad(values[name].sum(),(left,right),retain_graph=True)
                    near(ga,weights[name][0][:,None]*gl/denom,torch.float64,name+'/left_analytic_gradient',reports)
                    near(gb,weights[name][1][:,None]*gr/denom,torch.float64,name+'/right_analytic_gradient',reports)
                near(values['rho'][:,0],rho,torch.float64,'J_K/responsibilities',reports)
                require(float(values['J_K'][0]) <= float(values['W_K'][0])+1e-10, 'Jensen mixture bound fails')
                raw = values['J_K'][0]*denom
                require(float(total.min())-1e-10 <= float(raw) <= float(total.min())+log(members)+1e-10, 'Mixture min/logM bound fails')
                if int(zl.sum()) in (0,nl) or int(zr.sum()) in (0,nr):
                    near(values['J_K'],values['J_K_sep'],torch.float64,'unique_side_removes_association',reports)
                shifts = torch.arange(members,dtype=torch.float64)[:,None]*31.+17.
                shifted = training_pattern_losses(left+shifts,right-shifts*3.,torch.zeros(nl,dtype=torch.long),torch.zeros(nr,dtype=torch.long),labels,1)
                for name in ('J_K','J_K_sep','W_K','rho'):
                    near(shifted[name],values[name],torch.float64,name+'/independent_side_member_shift',reports)
                cases['mixture_pattern_cases'] += 1
                check_cap()
    # Nontrivial ragged placement plus empty/extreme queries; reduction retains all Q.
    lrows = torch.tensor([0,0,0,2,2,3],dtype=torch.long)
    rrows = torch.tensor([0,0,2,2,2,3],dtype=torch.long)
    left = eta_fixture(4,6,torch.float64).requires_grad_()
    right = eta_fixture(4,6,torch.float64).requires_grad_()
    labels = TrainPatternLabels(torch.tensor([1,0,0,0,0,1, 0,1,1,1,1,0],dtype=torch.float64),'complete_TRAIN_observation_membership')
    packed = training_pattern_losses(left,right,lrows,rrows,labels,4)
    require(packed['denominator']==[5,1,5,2] and packed['teacher_counts']==[[1,0,0,1],[1,0,3,0]], 'Ragged counts/denominators differ')
    require(bool((packed['J_K'][1:]==0).all()), 'Empty/extreme query loss not exact zero')
    for name in ('J_K','J_K_sep','W_K'):
        ga,gb = torch.autograd.grad(packed[name].mean(),(left,right),retain_graph=True)
        require(bool((ga[:,3:]==0).all()) and bool((gb[:,2:]==0).all()), 'Extreme ragged gradients not exact zero')
    near(packed['J_K'].mean(),packed['J_K'][0]/4,torch.float64,'all_queries_not_informative_only_mean',reports)
    rejected = 0
    for authority in ('VALID_observation_membership','TEST_observation_membership','member_predicted_pattern'):
        caught = False
        try:
            training_pattern_losses(left,right,lrows,rrows,TrainPatternLabels(labels.values,authority),4)
        except ValueError:
            caught = True
        require(caught, 'Non-TRAIN/member-pattern teacher authority admitted')
        rejected += 1
    caught = False
    try:
        training_pattern_losses(left,right,lrows,rrows,TrainPatternLabels(labels.values[None,:].expand(4,-1),'complete_TRAIN_observation_membership'),4)
    except ValueError:
        caught = True
    require(caught, 'Member-axis label patterns admitted')
    rejected += 1
    eta = eta_fixture(1,3,torch.float64).repeat(4,1)
    labels = TrainPatternLabels(torch.tensor([1,0,0,0,1,0],dtype=torch.float64),'complete_TRAIN_observation_membership')
    rows = torch.zeros(3,dtype=torch.long)
    collapsed = training_pattern_losses(eta,eta,rows,rows,labels,1)
    for name in ('W_K','J_K_sep'):
        near(collapsed['J_K'],collapsed[name],torch.float64,'identical_components/'+name,reports)
    near(collapsed['rho'],torch.full((4,1),.25,dtype=torch.float64),torch.float64,'identical_components/uniform_responsibility',reports)
    # Capable single: exhaustive feasible-pattern mass and its parameter gradient.
    head = ConditionalSingleHead().double()
    with torch.no_grad():
        for index,parameter in enumerate(head.parameters()):
            parameter.copy_((torch.arange(parameter.numel(),dtype=torch.float64).reshape(parameter.shape)%17-8)*.003/(index+1))
    require(HEAD_INPUT==523 and HEAD_PARAMETERS==33601, 'Frozen single dimensions differ')
    for nl,nr in product(range(4),repeat=2):
        slots = nl+nr
        context = SingleVisibleContext(torch.linspace(-.2,.2,max(slots*SLOT_FEATURES,1),dtype=torch.float64)[:slots*SLOT_FEATURES].reshape(slots,SLOT_FEATURES),
            torch.linspace(-.1,.1,QUERY_FEATURES,dtype=torch.float64)[None,:],
            torch.linspace(-.3,.3,max(slots*WIDTH,1),dtype=torch.float64)[:slots*WIDTH].reshape(slots,WIDTH),
            torch.zeros(nl,dtype=torch.long),torch.zeros(nr,dtype=torch.long),1)
        for kl,kr in product(range(nl+1),range(nr+1)):
            probabilities = []
            for zl,zr in product(subsets(nl,kl),subsets(nr,kr)):
                nll = head.training_nll(context,TrainPatternLabels(torch.cat((zl,zr)),'complete_TRAIN_observation_membership'))[0]*max(slots,1)
                probabilities.append(torch.exp(-nll))
                cases['single_enumerated_patterns'] += 1
                if kl in (0,nl) and kr in (0,nr):
                    require(float(nll)==0., 'Extreme single loss not exact zero')
                    grads = torch.autograd.grad(nll,tuple(head.parameters()),retain_graph=True)
                    require(all(bool((g==0).all()) for g in grads), 'Extreme single parameter gradients not exact zero')
            mass = torch.stack(probabilities).sum()
            near(mass,torch.ones((),dtype=torch.float64),torch.float64,'single/exhaustive_supported_mass',reports)
            grads = torch.autograd.grad(mass,tuple(head.parameters()))
            for grad in grads:
                near(grad,torch.zeros_like(grad),torch.float64,'single/normalization_parameter_gradient',reports)
            cases['single_normalization_groups'] += 1
            check_cap()
    # Label-free centering of the single's native unary context preserves shifts.
    h = torch.linspace(-.3,.3,8*WIDTH,dtype=torch.float64).reshape(8,WIDTH)
    rows = torch.zeros(3,dtype=torch.long)
    neighbors = SimpleNamespace(left=(rows,torch.tensor([2,3,4])),right=(rows,torch.tensor([5,6,7])),queries=1)
    q = torch.tensor([[0,1]])
    left = torch.tensor([.75,-.125,.375],dtype=torch.float64,requires_grad=True)
    right = torch.tensor([-.5,.25,1.],dtype=torch.float64,requires_grad=True)
    labels = TrainPatternLabels(torch.tensor([0,1,0,0,0,1],dtype=torch.float64),'complete_TRAIN_observation_membership')
    context = visible_context(h,q,neighbors,left,right)
    base_loss = head.training_nll(context,labels)
    base_grad = torch.autograd.grad(base_loss.sum(),(left,right))
    shifted_left = (left.detach()+512).requires_grad_()
    shifted_right = (right.detach()-256).requires_grad_()
    shifted_context = visible_context(h,q,neighbors,shifted_left,shifted_right)
    shifted_loss = head.training_nll(shifted_context,labels)
    shifted_grad = torch.autograd.grad(shifted_loss.sum(),(shifted_left,shifted_right))
    near(shifted_loss,base_loss,torch.float64,'single/visible_unary_shift_NLL',reports)
    for actual,reference in zip(shifted_grad,base_grad):
        near(actual,reference,torch.float64,'single/visible_unary_shift_gradient',reports)
        near(reference.sum(),torch.zeros((),dtype=torch.float64),torch.float64,'single/visible_unary_zero_sum_gradient',reports)
    # Same matching-side law from shared CB mixture and one identity-sensitive head.
    a = .9
    p = float(torch.sigmoid(torch.tensor(2*a,dtype=torch.float64)))
    matching_given_left = p*p+(1-p)*(1-p)
    b = log(matching_given_left/(1-matching_given_left))
    with torch.no_grad():
        for parameter in head.parameters():parameter.zero_()
        prefix_coordinate = SLOT_FEATURES+QUERY_FEATURES
        head.network[0].weight[0,prefix_coordinate]=1.
        head.network[0].weight[1,prefix_coordinate]=-1.
        head.network[2].weight[0,0]=b
        head.network[2].weight[0,1]=-b
    h = torch.zeros((6,WIDTH),dtype=torch.float64)
    h[2,0],h[3,0],h[4,0],h[5,0]=1.,-1.,1.,-1.
    rows = torch.zeros(2,dtype=torch.long)
    neighbors = SimpleNamespace(left=(rows,torch.tensor([2,3])),right=(rows,torch.tensor([4,5])),queries=1)
    context = visible_context(h,torch.tensor([[0,1]]),neighbors,torch.zeros(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64))
    eta = torch.tensor([[a,-a],[-a,a]],dtype=torch.float64)
    for zl,zr in product(subsets(2,1),repeat=2):
        same = bool(torch.equal(zl,zr))
        expected = torch.tensor(matching_given_left/2 if same else (1-matching_given_left)/2,dtype=torch.float64)
        labels = TrainPatternLabels(torch.cat((zl,zr)),'complete_TRAIN_observation_membership')
        values = training_pattern_losses(eta,eta,rows,rows,labels,1)
        near(torch.exp(-values['J_K'][0]*4),expected,torch.float64,'matching/shared_mixture_law',reports)
        near(torch.exp(-values['J_K_sep'][0]*4),torch.tensor(.25,dtype=torch.float64),torch.float64,'matching/independent_side_law',reports)
        nll = head.training_nll(context,labels)[0]*4
        near(torch.exp(-nll),expected,torch.float64,'matching/capable_single_law',reports)
        gradient = torch.autograd.grad(nll,head.network[2].weight)[0]
        actual_b_gradient = gradient[0,0]-gradient[0,1]
        expected_b_gradient = torch.tensor(matching_given_left-float(same),dtype=torch.float64)
        near(actual_b_gradient,expected_b_gradient,torch.float64,'matching/single_analytic_parameter_gradient',reports)
        cases['matching_law_patterns'] += 1
    check_cap()
    require(cases=={'ESP_pattern_cases':508,'mixture_pattern_cases':675,'single_normalization_groups':100,
                   'single_enumerated_patterns':225,'matching_law_patterns':4}, 'Declared exhaustive coverage differs')
    return {'cases':cases,'comparison_reports':reports,'head_input_dimensions':HEAD_INPUT,'auxiliary_head_parameters':HEAD_PARAMETERS,
            'non_TRAIN_or_member_label_contract_rejections':rejected,
            'law_oracles_only':True,'numerical_predictive_experiment':False,'real_data_or_graph_context_read':False,
            'native_neural_schedule_or_full_batch_resource_qualified':False,'novelty_or_served_ranking_claim':False}
