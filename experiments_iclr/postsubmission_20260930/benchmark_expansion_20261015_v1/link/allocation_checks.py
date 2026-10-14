"""Synthetic numerical checks for root to invoke in the admitted allocation.

Not executed locally; these checks do not establish actual backbone/pilot
qualification. No dataset, source model, benchmark, or fit loop is launched.
"""
from .objectives import build_objectives, native_record_losses, reduce_native, step_blocks
from .adapters import make_hidden_residual, make_late_gate
from .planning import authenticate, query_keys


def run(torch):
    groups = [1,1,0,0,0,0]
    logits = torch.tensor([[-3.,-1.,2.,-2.,1.,3.]]*4, requires_grad=True)
    records = native_record_losses(torch, logits, groups, 'NCN')
    ncn = -torch.nn.functional.logsigmoid(logits[:, :2]).mean(1) - torch.nn.functional.logsigmoid(-logits[:, 2:]).mean(1)
    buddy = torch.nn.functional.binary_cross_entropy_with_logits(logits,
        torch.tensor(groups, dtype=logits.dtype).expand_as(logits), reduction='none').mean(1)
    assert torch.allclose(reduce_native(torch, records, groups, 'NCN'), ncn)
    assert torch.allclose(reduce_native(torch, native_record_losses(torch, logits, groups, 'BUDDY'), groups, 'BUDDY'), buddy)
    for family in ('NCN', 'NCNC', 'BUDDY'):
        rows = native_record_losses(torch, logits, groups, family)
        assert torch.allclose(reduce_native(torch, rows, groups, family),
                              reduce_native(torch, rows, groups, family, torch.ones_like(rows)))
    preheads = (torch.arange(120, dtype=torch.float).view(4,6,5)/30).requires_grad_()
    query_logits = preheads.sum(-1)
    source_losses = [-torch.nn.functional.logsigmoid(row[:2]).mean() -
                     torch.nn.functional.logsigmoid(-row[2:]).mean() for row in query_logits]
    endpoints = [(1,2),(1,3),(1,4),(2,4),(3,5),(7,8)]
    signature = authenticate(query_keys(endpoints), groups, 'TRAIN')
    for contrast in ('endpoint', 'cosine', 'global'):
        bundle = build_objectives(torch, family='NCNC', logits=query_logits, preheads=preheads,
            endpoints=endpoints, groups=groups, native_member_losses=source_losses,
            route_query_signatures=[signature]*4, contrast=contrast, responsibility='endpoint',
            seed=7, anchor_limit=6, negative_limit=2)
        gradient = torch.autograd.grad(bundle['private_objective'], preheads, retain_graph=True)[0]
        assert bool(torch.isfinite(gradient).all())
        assert bool(torch.isfinite(bundle['query_auxiliary_loss']))
    shared = torch.nn.Parameter(torch.tensor(2.))
    private = torch.nn.Parameter(torch.tensor(3.))
    optimizer = torch.optim.SGD([shared, private], lr=.01)
    left, right = shared.square()+private.square(), 2*shared.square()+3*private.square()
    old_s, old_p = shared.detach().clone(), private.detach().clone()
    receipt = step_blocks(torch, optimizer, [shared], [private],
                          {'shared_objective': left, 'private_objective': right})
    assert torch.allclose(shared, old_s-.01*2*old_s)
    assert torch.allclose(private, old_p-.01*6*old_p)
    assert receipt['optimizer_transitions'] == 1 and receipt['reverse_collections'] == 2
    hidden = torch.arange(24, dtype=torch.float).view(4,2,3)/10
    adapter = make_hidden_residual(torch, 3, seed=7)
    for member in range(4):
        assert torch.equal(adapter(hidden[member], member), hidden[member])
    gate = make_late_gate(torch, 3, seed=7)
    scores = torch.arange(8, dtype=torch.float).view(4,2,1)/10
    served, weights = gate(scores, hidden)
    assert torch.allclose(served, scores.mean(0))
    assert torch.equal(weights, torch.full_like(weights, .25))
    return {'native_NCN_NCNC_positive_negative_reductions': 'passed',
        'native_BUDDY_joint_reduction': 'passed', 'unit_responsibility_recovers_native': 'passed',
        'conditional_global_cosine_auxiliary_gradients': 'passed',
        'old_state_block_gradient_and_one_step': 'passed', 'hidden_adapter_initial_identity': 'passed',
        'late_gate_initial_mean_logits': 'passed', 'scope': 'synthetic numerical checks only'}
