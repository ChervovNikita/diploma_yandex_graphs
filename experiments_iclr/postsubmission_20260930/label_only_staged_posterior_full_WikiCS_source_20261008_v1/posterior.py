"""New nonlinear single and bounded serving map; unchanged label operators."""
import math


def require(value, message):
    if not value:
        raise ValueError(message)


def nonlinear_single(bank, *, torch, core_module, controls_module, initializer_seed, device):
    """Replace only the new instance's readout and rebuild its full Adam owner."""
    require(bank.mode == 'multihead_single4' and bank.members == 1 and bank.attention_heads == 4,
            'Four existing label-message heads and one prediction required')
    with core_module._isolated_constructor_rng(torch, initializer_seed + 700001):
        readout = torch.nn.Sequential(torch.nn.Linear(256,256,bias=False,dtype=torch.float32), torch.nn.ReLU(),
                                      torch.nn.Linear(256,10,bias=False,dtype=torch.float32))
        with torch.no_grad(): readout[2].weight.zero_()
    bank.joint_output = readout.to(device=device, dtype=torch.float32)
    bank.optimizers = [torch.optim.Adam(bank.parameters(),lr=.001,eps=1e-8,weight_decay=0.)]
    bank.optimizer = controls_module.OptimizerBank(bank.optimizers, bank.mode)
    original = bank.descriptor
    def descriptor():
        value = original()
        value.update(joint_readout='bias_free_256_to256_ReLU_256_to10_zero_final',
                     learned_parameter_formula='4*(128*F+128*C+4096)+256*256',
                     adapter_readout_replaced=True, old_linear_readout_optimizer_retained=False)
        return value
    bank.descriptor = descriptor
    bank._check_roles(); bank._finite_state()
    owned = {id(p) for g in bank.optimizers[0].param_groups for p in g['params']}
    require(owned == {id(p) for p in bank.parameters()}, 'Complete nonlinear single Adam ownership')
    return bank


def anchor_reach(torch, bank):
    """Literal permitted-TRAIN support; no score threshold or heldout truth."""
    legal = torch.zeros(bank.nodes,dtype=torch.bool,device=bank.train_ids.device)
    legal[bank.train_ids] = True
    counts = torch.zeros(bank.nodes,dtype=torch.long,device=bank.train_ids.device)
    counts.index_add_(0,bank.edge_target,legal[bank.edge_source].long())
    return counts > 0


def bounded(torch, label_logits, native_logits, reach, ids):
    """Actual served route mixtures; stable log probabilities and exact fallback."""
    require(label_logits.ndim == 3 and label_logits.shape[1:] == (len(ids),10), 'Full route class logits')
    native = native_logits[ids]
    native_lp = torch.nn.functional.log_softmax(native,dim=-1)
    native_p = torch.softmax(native,dim=-1)
    label_lp = torch.nn.functional.log_softmax(label_logits,dim=-1)
    label_p = torch.softmax(label_logits,dim=-1)
    support = reach[ids]
    mix_lp = torch.logaddexp(native_lp[None]+math.log(.2),label_lp+math.log(.8))
    route_lp = torch.where(support[None,:,None],mix_lp,native_lp[None].expand_as(label_lp))
    route_p = torch.where(support[None,:,None],.2*native_p[None]+.8*label_p,
                          native_p[None].expand_as(label_p))
    pool_lp = torch.logsumexp(route_lp,dim=0)-math.log(len(route_lp))
    pool_lp = torch.where(support[:,None],pool_lp,native_lp)
    pool_p = torch.where(support[:,None],route_p.mean(0),native_p)
    require(bool(torch.isfinite(route_lp).all()) and bool(torch.isfinite(pool_lp).all())
            and bool(torch.isfinite(route_p).all()) and bool(torch.isfinite(pool_p).all()), 'Finite stable mixtures')
    require(torch.equal(pool_p[~support],native_p[~support])
            and torch.equal(route_p[:,~support],native_p[None,~support].expand(len(route_p),-1,-1))
            and bool((route_lp >= native_lp[None]+math.log(.2)-2e-5).all()),
            'Exact native fallback and declared probability-floor log bound')
    return dict(served_probabilities=pool_p,served_log_probabilities=pool_lp,
        member_probabilities=route_p,member_log_probabilities=route_lp,
        label_member_logits=label_logits,heldout_ids=ids.clone(),label_support=support)


def readout(torch, prediction, truth):
    require(truth.shape == (5274,) and prediction['served_probabilities'].shape == (5274,10),
            'Complete development readout')
    p,lp = prediction['served_probabilities'],prediction['served_log_probabilities']
    mp,mlp = prediction['member_probabilities'],prediction['member_log_probabilities']
    member_counts = [int((row.argmax(-1)==truth).sum()) for row in mp]
    member_nll = [float(-row.gather(1,truth[:,None]).mean()) for row in mlp]
    counts = int((p.argmax(-1)==truth).sum())
    stats = dict(correctcount=counts,served_accuracy=counts/5274,
        served_NLL=float(-lp.gather(1,truth[:,None]).mean()),
        Brier=float(((p-torch.nn.functional.one_hot(truth,10))**2).sum(-1).mean()),
        member_correctcount=member_counts,member_accuracy=[x/5274 for x in member_counts],
        member_NLL=member_nll,mean_member_accuracy=sum(member_counts)/(len(member_counts)*5274),
        worst_member_accuracy=min(member_counts)/5274,mean_member_NLL=sum(member_nll)/len(member_nll),
        worst_member_NLL=max(member_nll),member_metric_semantics='Actual per-route bounded mixtures')
    require(all(math.isfinite(x) for x in [stats['served_NLL'],stats['Brier'],*member_nll]),
            'Finite mixture scores, including native-floor rows')
    return stats
