"""Attributed exact CMCL overlap and own-floor terms, numerical calls disabled."""
import math
from .caps import CLOSED
from .native import require
from .plan import MEMBERS, CLASSES, OWNER_K, BETA, LAMBDA


def node_terms(torch, logits, truth, caps=CLOSED):
    caps.require('source_bound', 'data', 'model', 'runtime')
    require(logits.ndim == 2 and logits.shape == (len(truth), CLASSES)
            and logits.dtype == torch.float32 and truth.dtype == torch.long
            and torch.isfinite(logits).all().item(), 'Complete FP32 categorical PubMed TRAIN logits')
    logp = torch.log_softmax(logits, dim=-1)
    ce = -logp.gather(1, truth[:, None]).squeeze(1)
    kl = -math.log(CLASSES)-logp.mean(dim=-1)  # KL(U_C || P), not reverse KL.
    return ce, kl


def assignment(torch, ces, kls, caps=CLOSED):
    caps.require('source_bound', 'data', 'model', 'runtime')
    require(ces.shape == kls.shape and ces.shape[0] == MEMBERS
            and not ces.requires_grad and not kls.requires_grad
            and torch.isfinite(ces).all().item() and torch.isfinite(kls).all().item(), 'Frozen complete incoming four-member TRAIN score table')
    # Total cost for owner set O is beta*sum KL + sum_O(CE-beta*KL).
    owners = torch.argsort(ces-BETA*kls, dim=0, stable=True)[:OWNER_K]
    mask = torch.zeros_like(ces, dtype=torch.bool)
    mask.scatter_(0, owners, True)
    require((mask.sum(dim=0) == OWNER_K).all().item(), 'Exactly K3 owners with fixed member-index ties')
    return mask


def member_terms(torch, ce, kl, spec, owner, caps=CLOSED):
    caps.require('source_bound', 'data', 'model', 'runtime')
    floor = ce.mean()/MEMBERS if spec['own_floor'] else None
    if spec['auxiliary'] == 'CMCL':
        require(owner.shape == ce.shape and owner.dtype == torch.bool and not owner.requires_grad, 'Stopped same-state ownership')
        aux = LAMBDA*torch.where(owner, ce, BETA*kl).mean()
        # SUM these member terms; no M/K division in the published CMCL component.
    elif spec['auxiliary'] == 'uniform':
        # Match total nonowner mass M-K without choosing specialist assignments.
        aux = LAMBDA*BETA*(MEMBERS-OWNER_K)/MEMBERS*kl.mean()
    elif spec['auxiliary'] == 'constant':
        # Expected owner/nonowner mass without discrete assignments: separates
        # extra private CE gradient scale from state-dependent specialist credit.
        aux = LAMBDA*(OWNER_K/MEMBERS*ce.mean()
                      + BETA*(MEMBERS-OWNER_K)/MEMBERS*kl.mean())
    else:
        aux = None
    return floor, aux
