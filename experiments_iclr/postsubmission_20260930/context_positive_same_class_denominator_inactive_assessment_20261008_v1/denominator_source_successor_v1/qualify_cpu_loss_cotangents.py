"""Prepared CPU artificial loss/cotangent fixture; no model, role, fit or CUDA."""
import json
from pathlib import Path
from denominator_objective import (directional_target_ce, masked_context_alignment,
                                   retained_entries)


def main():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2)
    labels = torch.tensor([0,0,0,1,1,2], dtype=torch.long)
    weights = torch.zeros((4,6,6), dtype=torch.float64)
    for member in range(4):
        for row in range(6):
            same = torch.nonzero(labels == labels[row], as_tuple=True)[0]
            weights[member,row,row] = 1.
            if len(same) > 1 and (member+row)%2 == 0:
                pos = int(torch.nonzero(same == row, as_tuple=True)[0][0])
                other = int(same[(pos+1)%len(same)])
                weights[member,row,other] = 1.
            weights[member,row] /= weights[member,row].sum()
    retained = retained_entries(weights, labels)
    checks = []
    def check(value, name):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    def reference(scores):
        losses = []
        for member in range(scores.shape[0]):
            for row in range(scores.shape[1]):
                keep = retained[member,row]
                losses.append(-(weights[member,row,keep] * F.log_softmax(scores[member,row,keep], -1)).sum())
        return torch.stack(losses).mean()
    scores = (torch.arange(4*6*6, dtype=torch.float64).reshape(4,6,6).sin()*3).requires_grad_()
    safe = directional_target_ce(scores, weights, retained)
    gathered = reference(scores)
    safe_g = torch.autograd.grad(safe, scores, retain_graph=True)[0]
    reference_g = torch.autograd.grad(gathered, scores)[0]
    check(torch.isfinite(safe) and torch.isfinite(safe_g).all(), 'finite loss/cotangents with excluded zero-target entries')
    check(torch.allclose(safe, gathered, atol=1e-12, rtol=1e-12) and
          torch.allclose(safe_g, reference_g, atol=1e-12, rtol=1e-12), 'retained-set gathered CE loss and score cotangents agree')
    check(torch.equal(safe_g[~retained], torch.zeros_like(safe_g[~retained])), 'excluded direct directional score cotangents are exactly zero')
    naive = -(weights * F.log_softmax(scores.masked_fill(~retained, -torch.inf), -1)).sum(-1).mean()
    check(torch.isnan(naive), 'fixture exposes0times-negative-infinity hazard in naive masked target CE')
    a = torch.sin(torch.arange(4*6*5, dtype=torch.float64).reshape(4,6,5)/7).requires_grad_()
    b = torch.cos(torch.arange(4*6*5, dtype=torch.float64).reshape(4,6,5)/9).requires_grad_()
    symmetric = masked_context_alignment(a,b,weights,labels)
    s = torch.einsum('mbd,mcd->mbc', F.normalize(a,dim=-1), F.normalize(b,dim=-1))/.2
    symmetric_reference = .5*(reference(s)+reference(s.transpose(1,2)))
    g = torch.autograd.grad(symmetric,(a,b),retain_graph=True)
    rg = torch.autograd.grad(symmetric_reference,(a,b))
    check(torch.allclose(symmetric,symmetric_reference,atol=1e-12,rtol=1e-12) and
          all(torch.allclose(x,y,atol=1e-12,rtol=1e-12) for x,y in zip(g,rg)),
          'directed symmetric anchor convention and representation cotangents agree')
    # When every denominator entry is retained, recover literal full target CE.
    all_scores = scores.detach().clone().requires_grad_()
    full = -(weights*F.log_softmax(all_scores,-1)).sum(-1).mean()
    all_safe = directional_target_ce(all_scores,weights,torch.ones_like(retained))
    fg = torch.autograd.grad(full,all_scores,retain_graph=True)[0]
    ag = torch.autograd.grad(all_safe,all_scores)[0]
    check(torch.allclose(full,all_safe,atol=1e-12,rtol=1e-12) and torch.allclose(fg,ag,atol=1e-12,rtol=1e-12),
          'all-retained limit preserves original weighted CE and cotangents')
    single_class = torch.zeros(6,dtype=torch.long)
    self_weights = torch.eye(6,dtype=torch.float64)[None].repeat(4,1,1)
    singleton_scores = scores.detach().clone().requires_grad_()
    zero = directional_target_ce(singleton_scores,self_weights,retained_entries(self_weights,single_class))
    zg = torch.autograd.grad(zero,singleton_scores)[0]
    check(zero == 0 and torch.equal(zg,torch.zeros_like(zg)), 'self-only retained row without other classes has zero loss/cotangent')
    extreme = scores.detach().float().clone(); extreme[0,0,0] = 1000.; extreme[0,0,3] = -1000.
    extreme.requires_grad_(); extreme_loss = directional_target_ce(extreme,weights.float(),retained)
    eg = torch.autograd.grad(extreme_loss,extreme)[0]
    check(torch.isfinite(extreme_loss) and torch.isfinite(eg).all(), 'finite float32 extreme-score loss/cotangents')
    print(json.dumps(dict(status='CPU_ARTIFICIAL_LOSS_COTANGENT_FIXTURE_ONLY',checks=checks,
        scientific_data_access=False, model_or_optimizer_calls=0, serving_calls=0, native_or_CUDA_qualified=False,
        numerical_tolerances_scope='artificial helper equivalence only; no scientific or native parity gate'),indent=2))


if __name__ == '__main__':
    main()
