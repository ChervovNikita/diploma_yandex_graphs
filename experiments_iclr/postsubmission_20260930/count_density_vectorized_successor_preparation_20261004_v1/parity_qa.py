"""Fabricated CPU scalar-reference value, gradient and chunk parity.

No native model, teacher/data reader or scientific fit. Not executed here.
"""
from copy import deepcopy
import torch
from torch import nn
import hardened_count_density as scalar
import vector_density as vector
from vector_head import potential
import ragged_density as ragged


def close(actual, expected, message, atol=2e-10, rtol=2e-10):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(message + "; max_abs=" + str(float((actual - expected).abs().max()) if actual.numel() else 0.))


def scalar_head(head, context, swapped, nl, nr):
    k, l = torch.meshgrid(torch.arange(nl + 1, dtype=context.dtype), torch.arange(nr + 1, dtype=context.dtype), indexing="ij")
    def features(a, b, na, nb):
        return torch.stack((a / max(na, 1), b / max(nb, 1), torch.log1p(a), torch.log1p(b),
                            torch.log1p(na - a), torch.log1p(nb - b), (a == 0).to(a.dtype), (b == 0).to(a.dtype),
                            (a == na).to(a.dtype), (b == nb).to(a.dtype)), -1)
    a = head(torch.cat((context.expand(nl+1, nr+1, -1), features(k, l, nl, nr)), -1)).squeeze(-1)
    b = head(torch.cat((swapped.expand(nl+1, nr+1, -1), features(l, k, nr, nl)), -1)).squeeze(-1)
    return (.5 * (a + b)).to(torch.float64)


def run():
    torch.set_num_threads(2)
    gen = torch.Generator(device="cpu").manual_seed(2026100401)
    # Every0..8 support/partition shares one padded batch. Invalid unary slots
    # contain finite nonzero sentinels to expose accidentally used padding.
    sizes = [(a, size-a) for size in range(9) for a in range(size+1)]
    nl = torch.tensor([a for a,b in sizes]); nr = torch.tensor([b for a,b in sizes])
    left = torch.randn((len(sizes), 8), dtype=torch.float64, generator=gen)
    right = torch.randn((len(sizes), 8), dtype=torch.float64, generator=gen)
    g = torch.randn((len(sizes), 9, 9), dtype=torch.float64, generator=gen)
    zl = torch.zeros_like(left); zr = torch.zeros_like(right)
    for i,(a,b) in enumerate(sizes):
        zl[i,:a] = torch.arange(a) % 2
        zr[i,:b] = 1 - torch.arange(b) % 2
    z = vector.log_normalizers(left, right, nl, nr, g)
    lm, rm, _ = vector.marginals(left, right, nl, nr, g)
    loss = vector.nll(left, right, nl, nr, g, zl, zr)
    for i,(a,b) in enumerate(sizes):
        gs = g[i,:a+1,:b+1]
        close(z[i], scalar.log_normalizer(left[i,:a], right[i,:b], gs), "normalizer")
        mu,_ = scalar.unary_marginals(left[i,:a],right[i,:b],gs)
        close(lm[i,:a],mu[:a],"left marginal"); close(rm[i,:b],mu[a:],"right marginal")
        close(lm[i,a:],torch.zeros_like(lm[i,a:]),"left padding"); close(rm[i,b:],torch.zeros_like(rm[i,b:]),"right padding")
        close(loss[i],scalar.query_nll(left[i,:a],right[i,:b],gs,torch.cat((zl[i,:a],zr[i,:b]))),"NLL")
    lv,rv,gv = (x.clone().requires_grad_() for x in (left,right,g))
    vl = vector.nll(lv,rv,nl,nr,gv,zl,zr).mean()
    vd = torch.autograd.grad(vl,(lv,rv,gv))
    ls,rs,gs = (x.clone().requires_grad_() for x in (left,right,g))
    sl = torch.stack([scalar.query_nll(ls[i,:a],rs[i,:b],gs[i,:a+1,:b+1],torch.cat((zl[i,:a],zr[i,:b])))
                      for i,(a,b) in enumerate(sizes)]).mean()
    sd = torch.autograd.grad(sl,(ls,rs,gs))
    for a,b in zip(vd,sd):
        close(a,b,"padded all-query mean gradient")
        if not torch.isfinite(a).all():
            raise AssertionError("Impossible-cell gradient became nonfinite")
    # Cross-row padding and extreme unary values cannot alter any real row.
    extreme_l = torch.tensor([[1000.,-1000.],[1000.,1000.],[-1000.,-1000.]],dtype=torch.float64)
    extreme_r = extreme_l.flip(1)
    counts = torch.tensor([2,1,0]); eg = torch.zeros((3,3,3),dtype=torch.float64)
    el,er,_=vector.marginals(extreme_l,extreme_r,counts,counts,eg)
    for i,n in enumerate(counts.tolist()):
        mu,_=scalar.unary_marginals(extreme_l[i,:n],extreme_r[i,:n],eg[i,:n+1,:n+1])
        close(el[i,:n],mu[:n],"extreme left");close(er[i,:n],mu[n:],"extreme right")
    # Head parity uses the unchanged530->64->1 architecture. Neural maps are
    # FP32, so batched GEMM parity is tolerance-based rather than bitwise.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(610041)
        head=nn.Sequential(nn.Linear(530,64),nn.ReLU(),nn.Linear(64,1))
    context=torch.randn((7,520),generator=gen); swapped=torch.randn((7,520),generator=gen)
    lengths_l=torch.tensor([0,1,2,0,3,5,8]);lengths_r=torch.tensor([0,0,1,4,2,1,8])
    gb=potential(head,context,swapped,lengths_l,lengths_r,head_state_chunk=17)
    for i,(a,b) in enumerate(zip(lengths_l.tolist(),lengths_r.tolist())):
        close(gb[i,:a+1,:b+1],scalar_head(head,context[i],swapped[i],a,b),"neural head",atol=2e-6,rtol=1e-5)
    lo=torch.cat((torch.zeros(1,dtype=torch.long),lengths_l.cumsum(0)))
    ro=torch.cat((torch.zeros(1,dtype=torch.long),lengths_r.cumsum(0)))
    eta_l=torch.randn(int(lo[-1]),dtype=torch.float64,generator=gen)
    eta_r=torch.randn(int(ro[-1]),dtype=torch.float64,generator=gen)
    labels_l=(torch.arange(len(eta_l))%2).to(torch.float64)
    labels_r=(1-torch.arange(len(eta_r))%2).to(torch.float64)
    def gradients(workspace,recompute):
        h=deepcopy(head);c=context.clone().requires_grad_();cs=swapped.clone().requires_grad_()
        l=eta_l.clone().requires_grad_();r=eta_r.clone().requires_grad_()
        values=ragged.source_nll(l,r,lo,ro,c,cs,h,labels_l,labels_r,
                                  recompute=recompute,workspace_cells=workspace,max_queries=3)
        derivatives=torch.autograd.grad(values.mean(),(l,r,c,cs,*h.parameters()))
        return values,derivatives
    direct,dg=gradients(100000,False);checkpointed,cg=gradients(50,True)
    close(direct,checkpointed,"chunk/checkpoint NLL",atol=2e-6,rtol=1e-5)
    for a,b in zip(dg,cg):close(a,b,"chunk/checkpoint full neural gradient",atol=2e-6,rtol=2e-5)
    h=deepcopy(head);c=context.clone().requires_grad_();cs=swapped.clone().requires_grad_()
    l=eta_l.clone().requires_grad_();r=eta_r.clone().requires_grad_()
    reference=[]
    for i,(a,b) in enumerate(zip(lengths_l.tolist(),lengths_r.tolist())):
        reference.append(scalar.query_nll(l[lo[i]:lo[i+1]],r[ro[i]:ro[i+1]],scalar_head(h,c[i],cs[i],a,b),
                                          torch.cat((labels_l[lo[i]:lo[i+1]],labels_r[ro[i]:ro[i+1]]))))
    reference=torch.stack(reference)
    rg=torch.autograd.grad(reference.mean(),(l,r,c,cs,*h.parameters()))
    close(direct,reference,"scalar neural NLL",atol=2e-6,rtol=1e-5)
    for a,b in zip(dg,rg):close(a,b,"scalar neural gradient",atol=2e-6,rtol=2e-5)
    full_mu=ragged.unary_marginals(eta_l,eta_r,lo,ro,context,swapped,head,workspace_cells=100000)
    chunk_mu=ragged.unary_marginals(eta_l,eta_r,lo,ro,context,swapped,head,workspace_cells=50,max_queries=2)
    for a,b in zip(full_mu,chunk_mu):close(a,b,"chunk marginal parity",atol=2e-6,rtol=1e-5)
    if any(value.requires_grad for value in full_mu):
        raise AssertionError("Marginal route failed detachment")
    return {"status":"PASS","padded_geometries":len(sizes),"neural_ragged_geometries":7,
            "scalar_gradient_and_checkpoint_parity":True,"chunk_and_full_slot_coverage":True,
            "native_model_or_full_batch_feasibility_qualified":False}
