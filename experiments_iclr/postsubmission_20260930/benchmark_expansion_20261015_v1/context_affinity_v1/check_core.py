"""Small semantic checks; no native task or predictive evidence."""
import json
import torch
from core import context_auxiliary


def inputs():
    g = torch.Generator().manual_seed(41017)
    return dict(preheads=torch.randn(4, 6, 5, generator=g, dtype=torch.float64, requires_grad=True),
                labels=torch.tensor([0,0,0,1,1,1]), object_ids=torch.arange(6),
                context_a=torch.randn(6, 3, generator=g, dtype=torch.float64, requires_grad=True),
                context_b=torch.randn(6, 4, generator=g, dtype=torch.float64, requires_grad=True),
                reduction="class_balanced")


def run():
    checks = []
    x = inputs()
    common = context_auxiliary(torch, mode="common", **x)
    split = context_auxiliary(torch, mode="split", **x)
    assert common["diagnostics"]["eligible_anchors"] == 6
    ga, = torch.autograd.grad(common["loss"], x["preheads"], retain_graph=True)
    gb, = torch.autograd.grad(split["loss"], x["preheads"], retain_graph=True)
    assert ga.isfinite().all() and gb.isfinite().all() and not torch.equal(ga,gb)
    unused = torch.autograd.grad(split["loss"], (x["context_a"],x["context_b"]), allow_unused=True)
    assert unused == (None,None)
    checks.append("finite_live_prehead_gradients_different_assignments_and_detached_targets")
    # If factual channels coincide, common and split are the same intervention.
    y = {**x,"context_b":x["context_a"]}
    a=context_auxiliary(torch,mode="common",**y);b=context_auxiliary(torch,mode="split",**y)
    assert torch.equal(a["loss"], b["loss"])
    checks.append("identical_context_channels_exact_common_split_equivalence")
    # Disjoint singleton classes cannot invent donors.
    y={**x,"labels":torch.arange(6)}
    r=context_auxiliary(torch,mode="split",**y)
    assert float(r["loss"]) == 0 and r["diagnostics"]["eligible_anchors"] == 0
    assert torch.equal(torch.autograd.grad(r["loss"],x["preheads"])[0],torch.zeros_like(x["preheads"]))
    checks.append("unsupported_classes_zero_auxiliary_and_gradient")
    # Joint and separate source reductions differ by their declared group scale
    # when equal-size groups have full donor support.
    joint=context_auxiliary(torch,mode="split",**{**x,"reduction":"joint"})
    separate=context_auxiliary(torch,mode="split",**{**x,"reduction":"binary_separate"})
    assert torch.allclose(separate["loss"],2*joint["loss"],atol=1e-14,rtol=1e-14)
    checks.append("binary_native_separate_group_sum_not_joint_mean")
    # Autograd compared to a numerical derivative at a nonzero, nondegenerate point.
    ix=(2,3,1);step=1e-6
    vals=[]
    for direction in (-1,1):
        z=x['preheads'].detach().clone();z[ix]+=direction*step
        vals.append(context_auxiliary(torch,mode="split",**{**x,'preheads':z})['loss'])
    numeric=(vals[1]-vals[0])/(2*step)
    assert torch.allclose(numeric,gb[ix],atol=1e-8,rtol=1e-5)
    checks.append("assignment_gradient_finite_difference")
    # A relation partition may remove otherwise same-class donor opportunities.
    r=context_auxiliary(torch,mode='common',**{**x,'relation_ids':torch.arange(6)})
    assert r['diagnostics']['eligible_anchors'] == 0
    checks.append("different_relations_never_become_donors")
    r=context_auxiliary(torch,mode='common',**{**x,'context_b':torch.zeros_like(x['context_b'])})
    assert r['diagnostics']['zero_context_b'] == 6 and torch.isfinite(r['loss'])
    checks.append("zero_context_uniform_target_finite")
    try:context_auxiliary(torch,mode='split',**{**x,'object_ids':torch.zeros_like(x['object_ids'])})
    except ValueError:pass
    else:raise AssertionError('Duplicate object identities accepted')
    checks.append("duplicate_objects_rejected")
    return dict(status='passed',checks=checks,count=len(checks),native_constructions=0,
                optimizer_steps=0,scientific_fits=0,TEST_accessed=False,
                scope='tensor semantics only; actual factual targets/native ports pending')


if __name__=='__main__':print(json.dumps(run(),indent=2))
