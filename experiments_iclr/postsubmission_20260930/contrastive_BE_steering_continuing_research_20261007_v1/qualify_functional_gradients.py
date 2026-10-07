"""CPU Torch algorithm qualifier, not training or evidence from scientific data.

Prepared only; root may execute in the existing repo-owned numeric environment.
"""
import json
from pathlib import Path
import numpy as np
import torch
from context_positive_masks import graph_contexts,positive_masks,sampled_weights
from context_alignment_objective import context_alignment,objective


def main():
    torch.set_num_threads(2)
    torch.manual_seed(133721)
    rng=np.random.default_rng(99221)
    count,width,hidden,classes=18,5,7,3
    labels=np.repeat(np.arange(classes),6)
    x_np=rng.normal(size=(count,width))
    src=np.arange(count);dst=(src+1)%count
    edges=np.stack([np.concatenate([src,dst]),np.concatenate([dst,src])])
    masks=positive_masks(graph_contexts(x_np,edges,src),labels,neighbors=2)
    qr=sampled_weights(masks,src,'route');qc=sampled_weights(masks,src,'common')
    assert np.max(np.abs(qr-qc))>0
    x=torch.tensor(x_np,dtype=torch.float64);xb=x+torch.tensor(rng.normal(scale=.1,size=x_np.shape),dtype=torch.float64)
    y=torch.tensor(labels)
    w=torch.randn(hidden,width,dtype=torch.float64,requires_grad=True)
    head=torch.randn(classes,hidden,dtype=torch.float64,requires_grad=True)
    r=torch.ones(4,width,dtype=torch.float64,requires_grad=True)
    s=torch.ones(4,hidden,dtype=torch.float64,requires_grad=True)
    def bank(view):
        # Equal factors => identical actual member features/logits. They retain
        # distinct live private factor variables, around one shared nonlinear map.
        representation=torch.stack([torch.tanh((view*r[m])@w.T)*s[m] for m in range(4)])
        return representation@head.T,representation
    la,ha=bank(x);lb,hb=bank(xb)
    assert torch.equal(ha[0],ha[1]) and torch.equal(la[0],la[1])
    route,_=objective(la,ha,lb,hb,y,qr)
    common,_=objective(la,ha,lb,hb,y,qc)
    gr=torch.autograd.grad(route,(w,head,r,s),retain_graph=True)
    gc=torch.autograd.grad(common,(w,head,r,s),retain_graph=True)
    shared_error=max(float((gr[i]-gc[i]).abs().max()) for i in (0,1))
    private_difference=max(float((gr[i]-gc[i]).abs().max()) for i in (2,3))
    assert torch.allclose(route,common,atol=1e-12,rtol=1e-12)
    assert shared_error<1e-12 and private_difference>1e-8
    # Verify the M1 objective and untied loss scaling against an explicit sum
    # of separable one-member objectives on independently created output leaves.
    la_leaf=la.detach().clone().requires_grad_(True);ha_leaf=ha.detach().clone().requires_grad_(True)
    lb_leaf=lb.detach().clone().requires_grad_(True);hb_leaf=hb.detach().clone().requires_grad_(True)
    loss4,_=objective(la_leaf,ha_leaf,lb_leaf,hb_leaf,y,qr,independent=True)
    parts=[objective(la_leaf[m:m+1],ha_leaf[m:m+1],lb_leaf[m:m+1],hb_leaf[m:m+1],y,qr[m:m+1])[0] for m in range(4)]
    sum_parts=sum(parts)
    assert torch.allclose(loss4,sum_parts,atol=1e-12,rtol=1e-12)
    g4=torch.autograd.grad(loss4,(la_leaf,ha_leaf,lb_leaf,hb_leaf),retain_graph=True)
    gparts=torch.autograd.grad(sum_parts,(la_leaf,ha_leaf,lb_leaf,hb_leaf))
    for a,b in zip(g4,gparts):
        assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
    loss1,_=objective(la[:1],ha[:1],lb[:1],hb[:1],y,qc[:1])
    assert torch.isfinite(loss1)
    # Same-class label compatibility does not constitute competence protection.
    # This qualifier establishes only algebra and actual gradients in a helper.
    report={'status':'CPU_TORCH_FUNCTIONAL_HELPER_ONLY','scientific_data_access':False,
            'scientific_fits':0,'cuda_native_fullgraph_qualified':False,
            'torch':str(torch.__version__),'dtype':'float64',
            'identical_route_representation_and_logits':True,
            'common_route_loss_difference':float((route-common).abs()),
            'maximum_shared_parameter_gradient_difference':shared_error,
            'maximum_private_factor_gradient_difference':private_difference,
            'untied_four_sum_equals_four_separable_losses':True,
            'matched_single_common_objective_finite':True,
            'competence_or_quality_guaranteed':False}
    (Path(__file__).parent/'FUNCTIONAL_GRADIENT_QUALIFICATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
