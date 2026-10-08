"""One short artificial CPU qualification of the actual standard backend."""
import argparse
import os
import time
from common import *
from heads import train_rms, parameters, effective, objective, paired_start, fit
from cohorts import masks, transitions


def fixture(torch, np):
    rng=np.random.default_rng(1729)
    H=rng.normal(size=(2,12,5))+2
    H[:,:,-1]=0
    scale,zero=train_rms(np,H)
    require(np.array_equal(zero,np.array([4])) and scale[-1]==1, 'Exact zero-RMS convention')
    H=H/scale
    require(np.allclose((H[:,:,:4]**2).mean((0,1)),1) and np.max(abs(H.mean((0,1))))>.1, 'RMS without centering')
    W=rng.normal(scale=.1,size=(3,5)); r=np.ones((2,5)); s=np.ones((2,3))
    a=dict(W=W,r=r,s=s,A0=W[None]*r[:,None]*s[:,:,None],bias=np.zeros(3),
        H_train=H,H_dev=H.copy(),y_train=np.arange(12,dtype=np.int64)%3,y_dev=np.arange(12,dtype=np.int64)%3)
    paired=paired_start(torch,a)
    inputs=torch.from_numpy(H); y=torch.from_numpy(a['y_train'])
    gradient_errors={}
    for condition in ('BE_factor_refit','BE_full_refit'):
        p=parameters(torch,a,condition)
        total,_,_=objective(torch,inputs,y,effective(p),p['bias']); total.backward()
        direction={k:torch.from_numpy(rng.normal(size=v.shape)) for k,v in p.items()}
        norm=sum(float(v.square().sum()) for v in direction.values())**.5
        direction={k:v/norm for k,v in direction.items()}
        analytic=sum(float((p[k].grad*v).sum()) for k,v in direction.items())
        eps=1e-6
        plus={k:(v.detach()+eps*direction[k]) for k,v in p.items()}
        minus={k:(v.detach()-eps*direction[k]) for k,v in p.items()}
        fplus=float(objective(torch,inputs,y,effective(plus),plus['bias'])[0])
        fminus=float(objective(torch,inputs,y,effective(minus),minus['bias'])[0])
        error=abs(analytic-(fplus-fminus)/(2*eps))
        require(error<1e-7, 'Actual autograd directional derivative')
        gradient_errors[condition]=error
    A=torch.from_numpy(a['A0']); b=torch.from_numpy(a['bias'])
    shift=torch.from_numpy(rng.normal(size=(2,1,5)))
    old=objective(torch,inputs,y,A,b)
    shifted=objective(torch,inputs,y,A+shift,b)
    require(all(abs(float(x-z))<1e-12 for x,z in zip(old,shifted)), 'Class-gauge-invariant effective penalty/CE')
    endpoints={}
    for condition in ('BE_factor_refit','BE_full_refit'):
        endpoint,receipt=fit(torch,np,a,condition,time.monotonic()+120)
        declared='converged' if receipt['raw_parameter_gradient_infinity']<=1e-6 else 'finite_nonconverged'
        require(receipt['status']==declared and all(np.isfinite(v).all() for v in endpoint.values()),
                'Finite actual backend endpoint and truthful convergence declaration')
        if condition=='BE_full_refit':
            require(receipt['status']=='converged', 'Small convex synthetic full fit must meet the fixed gradient criterion')
        endpoints[condition]=receipt
    # All-rival qualification: different members cannot jointly supply one route.
    y0=np.array([0],dtype=np.int64)
    baseline=np.array([[[0.,2.,3.,-1.]],[[0.,2.,3.,-1.]]])
    candidate=np.array([[[1.,0.,2.,3.]],[[1.,2.,0.,3.]]])
    softmax=lambda x:np.exp(x-x.max(-1,keepdims=True))/np.exp(x-x.max(-1,keepdims=True)).sum(-1,keepdims=True)
    P0,P1=softmax(baseline),softmax(candidate)
    m0=masks(np,baseline,P0,P0.mean(0),y0); m1=masks(np,candidate,P1,P1.mean(0),y0)
    first=transitions(np,m0,m1,baseline,candidate,y0,np.ones(1,dtype=np.bool_))
    require(first['common_rival_order_reversal']==first['strict_common_rival_repair']==0, 'No mixed-member all-rival route')
    candidate=np.array([[[4.,2.,3.,5.]],[[0.,2.,3.,5.]]]); P1=softmax(candidate)
    m1=masks(np,candidate,P1,P1.mean(0),y0)
    second=transitions(np,m0,m1,baseline,candidate,y0,np.ones(1,dtype=np.bool_))
    require(second['common_rival_order_reversal']==1 and second['strict_common_rival_repair']==0,
            'Order reversal can remain wrong against a new class')
    return dict(paired_start=paired,directional_derivative_errors=gradient_errors,synthetic_fits=endpoints,
                RMS_no_centering=True,class_centered_penalty_gauge_check=True,all_rival_same_member_checks=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.environ.update(cpu_env())
    import numpy as np
    import torch
    started=time.monotonic(); signature=backend(torch,np)
    result=fixture(torch,np)
    write(args.output,dict(schema='Wiki24-direct12-CPU-backend-qualification-v1',passed=True,
        source_manifest_sha256=sha(ROOT/'MANIFEST.json'),backend=signature,checks=result,
        artificial_only=True,models_or_native_backbones_constructed=False,scientific_arrays_or_scores_opened=False,
        GPU_calls=0,inclusive_wall_seconds=time.monotonic()-started))
