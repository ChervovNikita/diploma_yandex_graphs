"""One fixed synthetic derivative diagnosis; no fitting or qualification pass."""
from pathlib import Path
import hashlib
import importlib.util
import json
import random
import resource
import signal
import socket
import sys
import time
import traceback

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
WORKER = PHASE/'learnability_responsibility_native_numerical_worker_preparation_20261005_v1/qualify.py'
WORKER_SHA = '976332545f78f1fe9642b2a4fa9d61127b427739cd85ea9a26301afc92cb61f7'

def main():
    output = Path(sys.argv[1]);output.mkdir(exist_ok=False)
    start=time.monotonic()
    result={'status':'RUNNING_DIAGNOSTIC','model_fits':0,'dataset_label_access':False,
            'predictive_evidence':False,'qualification_pass_authorized':False,
            'original_failed_qualifier_preserved':True,'fixed_deadline_seconds':180,
            'fixed_scales':[1e-3,3e-4,1e-4,3e-5,1e-5]}
    def save():
        result['elapsed_seconds']=time.monotonic()-start
        result['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    def deadline(signum,frame):raise TimeoutError('Fixed180-second derivative diagnosis exceeded')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(180)
    save()
    try:
        assert Path.cwd().resolve()==REPO and socket.gethostname()=='anogena-2-0'
        assert hashlib.sha256(WORKER.read_bytes()).hexdigest()==WORKER_SHA
        assert Path(sys.executable).absolute()==PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
        sys.path.insert(0,str(REPO/'.venv/lib/python3.11/site-packages'))
        import torch
        import numpy as np
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        spec=importlib.util.spec_from_file_location('diagnostic_qualifier',WORKER)
        q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
        q.torch=torch;q.np=np
        modules={n:q.load_module('diagnostic_'+n,PHASE/p,h) for n,(p,h) in q.PINS.items()}
        op,port=modules['operator'],modules['port']
        assert op.SOURCE_RELEASED is False and port.PORT_RELEASED is False
        generator=torch.Generator().manual_seed(126005)
        x=torch.randn(15,300,dtype=torch.float64,generator=generator)
        support=set((i,i) for i in range(15))
        for i in range(14):support.update({(i,i+1),(i+1,i)})
        for i,j in [(0,4),(1,6),(2,7),(3,8),(4,9),(0,9)]:support.update({(i,j),(j,i)})
        edges=torch.tensor(sorted(support),dtype=torch.long).T.contiguous()
        s,r=torch.arange(10),torch.arange(10,15)
        ys,yr=torch.arange(10)%5,torch.arange(5)
        family=q.build_family(modules['native'],modules['boundary'],torch.device('cpu'),torch.float64,106005)
        forward,theta,phis,_=port._native_callback_and_state(family,x,edges,expected_nodes=15)
        pairs=port._sparse_pairs(op,s,ys,edges,node_count=15,dtype=torch.float64)
        before=q.snapshot(family,x,edges,torch.device('cpu'))
        cfg=op.Config()
        live=lambda core:q.outer(op,core,phis,forward,pairs,s,ys,r,yr)
        stopped=lambda core:q.outer(op,core,phis,forward,pairs,s,ys,r,yr,'stop_q')
        gradient=torch.func.grad(live)(theta)
        generator=torch.Generator().manual_seed(116005)
        direction={n:torch.zeros_like(v) if n.startswith('local_head.') else
                   torch.randn(v.shape,dtype=torch.float64,generator=generator) for n,v in theta.items()}
        norm=sum(v.square().sum() for v in direction.values()).sqrt()
        direction={n:v/norm for n,v in direction.items()}
        analytic=float(sum((gradient[n]*direction[n]).sum() for n in theta))
        result['torch_func_directional']=analytic
        result['finite_differences']=[];save()
        base=float(live(theta))
        for eps in result['fixed_scales']:
            plus={n:v+eps*direction[n] for n,v in theta.items()}
            minus={n:v-eps*direction[n] for n,v in theta.items()}
            positive,negative=float(live(plus)),float(live(minus))
            result['finite_differences'].append({'epsilon':eps,'positive_value':positive,
                'negative_value':negative,'central':(positive-negative)/(2*eps),
                'right':(positive-base)/eps,'left':(base-negative)/eps,
                'absolute_error':abs((positive-negative)/(2*eps)-analytic)})
            save()

        # Independent ordinary-autograd private partial: Q depends on original
        # phi, while a separate equal-valued proxy owns the main private tangent.
        core={n:v.detach().clone().requires_grad_() for n,v in theta.items()}
        originals=tuple({n:v.detach().clone().requires_grad_() for n,v in phi.items()} for phi in phis)
        pre=torch.stack([forward(core,phi)[s] for phi in originals])
        probes=[]
        for m,phi in enumerate(originals):
            own=torch.nn.functional.cross_entropy(pre[m],ys)
            g=torch.autograd.grad(own,tuple(phi.values()),create_graph=True,allow_unused=True)
            probes.append({n:v-cfg.eta_probe*(torch.zeros_like(v) if d is None else d)
                           for (n,v),d in zip(phi.items(),g)})
        post=torch.stack([forward(core,phi)[s] for phi in probes])
        assignments=[]
        for pair in pairs:
            response=op._margin(post,pair)-op._margin(pre,pair)
            cost,_,_=op._normalized_cost(response,cfg.response_epsilon)
            assignments.append(op._assignment_map(cost,pair.laplacian,cfg))
        adapted=[]
        for m,phi in enumerate(originals):
            proxy={n:v.detach().clone().requires_grad_() for n,v in phi.items()}
            loss=op._main_loss(core,proxy,tuple(assignments),m,forward,s,ys,pairs,5,cfg)
            derivatives=torch.autograd.grad(loss,tuple(proxy.values()),create_graph=True,allow_unused=True)
            adapted.append({n:phi[n]-cfg.eta_private*(torch.zeros_like(v) if d is None else d)
                            for (n,v),d in zip(proxy.items(),derivatives)})
        query=op._query_objective(core,tuple(adapted),forward,r,yr,cfg)
        derivatives=torch.autograd.grad(query,tuple(core.values()),allow_unused=True)
        oracle={n:torch.zeros_like(v) if d is None else d for (n,v),d in zip(core.items(),derivatives)}
        result['ordinary_autograd_value']=float(query)
        result['torch_func_value']=base
        result['ordinary_autograd_directional']=float(sum((oracle[n]*direction[n]).sum() for n in theta))
        result['autograd_vs_func_all_coordinate_max_error']=max(float((oracle[n]-gradient[n]).abs().max()) for n in theta)
        stopped_gradient=torch.func.grad(stopped)(theta)
        result['live_Q_chain_gradient_norm']=float(sum((gradient[n]-stopped_gradient[n]).square().sum() for n in theta).sqrt())
        result['native_restoration']=q.unchanged(before,family,x,edges,torch.device('cpu'))
        result['status']='COMPLETE_DERIVATIVE_DIAGNOSIS_NOT_QUALIFICATION'
    except BaseException as error:
        result.update(status='FAIL_DIAGNOSTIC',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:signal.alarm(0);save()
    print(json.dumps(result));raise SystemExit(0 if result['status'].startswith('COMPLETE_') else 1)

if __name__=='__main__':main()
