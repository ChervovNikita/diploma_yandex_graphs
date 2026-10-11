"""ONE nonzero derived-weight CUDA GRU vs sequential CPUfloat64 AD fixture."""
from contextlib import nullcontext

def run(torch,backend):
    backend.require_runtime(torch)
    if torch.get_default_dtype()!=torch.float32 or not torch.cuda.is_available():raise ValueError('Admitted source float32 default and CUDA runtime')
    h=3;tokens=[0,1,2,1,0,3,4,3];reverse=[4,3,2,1,0,7,6,5]
    base=torch.arange(27,dtype=torch.float64).reshape(9,3)
    fixed=dict(H=[[.2,.1,.4],[.7,.2,.3],[.1,.8,.6],[.3,.5,.2],[.6,.4,.7]],W_ih=(base/71-.17).tolist(),W_hh=(base.flip(0)/83-.12).tolist(),r_x=(torch.arange(18,dtype=torch.float64).reshape(2,3,3)/37+.7).tolist(),r_h=(torch.arange(18,dtype=torch.float64).reshape(2,3,3).flip(-1)/43+.8).tolist(),s=(torch.arange(18,dtype=torch.float64).reshape(2,3,3)/53+.65).tolist(),b_i=[[[.1,-.1,.05],[.3,.2,.1],[-.1,.1,.2]],[[.04,.12,-.02],[.1,.3,.2],[.05,-.04,.1]]],b_h=[[[.02,.04,.07],[.05,.03,.01],[.1,.06,-.03]],[[.08,.02,.01],[.03,.1,.07],[.04,-.02,.1]]],W_o=[[.2,-.1,.05],[.1,.3,-.2],[-.15,.07,.25]])
    cpu_tour=dict(tokens=torch.tensor(tokens,dtype=torch.int64),reverse_index=torch.tensor(reverse,dtype=torch.int64),occurrences=torch.tensor([2,2,1,2,1],dtype=torch.int64),offsets=torch.tensor([0,5,8],dtype=torch.int64),components=2,length=8)
    packed=backend.to_device(backend.pack_tour(torch,cpu_tour),'cuda:0');results=[];operations={}
    for fused in (True,False):
        device='cuda:0' if fused else 'cpu'
        p={key:torch.tensor(value,dtype=torch.float64,device=device).requires_grad_() for key,value in fixed.items()}
        head=torch.tensor([[.4,-.2],[.1,.3],[-.3,.25]],dtype=torch.float64,device=device)
        target=torch.tensor([[.1,-.2],[.3,.1],[-.1,.2],[.25,-.15],[.05,.3]],dtype=torch.float64,device=device)
        zs=[]
        # Two private rows in one fixture exercise summed shared slow gradients.
        scope=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if fused else nullcontext()
        with scope as profiler:
            for member in range(2):
                if fused:
                    a=(p['s'][member,:, :,None]*p['W_ih'].reshape(3,h,h)*p['r_x'][member,:,None,:]).reshape(3*h,h)
                    b=(p['s'][member,:, :,None]*p['W_hh'].reshape(3,h,h)*p['r_h'][member,:,None,:]).reshape(3*h,h)
                    z=backend.node_memory(torch,p['H'],[a,b,p['b_i'][member].reshape(3*h),p['b_h'][member].reshape(3*h)],packed,True)
                else:
                    # Independent per-gate sequential reset-after equations.
                    aa=[p['s'][member,g][:,None]*p['W_ih'][g*h:(g+1)*h]*p['r_x'][member,g][None,:] for g in range(3)]
                    bb=[p['s'][member,g][:,None]*p['W_hh'][g*h:(g+1)*h]*p['r_h'][member,g][None,:] for g in range(3)]
                    directions=[]
                    for rev in (False,True):
                        states=[None]*8
                        for lo,hi in ((0,5),(5,8)):
                            state=torch.zeros(h,dtype=torch.float64)
                            positions=range(hi-1,lo-1,-1) if rev else range(lo,hi)
                            for position in positions:
                                x=p['H'][tokens[position]]
                                i=[aa[g]@x+p['b_i'][member,g] for g in range(3)]
                                hidden=[bb[g]@state+p['b_h'][member,g] for g in range(3)]
                                r=torch.sigmoid(i[0]+hidden[0]);gate=torch.sigmoid(i[1]+hidden[1]);n=torch.tanh(i[2]+r*hidden[2])
                                state=gate*state+(1-gate)*n;states[position]=state
                        directions.append(torch.stack(states))
                    values=(directions[0]+directions[1])*.5
                    z=torch.stack([torch.stack([values[t] for t,node in enumerate(tokens) if node==v]).mean(0) for v in range(5)])
                zs.append(z)
            hidden=torch.stack([p['H']+.25*(z@p['W_o']) for z in zs])
            hidden=hidden/torch.linalg.vector_norm(hidden,ord=2,dim=-1,keepdim=True).clamp_min(1.)
            loss=((hidden@head-target[None])**2).mean();loss.backward()
        if fused:
            operations={event.key:event.count for event in profiler.key_averages() if 'gru' in event.key or 'cudnn_rnn' in event.key}
            if operations.get('aten::_cudnn_rnn',0)!=4:raise ValueError('Short CUDA fixture must exercise four actual cuDNN forward calls')
        results.append(dict(states=torch.stack(zs).detach().cpu(),hidden=hidden.detach().cpu(),loss=loss.detach().cpu(),gradients={key:value.grad.detach().cpu() for key,value in p.items()}))
    native,reference=results;errors={};atol,rtol=1e-9,1e-7
    for name in ('states','hidden','loss'):
        errors[name]=float((native[name]-reference[name]).abs().max())
        if not torch.allclose(native[name],reference[name],atol=atol,rtol=rtol):raise ValueError('Fused reset-after forward differs: '+name)
    for name,value in native['gradients'].items():
        other=reference['gradients'][name];errors['grad_'+name]=float((value-other).abs().max())
        if not torch.isfinite(value).all() or not torch.allclose(value,other,atol=atol,rtol=rtol):raise ValueError('Derived GRU slow/fast derivative differs: '+name)
        if float(other.abs().max())<=1e-11:raise ValueError('Short fixture lacks nonzero gradient: '+name)
    return dict(complete=True,fixtures=1,CUDAfloat64_vs_CPUfloat64=True,nodes=5,tokens=8,components=2,private_rows=2,nonzero_readout=True,reference='ordinary sequential reset-after CPU autograd with independent occurrence reduction',checked_gradients=list(fixed),atol=atol,rtol=rtol,max_absolute_errors=errors,backend_operations=operations,cudnn_version=torch.backends.cudnn.version(),higher_order_support=False,model_optimizer_updates=0,scientific_quality_claim=False)
