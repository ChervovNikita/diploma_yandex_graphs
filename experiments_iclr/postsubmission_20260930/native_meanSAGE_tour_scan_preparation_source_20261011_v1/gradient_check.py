"""ONE short CPUfloat64 derivative fixture; root calls only after admission.

No numerical import, model/data access or execution at module import. This
checks the concrete custom-backward risk independently of zero W_o startup;
it is not an accuracy experiment, stochastic grid or full-input resource check.
"""

def run(torch,scan_backend):
    if torch.get_default_dtype()!=torch.float32:
        raise ValueError('Keep native default dtype; fixture explicitly uses float64 CPU')
    dtype=torch.float64;device='cpu';scan=scan_backend.make_segmented_scan(torch)
    tensor=lambda x:torch.tensor(x,dtype=dtype,device=device)
    tokens=torch.tensor([0,1,2,1,0,3,4,3],dtype=torch.int64,device=device)
    starts=torch.tensor([True,False,False,False,False,True,False,False],dtype=torch.bool,device=device)
    reverse=torch.tensor([4,3,2,1,0,7,6,5],dtype=torch.int64,device=device)
    tour=dict(tokens=tokens,starts=starts,reverse_index=reverse,occurrences=torch.bincount(tokens,minlength=5))
    fixed={
      'H':[[.2,.1,.4],[.7,.2,.3],[.1,.8,.6],[.3,.5,.2],[.6,.4,.7]],
      'Wu':[[.3,-.2,.1],[.15,.25,-.1],[-.2,.05,.35]],
      'Wa':[[.1,.2,-.15],[-.3,.25,.1],[.2,-.1,.3]],
      'bu':[.1,-.05,.03], 'ba':[.2,-.1,.15],
      'Wo':[[.2,-.1,.05],[.1,.3,-.2],[-.15,.07,.25]]}
    head=tensor([[.4,-.2],[.1,.3],[-.3,.25]])
    target=tensor([[.1,-.2],[.3,.1],[-.1,.2],[.25,-.15],[.05,.3]])
    outputs=[]
    for custom in (True,False):
        p={key:tensor(value).requires_grad_() for key,value in fixed.items()}
        u=torch.tanh(p['H']@p['Wu']+p['bu']);a=torch.sigmoid(p['H']@p['Wa']+p['ba']);q=(1-a)*u
        a.retain_grad();q.retain_grad()
        if custom:
            z=scan_backend.bidirectional_context(torch,scan,a,q,tour)
        else:
            # Independent ordinary sequential recurrence with autograd. This
            # Python token loop is ONLY the short CPU reference, never CUDA.
            aa=a.index_select(0,tokens);qq=q.index_select(0,tokens);forward=[];backward=[None]*8
            for lo,hi in ((0,5),(5,8)):
                state=torch.zeros(3,dtype=dtype,device=device)
                for t in range(lo,hi):
                    state=aa[t]*state+qq[t];forward.append(state)
                state=torch.zeros(3,dtype=dtype,device=device)
                for t in range(hi-1,lo-1,-1):
                    state=aa[t]*state+qq[t];backward[t]=state
            # Deliberately separate occurrence reduction from backend index_add.
            paired=(torch.stack(forward)+torch.stack(backward))*.5
            z=torch.stack([paired[tokens==v].mean(0) for v in range(5)])
        hidden=p['H']+.25*(z@p['Wo'])
        hidden=hidden/torch.linalg.vector_norm(hidden,ord=2,dim=1,keepdim=True).clamp_min(1.)
        loss=((hidden@head-target)**2).mean();loss.backward()
        outputs.append(dict(z=z.detach(),hidden=hidden.detach(),loss=loss.detach(),gradients={**{key:value.grad.detach() for key,value in p.items()},'original_a':a.grad.detach(),'q':q.grad.detach()}))
    atol,rtol=1e-9,1e-7;left,right=outputs;errors={}
    for field in ('z','hidden','loss'):
        errors[field]=float((left[field]-right[field]).abs().max())
        if not torch.allclose(left[field],right[field],atol=atol,rtol=rtol):raise ValueError('Short custom forward differs: '+field)
    for field,value in left['gradients'].items():
        reference=right['gradients'][field];errors['grad_'+field]=float((value-reference).abs().max())
        if not torch.isfinite(value).all() or not torch.allclose(value,reference,atol=atol,rtol=rtol):
            raise ValueError('Short manual adjoint differs: '+field)
    # This fixture actually exercises later input/gate learning, which initial
    # W_o0 in a full-input readiness update cannot establish.
    if any(float(right['gradients'][key].abs().max())<=1e-10 for key in ('Wu','Wa','Wo','original_a','q')):
        raise ValueError('Fixed short fixture did not exercise the intended live derivative')
    return dict(complete=True,fixtures=1,CPUfloat64=True,nodes=5,tokens=8,components=2,directions=2,nonzero_output_projection=True,forward_reference='ordinary sequential recurrence',gradient_reference='independent sequential ordinary autograd',atol=atol,rtol=rtol,max_absolute_errors=errors,higher_order_support=False,model_optimizer_updates=0,scientific_quality_claim=False)

if __name__=='__main__':raise SystemExit('Inactive source: root injects admitted torch/backend and runs exactly this one fixture.')
