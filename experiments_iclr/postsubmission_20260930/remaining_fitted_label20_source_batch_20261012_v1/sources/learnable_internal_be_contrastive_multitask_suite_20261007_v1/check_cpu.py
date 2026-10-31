"""Canonical synthetic CPU operators/gradients, no real labels or fits."""
import argparse
import copy
import json
from pathlib import Path
import time
from runtime import ROOT, PHASE, allocation, runtime_versions, verify_manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args()
    allocation(cpu_check=True); versions=runtime_versions()
    import torch
    from torch import nn
    from torch_geometric.data import Data,Batch
    from models import Ensemble,native_sources
    from factors import FactorLinear,factor_counts
    from objectives import objective,alignment_loss,residual_member_contrast
    from data import support_graph,pair_ids,require_keys,molecular_batch
    torch.manual_seed(617)
    sources=native_sources(PHASE,json.loads((ROOT/'DEPENDENCIES.json').read_text()))
    arms=json.loads((ROOT/'configs/wikics.json').read_text())['arms']
    started=time.monotonic();rows=[]
    wiki_options=dict(in_channels=5,hidden_channels=4,out_channels=3,local_layers=2,global_layers=1,
        in_dropout=0.,dropout=0.,global_dropout=0.,heads=1,beta=-1,pre_ln=False)
    wiki={'x':torch.randn(8,5,dtype=torch.float64),'edge_index':torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,0],[1,0,2,1,3,2,4,3,5,4,6,5,7,6,0,7]]),
          'ids':torch.arange(8)}
    wy=torch.tensor([0,1,2,0,1,2,0,1])
    # Deliberate duplicate records, target must vanish globally from sparse graph.
    positives=torch.tensor([[0,1],[0,1],[1,0],[1,2],[2,3],[3,4],[4,0],[0,2],[2,4],[4,1]])
    targets=torch.tensor([[0,1],[2,3]])
    adj=support_graph(positives,8,targets)
    row,col,_=adj.coo()
    assert not torch.isin(pair_ids(torch.stack((row,col),1),8),pair_ids(targets,8)).any()
    collab={'x':torch.randn(8,5,dtype=torch.float64),'adj':adj,
            'query':torch.tensor([[0,1],[2,3],[3,4],[3,7],[1,6],[5,7]])}
    cy=torch.tensor([1.,1.,1.,0.,0.,0.],dtype=torch.float64)
    gs=[]
    for i in range(4):
        gs.append(Data(x=torch.tensor([[i%2,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0],[2,0,0,0,0,0,0,0,0]]),
            edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),edge_attr=torch.zeros(4,3,dtype=torch.long)))
    mol={'graph':Batch.from_data_list(gs)};my=torch.tensor([0.,1.,0.,1.],dtype=torch.float64)
    cases=[('wikics',wiki_options,wiki,wy),('collab',dict(features=5,hidden=4),collab,cy),
           ('molhiv',dict(hidden=4,layers=2,dropout=0.),mol,my)]
    for task,options,batch,y in cases:
        for arm in arms:
            model=Ensemble(task,arm,617,options,sources).double().eval()
            a,h=model(batch);b,k=model(batch)
            assert a.shape==(model.members,len(y),3 if task=='wikics' else 1)
            assert h.shape[:2]==(model.members,len(y))
            total,components=objective(a,h,b,k,y,task,model.contrastive,model.independent)
            total.backward()
            grad=[(n,p) for n,p in model.named_parameters() if p.grad is not None]
            assert grad and all(torch.isfinite(p.grad).all() for _,p in grad)
            for module in model.modules():
                if isinstance(module,FactorLinear) and module.weight.grad is not None:
                    assert module.r.grad is not None and module.s.grad is not None
                    assert torch.isfinite(module.r.grad).all() and torch.isfinite(module.s.grad).all()
            if not model.independent:
                weights=[p for n,p in grad if n.endswith('.weight')]
                privates=[p for n,p in grad if n.endswith(('.r','.s','message_factors'))]
                assert any(p.grad.abs().sum()>0 for p in weights)
                assert any(p.grad.abs().sum()>0 for p in privates)
                before=[p.detach().clone() for p in (weights[0],privates[0])]
                opt=torch.optim.Adam(model.parameters(),lr=.001);opt.step()
                assert all(not torch.equal(old,p) for old,p in zip(before,(weights[0],privates[0])))
            rows.append({'task':task,'arm':arm,'shape':list(a.shape),'finite_gradient_parameters':len(grad),
                         'counts':factor_counts(model),'loss':float(total.detach())})
        # Unit factors preserve native serving and shared gradient, both native modes.
        for global_stage in ([False,True] if task=='wikics' else [False]):
            base=Ensemble(task,'single',617,options,sources).double().eval()
            unit=Ensemble(task,'be_unit',617,options,sources).double().eval()
            if task=='wikics':base.set_global(global_stage);unit.set_global(global_stage)
            la,ha=base(batch);lb,hb=unit(batch)
            assert torch.allclose(lb,la.expand_as(lb),rtol=1e-11,atol=1e-12)
            assert torch.allclose(hb,ha.expand_as(hb),rtol=1e-11,atol=1e-12)
            own_a,_=objective(la,ha,la,ha,y,task,False,False)
            own_b,_=objective(lb,hb,lb,hb,y,task,False,False)
            own_a.backward();own_b.backward()
            ub=dict(unit.models[0].named_parameters())
            checked=0
            for name,p in base.models[0].named_parameters():
                if p.grad is not None:
                    assert name in ub and ub[name].grad is not None,name
                    assert torch.allclose(p.grad,ub[name].grad,rtol=1e-9,atol=1e-11),name
                    checked+=1
            rows.append({'task':task,'unit_native_mode_global':global_stage,'shared_gradient_parameters_matched':checked})
    # Backbone is permutation equivariant; chemistry is untouched by stochastic views.
    mm=Ensemble('molhiv','be_init',617,cases[-1][1],sources).double().eval()
    prediction,_=mm(mol)
    permuted=[]
    for g in gs:
        order=torch.tensor([2,0,1]);inverse=torch.argsort(order)
        permuted.append(Data(x=g.x[order],edge_index=inverse[g.edge_index],edge_attr=g.edge_attr.clone()))
    got,_=mm({'graph':Batch.from_data_list(permuted)})
    assert torch.allclose(prediction,got,rtol=1e-10,atol=1e-11)
    # Class-mean shifts do not change residual term, but useful diversity is unproved.
    a=torch.randn(4,8,5,dtype=torch.float64,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
    labels=torch.tensor([0,0,0,0,1,1,1,1]);offset=torch.randn(4,2,5,dtype=torch.float64)
    value=residual_member_contrast(a,b,labels)
    shifted=residual_member_contrast(a+offset[:,labels],b+offset[:,labels],labels)
    assert torch.allclose(value,shifted,rtol=1e-11,atol=1e-12)
    assert residual_member_contrast(a[:1],b[:1],labels).item()==0
    assert residual_member_contrast(a[:,:2],b[:,:2],torch.tensor([0,1])).item()==0
    value.backward();assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    # Full-label own loss: changing any supervised label changes that member's gradient.
    logits=torch.randn(4,8,3,dtype=torch.float64,requires_grad=True)
    from objectives import own_supervision
    loss=own_supervision(logits,wy,'wikics').sum()
    grads=torch.autograd.grad(loss,logits)[0]
    assert grads.abs().sum(-1).gt(0).all()
    # Tensor-only role projection rejects an extra heldout field before values load.
    failed=False
    try:require_keys({'x':torch.zeros(1),'test_y':torch.ones(1)},('x',))
    except ValueError:failed=True
    assert failed
    # Canonical graph-local concatenation adapter, including a complete tail batch.
    payload={'x':torch.cat([g.x for g in gs]),'edge_index':torch.cat([g.edge_index for g in gs],1),
        'edge_attr':torch.cat([g.edge_attr for g in gs]),'node_ptr':torch.arange(5)*3,
        'edge_ptr':torch.arange(5)*4,'ids':torch.arange(4),'y':my}
    rebuilt,truth=molecular_batch(payload,torch.tensor([3,0,2]),'cpu')
    assert rebuilt['graph'].num_graphs==3 and torch.equal(truth,my[torch.tensor([3,0,2])])
    # No stateful normalization that could mix member/view running moments.
    assert not any(isinstance(x,nn.modules.batchnorm._BatchNorm) for x in mm.modules())
    result={'schema':'internal-be-synthetic-cpu-check-v1','passed':True,'runtime':versions,
        'source_manifest_sha256':verify_manifest(),'rows':rows,'chemistry_permutation_check':True,
        'duplicate_target_support_removal':True,'role_extra_heldout_field_rejected':True,
        'full_member_label_gradient_check':True,'class_center_loss_gradient_check':True,
        'real_data_access':False,'scientific_fits':0,'cuda_initialized':torch.cuda.is_initialized(),
        'seconds':time.monotonic()-started}
    assert result['cuda_initialized'] is False
    output=Path(args.output).resolve()
    if output.exists() or not output.is_relative_to(ROOT):raise ValueError('Fresh source-packet receipt only')
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'rows':len(rows),'real_data_access':False,'cuda_initialized':False}))


if __name__=='__main__':main()
