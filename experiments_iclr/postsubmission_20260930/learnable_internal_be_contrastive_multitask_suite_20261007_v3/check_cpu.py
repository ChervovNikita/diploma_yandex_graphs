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
            'query':torch.tensor([[0,1],[1,0],[2,3],[3,7],[1,6],[5,7]])}
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
            total,components=objective(a,h,b,k,y,task,model.contrastive,model.independent,identities=batch.get('query'))
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
    # Numeric NPZ directory check rejects a heldout field before np.load runs.
    import tempfile,numpy as np
    from data import load_npz
    with tempfile.TemporaryDirectory(dir=ROOT) as work:
        path=Path(work)/'bad.npz';np.savez(path,x=np.zeros(1),test_y=np.ones(1))
        original=np.load;calls=[]
        def spy(*a,**k):calls.append(True);return original(*a,**k)
        np.load=spy
        try:
            failed=False
            try:load_npz(path,('x',))
            except ValueError:failed=True
            assert failed and not calls
        finally:np.load=original
        safe=Path(work)/'safe.npz';np.savez(safe,x=np.arange(4,dtype=np.float32))
        loaded=load_npz(safe,('x',));assert torch.equal(loaded['x'],torch.arange(4,dtype=torch.float32))
    # Packed untied joint transition and ordinary own transitions share runner helper.
    from selection import local_transition,finite_predictions,finite_state
    for arm in ('independent4','independent4_contrastive'):
        bank=Ensemble('wikics',arm,719,wiki_options,sources).double()
        optimizers=[torch.optim.Adam(x.parameters(),lr=.001) for x in bank.models]
        la,ha=bank(wiki);loss,_=objective(la,ha,la,ha,wy,'wikics',False,True);loss.backward()
        for optimizer in optimizers:optimizer.step()
        saved_joint={'model':copy.deepcopy(bank.state_dict()),'optimizers':copy.deepcopy([x.state_dict() for x in optimizers])}
        own=[]
        for m,body in enumerate(bank.models):
            state=copy.deepcopy(body.state_dict())
            for name,value in state.items():
                if value.dtype.is_floating_point:state[name]=torch.full_like(value,float(m+11))
            own_optimizer=copy.deepcopy(optimizers[m].state_dict())
            for history in own_optimizer['state'].values():
                for key,value in history.items():
                    if isinstance(value,torch.Tensor):value.add_(7*(m+1))
            own.append({'model':state,'optimizer':own_optimizer})
        for value in bank.parameters():
            with torch.no_grad():value.add_(13)
        for optimizer in optimizers:
            for history in optimizer.state.values():
                for key,value in history.items():
                    if isinstance(value,torch.Tensor):value.add_(113)
        rng=torch.get_rng_state().clone();loaded_names=[]
        def load(name):
            loaded_names.append(name)
            return own[int(name.removeprefix('own_local_').removesuffix('.pt'))] if name.startswith('own_local_') else saved_joint
        local_transition(bank,optimizers,load,arm=='independent4')
        assert torch.equal(torch.get_rng_state(),rng)
        assert all(body.body._global for body in bank.models)
        if arm=='independent4':
            assert loaded_names==['own_local_'+str(m)+'.pt' for m in range(4)]
            for m,body in enumerate(bank.models):
                assert all(torch.equal(value,own[m]['model'][name]) for name,value in body.state_dict().items())
        else:
            assert loaded_names==['selected_local.pt']
            assert all(torch.equal(value,saved_joint['model'][name]) for name,value in bank.state_dict().items())
        expected_optimizers=[row['optimizer'] for row in own] if arm=='independent4' else saved_joint['optimizers']
        for optimizer,expected in zip(optimizers,expected_optimizers):
            current=optimizer.state_dict()
            assert current['param_groups']==expected['param_groups']
            for key,state in current['state'].items():
                assert all(torch.equal(value,expected['state'][key][field]) for field,value in state.items())
        finite_state(bank,optimizers)
        rows.append({'WikiCS_transition_arm':arm,'restored_files':loaded_names,'live_CPU_RNG_preserved':True,'optimizer_custody_checked':True,'live_Adam_states_perturbed':True,'both_branch_moments_steps_asserted':True})
    # Every fixed budget includes bounded cleanup inside its absolute cap.
    for task in ('wikics','collab','molhiv'):
        budget=json.loads((ROOT/'configs'/str(task+'.json')).read_text())['budget']
        assert type(budget['cell_cleanup_grace_seconds']) is int and budget['cell_cleanup_grace_seconds']==10
        assert type(budget['cell_active_compute_seconds']) is int
        assert budget['cell_active_compute_seconds']+budget['cell_cleanup_grace_seconds']==budget['cell_hard_seconds']
        assert budget['cell_active_compute_seconds']>budget['cell_soft_seconds']
    # Genuine measured resource metadata rejects bool and nonfinite values.
    from runtime import resource_measurements
    resource_measurements({'peak_GPU_bytes':1048576,'inclusive_seconds':1.25,'cell_identity':{'members':4,'own_views':2}})
    for value in (True,False,0,-1,1.5,float('nan'),float('inf')):
        failed=False
        try:resource_measurements({'peak_GPU_bytes':value,'inclusive_seconds':1.,'cell_identity':{'members':4,'own_views':2}})
        except ValueError:failed=True
        assert failed
    for value in (True,False,0.,-1.,float('nan'),float('inf'),'1.2'):
        failed=False
        try:resource_measurements({'peak_GPU_bytes':1048576,'inclusive_seconds':value,'cell_identity':{'members':4,'own_views':2}})
        except ValueError:failed=True
        assert failed
    for value in (True,0,1.5):
        failed=False
        try:resource_measurements({'peak_GPU_bytes':1048576,'inclusive_seconds':1.,'cell_identity':{'members':value,'own_views':2}})
        except ValueError:failed=True
        assert failed
    # Reject NaN/Inf predictors before an apparently finite argmax metric.
    for bad in (float('nan'),float('inf')):
        failed=False
        try:finite_predictions(torch.tensor([[[bad,0.]]]),torch.zeros(1,2))
        except FloatingPointError:failed=True
        assert failed
    # Duplicate/reversed TRAIN target records are same-edge positives, not negatives.
    reps=torch.randn(2,4,5,dtype=torch.float64)
    ids=torch.tensor([[0,1],[1,0],[2,3],[3,2]])
    duplicate_labels=torch.tensor([1.,1.,0.,0.])
    val=alignment_loss(reps,reps,duplicate_labels,'collab',identities=ids)
    assert torch.isfinite(val)
    failed=False
    try:alignment_loss(reps,reps,torch.tensor([1.,0.,0.,0.]),'collab',identities=ids)
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
        'pre_numeric_deserialization_role_guard':True,'joint_and_ordinary_transition_state_tests':True,
        'nonfinite_prediction_rejection':True,'nonbool_finite_resource_metadata_guard':True,'cleanup_inside_all_absolute_caps':True,'canonical_duplicate_contrastive_targets':True,
        'real_data_access':False,'scientific_fits':0,'cuda_initialized':torch.cuda.is_initialized(),
        'seconds':time.monotonic()-started}
    assert result['cuda_initialized'] is False
    output=Path(args.output).resolve()
    if output.exists() or not output.is_relative_to(ROOT):raise ValueError('Fresh source-packet receipt only')
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'rows':len(rows),'real_data_access':False,'cuda_initialized':False}))


if __name__=='__main__':main()
