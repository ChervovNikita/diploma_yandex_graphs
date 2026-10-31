"""Meaningful CPU model/objective/stream/serialization gates; no real inputs."""
import argparse
import copy
import io
import os
import sys
sys.dont_write_bytecode=True
import torch
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
from contract import LAYOUTS,OBJECTIVES,RECIPE,write_new,sources,refuse_real
from predictive_core import INHERITED,LOWRANK,model_for,optimizer_for,objective,state_hash,configure


def require(value,message):
    if not value:raise AssertionError(message)


def fixture():
    graphs=[]
    for index,n in enumerate((3,4,5,6)):
        x=torch.zeros((n,9),dtype=torch.long);x[:,0]=(torch.arange(n)+index)%10
        pairs=[(i,i+1) for i in range(n-1)]+[(i+1,i) for i in range(n-1)]
        edge=torch.tensor(pairs,dtype=torch.long).T.contiguous()
        graphs.append(Data(x=x,edge_index=edge,edge_attr=torch.zeros((len(pairs),3),dtype=torch.long),y=torch.tensor([[float(index%2)]]),gnnm_official_id=torch.tensor([100+index])))
    return graphs,Batch.from_data_list(graphs)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    configure();checks=[]
    z=torch.tensor([[[-2.,.4],[.7,2.1]],[[1.,-.9],[-.3,.5]],[[.4,2.2],[1.4,-1.1]],[[.9,-.2],[-2.,1.8]]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[1.,float('nan')],[0.,1.]],dtype=torch.float64);finite=torch.isfinite(y);safe=torch.where(finite,y,torch.zeros_like(y));counts=finite.sum(-1)
    for family in OBJECTIVES:
        gradient=torch.autograd.grad(objective(z,y,family),z,retain_graph=True)[0]
        probability=z.sigmoid() if family=='own' else z.mean(0).sigmoid()[None].expand_as(z)
        expected=(probability-safe[None])*finite[None]/counts[None,:,None]/(z.shape[0]*z.shape[1])
        require(torch.allclose(gradient,expected,atol=1e-14,rtol=1e-14),'Independent analytic objective gradient failed')
    require(abs(float(objective(z,y,'own')-objective(z,y,'pooled')))>1e-3,'Fixture must distinguish deployed and member objectives')
    require(torch.equal(objective(z[:1],y,'own'),objective(z[:1],y,'pooled')),'Single-member objective identity failed')
    checks.append('analytic own/pooled finite-target equal-graph gradients and nontrivial distinction')
    graphs,batch=fixture();models={};definitions={}
    for layout in LAYOUTS:
        models[layout],definitions[layout]=model_for(0,layout)
        optimizer_for(models[layout])
    tied=models['tied'];tied.eval();reference=tied(batch,batch.x).detach()
    for layout,model in models.items():
        model.eval();require(torch.allclose(model(batch,batch.x),reference,atol=1e-6,rtol=0),'Canonical initial member logits differ: '+layout)
        for other in models.values():
            if model is not other:require(not ({p.data_ptr() for p in model.parameters()} & {p.data_ptr() for p in other.parameters()}),'Cross-arm parameter storage alias')
        if layout.endswith('_private'):
            index={'first_private':0,'middle_private':1,'last_private':2}[layout];copies=model.private_blocks[str(index)]
            require(isinstance(model.blocks[index],torch.nn.Identity),'Unused shared block not removed')
            source=tied.blocks[index].state_dict()
            for block in copies:require(set(block.state_dict())==set(source) and all(torch.equal(block.state_dict()[n],v) for n,v in source.items()),'Private complete parameter/buffer copy differs')
            for i,block in enumerate(copies):
                for other in copies[i+1:]:require(not ({p.data_ptr() for p in block.parameters()} & {p.data_ptr() for p in other.parameters()}),'Private member storage aliases')
            require(definitions[layout]['counts']['parameter_scalars']-definitions['tied']['counts']['parameter_scalars']==3*sum(p.numel() for p in tied.blocks[index].parameters()),'Complete block capacity charge differs')
    require(len({definitions[l]['counts']['parameter_scalars'] for l in LAYOUTS if l.endswith('_private')})==1,'Placement capacities differ')
    checks.append('canonical functions, complete private state/buffers, isolated storage, removed unused block and actual optimizer identity')
    for model in models.values():
        model.train();torch.manual_seed(1771);values=model(batch,batch.x)
        if model is tied:drop=values.detach()
        else:require(torch.allclose(values,drop,atol=1e-6,rtol=0),'Paired initial dropout function differs')
    require(not torch.equal(drop,reference),'Dropout fixture is inactive')
    checks.append('live paired dropout stream across all five zero-delta initial models')
    for family in OBJECTIVES:
        for layout in LAYOUTS:
            model=copy.deepcopy(models[layout]);model.train();opt=optimizer_for(model);before=state_hash(model)
            torch.manual_seed(2112);loss=objective(model(batch,batch.x),batch.y.float(),family);loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Missing/nonfinite synthetic gradients')
            if layout=='declared_low_rank':
                require(all(torch.count_nonzero(layer.A.grad)==0 for layer in model.low_rank_layers),'A must have zero first derivative at zero B')
                require(any(torch.count_nonzero(layer.B.grad)>0 for layer in model.low_rank_layers),'Random A / zero B has no live B gradient')
            opt.step();require(before!=state_hash(model),'Synthetic optimizer failed to learn')
            if layout=='declared_low_rank':
                opt.zero_grad(set_to_none=True);torch.manual_seed(2113);objective(model(batch,batch.x),batch.y.float(),family).backward()
                require(any(torch.count_nonzero(layer.A.grad)>0 for layer in model.low_rank_layers),'Trainable A did not learn after B update')
            model.eval();expected=model(batch,batch.x).detach();stream=io.BytesIO();torch.save({'state':model.state_dict(),'optimizer':opt.state_dict()},stream);stream.seek(0)
            loaded=torch.load(stream,map_location='cpu',weights_only=False);fresh=copy.deepcopy(models[layout]);fresh.load_state_dict(loaded['state'],strict=True);fresh.eval()
            require(torch.equal(expected,fresh(batch,batch.x)),'Serialized selected full member forward does not replay')
    checks.append('both objectives actually learn on every layout; live low-rank factors and exact CPU checkpoint roundtrip')
    for family in OBJECTIVES:
        zero,_=model_for(0,'declared_low_rank',rank=0);base=copy.deepcopy(tied)
        for model in (zero,base):
            model.train();opt=optimizer_for(model);torch.manual_seed(2901);objective(model(batch,batch.x),batch.y.float(),family).backward();opt.step();model.eval()
        require(torch.equal(zero(batch,batch.x),base(batch,batch.x)),'Rankzero update differs from tied')
    checks.append('rankzero exact tied update for own and pooled')
    def order():
        return [b.gnnm_official_id.tolist() for b in DataLoader(graphs,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(19710),num_workers=0)]
    require(order()==order() and sorted(i for b in order() for i in b)==[100,101,102,103],'Inherited DataLoader stream coverage/replay failed')
    rows=[dict(epoch=3,rocauc=.7,pooled_bce=.5,mean_member_bce=.6),dict(epoch=2,rocauc=.7,pooled_bce=.4,mean_member_bce=.6),dict(epoch=1,rocauc=.7,pooled_bce=.4,mean_member_bce=.6)]
    require(max(rows,key=lambda r:INHERITED.selection_key(r,r['epoch']))['epoch']==1,'AUC/BCE/earliest selection priority failed')
    try:refuse_real()
    except ValueError:pass
    else:raise AssertionError('Real execution gate opened')
    checks.extend(['inherited DataLoader exact ordering/coverage','AUC then BCE then earliest selection oracle','real admission refuses before input access'])
    write_new(a.output,dict(status='passed',scope='synthetic CPU engineering only',checks=checks,recipe=RECIPE,sources=sources(),actual_definitions=definitions,real_data_or_scores=False))
    print('Synthetic primary runner gates passed; no predictive result.')


if __name__=='__main__':main()
