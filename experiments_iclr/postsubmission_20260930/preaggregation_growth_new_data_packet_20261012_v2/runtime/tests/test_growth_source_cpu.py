"""One narrow synthetic full-width source constructor/callback custody check."""
import copy,json,os,tempfile,unittest
from pathlib import Path
from adapters.preaggregation_growth import sources,original_boundary,boundary_episode,MultiAdam,optimizer

class GrowthCPU(unittest.TestCase):
 def test_source_constructors_exact3pass_transition_and_original_boundary(self):
  import torch
  from torch_sparse import SparseTensor
  torch.set_num_threads(1);root=Path(os.environ['GNNM_REFERENCE_PROJECT']).resolve();packet=Path(__file__).resolve().parents[2];B=json.loads((packet/'SOURCE_BINDINGS.json').read_text());s=sources(root,B);s.bindings=B
  torch.random.default_generator.manual_seed(97091);x=torch.randn(64,3703);edge=torch.tensor([[i for i in range(64) for _ in range(2)],[j for i in range(64) for j in ((i+1)%64,(i+3)%64)]])
  support=SparseTensor.from_edge_index(edge,sparse_sizes=(64,64)).to_symmetric().coalesce()
  q=(torch.tensor([[0,1,2,3],[1,2,3,4]]),torch.tensor([[0,1,2,3],[9,10,11,12]]));inner=[q]*4
  warm,_,_,_=s.models.build(s.native,s.heads,arm='capable_single',seed=99101,factor_seed=4099138,device=torch.device('cpu'));opt=optimizer(torch,warm,.001);drop=s.steps.DropoutStreams(torch,torch.device('cpu'),99101)
  values=s.warm.ordinary_step(torch,s.steps,warm,opt,x,support,inner,q,drop);self.assertEqual(len(values),3);self.assertTrue(all(v['step']==3 for v in opt.state.values()))
  # Explicit synthetic Stiefel values validate constructor shapes only; no claim of trained TRAIN-gradient bases.
  graph=torch.stack([torch.eye(256)[2*m:2*m+2] for m in range(4)]);basis={'graph':graph,'unfiltered':graph[:1].expand(4,-1,-1).clone(),'single':torch.eye(256)[:8][None]}
  for arm in ('no_growth','graph_growth','unfiltered_growth','capable_single_rank8','independent_graph_growth4'):
   model,part=s.growth.make(torch,s.native,s.heads,s.models,s.initializer,warm,arm,99101,4099138,torch.device('cpu'),basis)
   self.assertEqual(model.member_count,1 if arm=='capable_single_rank8' else 4)
   if arm!='no_growth':self.assertEqual(part['growth']['new_parameters'],4096)
   rows=list(model.routes) if arm=='independent_graph_growth4' else [model];owners=[optimizer(torch,m,.001) for m in rows];owner=owners[0] if len(owners)==1 else MultiAdam(owners)
   from engine import inspect_ownership
   inspect_ownership(type('Adapter',(),{'model':model,'optimizer':owner})())
   with torch.no_grad():
    p,n=s.steps.functional_forward(torch,model,dict(model.named_parameters()),x,support,q,training=False);wp,wn=s.steps.functional_forward(torch,warm,dict(warm.named_parameters()),x,support,q,training=False)
    self.assertTrue(torch.equal(p,wp.expand_as(p)));self.assertTrue(torch.equal(n,wn.expand_as(n)))
  bank,proof=original_boundary(torch,root,s,99201,4099238,torch.device('cpu'));o=optimizer(torch,bank,.001);drop=s.steps.DropoutStreams(torch,torch.device('cpu'),99201+200000)
  values=boundary_episode(torch,s,bank,o,x,support,inner,q,drop);self.assertEqual(len(values),3);self.assertTrue(all(v['step']==3 for v in o.state.values()));self.assertTrue(proof['complete_native_interior_retained'])

if __name__=='__main__':unittest.main()
