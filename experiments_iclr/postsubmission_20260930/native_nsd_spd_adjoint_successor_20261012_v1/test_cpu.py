"""Prospective matrix-function derivative proof and only two corrected source fixtures."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,sys,tempfile,unittest
ROOT=Path(os.environ['GNNM_REFERENCE_PROJECT']);HERE=ROOT/'native_nsd_spd_adjoint_successor_20261012_v1';sys.path.insert(0,str(HERE/'runtime'))
import torch
from adapters.nsd_spd_power import make_spd_inverse_sqrt
from adapters.nsd import make_adapter
from engine import capture_checkpoint,restore_checkpoint,_equal_tree
B=json.loads((ROOT/'native_nsd_five_new_data_packet_20261012_v2/SOURCE_BINDINGS.json').read_text());p=ROOT/B['native/lib/laplace.py']['path'];tree=ast.parse(p.read_text());definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='batched_sym_matrix_pow');env={'torch':torch};exec(compile(ast.Module(body=[definition],type_ignores=[]),str(p),'exec'),env);source=env['batched_sym_matrix_pow']
class SPDProof(unittest.TestCase):
 def test_repeated_spectrum_forward_identity_and_composite_gradcheck(self):
  torch.set_num_threads(2);stable=make_spd_inverse_sqrt(torch)
  for dtype in (torch.float32,torch.float64):
   A=torch.stack([torch.eye(4,dtype=dtype)*2471.4504,torch.diag(torch.tensor([1.,1.,2.,2.],dtype=dtype)),torch.diag(torch.tensor([1.,2.,3.,5.],dtype=dtype))]).requires_grad_();native=source(A,-.5);fixed=stable(A,-.5);self.assertTrue(torch.equal(native,fixed));fixed.sum().backward();self.assertTrue(torch.isfinite(A.grad).all())
   bad=A.detach().clone().requires_grad_();source(bad,-.5).sum().backward();self.assertTrue(torch.isnan(bad.grad[0]).any())
  for scale in (0.,1.):
   X=(torch.eye(4,dtype=torch.float64)*scale).requires_grad_()
   self.assertTrue(torch.autograd.gradcheck(lambda Q:stable((Q@Q.T+torch.eye(4,dtype=Q.dtype)).unsqueeze(0),-.5),X,eps=1e-6,atol=2e-5,rtol=2e-4))
  with self.assertRaises(ValueError):stable(-torch.eye(4).unsqueeze(0),-.5)
  with self.assertRaises(ValueError):stable(torch.eye(4).unsqueeze(0),-1.)
  # Distinct spectrum adjoint agrees with the original SVD on symmetric tangents.
  X=torch.tensor([[1.,.2,.1,0.],[.2,2.,.3,.1],[.1,.3,3.,.2],[0.,.1,.2,5.]],dtype=torch.float64).unsqueeze(0).requires_grad_();G=torch.arange(16,dtype=torch.float64).reshape(1,4,4);G=(G+G.transpose(-1,-2))/2
  old=torch.autograd.grad(source(X,-.5),X,G)[0];new=torch.autograd.grad(stable(X,-.5),X,G)[0];self.assertTrue(torch.allclose((old+old.transpose(-1,-2))/2,new,atol=1e-10,rtol=1e-9))
 def test_only_two_corrected_native_source_callbacks_and_complete_restore(self):
  with tempfile.TemporaryDirectory(dir=HERE) as tmp:
   folder=Path(tmp);N=12;ids=torch.arange(N);x=torch.rand(N,3170,generator=torch.Generator().manual_seed(71));edge=torch.cat([torch.stack((ids,(ids+k)%N)) for k in (1,3,N-1,N-3)],1);keys=(edge[0]*N+edge[1]).unique(sorted=True);edge=torch.stack((keys//N,keys%N));inputs={}
   for role,data in [('graph',{'x':x,'edge_index':edge}),('train',{'ids':ids[:6],'y':ids[:6]%2}),('select',{'ids':ids[6:],'y':ids[6:]%2})]:
    f=folder/(role+'.pt');torch.save(data,f);raw=f.read_bytes();inputs[role]={'path':str(f.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
   for arm in ('layer_shared_incidence','node_capacity_right'):
    c={'nsd_arm':arm,'nsd_sources':B,'node_inputs':inputs,'classes':2,'learning_rate':.01,'dropout':.2,'initialization_seed':98101,'members':4,'source_complete_native_paths':4};a=make_adapter(torch=torch,project_root=ROOT,study={'dataset':'twitch-de-musae-raw'},run={'run_id':'synthetic-'+arm,'config':c},device='cpu');before=capture_checkpoint(torch,a,fit_id=arm,step=0,validation={},decision={});path=folder/(arm+'.pt');torch.save(before,path)
    for member in a.model.members:self.assertEqual(member.laplacian_builder.normalise.__code__,type(member.laplacian_builder).normalise.__code__)
    def denied_step(*args,**kwargs):return None
    a.optimizer.step=denied_step;z=a.train_step(1);self.assertEqual(z['operation_counts']['backward_calls'],1);self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters()));restore_checkpoint(torch,path,a,expected_fit_id=arm);after=capture_checkpoint(torch,a,fit_id=arm,step=0,validation={},decision={});self.assertTrue(_equal_tree(torch,before,after))
if __name__=='__main__':unittest.main()
