"""New endpoint/count source CPU risks: tiny graph, source dimensions, replay."""
import copy,json,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parents[2];ROOT=P.parent;sys.path.insert(0,str(P/'runtime'))
from adapters import link_prototypes as port
from adapters.common_native_wrapper import receipt

def equal(torch,a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(torch,a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(torch,x,y) for x,y in zip(a,b))
 return a==b

def main():
 import torch,numpy
 from torch_geometric.utils import negative_sampling
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 bindings={k:receipt(ROOT,P/'sources'/v) for k,v in port.PATHS.items()};prior=json.loads((P/'LINK_PROTOTYPES_CPU_QA_TRANSPORT_attempt2.json').read_text());rows=[json.loads(line)['case'] for line in prior['stdout'].splitlines() if line.startswith('{')]
 assert len(rows)==7
 for row in rows:row['preserved_pass_receipt']='LINK_PROTOTYPES_CPU_QA_TRANSPORT_attempt2.json'
 steps=0
 for arm in port.ARMS:
  if arm!='C_mu':continue
  torch.manual_seed(614);n=12;x=torch.randn(n,128);pairs=torch.tensor([[0,1],[0,2],[1,2],[2,3],[3,4],[3,5],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11]],dtype=torch.long);raw=torch.cat((pairs,pairs.flip(1)),0).T.contiguous();records=torch.tensor([0,4,7]);negative=torch.tensor([[0,4],[1,6],[2,7]],dtype=torch.long)
  model,opt,modules,details=port.construct(torch,ROOT,bindings,arm=arm,seed=88101,body=0,lr_multiplier=1.,dropout=.3,device='cpu');graph=modules['graph_ops'].Graph.mask_train_batch(pairs,records,n)
  if arm in ('P0','J_P','F_P'):assert sum(p.numel() for p in model.parameters())==43790
  if arm=='C_mu':assert sum(p.numel() for p in model.parameters())==88900 and details['unstepped_discarded_optimizer_objects']==1
  if arm.startswith('endpoint_'):
   assert tuple(model.decoder.v.shape)==((4,64) if arm=='endpoint_private4' else (1,64))
   h=torch.randn(n,64,requires_grad=True);q=pairs[:3];product=model.decoder.endpoint_product(h,q,0);grad=torch.autograd.grad(product.square().sum(),h)[0];assert torch.isfinite(grad).all() and grad.abs().sum()>0
   # Symmetric endpoint operation with exact unit-axis reflection, no clipping.
   assert torch.equal(product,model.decoder.endpoint_product(h,q.flip(1),0))
  if arm in ('P0','J_P','F_P','C_mu'):
   teacher=modules['pattern_teacher'].ObservationTeacher.from_train(pairs,n);neighbors=modules['graph_ops'].enumerate_neighbors(graph,pairs[records]);_,labels=teacher.labels(pairs[records],neighbors);assert labels.ndim==1
   class TinyReplay:
    receipt={'master_seed':2026100401}
    data={'pairs':pairs,'nodes':n}
    def query_batch(self,batch):return records,torch.cat((negative,negative,negative,negative,negative),0)[:len(pairs)],{}
    def assert_actual(self,batch,value):
     assert torch.equal(value['record_ids'],records)
     expected=modules['graph_ops'].Graph.mask_train_batch(pairs,records,n);assert torch.equal(value['graph'].row,expected.row) and torch.equal(value['graph'].col,expected.col)
     # Source checks the complete ordered support tuple before teacher access.
     for key in ('positive','negative'):
      q=value[key+'_queries'];expected_n=modules['graph_ops'].enumerate_neighbors(expected,q)
      for role in ('common','left','right'):assert all(torch.equal(a,b) for a,b in zip(getattr(value[key+'_neighbors'],role),getattr(expected_n,role)))
   replay=TinyReplay();learning=modules['learning'];ns=dict(vars(learning),TRAIN_BATCH=3);tree=__import__('ast').parse(Path(learning.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,__import__('ast').FunctionDef) and n.name=='train_replay_batch');exec(compile(__import__('ast').fix_missing_locations(__import__('ast').Module(body=[copy.deepcopy(fn)],type_ignores=[])),'<tiny_cardinality_source_fixture>','exec'),ns)
   update=lambda:ns['train_replay_batch'](model,opt,x,replay,0,teacher,arm)
  else:
   def update():
    model.train();opt.zero_grad(set_to_none=True);loss=modules['prototype'].training_batch_loss(model,x,pairs,records,negative,'private');loss.backward();opt.step();return float(loss.detach())
  state=(copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),port.bridge.rng_state(torch,numpy));value=update();steps+=1;after=(copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),port.bridge.rng_state(torch,numpy));model.load_state_dict(state[0]);opt.load_state_dict(state[1]);port.bridge.restore_rng(torch,numpy,state[2]);again=update();steps+=1
  assert equal(torch,value,again) and equal(torch,after[0],model.state_dict()) and equal(torch,after[1],opt.state_dict()) and equal(torch,after[2],port.bridge.rng_state(torch,numpy))
  if arm in ('P0','J_P','F_P'):
   model.eval();h=model.encode(x,graph);z,detail=model.query_forward(h,graph,pairs[records],auxiliary_grad=False);assert len(detail['neighbors'].common[0])>=0 and z.shape==(3,4)
  if arm=='C_mu':
   model.eval();h=model.encode(x,graph);z,detail=model.query_forward(h,graph,pairs[records]);assert not detail['mu'].requires_grad and detail['eta'].dtype==torch.float64 and z.shape==(3,1)
  rows.append({'arm':arm,'source_initialization_parameter_census':True,'outer_endpoint_feature_gradients':True,'source_support_teacher_membership':True,'new_API_and_Adam_RNG_replay':True});print(json.dumps({'case':rows[-1],'completed_toy_steps':steps}),flush=True)
 assert len(rows)==8 and steps==2 and not torch.cuda.is_initialized()
 result={'schema':'link-prototype-new-source-CPU-QA-v1','status':'passed','cases':rows,'toy_optimizer_steps':steps,'prior_attempt_known_toy_steps':14,'all_attempt_toy_steps':steps+14,'benchmark_fits':0,'actual_input_arrays':0,'CUDA_compute':0,'existing_kernel_tests_repeated':False};(P/'LINK_PROTOTYPES_CPU_QA.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','cases':8,'toy_steps':steps,'scientific_fits':0}))

if __name__=='__main__':main()
