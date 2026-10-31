"""New synthetic CPU integration checks; no scientific inputs or CUDA."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from adapters.native_label_families import make_adapter,feature_graph,receipt,NEIGHBOR_ARMS,SPARSE_ARMS
from adapters.shared_message_port import make_adapter as shared_adapter,prepare_operator
from adapters.private_hop_port import source_environment
from compile_label_family_batch import make_study,FAMILIES

ROOT=Path(os.environ['GNNM_REFERENCE_PROJECT']).resolve();PACKET=Path(__file__).resolve().parents[2];B=json.loads((PACKET/'SOURCE_BINDINGS.json').read_text())


def fixture(directory,family,arm,features=300,classes=10,body=0):
 import torch
 n=24;x=torch.arange(n*features,dtype=torch.float32).reshape(n,features)/(n*features)+.001
 edges=torch.tensor([[i for i in range(n) for _ in range(2)],[j for i in range(n) for j in (i,(i+1)%n)]])
 ids=torch.arange(10);train={'ids':ids,'y':ids%classes};select={'ids':torch.tensor([10,11,12]),'y':torch.tensor([0,1,2])}
 inputs={}
 for role,payload in [('graph',{'x':x,'edge_index':edges}),('train',train),('select',select)]:
  path=directory/(role+'.pt');torch.save(payload,path);inputs[role]=receipt(ROOT,str(path.relative_to(ROOT)))
 derived={'sparse_supports':{},'mean_operators':{}}
 if family=='sparse_feature_kernel':
  cfg={'family_sources':B[family]};h,edits=feature_graph(ROOT,cfg,features,n);support,meta=h.build_support(torch,x,edges)
  path=directory/'support.pt';torch.save({'edge_index':support,'source_graph':inputs['graph'],'feature_graph_source':B[family]['feature_graph'],'provider_sites':edits,'preparation':meta},path)
  derived['sparse_supports']['wikics']=receipt(ROOT,str(path.relative_to(ROOT)))
 if family=='shared_message':
  p,a=prepare_operator(torch,x,edges);path=directory/'mean.pt';torch.save({'forward':p,'adjoint':a,'source_graph':inputs['graph'],'policy':'incoming_record_mean_float32;factualCOO;no new loops'},path)
  derived['mean_operators']['wikics']=receipt(ROOT,str(path.relative_to(ROOT)))
 # Metadata compiler supplies the same contracts as actual disabled rows.
 study,_,_=make_study(ROOT,PACKET,family,'wikics',B,{'sparse_supports':{'wikics':derived['sparse_supports'].get('wikics',{'path':'synthetic','bytes':1,'sha256':'0'*64})},
  'mean_operators':{'wikics':derived['mean_operators'].get('wikics',{'path':'synthetic','bytes':1,'sha256':'0'*64})}})
 row=copy.deepcopy(next(r for r in study['runs'] if r['kind']=='model_fit' and r['config']['family_arm']==arm))
 row.update(seed=99101,body=body);row['config'].update(node_inputs=inputs,classes=classes,dropout=.2,learning_rate=.001,
  provider_graph={'nodes':n,'features':features,'classes':classes,'edges':len(edges[0])},initialization_seed=99101+1000003*body)
 if family=='sparse_feature_kernel':row['config']['sparse_support']=derived['sparse_supports']['wikics']
 if family=='shared_message':row['config']['mean_operator']=derived['mean_operators']['wikics']
 study['data']={'scope':'synthetic_CPU_engineering_only'};study['source']={'files':list(B['common'].values())}
 return study,row


class SourceBatchCPU(unittest.TestCase):
 def setUp(self):
  import torch
  self.t=torch;self.threads=torch.get_num_threads();torch.set_num_threads(2)
 def tearDown(self):self.t.set_num_threads(self.threads)
 def equal(self,a,b):
  if self.t.is_tensor(a):self.assertTrue(self.t.equal(a,b))
  elif isinstance(a,dict):
   self.assertEqual(set(a),set(b))
   for k in a:self.equal(a[k],b[k])
  elif isinstance(a,(list,tuple)):
   self.assertEqual(len(a),len(b))
   for x,y in zip(a,b):self.equal(x,y)
  else:self.assertEqual(a,b)

 def test_neighborhood_complete_constructor_dual_CE_gradients_native_readout_and_roundtrip(self):
  t=self.t
  for arm in NEIGHBOR_ARMS[:4]:
   with tempfile.TemporaryDirectory(dir=ROOT,prefix='label-neighborhood-CPU-') as d:
    study,row=fixture(Path(d),'neighborhood_distribution',arm);a=make_adapter(torch=t,project_root=ROOT,study=study,run=row,device='cpu')
    f=a.source_family;m,o,s=f.make(row['seed'],row['body'],'baseline',True,row['config']['members'])
    self.equal(a.model.state_dict(),m.state_dict());self.equal(a.source_streams,s)
    got=a.train_step(1);expected=f.train_update(m,o,s,row['config']['members']>1,row['config']['members'])
    self.assertEqual(got['objective'],expected);self.equal(a.model.state_dict(),m.state_dict());self.equal(a.optimizer.state_dict(),o.state_dict())
    a.model.eval();logits=a.logits_fn(a.model,False,1);self.assertEqual(logits.shape[-1],10);self.assertEqual(f._last_native_logits.shape,logits.shape)
    p=copy.deepcopy(a.model.state_dict());o=copy.deepcopy(a.optimizer.state_dict());state=copy.deepcopy(a.capture_state());a.train_step(2);after=copy.deepcopy(a.model.state_dict())
    a.model.load_state_dict(p);a.optimizer.load_state_dict(o);a.restore_state(state);a.train_step(2);self.equal(after,a.model.state_dict())
  for arm in ('ordinary_M1','factorized_plain_M1','original_gnnm'):
   with tempfile.TemporaryDirectory(dir=ROOT,prefix='label-native-controls-CPU-') as d:
    st,row=fixture(Path(d),'neighborhood_distribution',arm,features=7,classes=3);a=make_adapter(torch=t,project_root=ROOT,study=st,run=row,device='cpu')
    self.assertEqual(a.train_step(1)['operation_counts']['backward_calls'],4 if arm=='original_gnnm' else 1)
  with tempfile.TemporaryDirectory(dir=ROOT,prefix='label-neighborhood-I4-CPU-') as d:
   seeds=[]
   for b in range(4):
    st,row=fixture(Path(d),'neighborhood_distribution',NEIGHBOR_ARMS[3],features=7,classes=3,body=b);a=make_adapter(torch=t,project_root=ROOT,study=st,run=row,device='cpu')
    seeds.append(a.model._distribution_seeds);self.assertEqual(a.logits_fn(a.model,False,0).shape,(1,24,3))
   self.assertEqual(len({tuple(s['directions']) for s in seeds}),4);self.assertEqual(len({s['joint_residual_head'] for s in seeds}),4)

 def test_sparse_complete_source_modes_support_mean_and_parameter_moment_stream_restore(self):
  t=self.t
  for arm in SPARSE_ARMS[:5]:
   with tempfile.TemporaryDirectory(dir=ROOT,prefix='label-sparse-CPU-') as d:
    st,row=fixture(Path(d),'sparse_feature_kernel',arm);a=make_adapter(torch=t,project_root=ROOT,study=st,run=row,device='cpu')
    f=a.source_family;mode='private' if arm==SPARSE_ARMS[0] else 'fixed' if arm==SPARSE_ARMS[2] else 'common';members=row['config']['members']
    m,o,s=f.make(row['seed'],row['body'],'baseline',arm!=SPARSE_ARMS[3],members,mode)
    self.equal(a.model.state_dict(),m.state_dict());loss=a.train_step(1);expected=f.source_train(m,o,s,members,members>1)
    self.assertEqual(loss['objective'],expected);self.equal(a.model.state_dict(),m.state_dict());self.equal(a.optimizer.state_dict(),o.state_dict())
    self.assertEqual(loss['operation_counts']['backward_calls'],1);self.assertEqual(loss['operation_counts']['feature_kernel_contexts'],1)
    p=copy.deepcopy(a.model.state_dict());o=copy.deepcopy(a.optimizer.state_dict());state=copy.deepcopy(a.capture_state());a.train_step(2);after=copy.deepcopy(a.model.state_dict())
    a.model.load_state_dict(p);a.optimizer.load_state_dict(o);a.restore_state(state);a.train_step(2);self.equal(after,a.model.state_dict())
  with tempfile.TemporaryDirectory(dir=ROOT,prefix='label-sparse-provider-CPU-') as d:
   st,row=fixture(Path(d),'sparse_feature_kernel',SPARSE_ARMS[4],features=7,classes=3);a=make_adapter(torch=t,project_root=ROOT,study=st,run=row,device='cpu')
   self.assertEqual(a.model._feature_kernel.projection.shape,(7,16));a.train_step(1)

 def test_shared_message_full_source_paths_native_reference_final_selector_and_state(self):
  t=self.t
  for arm in FAMILIES['shared_message']['arms']:
   with tempfile.TemporaryDirectory(dir=ROOT,prefix='label-shared-CPU-') as d:
    st,row=fixture(Path(d),'shared_message',arm,features=7,classes=3);a=shared_adapter(torch=t,project_root=ROOT,study=st,run=row,device='cpu')
    a.train_step(1);p=copy.deepcopy(a.model.state_dict());o=copy.deepcopy(a.optimizer.state_dict());state=copy.deepcopy(a.capture_state());a.train_step(2);after=copy.deepcopy(a.model.state_dict())
    a.model.load_state_dict(p);a.optimizer.load_state_dict(o);a.restore_state(state);a.train_step(2);self.equal(after,a.model.state_dict())
    a.model.eval();self.assertEqual(a.logits_fn(a.model,False,2).shape,(row['config']['members'],24,3));self.assertEqual(a.fusion,'mean_logits')
    self.assertEqual(a.selector['metric'],'pooled_nll');self.assertEqual(a.selector['direction'],'min')
    if arm=='exact_member':
     with t.random.fork_rng(devices=[]):
      t.random.default_generator.manual_seed(row['seed']);native=a.source_models.TABMModel('SAGE',2,7,64,3,1,1,'LayerNorm',.2,tabm_inits=4,device='cpu')
     # Compare source portable and literal native initialization/readout.
     env=dict(a.source_models.construct.__globals__);env['SEEDS']=(row['seed'],)
     from types import FunctionType
     fresh=FunctionType(a.source_models.construct.__code__,env,closure=a.source_models.construct.__closure__)(
       {'features':7,'classes':3},'exact_member',seed=row['seed'],checkpoint_blocks=False)
     fresh.eval();native.eval();graph=t.load(Path(d)/'graph.pt',weights_only=True)
     expected=t.stack([native(type('Graph',(),{'edge_index':graph['edge_index']})(),graph['x'],m) for m in range(4)])
     actual=a.source_models.forward_logits(fresh,a.source_family,graph['x'])
     self.assertTrue(t.allclose(actual,expected,rtol=1e-5,atol=1e-5))

 def test_private_hop_full_23site_body_gather_commit_and_factual_aux_stream_custody(self):
  t=self.t;c={'hop_sources':B['private_hop'],'polyformer_files':B['polyformer_files'],'boundary_source':B['common']['native'],'dropout':.5,'learning_rate':.005}
  env,h,recipe,proof=source_environment(t,ROOT,c,nodes=136)
  x=t.arange(136*500,dtype=t.float32).reshape(136,500)/(136*500)+.001
  edges=t.tensor([[i for i in range(136) for _ in range(2)],[j for i in range(136) for j in (i,(i+1)%136)]])
  ids=t.arange(12);y=ids%3;s=env['Session']('shared4_own',99101,x,edges,ids,y);credit=h['HopCredit'](s,'private_missinghop')
  before=copy.deepcopy(s.streams.states);g=credit.gather();self.assertEqual(credit.counters['factual_forwards'],4);self.assertEqual(credit.counters['auxiliary_forwards'],3)
  self.assertEqual(credit.counters['gradient_calls'],7);self.assertEqual(credit.counters['optimizer_steps'],0);credit.commit(g);self.assertEqual(credit.counters['optimizer_steps'],1)
  self.assertEqual(len([m for m in credit.body.modules() if isinstance(m,s.factors.FactorLinear)]),23)
  for condition in ('single_native','factor1_native'):
   session=env['Session']('single_native',99101,x,edges,ids,y)
   if condition=='factor1_native':h['prepare_factor1'](session);q=h['HopCredit'](session,'factor1_native');q.step();self.assertEqual(q.counters['gradient_calls'],1)
   else:session.train_step();self.assertEqual(session.counters['backwards'],1)


if __name__=='__main__':unittest.main()
