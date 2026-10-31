"""Synthetic CPU-only Q21 source/gradient/correction/state/reference checks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from adapters.private_bootstrap_source import ARMS,I4,SOURCES,CONTRACT,bootstrap_module,source_build,make_adapter,receipt
from compile_private_bootstrap_matrix import make_study,DOMAIN,SEARCH_SEEDS,COMPARATIVE_SEEDS

ROOT=Path(os.environ['GNNM_REFERENCE_PROJECT'])

def bindings():return {k:receipt(ROOT,v[0]) for k,v in SOURCES.items()}

def fixture(directory,backbone='SAGE',arm='shared4_coherent_bootstrap',body=0):
 import torch,numpy as np
 b=bindings();x=torch.arange(84,dtype=torch.float32).reshape(12,7)/84;edges=torch.tensor([[i for i in range(12) for _ in range(2)],[j for i in range(12) for j in (i,(i+1)%12)]])
 train={'ids':torch.tensor([4,0,5,1,3,2]),'y':torch.tensor([1,0,2,0,1,2])};select={'ids':torch.tensor([6,7,8]),'y':torch.tensor([0,1,2])}
 inputs={}
 for role,data in [('graph',{'x':x,'edge_index':edges}),('train',train),('select',select)]:
  p=directory/(role+'.pt');torch.save(data,p);inputs[role]=receipt(ROOT,str(p.relative_to(ROOT)))
 helper=bootstrap_module(ROOT,b,6,3);raw,w,record=helper.draw_weights(np,77101,train['ids'].numpy(),train['y'].numpy());weights=torch.from_numpy(w.copy())
 from adapters.context_exclusion import tensor_hash
 p=directory/'weights.pt';torch.save({'schema':'private-bootstrap-fixed-TRAIN-v1','seed':77101,'train':inputs['train'],'bootstrap_source':b['bootstrap'],
  'ids':train['ids'],'y':train['y'],'raw_exponential':torch.from_numpy(raw),'weights':weights,'weights_tensor_sha256':tensor_hash(weights),'source_record':record},p)
 c={'bootstrap_arm':arm,'members':ARMS[arm][2],'physical_slow_bodies':1,'source_optimizer_count':1,'backbone':backbone,'width':128,'depth':2,
  'classes':3,'weight_decay':0,'learning_rate':.001,'dropout':.2,'source_contract':CONTRACT,'bootstrap_sources':b,'node_inputs':inputs,
  **({'bootstrap_weights':receipt(ROOT,str(p.relative_to(ROOT)))} if ARMS[arm][3] else {})}
 run={'run_id':'synthetic-Q21-'+arm,'seed':77101,'body':body,'config':c}
 study={'dataset':'synthetic-Q21','data':{'scope':'synthetic_CPU_engineering_only'},'source':{'files':list(b.values())}}
 return study,run


class PrivateBootstrapSourceTests(unittest.TestCase):
 def setUp(self):
  import torch
  self.t=torch;self.threads=torch.get_num_threads();torch.set_num_threads(2)
 def tearDown(self):self.t.set_num_threads(self.threads)

 def test_fixed_source_weights_class_mass_local_RNG_and_dimension_delta(self):
  import numpy as np
  b=bindings();h=bootstrap_module(ROOT,b,6,3);ids=np.array([4,0,5,1,3,2],dtype=np.int64);y=np.array([1,0,2,0,1,2],dtype=np.int64)
  state=copy.deepcopy(np.random.get_state());r,w,record=h.draw_weights(np,77101,ids,y);r2,w2,_=h.draw_weights(np,77101,ids,y)
  self.assertTrue(np.array_equal(r,r2));self.assertTrue(np.array_equal(w,w2));self.assertTrue((w>0).all());self.assertEqual(w.dtype,np.float64)
  self.assertTrue(np.array_equal(np.random.get_state()[1],state[1]));self.assertEqual(np.random.get_state()[2:],state[2:])
  for label in range(3):self.assertTrue(np.allclose(w[:,y==label].sum(1),2,rtol=0,atol=1e-13))
  _,_,rendered=source_build(ROOT,b,7,3);self.assertEqual(rendered['provider_changes'],['native_features','native_classes','stem_factor_features','bootstrap.TRAIN_NODES','bootstrap.CLASSES'])
  bad=copy.deepcopy(b);bad['bootstrap']['sha256']='0'*64
  with self.assertRaises(ValueError):source_build(ROOT,bad,7,3)

 def test_real_GAT_SAGE_shared_private_block_gradients_and_corrections(self):
  t=self.t
  for backbone in ('GAT','SAGE'):
   for arm in list(ARMS)[:3]:
    with tempfile.TemporaryDirectory(dir=ROOT,prefix='Q21-synthetic-gradient-') as tmp:
     study,run=fixture(Path(tmp),backbone,arm);a=make_adapter(torch=t,project_root=ROOT,study=study,run=run,device='cpu')
     f=a.source_family;roles,shared,private=f.bootstrap.parameter_roles(t,f.factors,a.model)
     logits=f.logits(a.model,a.source_streams,True,a.train_ids);flat=logits.flatten(0,1);targets=a.train_y.repeat(4)
     own=t.nn.functional.cross_entropy(flat,targets);weighted=(t.nn.functional.cross_entropy(flat,targets,reduction='none').reshape(4,6).double()*a.bootstrap_weights).mean()
     sg=t.autograd.grad(own,[p for _,p in shared],retain_graph=True,allow_unused=True);pg=t.autograd.grad(weighted,[p for _,p in private],retain_graph=True,allow_unused=True)
     f.bootstrap.bank_gradients(t,t.nn.functional,logits,a.train_y,a.bootstrap_weights,shared,private)
     for named,expected in ((shared,sg),(private,pg)):
      for (_,p),g in zip(named,expected):
       if g is None:self.assertIsNone(p.grad)
       else:self.assertTrue(t.equal(p.grad,g))
     self.assertEqual(roles['reverse_mode_calls_per_update'],2)
     extra=roles['extra_private_parameters'];self.assertEqual(extra,0 if arm==list(ARMS)[0] else (2 if backbone=='GAT' else 4)*2*4*128)
     for m in a.model.modules():
      if hasattr(m,'u1'):self.assertTrue(t.equal(m.u1,m.u2));self.assertNotEqual(m.u1.data_ptr(),m.u2.data_ptr())
      if hasattr(m,'lora_b'):self.assertEqual(int(t.count_nonzero(m.lora_b)),0)
     result=a.train_step(1);self.assertEqual(result['operation_counts']['autograd_grad_VJP_calls'],2)
     self.assertEqual(result['operation_counts']['optimizer_updates'],1)

 def test_selected_state_stream_moment_roundtrip_and_immutable_table(self):
  t=self.t
  with tempfile.TemporaryDirectory(dir=ROOT,prefix='Q21-synthetic-state-') as tmp:
   study,run=fixture(Path(tmp),'SAGE','shared4_paired_graph_bootstrap');a=make_adapter(torch=t,project_root=ROOT,study=study,run=run,device='cpu')
   a.train_step(1);parameters=copy.deepcopy(a.model.state_dict());moments=copy.deepcopy(a.optimizer.state_dict());state=copy.deepcopy(a.capture_state())
   a.train_step(2);after=copy.deepcopy(a.model.state_dict());after_moments=copy.deepcopy(a.optimizer.state_dict())
   a.model.load_state_dict(parameters);a.optimizer.load_state_dict(moments);a.restore_state(copy.deepcopy(state));a.train_step(2)
   for name,v in after.items():self.assertTrue(t.equal(v,a.model.state_dict()[name]))
   for key,v in after_moments['state'].items():
    for name,field in v.items():
     if t.is_tensor(field):self.assertTrue(t.equal(field,a.optimizer.state_dict()['state'][key][name]))
   a.bootstrap_weights.add_(.1)
   with self.assertRaisesRegex(ValueError,'Immutable'):a.capture_state()

 def test_real_original_RSB_and_native_equipment_controls(self):
  t=self.t
  for backbone in ('GAT','SAGE'):
   for arm in ('ordinary_M1','factorized_allmap_M1','paired_graph_M1','rank1_lora_graph_M1','original_gnnm','shared4_unchanged'):
    with tempfile.TemporaryDirectory(dir=ROOT,prefix='Q21-synthetic-control-') as tmp:
     study,run=fixture(Path(tmp),backbone,arm);a=make_adapter(torch=t,project_root=ROOT,study=study,run=run,device='cpu')
     self.assertEqual(a.logits_fn(a.model,False,0).shape,(ARMS[arm][2],12,3))
     if arm=='original_gnnm':
      self.assertTrue(set(a.model.input_be_block.R.flatten().tolist())<={-1.,1.})
      self.assertTrue(t.equal(a.model.output_be_block.R,t.ones_like(a.model.output_be_block.R)))
      for m in (a.model.input_be_block,a.model.output_be_block):
       self.assertTrue(t.equal(m.S,t.ones_like(m.S)));self.assertEqual(int(t.count_nonzero(m.B)),0);self.assertTrue(m.R.requires_grad and m.S.requires_grad and m.B.requires_grad)
     value=a.train_step(1);self.assertEqual(value['operation_counts']['backward_calls'],4 if arm=='original_gnnm' else 1)
     self.assertEqual(value['operation_counts']['optimizer_updates'],1)

 def test_finite_all_masks_source_reference_roster_and_paid_I4(self):
  totals=[]
  for dataset,count in [('wikics',20),('amazon-photo',10)]:
   files=[{'path':'graph.pt','bytes':0,'sha256':'0'*64}];splits=[]
   for i in range(count):
    fields={r:r+str(i)+'.pt' for r in ('train','select')};end={r:r+str(i)+'.pt' for r in ('cal_ids','evaluate_ids','cal_targets','evaluate_targets')}
    for path in [*fields.values(),*end.values()]:files.append({'path':path,'bytes':0,'sha256':'0'*64})
    splits.append({'split_id':i,'evaluate_purpose':'development','node_inputs':{'graph':'graph.pt',**fields},'endpoint_inputs':end})
   custody={'dataset_id':dataset,'splits':splits,'files':files,'model_graph':{'classes':3},'request':{'graph_policy':{},'evaluation_exposure':{}},'request_hash':'synthetic'}
   weights={'tables':[{'dataset':dataset,'split':i,'seed':seed,'binding':{'path':'w.pt','bytes':0,'sha256':'0'*64}} for i in range(count) for seed in (*SEARCH_SEEDS,*COMPARATIVE_SEEDS)]}
   for backbone in ('GAT','SAGE'):
    value,n,banks=make_study(ROOT,custody,ROOT/'tiny',{'files':[]},{},'synthetic-runtime',backbone,weights);totals.append(n)
    self.assertTrue(value['study_id'].startswith('private-bootstrap-'));self.assertEqual(len([r for r in value['runs'] if r['kind']=='selection']),12)
    expected={(i,s) for i in range(count) for s in COMPARATIVE_SEEDS}
    exports=[r for r in value['runs'] if r['kind']=='ensemble_assembly']
    for cmp in value['comparisons']:
     for method in cmp['references']+cmp['candidates']:self.assertEqual({(r['split'],r['seed']) for r in exports if r['method']==method},expected)
    runmap={r['run_id']:r for r in value['runs']}
    for bank in banks:
     if bank['genuine_I4']:
      fits=[runmap[p] for p in bank['fit_parents']];self.assertEqual([r['body'] for r in fits],[0,1,2,3]);self.assertTrue(all(r['config']['members']==1 for r in fits))
      self.assertEqual(len({r['seed']+1000003*r['body'] for r in fits}),4)
  self.assertEqual(sum(n['atomic_fit_groups'] for n in totals),5184);self.assertEqual(sum(n['all_run_records'] for n in totals),13872)
  self.assertEqual(sum(n['full_budget_optimizer_updates'] for n in totals),25920000);self.assertEqual(sum(n['genuine_I4_banks'] for n in totals),720)


if __name__=='__main__':unittest.main()
