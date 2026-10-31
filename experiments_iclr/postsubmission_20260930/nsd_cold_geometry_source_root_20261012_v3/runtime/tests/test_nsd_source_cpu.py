"""One synthetic fullGeneral source closure/optimizer/checkpoint qualification."""
from pathlib import Path
import copy,hashlib,json,os,tempfile,unittest
import torch
from adapters.nsd import make_adapter,factors,factory,construct,helpers
from engine import inspect_ownership,capture_checkpoint,restore_checkpoint,_equal_tree
ROOT=Path(os.environ['GNNM_REFERENCE_PROJECT']);PACKET=ROOT/'native_nsd_five_new_data_packet_20261012_v1'
class NativeNSDCPU(unittest.TestCase):
 def test_full_native_source_arms_and_frozen_warm_snapshot(self):
  torch.set_num_threads(2);B=json.loads((PACKET/'SOURCE_BINDINGS.json').read_text());seed=98101
  with tempfile.TemporaryDirectory(dir=PACKET) as tmp:
   folder=Path(tmp);N=24;g=torch.Generator().manual_seed(881);x=torch.randn(N,128,generator=g);ids=torch.arange(N);edges=torch.cat([torch.stack((ids,(ids+k)%N)) for k in (1,3,N-1,N-3)],1);keys=(edges[0]*N+edges[1]).unique(sorted=True);edge=torch.stack((keys//N,keys%N))
   inputs={}
   for role,data in [('graph',{'x':x,'edge_index':edge}),('train',{'ids':ids[:12],'y':ids[:12]%2}),('select',{'ids':ids[12:],'y':ids[12:]%2})]:
    path=folder/(role+'.pt');torch.save(data,path);raw=path.read_bytes();inputs[role]={'path':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
   def row(arm):
    paths=1 if arm in ('ordinary_M1','factorized_M1','original_gnnm') else 4;members=1 if arm in ('ordinary_M1','factorized_M1','joint_multi_transport_single') else 4
    return {'run_id':'synthetic-'+arm,'config':{'nsd_arm':arm,'nsd_sources':B,'node_inputs':inputs,'classes':2,'learning_rate':.01,'dropout':.2,'initialization_seed':seed,'members':members,'source_complete_native_paths':paths}}
   for arm in ('assigned_fixed','assigned_resampled','layer_shared_incidence','node_capacity_right','joint_multi_transport_single','ordinary_M1','factorized_M1','original_gnnm','geometry_warm'):
    r=row(arm);a=make_adapter(torch=torch,project_root=ROOT,study={'dataset':'twitch-de'},run=r,device='cpu');inspect_ownership(a)
    if arm=='geometry_warm':self.assertTrue(all(not p.requires_grad and bool(torch.all(p==1)) for p in factors(a.model)))
    metrics=a.train_step(1);self.assertEqual(metrics['operation_counts']['optimizer_updates'],1);v=a.validate(1);d=a.observe_validation(1,v);self.assertTrue(d['save_selected']);before=capture_checkpoint(torch,a,fit_id=r['run_id'],step=1,validation=v,decision=d);path=folder/(arm+'_selected.pt');torch.save(before,path)
    if arm=='geometry_warm':self.assertTrue(all(not p.requires_grad and p.grad is None and p not in a.optimizer.state and bool(torch.all(p==1)) for p in factors(a.model)))
    # One continuation call followed by full serializer restore checks source plain L/RNG/Adam and modes.
    a.train_step(2);restore_checkpoint(torch,path,a,expected_fit_id=r['run_id']);after=capture_checkpoint(torch,a,fit_id=r['run_id'],step=1,validation=v,decision=d);self.assertTrue(_equal_tree(torch,before,after),arm)
    if arm=='geometry_warm':
     for p in factors(a.model):p.requires_grad_(True)
     a.train_step(2);self.assertTrue(all(p in a.optimizer.state for p in factors(a.model)));inspect_ownership(a)
     # The exact actual normalized first-action hook executes on synthetic native geometry.
     counts={k:0 for k in ('screen_native_builder_returns','extra_offnode_spmm_attempts','extra_offnode_spmm_completed','offnode_sparse_copy_bytes_total','screen_action_CPU_bytes_total')};member=a.model.members[0];member.eval()
     with torch.no_grad(),a.e.init.FirstActionCapture(torch,member,a.train_ids,counts,None) as action:member(x)
     self.assertEqual(counts['extra_offnode_spmm_completed'],1);self.assertTrue(torch.isfinite(action.action).all());self.assertGreater(action.baseline_value.numel(),0)
    del a

if __name__=='__main__':unittest.main()
