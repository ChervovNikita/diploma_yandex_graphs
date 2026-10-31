"""Only the changed raw3170 input shape;source callback mechanics inherited."""
from pathlib import Path
import hashlib,json,os,tempfile,unittest
import torch
from adapters.nsd import make_adapter
from engine import inspect_ownership
ROOT=Path(os.environ['GNNM_REFERENCE_PROJECT']);P=ROOT/'native_nsd_five_new_data_packet_20261012_v2'
class RawInputShape(unittest.TestCase):
 def test_full_source_raw3170_stem_factor_and_literal_boundary_shapes(self):
  torch.set_num_threads(2);B=json.loads((P/'SOURCE_BINDINGS.json').read_text())
  with tempfile.TemporaryDirectory(dir=P) as tmp:
   folder=Path(tmp);N=12;ids=torch.arange(N);x=torch.randn(N,3170,generator=torch.Generator().manual_seed(881));edge=torch.cat([torch.stack((ids,(ids+k)%N)) for k in (1,3,N-1,N-3)],1);keys=(edge[0]*N+edge[1]).unique(sorted=True);edge=torch.stack((keys//N,keys%N));inputs={}
   for role,data in [('graph',{'x':x,'edge_index':edge}),('train',{'ids':ids[:6],'y':ids[:6]%2}),('select',{'ids':ids[6:],'y':ids[6:]%2})]:
    f=folder/(role+'.pt');torch.save(data,f);raw=f.read_bytes();inputs[role]={'path':str(f.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
   for arm in ('assigned_fixed','assigned_resampled','layer_shared_incidence','node_capacity_right','joint_multi_transport_single','ordinary_M1','factorized_M1','original_gnnm','geometry_warm'):
    paths=1 if arm in ('ordinary_M1','factorized_M1','original_gnnm') else 4;members=1 if arm in ('ordinary_M1','factorized_M1','joint_multi_transport_single') else 4
    c={'nsd_arm':arm,'nsd_sources':B,'node_inputs':inputs,'classes':2,'learning_rate':.01,'dropout':.2,'initialization_seed':98101,'members':members,'source_complete_native_paths':paths}
    a=make_adapter(torch=torch,project_root=ROOT,study={'dataset':'twitch-de-musae-raw'},run={'run_id':'synthetic-raw-shape-'+arm,'config':c},device='cpu');inspect_ownership(a);a.model.eval()
    with torch.no_grad():z=a.logits_fn(a.model,False,0)
    self.assertEqual(tuple(z.shape),(members,N,2));self.assertTrue(torch.isfinite(z).all());self.assertEqual(a.provenance['native_args']['input_dim'],3170)
    del a
if __name__=='__main__':unittest.main()
