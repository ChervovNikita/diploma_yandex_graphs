"""Two tiny synthetic callbacks through the literal HGT train/serialize protocol."""
import copy,importlib.util,json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
OUT=Path(__file__).resolve().parents[1];PHASE=OUT.parent;sys.path[:0]=[str(OUT/'runtime'),str(PHASE/'native_hgt_mixed_newdata_packet_20261012_v2/runtime')]
import torch
from torch.nn import functional as F
from adapters.hgt_native import HGTAdapter
from hgt_resource import run_protocol
class Streams:
 def __init__(self):self.states=[torch.get_rng_state().clone()]
 def state_dict(self):return {'states':[s.clone() for s in self.states]}
 def load_state_dict(self,v):self.states=[s.clone() for s in v['states']]
def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU synthetic protocol only')
 torch.set_num_threads(2);torch.manual_seed(9281)
 class Tiny(torch.nn.Module):
  def __init__(self):super().__init__();self.linear=torch.nn.Linear(3,4);self.unused=torch.nn.Parameter(torch.ones(2));self.x=torch.arange(18).float().reshape(6,3)/20
  def forward(self):return self.linear(F.dropout(self.x,p=.2,training=self.training)).unsqueeze(0)
  def operations(self,training):return {'synthetic_source_path':1}
 model=Tiny();optimizer=torch.optim.AdamW(model.parameters(),weight_decay=.0001);scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,total_steps=300,max_lr=.001,pct_start=.05)
 source=PHASE/'native_hgt_mixed_newdata_packet_20261012_v2/sources/onecycle.py';sp=importlib.util.spec_from_file_location('literal_HGT_onecycle_state',source);onecycle=importlib.util.module_from_spec(sp);sp.loader.exec_module(onecycle)
 adapter=HGTAdapter(torch,F,model,optimizer,logits_fn=lambda m,*_:m(),train_ids=torch.tensor([0,1,2,3]),train_y=torch.arange(4),select_ids=torch.tensor([4]),select_y=torch.tensor([0]),
  regularizer=lambda m:next(m.parameters()).new_zeros(()),fusion='mean_logits',selector={'metric':'pooled_nll','direction':'min','ties':'last','patience_steps':30,'save_every':1},provenance={'config':{'hgt_arm':'native_M1'}})
 adapter.scheduler=scheduler;adapter.onecycle=onecycle;adapter.streams=Streams();adapter.custody={'fixture':'synthetic'};adapter.blocks=None;adapter.preparation_charged=False;adapter.preparation_counts={}
 adapter.implementation=SimpleNamespace(mean_member_ce=lambda z,ids,y:torch.stack([F.cross_entropy(row[ids],y) for row in z]).mean())
 before=copy.deepcopy((model.state_dict(),optimizer.state_dict(),onecycle.portable_state(scheduler)));owned_step=optimizer.step;class_step_before=type(scheduler).__dict__.get('step');has_class_step='step' in type(scheduler).__dict__
 with tempfile.TemporaryDirectory(dir=OUT/'tests') as temp:proof=run_protocol(torch,adapter,Path(temp),'synthetic-HGT-own','cpu')
 from engine import _equal_tree
 assert _equal_tree(torch,before,(model.state_dict(),optimizer.state_dict(),onecycle.portable_state(scheduler)))
 assert not optimizer.state and all(p.grad is None for p in model.parameters()) and optimizer.step==owned_step
 assert ('step' in type(scheduler).__dict__)==has_class_step and type(scheduler).__dict__.get('step')==class_step_before
 assert len(proof['callbacks'])==2 and proof['callbacks'][1]['moment_scope']=='synthetic_active_only_mature_buffers' and all(r['source_None_gradient_parameter_objects']==1 for r in proof['callbacks'])
 assert proof['nonlearned_mature_serializer_proxy']['exact_engine_restore_verified'] and proof['nonlearned_mature_serializer_proxy']['synthetic_active_only_AdamW_moment_elements']==32
 result={'schema':'HGT-resource-protocol-synthetic-denied-CPU-proof-v1','callbacks':2,'source_HGT_train_step_body_used':True,'native_Torch_OneCycle300_and_literal_state_helper':True,
  'source_unused_None_leaf_preserved':True,'active_only_nonlearned_mature_moment_second_callback_and_serializer':True,'model_optimizer_modes_flags_global_private_RNG_restored':True,
  'synthetic_optimizer_updates':0,'actual_input_models':0,'actual_input_callbacks':0,'scores':0,'GO':False,'CUDA_initialized':torch.cuda.is_initialized()}
 p=OUT/'SYNTHETIC_PROTOCOL_PROOF.json'
 if p.exists():raise FileExistsError('Preserve proof')
 p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
