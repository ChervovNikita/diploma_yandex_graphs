"""Synthetic gate for native terminal-stop recovery and legacy full-budget law."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from adapters.classifier import ClassifierAdapter
from adapters.meansage_native import BUDGET,SELECTION
from engine import run_trajectory,resume_trajectory

class Toy(ClassifierAdapter):
 def train_step(self,step):
  self.optimizer.zero_grad();loss=sum(p.square().sum() for p in self.model.parameters());loss.backward();self.optimizer.step()
  return {'objective':float(loss.detach()),'ce':float(loss.detach()),'regularization':0.,'operation_counts':{'optimizer_updates':1,'backward_calls':1,'model_forwards':1}}
 def validate(self,step):
  v=0. if step<=2 else 1.
  return {'objective':v,'ce':v,'regularization':0.,'selection_metric':v,'operation_counts':{'optimizer_updates':0}}

def make():
 model=torch.nn.Linear(1,1);optimizer=torch.optim.SGD(model.parameters(),lr=.001)
 return Toy(torch,F,model,optimizer,logits_fn=None,train_ids=None,train_y=None,select_ids=None,select_y=None,regularizer=None,fusion='mean_probabilities',
  selector={'metric':'objective','direction':'min','ties':'last','patience_steps':50,'save_every':1},provenance={})

def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True);args=p.parse_args();packet=Path(args.project_root)/args.packet;folder=packet/'synthetic_stopping_recovery';folder.mkdir()
 c={'meansage_arm':'P1','meansage_family':'peer','architecture':'native_meansage_peer','source_optimizer_count':1,'physical_slow_bodies':1,'members':1,
  'learning_rate':.001,'dropout':.4,'weight_decay':0,'width':32,'depth':2,'classes':8,'backbone':'SAGE','is_original_gnnm':False,'historical_state_inherited':False,
  'initialization_seed':98401,'source_TRAIN_forwards_per_update':1,'source_TRAIN_backwards_per_update':1,'source_complete_body_constructors':1,
  'source_constructor_selection':'whole_single_source_factory','fusion':'mean_probabilities'}
 run={'kind':'model_fit','config':c,'budget':BUDGET,'selection':SELECTION,'seed':98401,'body':0}
 spec={'fit_id':'native-stop-recovery','budget_steps':100000,'budget_unit':'optimizer_updates','validation_interval':1,'validate_initial':False,'prefixes':[],
  'inherited_fit_ids':[],'adapter_spec':{'run':run},'completion_policy':'native_selector_stop'}
 def interruption(phase,step,result):
  if phase=='before_restore':raise RuntimeError('Synthetic controlled interruption after native stop')
 try:run_trajectory(torch,make(),spec,folder/'native',resource_check=interruption)
 except RuntimeError as error:assert 'Synthetic controlled' in str(error)
 else:raise AssertionError('Interruption not exercised')
 result=resume_trajectory(torch,make(),spec,folder/'native',admission={'schema':'synthetic-ended-owner-v1','previous_owner_terminal':True,'previous_owner':{'synthetic':True,'pid':0}})
 assert result['completed_steps']==52 and result['selected']['step']==2 and result['trajectory_operation_counts']['optimizer_updates']==52
 assert result['attempts'][-1]['operation_counts'].get('optimizer_updates',0)==0 and result['completion_reason']=='native_patience50'
 legacy={k:v for k,v in spec.items() if k!='completion_policy'};legacy['fit_id']='legacy-full-budget';legacy['budget_steps']=55;legacy['adapter_spec']={}
 value=run_trajectory(torch,make(),legacy,folder/'legacy')
 assert value['completed_steps']==55 and value['first_selector_stop']==52 and value['trajectory_operation_counts']['optimizer_updates']==55
 record={'schema':'native-stopping-recovery-synthetic-proof-v1','passed':True,'native_completed_updates':52,'native_recovery_new_optimizer_updates':0,
  'legacy_full_budget_completed_updates':55,'legacy_selector_stop_only_diagnostic':52,'original_failed_attempt_retained':True,
  'actual_input_updates':0,'actual_input_fits':0,'scores':0,'GO':False}
 (packet/'STOPPING_RECOVERY_CPU_PROOF.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');print(json.dumps(record))

if __name__=='__main__':main()
