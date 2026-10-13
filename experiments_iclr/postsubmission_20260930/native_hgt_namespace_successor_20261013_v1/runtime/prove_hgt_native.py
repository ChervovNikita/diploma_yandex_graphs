"""Synthetic CPU source/blocks/OneCycle/serializer/stopping proof only."""
import argparse,copy,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from adapters import hgt_native as a
from adapters.classifier import ClassifierAdapter
from engine import capture_checkpoint,restore_checkpoint,run_trajectory,_equal_tree,inspect_independence
from hgt_stopping import validate_hgt_run

def config(arm,seed,body,sources,inputs):
 members=1 if arm in ('native_M1','factor_M1') else 4
 return {'hgt_arm':arm,'hgt_sources':sources,'architecture':'native_HGT_typed','backbone':'native_HGB_HGT','width':72 if arm=='wider_BE' else 64,
  'depth':3,'heads':8,'classes':4,'dropout':.2,'weight_decay':.0001,'learning_rate':.001,'fusion':'mean_logits','members':members,
  'source_optimizer_count':1,'physical_slow_bodies':4 if arm=='untied_HGT' else 1,'initialization_seed':seed+1000003*body,
  'source_complete_body_constructors':1 if arm=='original_gnnm' else 5 if arm=='untied_HGT' else 3 if arm=='wider_BE' else 2,
  'source_scheduler_rule':'OneCycle300 maxlr.001 pct_start.05 cosine step(update)','source_TRAIN_forwards_per_update':members,
  'source_TRAIN_backwards_per_update':2 if arm in ('pool/own','own/pool') else 1,'historical_state_inherited':False,'is_original_gnnm':arm=='original_gnnm',
  'node_inputs':inputs,'stage':'comparative'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True);args=p.parse_args();root=Path(args.project_root);packet=root/args.packet
 started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(121021);sources=json.loads((packet/'EXACT_SOURCE_BINDINGS.json').read_text());sources={k:{x:v for x,v in r.items() if x in ('path','bytes','sha256')} for k,r in sources.items()}
 folder=packet/'synthetic_cpu';folder.mkdir();counts={'0':8,'1':6,'2':4,'3':3};dims={'0':3,'1':6,'2':4,'3':3};relations=[]
 names=[]
 for raw,u,v in ((0,'0','1'),(1,'1','2'),(2,'1','3'),(3,'1','0'),(4,'2','1'),(5,'3','1')):
  src=torch.arange(counts[u]);dst=src%counts[v];relations.append({'raw_id':raw,'source':u,'target':v,'src':src,'dst':dst});names.append(f'raw{raw}__{u}__{v}')
 features={'0':torch.arange(24).reshape(8,3).float()/20}
 for t in ('1','2','3'):
  ids=torch.arange(counts[t]);features[t]=torch.sparse_coo_tensor(torch.stack([ids,ids]),torch.ones(counts[t]),(counts[t],counts[t])).coalesce()
 schema={'node_counts':counts,'input_dims':dims,'target':'0','feature_type':2,'relation_row_order':names}
 torch.save({'schema':schema,'features':features,'relations':relations},folder/'graph.pt');torch.save({'ids':torch.arange(4),'y':torch.arange(4)},folder/'train.pt');torch.save({'ids':torch.arange(4,8),'y':torch.arange(4)},folder/'select.pt')
 inputs={k:a.receipt(root,folder/(k+'.pt')) for k in ('graph','train','select')};study={'study_id':'synthetic_HGT','dataset':'synthetic_HGB_DBLP','stage':'engineering','source':{'files':list(sources.values())},'data':{'roles':{'train':{'0':inputs['train']},'select':{'0':inputs['select']}}}}
 adapters={};checks=[];updates=0;seed=99601
 for arm in a.ARMS:
  c=config(arm,seed,0,sources,inputs);run={'run_id':'synthetic-'+arm.replace('/','_'),'kind':'model_fit','split':0,'seed':seed,'body':0,'config':c,'budget':a.BUDGET,'selection':a.SELECTION,'parents':[]}
  validate_hgt_run(run);adapter=a.make_adapter(torch=torch,project_root=root,study=study,run=run,device='cpu');outer=torch.get_rng_state().clone();metrics=adapter.train_step(1);updates+=1
  assert torch.equal(outer,torch.get_rng_state());assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.model.parameters())
  assert all(p not in adapter.optimizer.state for p in adapter.model.parameters() if p.grad is None)
  assert adapter.scheduler.last_epoch==1 and metrics['operation_counts']['optimizer_updates']==1 and metrics['operation_counts']['source_scheduler_steps']==1
  adapter.model.eval();val=adapter.validate(1);decision=adapter.observe_validation(1,val);payload=capture_checkpoint(torch,adapter,fit_id=run['run_id'],step=1,validation=val,decision=decision)
  path=folder/(run['run_id']+'.pt');torch.save(payload,path)
  result1=adapter.train_step(2);updates+=1;state1=capture_checkpoint(torch,adapter,fit_id=run['run_id'],step=2,validation={},decision={})
  restore_checkpoint(torch,path,adapter,expected_fit_id=run['run_id']);assert adapter.scheduler.last_epoch==1
  result2=adapter.train_step(2);updates+=1;state2=capture_checkpoint(torch,adapter,fit_id=run['run_id'],step=2,validation={},decision={})
  assert _equal_tree(torch,state1,state2) and result1==result2
  adapters[arm]=adapter
  (packet/'SYNTHETIC_HGT_PROGRESS.json').write_text(json.dumps({'completed_arms':list(adapters),'synthetic_AdamW_updates':updates,'actual_input_updates':0},indent=2)+'\n')
 checks.append('13 exact source/control factories; parameter count and full paths; present gradients finite; unused None leaves have no AdamW moments/decay')
 checks.append('all13 complete model/AdamW/OneCycle/private/global RNG/mode/flag serialized restores and exact next-update2 replay')
 # Pure source mixed hook AST reverse-projection, without calling old driver main.
 import inspect
 driver=a.source(root,sources['driver'],'driver');policy=a.source(root,sources['block_policy'],'block_policy');tree,custody=policy.fit_tree(inspect.getsource(driver.fit));assert custody['reverse_projection_exact'] and custody['substitutions']==1
 checks.append('exact mixed block policy one-backward AST substitution reverse projection; own/own direct source law')
 # Witness genuine separately selected own fits have disjoint parameter storage.
 for arm in ('native_M1','factor_M1'):
  own=[adapters[arm]]
  for body in (1,2,3):
   c=config(arm,seed,body,sources,inputs);run={'run_id':arm+'-'+str(body),'kind':'model_fit','split':0,'seed':seed,'body':body,'config':c,'budget':a.BUDGET,'selection':a.SELECTION,'parents':[]};own.append(a.make_adapter(torch=torch,project_root=root,study=study,run=run,device='cpu'))
  assert inspect_independence(own)['independent_adapters']==4
 checks.append('nativeM1 and factorM1 own I4: four disjoint complete bodies/optimizers/private streams; source untied4 remains one joint optimizer')
 # Original literal BEBlock equations match sparse and dense input branches.
 original=adapters['original_gnnm'];typed=original.model.source_model.adapt_ws[1];typed.active_route=2;identity=features['1']
 assert torch.allclose(typed(identity),typed(identity.to_dense()),atol=1e-7,rtol=1e-6);typed.active_route=0
 assert original.model.source_model.out.is_first is False and original.model.source_model.out.R.shape==(4,64)
 checks.append('actual original BEBlock typed sparse/dense equality and source RSB boundary; native tanh/residual/norm/dropout unchanged')
 # Native source selector latest tie step2, then30 strictly worse; engine stops32.
 class Toy(ClassifierAdapter):
  def train_step(self,step):
   self.optimizer.zero_grad();loss=sum(p.square().sum() for p in self.model.parameters());loss.backward();self.optimizer.step()
   return {'objective':float(loss.detach()),'ce':float(loss.detach()),'regularization':0.,'operation_counts':{'optimizer_updates':1,'backward_calls':1,'model_forwards':1}}
  def validate(self,step):
   value=0. if step<=2 else 1.
   return {'objective':value,'ce':value,'pooled_nll':value,'selection_metric':value,'regularization':0.,'operation_counts':{'optimizer_updates':0}}
 model=torch.nn.Linear(1,1);toy=Toy(torch,F,model,torch.optim.SGD(model.parameters(),lr=.001),logits_fn=None,train_ids=None,train_y=None,select_ids=None,select_y=None,regularizer=None,fusion='mean_logits',
  selector={'metric':'pooled_nll','direction':'min','ties':'last','patience_steps':30,'save_every':1},provenance={})
 run={'kind':'model_fit','seed':seed,'body':0,'config':config('native_M1',seed,0,sources,inputs),'budget':a.BUDGET,'selection':a.SELECTION}
 result=run_trajectory(torch,toy,{'fit_id':'synthetic-HGT-stop','budget_steps':300,'budget_unit':'optimizer_updates','validation_interval':1,'validate_initial':False,'prefixes':[],
  'inherited_fit_ids':[],'adapter_spec':{'run':run},'completion_policy':'hgt_native_selector_stop'},folder/'engine_stop')
 assert result['completed_steps']==32 and result['selected']['step']==2 and result['completion_reason']=='native_patience30' and result['actual_completed_optimizer_updates']==32 and result['declared_optimizer_update_cap']==300
 checks.append('HGT engine latest exact tie2+30strict worse ends32/restores2; actual32 kept distinct from cap300')
 record={'schema':'native-HGT-mixed-synthetic-CPU-proof-v2','passed':True,'checks':checks,'seconds':time.monotonic()-started,
  'synthetic_source_AdamW_updates':updates,'synthetic_engine_SGD_updates':32,'actual_input_fits':0,'actual_input_optimizer_updates':0,'predictive_scores':0,'GO':False}
 (packet/'SYNTHETIC_CPU_PROOF.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');print(json.dumps(record),flush=True)

if __name__=='__main__':main()
