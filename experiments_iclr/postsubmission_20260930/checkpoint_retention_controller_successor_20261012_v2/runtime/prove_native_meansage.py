"""One synthetic CPU engineering proof; no actual-input fits or scoring."""
import argparse,copy,hashlib,json,math,random,tempfile,time
from pathlib import Path
import torch
from torch.nn import functional as F
from adapters import meansage_native as a
from adapters.meansage_endpoints import fit_temperature,log_pool
from adapters.classifier import ClassifierAdapter
from engine import capture_checkpoint,restore_checkpoint,run_trajectory,_equal_tree,inspect_independence
from native_stopping import validate_native_run

def config(family,arm,seed,body,sources,inputs,prep):
 original=arm=='original_gnnm';members=4 if arm in ('S4','SA4','SM4','SX4','UX4','PV1','V1','SW4','C4','R4','original_gnnm') else 1
 return {'meansage_family':family,'meansage_arm':arm,'meansage_sources':sources,'architecture':'original_gnnm' if original else 'native_meansage_'+family,
  'backbone':'SAGE','width':128 if original else 32,'depth':2,'classes':8,'learning_rate':.001,'dropout':.4,'weight_decay':0,
  'source_optimizer_count':1,'physical_slow_bodies':4 if arm=='UX4' else 1,'members':members,'is_original_gnnm':original,
  'initialization_seed':seed+1000003*body,'source_TRAIN_forwards_per_update':4 if original else 1,'source_TRAIN_backwards_per_update':4 if original else 1,
  'source_complete_body_constructors':4 if arm in ('I4','IA4','WI4','UX4') else 1,
  'source_constructor_selection':'full_four_factory_choose_run_body' if arm in ('I4','IA4','WI4') else 'whole_single_source_factory',
  'historical_state_inherited':False,'fusion':'mean_logits' if original else 'mean_probabilities','node_inputs':inputs,
  'meansage_incidence':prep['incidence'] if family=='peer' else None,
  'meansage_tours':prep['tours'][str(seed)] if family=='tour' and arm not in ('P1','F1','I4','S4','original_gnnm') else None,'stage':'comparative'}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--project-root',required=True);parser.add_argument('--packet',required=True);args=parser.parse_args()
 root=Path(args.project_root).resolve();packet=root/args.packet;sources=json.loads((packet/'EXACT_SOURCE_BINDINGS.json').read_text())
 sources={k:{x:v for x,v in row.items() if x in ('path','bytes','sha256')} for k,row in sources.items()}
 torch.set_num_threads(2);torch.manual_seed(11897);torch.backends.cuda.matmul.allow_tf32=False
 started=time.monotonic();checks=[];seed=98301
 folder=packet/'synthetic_cpu_proof';folder.mkdir()
 n,f=16,7;x=(torch.arange(n*f).reshape(n,f)%3!=0).float();edges={(i,i) for i in range(n)}|{(i,i+1) for i in range(n-1)}|{(i+1,i) for i in range(n-1)}
 edge=torch.tensor(sorted(edges),dtype=torch.long).T;torch.save({'x':x,'edge_index':edge},folder/'graph.pt')
 torch.save({'ids':torch.arange(8),'y':torch.arange(8)},folder/'train.pt');torch.save({'ids':torch.arange(8,16),'y':torch.arange(8)},folder/'select.pt')
 inputs={k:a.receipt(root,folder/(k+'.pt')) for k in ('graph','train','select')}
 prep=a.prepare_graph(torch,root,inputs['graph'],sources,[seed],folder/'graph_only')
 checks.append('synthetic raw binary GRAPH-only incidence/four tours; private RNG unchanged')
 study={'study_id':'synthetic_native_meansage','stage':'engineering','dataset':'synthetic_photo','source':{'files':list(sources.values())},
  'data':{'roles':{'train':{'0':inputs['train']},'select':{'0':inputs['select']}}}}
 adapters=[]
 cases=[('peer',arm,3 if arm in ('I4','IA4') else 0) for arm in a.PEER_ARMS]+[('tour',arm,3 if arm in ('I4','WI4') else 0) for arm in a.TOUR_ARMS]+[('peer','original_gnnm',0)]
 for family,arm,body in cases:
  c=config(family,arm,seed,body,sources,inputs,prep);run={'run_id':family+'-'+arm,'kind':'model_fit','seed':seed,'body':body,'split':0,'config':c,'budget':a.BUDGET,'selection':a.SELECTION,'parents':[]}
  validate_native_run(run);adapter=a.make_adapter(torch=torch,project_root=root,study=study,run=run,device='cpu')
  outer=copy.deepcopy(adapter.streams.capture());adapter.model.train();z=adapter.model()
  loss=torch.stack([F.cross_entropy(v[adapter.train_ids],adapter.train_y) for v in z]).mean()+adapter.model.regularizer();loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.model.parameters())
  assert _equal_tree(torch,outer,adapter.streams.capture());assert z.shape==(c['members'],n,8)
  if not adapter.model.original:
   # Serializer proof with all native parameter moments populated, no step.
   for p in adapter.model.parameters():adapter.optimizer.state[p]={'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.ones_like(p)*.01}
   g=adapter.optimizer.state['tf1_global'];g['updates']=7;g['beta1_power'].copy_(torch.tensor(.9**8));g['beta2_power'].copy_(torch.tensor(.999**8))
   adapter.model.eval();metrics=adapter.validate(7);decision=adapter.observe_validation(7,metrics)
   payload=capture_checkpoint(torch,adapter,fit_id=run['run_id'],step=7,validation=metrics,decision=decision);path=folder/(run['run_id']+'.pt');torch.save(payload,path)
   saved=copy.deepcopy(adapter.streams.states)
   adapter.model.train();out1=adapter.model().detach();streams1=copy.deepcopy(adapter.streams.states)
   with torch.no_grad():next(adapter.model.parameters()).add_(.25)
   restore_checkpoint(torch,path,adapter,expected_fit_id=run['run_id']);assert _equal_tree(torch,saved,adapter.streams.states)
   assert adapter.optimizer.state['tf1_global']['updates']==7
   adapter.model.train();out2=adapter.model().detach();assert torch.equal(out1,out2);assert _equal_tree(torch,streams1,adapter.streams.states)
  adapters.append(adapter)
 checks.append('21 synthetic source/control factories; parameter counts/ties/disjoint-own selection; finite gradients; full native TF1Adam/RNG serializer; exact dropout replay')
 # Direct ordinary equation/clip/penalty check, full source unchanged.
 ordinary=adapters[0];ordinary.model.eval();body=ordinary.model.native.bodies[0];A=ordinary.model.fixed[1].to_dense();d=ordinary.model.fixed[2]
 def layer(v,L):
  z=(A@(v@L.aggregate_weight))/d+v@L.skip_weight+L.bias[0]
  return z/torch.linalg.vector_norm(z,dim=1,keepdim=True).clamp_min(1)
 expected=layer(F.relu(layer(x,body.layers[0])),body.layers[1]);actual=ordinary.model()[0]
 assert torch.allclose(actual,expected,rtol=2e-6,atol=2e-7)
 assert torch.equal(ordinary.model.regularizer(),.05*(body.layers[0].aggregate_weight.square().sum()+body.layers[0].skip_weight.square().sum()))
 checks.append('manual two-layer ordinary aggregate/skip/raw-X/native norm clip and first-layer penalty equality')
 # Complete live donor gradient credit after nonzero synthetic feedback map.
 sm=next(v for v in adapters if v.provenance['config']['meansage_arm']=='SM4');sm.model.eval();sm.optimizer.zero_grad()
 with torch.no_grad():sm.model.native.feedback_maps[0].Vmap.fill_(.01)
 out=sm.model();out[0].square().sum().backward();grad=sm.model.native.bodies[0].layers[0].aggregate_r.grad
 assert grad is not None and bool(grad[1:].abs().sum()>0)
 checks.append('SM4 live donor first-factor gradients reach receiver own logits; no detach')
 scan=a.source(root,sources['scan']);fn=scan.make_segmented_scan(torch);aa=torch.rand(9,3,dtype=torch.double,requires_grad=True)*.9;qq=torch.randn(9,3,dtype=torch.double,requires_grad=True);starts=torch.tensor([1,0,0,1,0,0,0,1,0],dtype=torch.bool)
 assert torch.autograd.gradcheck(lambda u,v:fn(u,v,starts),(aa,qq),eps=1e-6,atol=2e-5,rtol=2e-4)
 with torch.no_grad():
  state=torch.zeros(3,dtype=torch.double);rows=[]
  for i in range(9):state=(0 if starts[i] else aa[i])*state+qq[i];rows.append(state)
 assert torch.allclose(fn(aa,qq,starts),torch.stack(rows),atol=1e-12,rtol=1e-12)
 checks.append('source segmented affine scan forward and first derivative gradcheck; starts cut carry only')
 # Actual source TF1Adam kernel versus an independent float32 scalar recipe.
 cls=a.source(root,sources['author']).optimizer_class(torch);p=torch.nn.Parameter(torch.tensor([1.,-.5]));opt=cls([p]);ref=p.detach().clone();m=torch.zeros_like(ref);v=m.clone();b1=torch.tensor(.9);b2=torch.tensor(.999)
 for g in (torch.tensor([.3,-.7]),torch.tensor([-.2,.4])):
  p.grad=g.clone();m=.9*m+.1*g;v=.999*v+.001*g*g;ref=ref-.001*torch.sqrt(1-b2)/(1-b1)*m/(torch.sqrt(v)+1e-8);opt.step();b1.mul_(.9);b2.mul_(.999)
 assert torch.equal(p,ref) and opt.state['tf1_global']['updates']==2 and torch.equal(opt.state['tf1_global']['beta1_power'],b1) and torch.equal(opt.state['tf1_global']['beta2_power'],b2)
 checks.append('two synthetic source TF1Adam transitions; epsilon before correction; exact float32 global beta powers')
 # Engine source selector: step2 exact tie replaces1, then 50 strictly worse.
 model=torch.nn.Linear(1,1);optimizer=torch.optim.SGD(model.parameters(),lr=.001)
 class Toy(ClassifierAdapter):
  def train_step(self,step):
   self.optimizer.zero_grad();loss=sum(p.square().sum() for p in self.model.parameters());loss.backward();self.optimizer.step()
   return {'objective':float(loss.detach()),'ce':float(loss.detach()),'regularization':0.,'operation_counts':{'optimizer_updates':1,'backward_calls':1,'model_forwards':1}}
  def validate(self,step):return {'objective':0. if step<=2 else 1.,'ce':0. if step<=2 else 1.,'regularization':0.,'selection_metric':0. if step<=2 else 1.,'operation_counts':{'optimizer_updates':0}}
 toy=Toy(torch,F,model,optimizer,logits_fn=None,train_ids=None,train_y=None,select_ids=None,select_y=None,regularizer=None,fusion='mean_probabilities',selector={'metric':'objective','direction':'min','ties':'last','patience_steps':50,'save_every':1},provenance={})
 run={'run_id':'toy-native-stop','kind':'model_fit','seed':seed,'body':0,'split':0,'config':config('peer','P1',seed,0,sources,inputs,prep),'budget':a.BUDGET,'selection':a.SELECTION,'parents':[]}
 spec={'fit_id':run['run_id'],'budget_steps':100000,'budget_unit':'optimizer_updates','completion_policy':'native_selector_stop','validation_interval':1,'validate_initial':False,'prefixes':[],'inherited_fit_ids':[],'adapter_spec':{'run':run}}
 result=run_trajectory(torch,toy,spec,folder/'engine_stop')
 assert result['completed_steps']==52 and result['selected']['step']==2 and result['first_selector_stop']==52 and result['trajectory_operation_counts']['optimizer_updates']==52 and result['completion_reason']=='native_patience50'
 assert result['actual_completed_optimizer_updates']==52 and result['declared_optimizer_update_cap']==100000
 checks.append('engine completes at52 after latest exact tie step2+50 strictly worse; selected2 restored; actual52/cap100000 kept separate')
 # Synthetic CAL constant logits force exact ties; source latest500 must win.
 lp=torch.zeros(4,8,1);targets=torch.zeros(8,dtype=torch.long);cal=fit_temperature(torch,lp,targets,fusion='mean_probabilities')
 assert cal['selected_step']==500 and cal['scalar_optimizer_updates']==500 and cal['objective_forward_evaluations']==1001 and cal['temperature']==1.
 general=torch.tensor([[[5.,0.],[1.,0.]],[[0.,1.],[0.,5.]]],dtype=torch.double).log_softmax(-1);T=torch.tensor(math.log(2.),dtype=torch.double)
 before=log_pool(torch,general,T);after=(torch.logsumexp(general,dim=0)-math.log(2))/2;after=after.log_softmax(-1)
 assert not torch.allclose(before,after)
 checks.append('source before-member-pool CAL500/latest ties/1001 objectives; distinct from post-pool scalar temperature')
 record={'schema':'native-meansage-peer-tour-synthetic-cpu-proof-v1','passed':True,'checks':checks,'seconds':time.monotonic()-started,
  'actual_input_optimizer_updates':0,'actual_input_fits':0,'predictive_scores':0,'synthetic_TF1Adam_updates':2,'synthetic_engine_SGD_updates':52,'synthetic_scalar_CAL_updates':500,'source_model_optimizer_commits':0,'GO':False}
 (packet/'SYNTHETIC_CPU_PROOF.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');print(json.dumps(record),flush=True)

if __name__=='__main__':main()
