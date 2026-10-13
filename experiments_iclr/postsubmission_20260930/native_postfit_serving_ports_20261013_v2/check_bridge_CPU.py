"""One targeted new adapter/custody bridge gate; inherits closed source kernels."""
import copy,gzip,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;sys.path.insert(0,str(P/'runtime'))
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from adapters.postfit import DirectHead,AffineHead,ServingHead,fold_assignment
from engine import inspect_ownership,capture_checkpoint,restore_checkpoint,_metrics
from run_study import load_factory
from postfit_semantics import paid_feature_parents
B=json.loads((P/'SOURCE_BINDINGS.json').read_text());specs=[json.loads(gzip.decompress(q.read_bytes())) for q in sorted((P/'matrix/studies').glob('*.gz'))];cases=[];maps=0;out=P/'synthetic_bridge_qa_01';out.mkdir(exist_ok=False)
def rows(family):return [r for s in specs for r in s['runs'] if r['kind']=='model_fit' and r['config']['postfit_family']==family]
def save_restore(a,r,step,name):
 inspect_ownership(a);v=_metrics(torch,a.validate(step));d=a.observe_validation(step,v);payload=capture_checkpoint(torch,a,fit_id=r['run_id'],step=step,validation=v,decision=d);q=out/(name+'.pt');torch.save(payload,q);before={k:v.clone() for k,v in a.model.state_dict().items()}
 with torch.no_grad():
  for p in a.model.parameters():
   if p.requires_grad:p.add_(.13)
 restore_checkpoint(torch,q,a,expected_fit_id=r['run_id']);assert all(torch.equal(v,a.model.state_dict()[k]) for k,v in before.items());return q
# Direct bridge uses balanced zero features: no scientific numerical problem.
N=594;train=torch.arange(580);select=torch.arange(580,N);base={'H':torch.zeros(4,N,512),'Z':torch.zeros(4,N,10),'train_ids':train,'train_y':train%10,'select_ids':select,'select_y':select%10}
factor={'weight':torch.zeros(10,512),'r':torch.ones(4,512),'s':torch.ones(4,10),'bias':torch.zeros(10)}
for condition in ('BE_factor_refit','BE_full_refit','ordinary_full_refit','single_full_refit'):
 r=next(r for r in rows('direct') if r['config']['postfit_phase']=='head' and r['config']['postfit_condition']==condition);d=copy.deepcopy(base)
 if condition.startswith('BE_'):d['parts']=[{'head_factorized':True,'head':factor}]
 else:
  members=1 if condition=='single_full_refit' else 4;d['H']=d['H'][:members];d['Z']=d['Z'][:members];d['parts']=[{'head_factorized':False,'head':{'weight':torch.zeros(10,512),'bias':torch.zeros(10)}} for _ in range(members)]
 a=DirectHead(torch,R,d,r,{'synthetic':condition});inspect_ownership(a);assert a.predict_rows(select).shape==(r['config']['members'],14,10)
 if condition=='BE_factor_refit':
  raw=a.train_step(1);maps+=1;_metrics(torch,raw);assert raw['operation_counts']['optimizer_updates']==1 and raw['operation_counts']['source_LBFGS_iterations']<=1000 and raw['operation_counts']['source_LBFGS_closures_including_final_check']<=50000;save_restore(a,r,1,'direct_source_solver_bridge');cases.append({'case':'one registered LBFGS source injection and checkpoint restore','optimizer_maps':1,'source_solver_receipt':a.last_receipt})
 else:cases.append({'case':'unstepped registered '+condition+' source start','optimizer_maps':0})
 del a
# Literal zero-delta constructors and one fresh affine update exercise new state ownership.
g=torch.Generator().manual_seed(83013);N=20
for condition in ('BE_head_continuation','common_affine_residual','private_affine_residual'):
 r=next(r for r in rows('affine') if r['config']['postfit_phase']=='head' and r['config']['postfit_bank_arm']=='E4' and r['config']['postfit_condition']==condition);head={'weight':torch.randn(10,128,generator=g),'r':torch.ones(4,128),'s':torch.ones(4,10),'bias':torch.zeros(10)};d={'H':torch.randn(4,N,128,generator=g)*.01,'Z':torch.randn(4,N,10,generator=g),'train_ids':torch.arange(12),'train_y':torch.arange(12)%10,'select_ids':torch.arange(12,N),'select_y':torch.arange(12,N)%10,'parts':[{'head_members':4,'head':head}]};a=AffineHead(torch,R,d,r,{'synthetic':condition});inspect_ownership(a)
 if condition=='private_affine_residual':a.train_step(1);maps+=1;save_restore(a,r,1,'affine_delta_bridge')
 cases.append({'case':'literal zero-delta '+condition+' ownership'+(' and one update/restore' if condition=='private_affine_residual' else ''),'optimizer_maps':int(condition=='private_affine_residual')});del a
# New fold adapter plumbing is verified with one map per source operator; no500-update repeats.
for rule in ('temperature_global','temperature_member','reliability_self','linear_stacking','reliability_graph'):
 r=next(r for r in rows('serving') if r['config']['postfit_phase']=='head' and r['config']['postfit_bank_arm']=='ordinary_I4' and r['config']['postfit_rule']==rule and r['config']['postfit_fold']==0);z=torch.randn(4,N,10,generator=g);neighbor=z.double().log_softmax(-1).exp().roll(1,1);d={'Z':z,'neighbor':neighbor,'train_ids':torch.arange(10),'train_y':torch.arange(10)%10,'select_ids':torch.arange(10,N),'select_y':torch.arange(10,N)%10,'parts':[]};a=ServingHead(torch,R,d,r,{'synthetic':rule});before={k:v.clone() for k,v in a.model.state_dict().items()};raw=a.train_step(1);maps+=1;_metrics(torch,raw);assert raw['operation_counts']['optimizer_updates']==1;save_restore(a,r,500,'fold_'+rule);cases.append({'case':'one-map fold '+rule+' registered ownership/restore','optimizer_maps':1,'no500update_source_kernel_repeat':True});del a
ids=torch.arange(10,30);fold=fold_assignment(torch,ids);query=torch.tensor([10,17,29,30,75]);routed=fold_assignment(torch,ids,query);assert torch.equal(routed[:3],fold[torch.tensor([0,7,19])]) and torch.equal(routed[3:],(query[3:]+11709)%5);cases.append({'case':'SELECT source folds and label-free unseen ID routing','optimizer_maps':0})
# Ensure frozen factories import in the actual namespaced loader.
for family in ('direct','affine','serving'):
 r=rows(family)[0];factory=load_factory(R,r['adapter']);assert callable(factory);cases.append({'case':'actual frozen factory import '+family,'optimizer_maps':0})
assert maps==7
report={'schema':'targeted-native-postfit-bridge-CPU-v1','status':'passed','cases':cases,'new_optimizer_step_calls':maps,'new_LBFGS_trajectories':1,'new_AdamW_maps':1,'new_Adam_maps':5,'closed_source_kernel_or_training_gates_repeated':False,'actual_input_arrays_opened':0,'actual_input_forwards':0,'scientific_fits':0,'CUDA_compute':0,'GO':False};(P/'BRIDGE_CPU_QA.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','cases':len(cases),'new_optimizer_calls':maps,'GO':False}))
