
import copy,gzip,json,sys
from pathlib import Path
packet=Path(__file__).resolve().parent;phase=packet.parent;sys.path.insert(0,str(packet/'runtime'))
from adapters import nsd as port
from engine import inspect_ownership,_equal_tree
import torch,numpy as np
torch.set_num_threads(1);assert not torch.cuda.is_initialized()
plan=json.loads((packet/'matrix/PLAN.json').read_text());s=json.loads(gzip.decompress((phase/plan['studies'][0]['path']).read_bytes()))
torch.manual_seed(681);x=torch.randn(12,3170);pairs=torch.tensor([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,0],[1,8],[2,9]])
edges=torch.cat((pairs,pairs.flip(1)),0).T.contiguous();graph={'x':x,'edge_index':edges};train={'ids':torch.arange(8),'y':torch.tensor([0,1]*4)};select={'ids':torch.arange(8,12),'y':torch.tensor([0,1,0,1])}
rows=[];maps=0
for arm in ('geometry_only_source','ordinary_M1','original_gnnm'):
 r=next(r for r in s['runs'] if r['kind']=='model_fit' and r['config']['nsd_arm']==arm);c=copy.deepcopy(r['config']);c['device']='cpu';B=c['nsd_sources'];e,f,args=port.factory(torch,phase,B,c,graph,'cpu');h=port.helpers(torch,phase,B);seed=c['initialization_seed'];h.seed_all(np,torch,seed);model=port.construct(torch,phase,B,e,f,c,seed);opt=port.optimizer(torch,model,e,c)
 if arm=='geometry_only_source':
  model.assert_ownership();fs=port.factors(model);assert all(v.requires_grad and torch.equal(v,torch.ones_like(v)) for v in fs);assert len({v.data_ptr() for v in fs})==len(fs)
 inspect_ownership(__import__('types').SimpleNamespace(model=model,optimizer=opt));rng=h.capture_rng(np,torch);before={'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'rng':copy.deepcopy(rng),'plain':port.plain_capture(torch,model)}
 def step():
  model.train()
  if arm=='geometry_only_source':
   counts={k:0 for k in ('train_forward_attempts','train_forwards_completed','backward_attempts','backwards_completed','Adam_attempts','Adam_steps_completed')}
   values=e.train.train_update(torch,model,opt,{'x':x,'train_index':train['ids'],'train_y':train['y']},counts);assert counts['backwards_completed']==4 and counts['Adam_steps_completed']==1
  elif arm=='original_gnnm':
   opt.zero_grad(set_to_none=True);values=[]
   for m in range(4):
    z=model.forward_route(m,x);loss=torch.nn.functional.cross_entropy(z[train['ids']],train['y']);(loss/4).backward();values.append(float(loss.detach()))
   opt.step()
  else:
   opt.zero_grad(set_to_none=True);loss=torch.nn.functional.nll_loss(model(x)[train['ids']],train['y']);loss.backward();opt.step();values=[float(loss.detach())]
  assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
  return values,{'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'rng':h.capture_rng(np,torch),'plain':port.plain_capture(torch,model)}
 values,after=step();maps+=1
 if arm=='geometry_only_source':assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in port.factors(model))
 model.load_state_dict(before['model']);opt.load_state_dict(before['optimizer']);h.restore_rng(np,torch,before['rng']);port.plain_restore(model,before['plain'],'cpu')
 again,state=step();maps+=1;assert _equal_tree(torch,values,again) and _equal_tree(torch,after,state)
 rows.append({'arm':arm,'literal_source_train_primitive':True,'finite_gradient':True,'live_factors_from_first_update':arm=='geometry_only_source','full_next_step_model_optimizer_RNG_plain_cache_replay':True,'synthetic_maps':2,'benchmark_arrays_read':0})
result={'status':'PASS','cases':rows,'synthetic_optimizer_maps':maps,'scientific_fits':0,'benchmark_arrays':0,'CUDA_compute':False,'SELECT_CAL_EVALUATE_scored':False};(packet/'CPU_QA.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
